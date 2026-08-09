# -*- coding: utf-8 -*-
"""generate_data_rhofold.py — 独立 circRNA 数据生成管线 (RhoFold+ MSA + OpenMM)

Level 0: ViennaRNA 粗筛 (vienna_pair_probs) → 近程/远端配对
Level 1: 分段 RhoFold+ MSA (常驻引擎, 分项策略) → 拼装 P 坐标
Level 1.5: OpenMM CG 弛豫 (平滑分段拼装边界) → 预热坐标
Level 2: OpenMM 3-bead CG 退火 (BSJ 闭合 + 精修) → 精修 P 坐标
输出: P 坐标 + 特征 (features), 与 generate_training_data.py 兼容

与 run_2013nt.py (验证管线) 分离: 本管线只做数据生成, 不跑 isRNAcirc/Level 3-5.
32 worker 并行, 每条序列独立进程 (RhoFold 子进程 + OpenMM CPU).

用法:
  C:/ana/envs/comfyui/python.exe generate_data_rhofold.py --n-workers 8 --n-samples 100 --max-len 2000 --min-len 50
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ── 路径配置 (与 generate_training_data.py 一致) ──
_ROOT = Path(__file__).resolve().parent
sys_path_inserted = False


def _ensure_src():
    """确保 src 在 sys.path (RhoFold/OpenMM 复用)."""
    global sys_path_inserted
    if not sys_path_inserted:
        import sys
        sys.path.insert(0, str(_ROOT / "src"))
        sys_path_inserted = True


# 数据/模型路径可用环境变量覆盖 (便于迁移到 autodl 4090)
_FASTA_GZ = Path(os.environ.get(
    "CIRCBASE_FASTA", "C:/Users/颜子壹/Documents/circbase_seqs.fa.gz"))
_DEFAULT_OUT = Path(os.environ.get("DATA_OUT", "C:/tmp/test_isrna/rhofold_data"))


# ═══════════════════════════════════════════════════════════
#  FASTA
# ═══════════════════════════════════════════════════════════
def load_fasta_gz(fp, max_len=2000, min_len=50):
    seqs, h, s = [], "", ""
    with gzip.open(fp, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if s and min_len <= len(s) <= max_len: seqs.append((h, s))
                h = line[1:].split("|")[0]; s = ""
            elif line: s += line.upper().replace("T", "U")
    if s and min_len <= len(s) <= max_len: seqs.append((h, s))
    return seqs


# ═══════════════════════════════════════════════════════════
#  Level 0: ViennaRNA 粗筛
# ═══════════════════════════════════════════════════════════
def level0_pairs(seq, threshold=0.5):
    """Level 0: 近程/远端配对."""
    from torusfold.scheme2.refine import vienna_pair_probs
    from torusfold.scheme2.pair_graph import build_full_pair_graph, extract_stem_blocks
    pairs, bpp = vienna_pair_probs(seq, threshold)
    _, scan_pairs, far_pairs = build_full_pair_graph(seq, pairs, do_scan=True)
    stem_blocks = extract_stem_blocks(pairs, scan_pairs)
    return pairs, far_pairs, stem_blocks


# ═══════════════════════════════════════════════════════════
#  Level 1: 分段 RhoFold+ MSA (常驻引擎, 只加载一次模型)
# ═══════════════════════════════════════════════════════════
_RHOFOLD_ROOT = Path(os.environ.get(
    "RHOFOLD_ROOT", "C:/Users/颜子壹/deploy/IGEM集成方案/tools/RhoFold"))
_RHOFOLD_CKPT = Path(os.environ.get(
    "RHOFOLD_CKPT", str(_RHOFOLD_ROOT / "pretrained" / "rhofold_pretrained_params.pt")))


class RhoFoldEngine:
    """常驻 RhoFold+ 推理引擎 (进程内加载一次模型).

    worker 进程启动时实例化, 处理该 worker 的所有序列/段.
    替代 rhofold_wrapper.rhofold_predict_chunk 的每次子进程加载 (12-120s/次).
    """

    def __init__(self, device: Optional[str] = None, verbose: bool = False):
        import torch
        sys_path = f"{_RHOFOLD_ROOT}"
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from rhofold.rhofold import RhoFold
        from rhofold.config import rhofold_config

        self.verbose = verbose
        t0 = time.time()
        self.model = RhoFold(rhofold_config)
        sd = torch.load(str(_RHOFOLD_CKPT), map_location="cpu")
        self.model.load_state_dict(sd["model"])
        dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # 混合精度: autocast 自动决定哪些层用 fp16 (embedding 保持 fp32, matmul 用 fp16)
        self.use_fp16 = dev.type == "cuda"
        if self.use_fp16 and verbose:
            print(f"  [RhoFoldEngine] 混合精度模式 (autocast)")

        self.model = self.model.to(dev).eval()
        self.device = dev
        if verbose:
            print(f"  [RhoFoldEngine] 模型加载完成: {time.time()-t0:.1f}s, device={dev}")

    def predict(self, seq: str, ss: str, out_dir: str, name: str,
                msa_path: Optional[str] = None) -> np.ndarray:
        """预测单个序列/段的 P 坐标 (不写子进程)."""
        import torch
        from rhofold.utils.alphabet import get_features

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 写 FASTA (转全大写, 避免 remove_insertions 删小写字符)
        upper_seq = seq.upper()
        fa_path = out_dir / f"{name}.fa"
        with open(fa_path, "w") as f:
            f.write(f">{name}\n{upper_seq}\n")
        msa_src = str(msa_path if msa_path else fa_path)

        if self.verbose:
            mode = "MSA" if msa_path else "SINGLE"
            print(f"    [{name}] RhoFold 模式: {mode}")

        fea = get_features(str(fa_path), msa_src)
        # 混合精度: autocast 自动处理 dtype 转换 (整数保持 Long, 浮点转 fp16)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                tokens=fea["tokens"].to(self.device),
                rna_fm_tokens=fea["rna_fm_tokens"].to(self.device),
                seq=fea["seq"],
            )
        frames = out[-1]["frames"][0, 0].data.cpu().numpy()
        p_coords = frames[:, 4:7]
        return np.asarray(p_coords, dtype=np.float64)


def level1_rhofold(seq, ss, out_dir, engine: Optional[RhoFoldEngine] = None,
                   max_seg_len=200, overlap=20, use_msa=True,
                   rfam_dir="", rfam_cm="", msa_blocks=None):
    """Level 1: 分段 RhoFold+ (常驻引擎) → 拼装 P 坐标.

    engine 提供时用常驻引擎 (模型只加载一次); 否则走 segmented_vfold3d_pipeline (子进程).
    """
    if engine is None:
        from torusfold.scheme2.segmented_vfold3d import segmented_vfold3d_pipeline
        coords, pdb_path, confs, unc = segmented_vfold3d_pipeline(
            seq, ss, out_dir,
            max_seg_len=max_seg_len, overlap=overlap,
            n_candidates=1, use_ensemble=False,
            use_rhofold=True, use_trrosetta=False,
            use_msa=use_msa, rfam_cm=rfam_cm, rfam_dir=rfam_dir,
            msa_blocks=msa_blocks,
        )
        return np.asarray(coords, dtype=np.float64)

    # ── 常驻引擎路径: 分项策略 ──
    # <300nt: 整条直接预测 (单段, 避免分段拼装误差)
    # ≥300nt: 复用预测管线分段 (split_sequence + 每段引擎预测 + 拼装 + 平滑)
    from torusfold.scheme2.segmented_vfold3d import (
        split_sequence, confidence_weighted_assemble, spline_smooth_dihedral,
    )
    L = len(seq)
    single_cutoff = 300  # <300nt 整条预测
    if L < single_cutoff:
        # 整条直接预测
        msa_path = None
        if use_msa:
            try:
                from torusfold.scheme2.segmented_vfold3d import _resolve_chunk_msa
                seg0 = {"seq": seq, "ss": ss, "start": 0, "end": L}
                msa_path = _resolve_chunk_msa(seg0, 0, Path(out_dir), out_dir,
                                              rfam_cm=rfam_cm, rfam_dir=rfam_dir)
            except Exception:
                msa_path = None
        if engine.verbose:
            print(f"  Level1: {L}nt < {single_cutoff}, 整条直接预测")
        coords = engine.predict(seq, ss, str(Path(out_dir) / "full"),
                                "full", msa_path=msa_path)
        return np.asarray(coords, dtype=np.float64)

    segments = split_sequence(seq, ss, max_seg_len, overlap, msa_blocks=msa_blocks)
    if engine.verbose:
        print(f"  Level1: {L}nt ≥ {single_cutoff}, {len(segments)} 段, "
              f"长度 {[s['end']-s['start'] for s in segments]}")

    segment_coords = []
    confs = []
    for idx, seg in enumerate(segments):
        seg_dir = Path(out_dir) / f"seg_{idx}"
        msa_path = None
        if use_msa:
            try:
                from torusfold.scheme2.segmented_vfold3d import _resolve_chunk_msa
                msa_path = _resolve_chunk_msa(seg, idx, seg_dir, out_dir,
                                              rfam_cm=rfam_cm, rfam_dir=rfam_dir)
            except Exception:
                msa_path = None
        coords = engine.predict(seg["seq"], seg["ss"], str(seg_dir), f"seg_{idx}",
                                msa_path=msa_path)
        segment_coords.append(coords)
        confs.append(0.7)

    full_coords = confidence_weighted_assemble(segment_coords, segments, confs, L)
    boundaries = [seg["end"] for seg in segments[:-1]]
    full_coords = spline_smooth_dihedral(full_coords, boundaries)
    return np.asarray(full_coords, dtype=np.float64)


# ═══════════════════════════════════════════════════════════
#  Level 1.5: CG 弛豫 (平滑分段拼装边界)
# ═══════════════════════════════════════════════════════════
def _openmm_cg_relax_pre(
    p_coords: np.ndarray,
    pairs: list,
    n_threads: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Level 1.5: 短 MD CG 弛豫, 平滑分段拼装边界不连续.

    镜像 prediction 管线 (isrnaclong.py) 的 Level 1.5 实现:
    _build_3bead_system_gpu(pair_scale=0.5, bsj_k_scale=0.3) + 最小化 + 2000 step 短 MD.
    主要消除分段拼装的接缝处坐标不连续, 为 Level 2 退火预热.

    Args:
        p_coords: (L,3) Level 1 拼装后的 P 坐标 (Å)
        pairs: 配对列表 [(i,j,score)]

    Returns:
        (L,3) 弛豫后的 P 坐标 (Å), 失败时原样返回
    """
    import openmm as mm
    from openmm import Platform as _Plat
    from openmm import LangevinMiddleIntegrator as _LMI
    from openmm.app import Simulation as _Sim
    from torusfold.scheme2.openmm_gpu_refiner import (
        _build_3bead_system_gpu, _create_3bead_topology,
        _generate_compact_coords, _sanitize_p_coords,
    )

    L = len(p_coords)
    if L < 3:
        return p_coords

    try:
        relax_coords = _sanitize_p_coords(p_coords.copy())
        # 只检查 P-P 键长异常才替换 (环状结构首末距天然可大, 用首末距会误判)
        avg_pp = 0.0
        if L > 1:
            ppd = np.linalg.norm(np.diff(relax_coords, axis=0), axis=1)
            avg_pp = float(np.mean(ppd[:min(L - 1, 500)]))
        if (not np.isfinite(avg_pp)) or avg_pp > 20.0 or avg_pp < 1.0:
            relax_coords = _generate_compact_coords(L, pairs)

        system, coords_nm, pf, sf, bjf, bjg = _build_3bead_system_gpu(
            relax_coords, pairs, pair_scale=0.5, bsj_k_scale=0.3)
        topo = _create_3bead_topology(L)
        integrator = _LMI(300 * mm.unit.kelvin, 1.0 / mm.unit.picosecond,
                          0.002 * mm.unit.picosecond)
        plat = _Plat.getPlatformByName("CPU")
        sim = _Sim(topo, system, integrator, plat, {"CpuThreads": str(n_threads)})
        sim.context.setPositions(coords_nm * mm.unit.nanometer)

        sim.minimizeEnergy(
            tolerance=100.0 * mm.unit.kilojoules_per_mole / mm.unit.nanometer,
            maxIterations=1000)
        sim.step(2000)  # 4ps 短 MD

        state = sim.context.getState(getPositions=True, getEnergy=True)
        e_relax = state.getPotentialEnergy()._value
        pos = state.getPositions(asNumpy=True)._value  # nm
        p_relaxed = np.asarray(pos[0::3]) * 10.0  # nm → Å, P beads

        if len(p_relaxed) == L:
            if verbose:
                print(f"    L1.5 弛豫: E={e_relax:.0f} kJ/mol")
            return p_relaxed
        return p_coords
    except Exception as e:
        if verbose:
            print(f"    L1.5 弛豫失败: {e}, 用原始坐标")
        return p_coords


# ═══════════════════════════════════════════════════════════
#  Level 2: OpenMM 3-bead CG 退火 (close + 精修)
# ═══════════════════════════════════════════════════════════
def _openmm_cg_relax(
    p_coords: np.ndarray,
    pairs: list,
    n_anneal: int = 500,
    n_threads: int = 1,
    bsj_k_scale: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """纯 OpenMM 3-bead CG 退火精修 (BSJ 闭合 + 配对 + 退火).

    替代 isRNAcirc.exe 的 LAMMPS close+REMD.
    复用 openmm_gpu_refiner 的 3-bead 力场 (与验证管线 Level 1.5 同源).

    Returns:
        (refined_p_coords_ang, final_energy) — (L,3) Å
    """
    import openmm as mm
    from openmm import Platform as _Plat
    from openmm import LangevinMiddleIntegrator as _LMI
    from openmm.app import Simulation as _Sim
    from torusfold.scheme2.openmm_gpu_refiner import (
        _build_3bead_system_gpu, _create_3bead_topology, _run_annealing,
    )

    p_coords = np.asarray(p_coords, dtype=np.float64)
    L = len(p_coords)
    if L < 3:
        return p_coords, 0.0

    system, coords_nm, pf, sf, bjf, bjg = _build_3bead_system_gpu(
        p_coords, pairs, pair_scale=1.0, bsj_k_scale=bsj_k_scale)
    topo = _create_3bead_topology(L)
    integrator = _LMI(300 * mm.unit.kelvin, 1.0 / mm.unit.picosecond,
                      0.002 * mm.unit.picosecond)
    plat = _Plat.getPlatformByName("CPU")
    sim = _Sim(topo, system, integrator, plat, {"CpuThreads": str(n_threads)})
    sim.context.setPositions(coords_nm * mm.unit.nanometer)

    try:
        sim.minimizeEnergy(maxIterations=1000)
    except Exception:
        pass

    e_final, pos_nm = _run_annealing(
        sim, pf, bjf, bjg, L, n_anneal=n_anneal, verbose=False)
    pos_p = np.asarray(pos_nm)[0::3] * 10.0  # P beads, nm → Å

    # 键长校正: 退火后 P-P 键长可能被拉伸, 缩放到标准 A-form 5.9Å
    if L > 1:
        _pp = np.linalg.norm(np.diff(pos_p, axis=0), axis=1)
        _mean = float(_pp.mean())
        if 0.1 < _mean < 20.0 and abs(_mean - 5.9) > 0.1:
            pos_p = pos_p * (5.9 / _mean)
    return pos_p, e_final


def _dotbracket_to_pairs_list(ss: str) -> list:
    """从 dot-bracket 提取配对列表 [(i,j,1.0)] (近程 + 远端)."""
    pairs, stack, stack_sq = [], [], []
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')' and stack:
            j = stack.pop()
            pairs.append((j, i, 1.0))
        elif ch == '[':
            stack_sq.append(i)
        elif ch == ']' and stack_sq:
            j = stack_sq.pop()
            pairs.append((j, i, 1.0))
    return pairs


# ═══════════════════════════════════════════════════════════
#  特征提取 (与 generate_training_data.py 兼容)
# ═══════════════════════════════════════════════════════════
def read_pdb_p(pdb):
    cs = []
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                cs.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(cs) if len(cs) >= 3 else None


def features(coords, fp):
    L = len(coords)
    f = np.zeros(11, dtype=np.float32)
    if L < 3 or not fp:
        return f
    ds = [np.linalg.norm(coords[i] - coords[j]) for i, j in fp if i < L and j < L]
    if ds:
        f[0], f[1], f[2], f[3] = np.mean(ds), np.std(ds), np.min(ds), np.max(ds)
    nc = sum(1 for i in range(L) for j in range(i + 5, L) if np.linalg.norm(coords[i] - coords[j]) < 3.0)
    nn = sum(1 for i in range(L) for j in range(i + 5, L) if 3.0 <= np.linalg.norm(coords[i] - coords[j]) < 8.0)
    f[4] = min(nc / max(L, 1), 1.0)
    f[5] = min(nn / max(L * (L - 1) // 2, 1), 1.0)
    c = coords.mean(0)
    rog = np.sqrt(np.mean(np.sum((coords - c) ** 2, 1)))
    f[6] = min(rog / max(0.35 * L, 1.0), 3.0)
    f[7] = sum(1 for i, j in fp if i < L and j < L and np.linalg.norm(coords[i] - coords[j]) < 15.0) / max(len(fp), 1)
    f[8] = np.linalg.norm(coords[0] - coords[-1]) / (L * 5.9 + 1e-8)
    f[9] = np.log(L + 1) / 10.0
    f[10] = len(fp) / max(L, 1)
    return f


# ═══════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════
def _predict_ss(seq):
    """ViennaRNA 二级结构预测 (带 fallback)."""
    try:
        from ViennaRNA import RNA
        return RNA.fold_compound(seq).mfe()[0]
    except Exception:
        pass
    try:
        import viennaRNA
        ss, _ = viennaRNA.RNA.fold(seq)
        return ss
    except Exception:
        pass
    n = len(seq)
    ss = ["."] * n
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    for i in range(n):
        if ss[i] != ".":
            continue
        for j in range(min(i + 40, n - 1), i + 3, -1):
            if ss[j] != ".":
                continue
            if seq[i] in comp and comp[seq[i]] == seq[j] and all(ss[k] == "." for k in range(i + 1, j)):
                ss[i] = "("
                ss[j] = ")"
                break
    return "".join(ss)


def _get_far_pairs(ss, gap=30):
    pairs, stk = [], []
    for i, ch in enumerate(ss):
        if ch == "(":
            stk.append(i)
        elif ch == ")" and stk:
            pairs.append((stk.pop(), i))
    return [(i, j) for i, j in pairs if abs(j - i) > gap]


def _l1_worker_producer(args):
    """Level 1 生产者: 常驻 RhoFold 引擎, 处理任务列表.

    单进程顺序执行 (单 GPU 上顺序跑最快, 不存在并行加速).
    每条序列做完 Level 0+1 后把结果塞队列, 由 L2 消费者取走.

    Args:
        args: (tasks, work_root, max_seg_len, overlap, rfam_dir, queue)
    """
    tasks, work_root, max_seg_len, overlap, rfam_dir, queue = args
    _ensure_src()

    # 显式初始化 CUDA (Windows spawn 下可能未初始化)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(0)
            _ = torch.zeros(1).cuda()  # 触发 CUDA context 初始化
            print(f"  [L1-GPU] CUDA 初始化成功: {torch.cuda.get_device_name(0)}")
        else:
            print("  [L1-GPU] CUDA 不可用, 回退 CPU 模式 (会很慢)")
    except Exception as e:
        print(f"  [L1-GPU] CUDA 初始化失败: {e}")

    # RhoFold 引擎加载一次, 整批序列复用
    engine = None
    try:
        engine = RhoFoldEngine(verbose=False)
        print("  [L1-GPU] RhoFold 引擎加载成功")
    except Exception as e:
        print(f"  [L1-GPU] RhoFold 引擎加载失败: {e}, 回退子进程模式")
        engine = None

    for (idx, header, seq) in tasks:
        try:
            _l1_produce_one(
                idx, header, seq, work_root,
                max_seg_len, overlap, rfam_dir, engine, queue)
        except Exception as e:
            print(f"  [s{idx:05d}] Level1 生产者异常: {e}")
            queue.put(None)  # 失败标记, 保证队列恒有 total 条
        finally:
            # 每条序列后释放显存 (RhoFold 中间激活的 GPU 缓存)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _l1_produce_one(idx, header, seq, work_root,
                    max_seg_len, overlap, rfam_dir, engine, queue):
    """Level 0 + Level 1 (单个样本), 结果塞队列."""
    name = f"s{idx:05d}"
    wdir = Path(work_root) / name
    wdir.mkdir(parents=True, exist_ok=True)

    # Level 0: 二级结构 + 配对
    ss = _predict_ss(seq)
    if len(ss) != len(seq):
        ss = ".".join(seq)
    fp = _get_far_pairs(ss)
    pairs_near = _dotbracket_to_pairs_list(ss)

    # Level 1: 分段 RhoFold+ MSA (常驻引擎, 或子进程 fallback)
    l1_dir = wdir / "l1_rhofold"
    try:
        coords_vfold = level1_rhofold(
            seq, ss, str(l1_dir), engine=engine,
            max_seg_len=max_seg_len, overlap=overlap,
            use_msa=True, rfam_dir=rfam_dir,
        )
    except Exception as e:
        print(f"  [{name}] Level1 RhoFold 失败: {e}")
        queue.put(None)
        return

    if coords_vfold is None or len(coords_vfold) != len(seq):
        print(f"  [{name}] RhoFold 坐标异常: {None if coords_vfold is None else coords_vfold.shape}")
        queue.put(None)
        return

    # 配对列表 (近程 + 远端), L1.5 和 L2 共用
    all_pairs = pairs_near + [(int(i), int(j), 1.0) for (i, j) in fp]
    seen = set()
    all_pairs = [p for p in all_pairs
                 if not (p[0], p[1]) in seen and not seen.add((p[0], p[1]))]

    queue.put({
        "idx": idx, "header": header, "seq": seq, "ss": ss,
        "far_pairs": fp, "all_pairs": all_pairs,
        "coords_vfold": coords_vfold,
        "wdir": str(wdir), "name": name,
    })


def _l2_consumer(queue, done_queue, total, n_anneal, work_root):
    """Level 2 消费者: 单进程从队列取 Level 1 结果, OpenMM 精修.

    Args:
        queue: Level 1 → Level 2 任务队列
        done_queue: L2 → main 完成队列 (聚合进度)
        total: 总任务数
        n_anneal: 退火步数
        work_root: 输出根目录

    Returns:
        None (结果通过 done_queue 返回)
    """
    _ensure_src()
    results = []
    for _ in range(total):
        job = queue.get()
        if job is None:
            done_queue.put(None)
            continue
        try:
            r = _l2_process_one(job, n_anneal, work_root)
        except Exception as e:
            name = job.get("name", "?")
            print(f"  [{name}] Level2 消费者异常: {e}")
            r = None
        done_queue.put(r)
        if r is not None:
            results.append(r)
    return results


def _l2_process_one(job, n_anneal, work_root):
    """Level 1.5 + Level 2: CG 弛豫 + BSJ 闭合退火 + 特征.

    在单线程消费者进程里执行 (OpenMM CPU).
    """
    idx = job["idx"]
    header = job["header"]
    seq = job["seq"]
    ss = job["ss"]
    fp = job["far_pairs"]
    all_pairs = job["all_pairs"]
    coords_vfold = job["coords_vfold"]
    wdir = Path(job["wdir"])
    name = job["name"]

    # Level 1.5: CG 弛豫 (平滑分段拼装边界, 为 Level 2 预热)
    try:
        coords_vfold = _openmm_cg_relax_pre(
            coords_vfold, all_pairs, n_threads=1, verbose=False)
    except Exception as e:
        print(f"  [{name}] Level1.5 弛豫跳过: {e}")

    # Level 2: OpenMM close + 精修
    try:
        coords_refined, energy = _openmm_cg_relax(
            coords_vfold, all_pairs, n_anneal=n_anneal, n_threads=1)
    except Exception as e:
        print(f"  [{name}] Level2 OpenMM 失败: {e}")
        return None

    if coords_refined is None or len(coords_refined) != len(seq):
        return None

    feat = features(coords_refined, fp)
    np.save(str(wdir / f"{name}_p.npy"), coords_refined)
    np.save(str(wdir / f"{name}_rhofold_p.npy"), coords_vfold)

    return {
        "header": header, "seq": seq, "ss": ss,
        "far_pairs": fp, "features": feat.tolist(),
        "n_P": len(coords_refined),
        "energy_openmm": float(energy),
        "pdb": str(wdir / f"{name}_refined.pdb"),
        "npy": str(wdir / f"{name}_p.npy"),
        "npy_rhofold": str(wdir / f"{name}_rhofold_p.npy"),
    }


# ═══════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════
def main():
    import gzip
    pa = argparse.ArgumentParser()
    pa.add_argument("--n-workers", type=int, default=8)
    pa.add_argument("--n-samples", type=int, default=100)
    pa.add_argument("--max-len", type=int, default=2000)
    pa.add_argument("--min-len", type=int, default=50)
    pa.add_argument("--max-seg-len", type=int, default=200)
    pa.add_argument("--overlap", type=int, default=20)
    pa.add_argument("--n-anneal", type=int, default=500)
    pa.add_argument("--rfam-dir", default="")
    pa.add_argument("--output", default=str(_DEFAULT_OUT))
    pa.add_argument("--resume", action="store_true")
    a = pa.parse_args()
    _ensure_src()

    data = Path(a.output) / "rhofold_data"
    data.mkdir(parents=True, exist_ok=True)
    print(f"loading {a.min_len}-{a.max_len}nt...")
    alls = load_fasta_gz(str(_FASTA_GZ), a.max_len, a.min_len)
    print(f"  candidates: {len(alls)}")
    np.random.seed(42)
    if len(alls) > a.n_samples:
        idx = np.random.choice(len(alls), a.n_samples, replace=False)
        seqs = [alls[i] for i in idx]
    else:
        seqs = alls
    print(f"  sampled: {len(seqs)}")
    if a.resume:
        done = {d.name for d in data.iterdir() if (d / f"{d.name}_p.npy").exists()}
        seqs = [(h, s) for i, (h, s) in enumerate(seqs) if f"s{i:05d}" not in done]
        print(f"  remaining: {len(seqs)}")
    if not seqs:
        print("all done!")
        return

    # 流水线: 1 L1 GPU Worker (顺序 RhoFold) + 1 L2 CPU Consumer (OpenMM)
    # 单 GPU 上顺序跑 RhoFold 最快, 不存在并行加速; 8 worker 只会争抢 GPU + 炸内存
    tasks = [(i, h, s) for i, (h, s) in enumerate(seqs)]
    n_w = 1  # L1 GPU worker 固定 1 个 (单卡)

    # Manager().Queue 走代理, 可跨进程 (Windows spawn) 共享
    manager = Manager()
    queue = manager.Queue(maxsize=4)  # 小缓冲, L1 做完一条 L2 就取走
    done_queue = manager.Queue()

    # L1 全部任务给一个 worker
    all_tasks_args = (tasks, str(data), a.max_seg_len, a.overlap, a.rfam_dir, queue)

    est = len(tasks) * 20 / 60  # 单 GPU, 每样本 ~20s
    print(f"\n>>> 流水线 1 L1-GPU-Worker x {len(tasks)} seqs (RhoFold+MSA 常驻引擎, L1.5+2 OpenMM CPU), ~{est:.0f} min\n")

    from multiprocessing import Process

    # L1 GPU Worker: 常驻 RhoFold 引擎, 顺序处理所有序列
    producer = Process(
        target=_l1_worker_producer,
        args=(all_tasks_args,),
        name="l1-gpu-producer",
    )

    # L2 CPU Consumer: 单进程 OpenMM 精修 (纯 CPU, 用户指定单线程)
    consumer = Process(
        target=_l2_consumer,
        args=(queue, done_queue, len(tasks), a.n_anneal, str(data)),
        name="l2-cpu-consumer",
    )

    producer.start()
    consumer.start()

    t0 = time.time()
    results = []
    ok = fail = 0
    n_done = 0

    # 聚合消费端结果 (阻塞等消费者跑完; 队列恒有 total 条, 自动停)
    while n_done < len(tasks):
        r = done_queue.get()
        if r is not None:
            results.append(r)
            ok += 1
        else:
            fail += 1
        n_done += 1
        d = n_done
        if d % 5 == 0 or d == len(tasks):
            el = time.time() - t0
            rate = d / max(el, 1) * 60
            print(f"  [{d}/{len(tasks)}] OK:{ok} FAIL:{fail} {rate:.1f}/min {el/60:.1f}min")
        if not consumer.is_alive() and n_done < len(tasks):
            print(f"\n  [消费者进程已退出, 未完成 {len(tasks)-n_done} 条] 提前终止")
            break

    producer.join()
    consumer.join()
    with open(data / "dataset_index.json", "w") as f:
        json.dump(results, f, indent=2)
    el = time.time() - t0
    print(f"\n{'='*50}\ndone! OK:{ok} FAIL:{fail}  {el/60:.1f}min\nindex: {data/'dataset_index.json'}")


if __name__ == "__main__":
    # 无缓冲输出: 顶层 reconfigure, 子进程 spawn 重新 import 时继承无缓冲, 日志实时可见
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
