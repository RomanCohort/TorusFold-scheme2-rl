# -*- coding: utf-8 -*-
"""inference_engine.py — TorusFold 推理端统一入口（物理 / DL 双引擎）。

规划：TorusFold 推理端分两个独立引擎：
  - physics 模式: 物理管线（ViennaRNA → scheme2 CG 几何 → OpenMM 弛豫 → 全原子重建 → amber 精修）
  - dl 模式:      DL 主模型（TorusFold v2 等变 GNN）预测结构，只出 CG 坐标

DL 主模型 checkpoint 等训练完成后导入。当前无权重时 dl 分支降级
（available=False, reason="no_checkpoint"），接口先搭好。

用法:
    from torusfold.inference_engine import predict_structure
    res = predict_structure("AGCUAGCU...", mode="physics")
    res = predict_structure("AGCUAGCU...", mode="dl", checkpoint="models/s10_final.pt")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 物理管线（scheme2 快速版 + isrnaclong 高精度） ──
from .scheme2 import predict_3d_allatom
from .scheme2.refine import vienna_pair_probs, BOND_LEN

# 默认 checkpoint 搜索路径（S10 final，训练完成后放这里）
_DEFAULT_DL_CKPT = Path(__file__).resolve().parent.parent.parent / "models" / "s10_final.pt"

# 默认临时输出目录（isrnaclong 需要 output_dir，用系统临时目录）
_DEFAULT_PHYSICS_DIR = Path(__file__).resolve().parent.parent.parent / "output_physics"


def pairs_to_dotbracket(pairs, length):
    """从配对列表生成 dot-bracket3 二级结构字符串。

    Args:
        pairs: [(i, j, weight), ...] 0-based 配对列表
        length: 序列长度

    Returns:
        dot-bracket3 字符串 (e.g. "...(((..))).")
    """
    db = list("." * length)
    for (i, j, _w) in pairs:
        if 0 <= i < length and 0 <= j < length:
            db[i] = "("
            db[j] = ")"
    return "".join(db)

# 默认 checkpoint 搜索路径（S10 final，训练完成后放这里）
_DEFAULT_DL_CKPT = Path(__file__).resolve().parent.parent.parent / "models" / "s10_final.pt"


def _default_gene_expr(sequence: str) -> Dict[str, float]:
    """DL 主模型 gene_expr 兜底（缺省 0.5，与 predict_single 内部一致）。"""
    return {}


def build_torusfold_model(
    checkpoint: Optional[str] = None,
    device: str = "cpu",
):
    """加载 DL 主模型（TorusFold v2）。

    Args:
        checkpoint: 显式权重路径；None 则尝试默认路径 models/s10_final.pt
        device: "cpu" / "cuda"

    Returns:
        TorusFold 模型实例；无可用 checkpoint 返回 None（调用方降级）。
    """
    path = checkpoint or _DEFAULT_DL_CKPT
    if not Path(path).exists():
        print(f"[inference_engine] DL checkpoint not found: {path}")
        print("[inference_engine] DL 主模型等训练完成后再导入，当前 dl 分支降级")
        return None

    try:
        from .torusfold import TorusFold, TorusFoldConfig
        model = TorusFold(TorusFoldConfig())
        model.load(str(path), device=device)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[inference_engine] DL model load failed: {e}")
        return None


def _dl_cg_signals(
    cg_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    sequence: str,
) -> Dict[str, float]:
    """DL 分支的 CG 级结构信号（轻量，不全原子）。

    只算几何派生的标量：BSJ 闭合距离、键长偏差、配对距离偏差。
    """
    signals: Dict[str, float] = {}
    if cg_coords is None or len(cg_coords) < 2:
        return signals

    # BSJ 闭合距离 (理想 ~BOND_LEN)
    bsj_dist = float(np.linalg.norm(cg_coords[0] - cg_coords[-1]))
    signals["bsj_distance"] = round(bsj_dist, 4)
    signals["closure_err"] = round(abs(bsj_dist - BOND_LEN) / BOND_LEN, 4)

    # 相邻 P-P 键长偏差 (理想 BOND_LEN)
    if len(cg_coords) > 2:
        bonds = np.linalg.norm(np.diff(cg_coords, axis=0), axis=1)
        signals["bond_len_mean"] = round(float(bonds.mean()), 4)
        signals["bond_len_std"] = round(float(bonds.std()), 4)
        signals["bond_err_rmsd"] = round(
            float(np.sqrt(((bonds - BOND_LEN) ** 2).mean())), 4)

    # 配对距离偏差 (WC pair C1'-C1' 理想 ~10.6 A；此处 P-P 近似评估相对一致性)
    if pairs:
        pair_dists = []
        for (i, j, _w) in pairs:
            if 0 <= i < len(cg_coords) and 0 <= j < len(cg_coords):
                pair_dists.append(float(np.linalg.norm(cg_coords[i] - cg_coords[j])))
        if pair_dists:
            signals["pair_dist_mean"] = round(float(np.mean(pair_dists)), 4)

    signals["n_pairs"] = len(pairs)
    return signals


def _predict_physics_high(
    seq: str,
    *,
    pair_threshold: float = 0.5,
    use_rl: bool = False,
    output_dir: Optional[str] = None,
    n_relax_rounds: int = 10,
    n_rest2_replicas: int = 8,
    md_step_scale: float = 0.1,
    resume: bool = True,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, object]:
    """高精度物理管线（Level 0-5, isrnaclong_pipeline）。

    封装 isrnaclong_pipeline 到 predict_structure 统一输出格式。
    secondary_structure 从 ViennaRNA 配对自动转 dot-bracket。
    output_dir 不传则用临时目录（每次运行独立，不支持 checkpoint 恢复）。
    """
    import tempfile
    import time

    # Level 0: ViennaRNA 配对
    pairs, bpp = vienna_pair_probs(seq, pair_threshold)
    secondary_structure = pairs_to_dotbracket(pairs, len(seq))

    # 输出目录
    if output_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="torusfold_physics_")
        output_dir = tmp_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 调用 isrnaclong Pipeline（Level 0-5）
    from .scheme2.isrnaclong import isrnaclong_pipeline

    t0 = time.time()
    try:
        result = isrnaclong_pipeline(
            seq,
            secondary_structure,
            output_dir,
            n_relax_rounds=n_relax_rounds,
            n_rest2_replicas=n_rest2_replicas,
            md_step_scale=md_step_scale,
            use_rl_relax=use_rl,
            use_rl_mcts=use_rl,
            resume=resume,
            verbose=True,
        )
        dt = time.time() - t0

        # LongPipelineResult → 统一输出
        return {
            "ok": True, "mode": "physics", "engine": "physics",
            "available": True,
            "sequence": seq,
            "secondary_structure": secondary_structure,
            "coords_cg": result.coords_cg,
            "coords_aa": result.coords_aa,
            "pairs": pairs,
            "pair_probs": bpp,
            "pair_rate": result.pair_rate,
            "cross_segment_ok_rate": result.cross_segment_ok_rate,
            "energy_cg": result.energy_cg,
            "energy_aa": result.energy_aa,
            "n_segments": result.n_segments,
            "runtime_seconds": dt,
            "fidelity_history": result.fidelity_history,
            "output_dir": output_dir,
            "structure_method": "isrnaclong_level5",
            "reason": "ok",
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            "ok": False, "mode": "physics", "engine": "physics",
            "available": False, "reason": "isrnaclong_error",
            "sequence": seq, "secondary_structure": secondary_structure,
            "error": str(e)[:500],
            "runtime_seconds": dt,
            "output_dir": output_dir,
            "pairs": pairs, "pair_probs": bpp,
        }


def predict_structure(
    sequence: str,
    *,
    mode: str = "physics",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    pair_threshold: float = 0.5,
    use_rl: bool = False,
    use_relaxation: bool = True,
    output_dir: Optional[str] = None,
    n_relax_rounds: int = 10,
    n_rest2_replicas: int = 8,
    md_step_scale: float = 0.1,
    resume: bool = True,
    **kwargs,
) -> Dict[str, object]:
    """TorusFold 推理统一入口。

    Args:
        sequence: circRNA 序列 (ACGU)
        mode:
          - "physics":      高精度物理管线（Level 0-5, isrnaclong, 慢但准）
          - "physics_fast": 精简物理管线（predict_3d_allatom, 快但略粗）
          - "dl":           DL 主模型（等变 GNN, CG only）
        checkpoint: DL 主模型权重路径（dl 模式用）
        device: 计算设备
        pair_threshold: ViennaRNA 配对阈值
        use_rl: 物理版是否启用 RL（physics: use_rl_mcts; physics_fast: RL 远端优化）
        use_relaxation: 物理版是否启用弛豫后处理
        output_dir: isrnaclong 输出目录（默认临时目录）
        n_relax_rounds: Level 2 迭代弛豫轮数（physics 模式）
        n_rest2_replicas: Level 4 REST2 副本数（physics 模式）
        md_step_scale: Level 2 MD 步数缩放因子（physics 模式）
        resume: 是否断点续跑（physics 模式 checkpoint）

    Returns:
        dict:
          三模式共用键：mode / engine / available / ok
          physics/physics_fast 额外有：coords_cg / coords_aa / pairs / pair_probs
          dl 额外有：coords_cg / pairs / pair_probs / structure_signals
    """
    seq = str(sequence).strip().upper()
    if not seq or not all(c in "ACGU" for c in seq):
        return {"ok": False, "mode": mode, "available": False,
                "reason": "invalid_sequence", "error": "sequence 只能包含 A/C/G/U"}

    if mode == "physics":
        return _predict_physics_high(
            seq, pair_threshold=pair_threshold, use_rl=use_rl,
            output_dir=output_dir, n_relax_rounds=n_relax_rounds,
            n_rest2_replicas=n_rest2_replicas, md_step_scale=md_step_scale,
            resume=resume, device=device, **kwargs)

    elif mode == "physics_fast":
        res = predict_3d_allatom(
            seq,
            pair_threshold=pair_threshold,
            use_rl=use_rl,
            use_relaxation=use_relaxation,
            **kwargs,
        )
        # 顶层对齐：确保有 mode / available
        res["mode"] = "physics_fast"
        res["engine"] = "physics_fast"
        res["ok"] = bool(res.get("available", True))
        return res

    elif mode == "dl":
        model = build_torusfold_model(checkpoint, device=device)
        if model is None:
            return {
                "ok": False, "mode": "dl", "engine": "dl",
                "available": False, "reason": "no_checkpoint",
                "sequence": seq,
                "coords_cg": None, "pairs": [], "pair_probs": None,
                "structure_signals": {},
                "message": "DL 主模型 checkpoint 缺失，等训练完成后导入再启用 dl 模式",
            }

        import torch
        # ViennaRNA 配对（与物理版一致）
        pairs, bpp = vienna_pair_probs(seq, pair_threshold)

        try:
            with torch.no_grad():
                outputs = model.forward(
                    [seq], gene_expr=None, device=device, predict_structure=True)
            coords = outputs.get("coords")
            if coords is None:
                return {
                    "ok": False, "mode": "dl", "engine": "dl",
                    "available": False, "reason": "no_coords",
                    "sequence": seq, "pairs": pairs, "pair_probs": bpp,
                }
            # coords → numpy (L, 3)
            cg_coords = coords.cpu().numpy() if hasattr(coords, "cpu") else np.asarray(coords)
            if cg_coords.ndim == 3:
                cg_coords = cg_coords[0]  # (B, L, 3) -> (L, 3)
            cg_coords = np.asarray(cg_coords, dtype=np.float64).reshape(-1, 3)

            signals = _dl_cg_signals(cg_coords, pairs, seq)

            return {
                "ok": True, "mode": "dl", "engine": "dl",
                "available": True,
                "sequence": seq,
                "coords_cg": cg_coords.tolist(),
                "pairs": pairs,
                "pair_probs": bpp,
                "structure_signals": signals,
                "structure_method": "torusfold_dl",
                "reason": "ok",
                "closure_distance": signals.get("bsj_distance"),
            }
        except Exception as e:
            return {
                "ok": False, "mode": "dl", "engine": "dl",
                "available": False, "reason": "dl_error",
                "sequence": seq, "error": str(e)[:300],
            }

    else:
        return {"ok": False, "mode": mode, "available": False,
                "reason": "unknown_mode",
                "error": f"mode 必须是 physics|physics_fast|dl，收到 {mode!r}"}


__all__ = [
    "predict_structure",
    "build_torusfold_model",
    "pairs_to_dotbracket",
]
