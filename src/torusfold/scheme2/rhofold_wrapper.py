"""
rhofold_wrapper.py — RhoFold+ RNA 3D 预测封装

在 comfyui conda env (AMD ROCm GPU) 下运行 RhoFold+ 单序列预测,
输出 P 原子坐标供 Kabsch 拼装使用。

依赖:
  conda activate comfyui
  pip install -e C:/Users/颜子壹/deploy/IGEM集成方案/tools/RhoFold --no-deps
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


_RHOFOLD_ROOT = Path("C:/Users/颜子壹/deploy/IGEM集成方案/tools/RhoFold")
_RHOFOLD_CKPT = _RHOFOLD_ROOT / "pretrained" / "rhofold_pretrained_params.pt"
_RHOFOLD_PYTHON = Path("C:/ana/envs/comfyui/python.exe")


def rhofold_predict_chunk(
    sequence: str,
    ss: str,
    output_dir: str,
    name: str = "chunk",
    msa_path: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """用 RhoFold+ 预测单个 chunk 的 3D 坐标.

    通过子进程调用 comfyui 环境的 Python 执行推理,
    避免主进程的 torch/CUDA 环境冲突.

    支持 MSA 输入: 传 msa_path (真 MSA 或伪 MSA 的 fasta) 时,
    RhoFold+ 用多序列共变信号折叠; 不传则单序列模式.
    注意: 单序列模式在工程序列上可能"塌缩"(所有残基挤成一团, 相邻残基 <3Å),
    喂 MSA 是避免塌缩的关键.

    Args:
        sequence: RNA 序列
        ss: 二级结构 (dot-bracket, 仅用于记录)
        output_dir: 输出目录
        name: chunk 名称
        msa_path: 可选, 多序列 MSA 的 fasta 路径 (与 sequence 长度匹配)
        verbose: 是否打印 MSA 加载/推理日志

    Returns:
        (L, 3) P 原子坐标
    """
    import subprocess

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写 FASTA 供 RhoFold+ 读取
    # 关键: 转全大写 (不删字符!).
    # RhoFold+ 的 remove_insertions 会删小写字符, 如果保留小写会导致
    # msa_tokens 和 rna_fm_tokens 长度不一致 (tensor mismatch).
    # 转全大写后 remove_insertions 不删任何字符, 两个 token 长度一致.
    upper_seq = sequence.upper()
    fa_path = out_dir / f"{name}.fa"
    with open(fa_path, "w") as f:
        f.write(f">{name}\n{upper_seq}\n")

    npy_out = out_dir / f"{name}_rhofold_p.npy"

    # 构造推理脚本
    # msa_path 存在时用多序列 MSA, 否则单序列 (a3m = fas)
    # 关键: 路径用 raw string (r"...") 表示, 避免 Windows 反斜杠被当 Unicode 转义
    msa_src = str(msa_path if msa_path else fa_path)
    msa_arg = f'r"{msa_src}"'
    msa_mode = "MSA" if msa_path else "SINGLE"
    script = f'''# -*- coding: utf-8 -*-
import torch, sys, numpy as np
sys.path.insert(0, r"{_RHOFOLD_ROOT}")
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features

# Load model
model = RhoFold(rhofold_config)
sd = torch.load(r"{_RHOFOLD_CKPT}", map_location="cpu")
model.load_state_dict(sd["model"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Features: msa_path ? 多序列 MSA : 单序列
print("[RhoFold] 模式: {msa_mode}", flush=True)
fea = get_features(r"{fa_path}", {msa_arg})

# Forward
with torch.no_grad():
    out = model(tokens=fea["tokens"].to(device),
                rna_fm_tokens=fea["rna_fm_tokens"].to(device),
                seq=fea["seq"])

# Extract P coords from frames
# frames shape: (N_models, batch, L, 7) — [qx,qy,qz,qw, tx,ty,tz]
# P = translation = frames[:, 4:7]
output = out[-1]
frames = output["frames"][0, 0].data.cpu().numpy()  # (L, 7)
p_coords = frames[:, 4:7]  # translation = P atom position

np.save(r"{npy_out}", p_coords)
print(f"[RhoFold] OK: {{len(p_coords)}} P atoms saved, {{len(p_coords)}} L", flush=True)
'''

    # 脚本写到无中文路径 (避免 Windows GBK 编码问题)
    script_dir = Path(tempfile.mkdtemp(prefix="rhofold_"))
    script_path = script_dir / f"{name}_infer.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    # 子进程调用 comfyui env Python
    result = subprocess.run(
        [str(_RHOFOLD_PYTHON), str(script_path)],
        capture_output=True, text=True,
    )

    # 打印推理日志 (可见 MSA 加载/模式/推理过程)
    if verbose:
        for line in (result.stdout or "").splitlines():
            if "[RhoFold]" in line or "OK:" in line:
                print(f"  {line.strip()}")

    if result.returncode != 0:
        raise RuntimeError(
            f"RhoFold+ inference failed (rc={result.returncode}): "
            f"{result.stderr[-500:]}"
        )

    # 读取结果
    if not npy_out.exists():
        raise FileNotFoundError(f"RhoFold+ output not found: {npy_out}")

    p_coords = np.load(str(npy_out))

    # 物理合理性检查: 检测"塌缩" (所有残基挤成一团, 相邻残基 <3Å 物理不可能).
    # 塌缩时 MFE 配对距离天然小, 但那是假象不是折叠. 喂 MSA 可避免.
    check_p_coords_physical(p_coords, name=name, verbose=verbose)

    return p_coords


def check_p_coords_physical(
    p_coords: np.ndarray,
    name: str = "chunk",
    verbose: bool = True,
) -> dict:
    """物理合理性检查: 相邻残基距离 + 碰撞比例 + 结构扩展范围.

    塌缩判据:
      - 相邻残基中位距离 < 3.0Å (正常 RNA 骨架 ~5.9Å) → 塌缩
      - 碰撞比例 (P-P < 3Å) > 5% → 塌缩/clash
      - 结构范围 (max pairwise dist) < 30Å (对 >50nt 序列) → 挤成一团

    Returns:
        dict: 各项指标 + 是否塌缩
    """
    L = len(p_coords)
    if L < 2:
        return {"collapsed": False}
    c = np.asarray(p_coords, dtype=float)

    # 相邻残基距离
    adj = np.linalg.norm(c[1:] - c[:-1], axis=1)
    adj_med = float(np.median(adj))

    # 碰撞: 非相邻残基 P-P < 3Å
    # 采样加速: 若 L 大, 只查相邻 20 以内的 + 全局抽样
    clash = 0.0
    n_check = 0
    if L <= 400:
        for i in range(L):
            for j in range(i + 2, L):
                d = np.linalg.norm(c[i] - c[j])
                if d < 3.0:
                    clash += 1
                n_check += 1
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(L, size=min(2000, L), replace=False)
        for i in idx:
            for j in idx:
                if j <= i + 1:
                    continue
                d = np.linalg.norm(c[i] - c[j])
                if d < 3.0:
                    clash += 1
                n_check += 1
    clash_ratio = clash / max(n_check, 1)

    # 结构扩展范围
    if L <= 1000:
        span = float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1)) * 2)
    else:
        rng = np.random.default_rng(0)
        sample = c[rng.choice(L, size=min(500, L), replace=False)]
        span = float(np.max(np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)))

    collapsed = (adj_med < 3.0) or (clash_ratio > 0.05 and span < 50.0)

    result = {
        "name": name,
        "adj_median": adj_med,
        "clash_ratio": clash_ratio,
        "span": span,
        "collapsed": bool(collapsed),
    }
    if verbose:
        flag = " [塌缩!]" if collapsed else ""
        print(f"  [物理检查] {name}: 相邻中位 {adj_med:.2f}Å, 碰撞 {clash_ratio:.1%}, "
              f"范围 {span:.1f}Å{flag}")
    return result


def rhofold_available() -> bool:
    """检查 RhoFold+ 是否可用 (comfyui env + GPU)."""
    import subprocess
    try:
        result = subprocess.run(
            [str(_RHOFOLD_PYTHON), "-c",
             "import torch; import rhofold; "
             "print('OK' if torch.cuda.is_available() else 'NO_CUDA')"],
            capture_output=True, text=True,
        )
        return "OK" in result.stdout
    except Exception:
        return False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="RhoFold+ 单 chunk 3D 预测 (支持真/伪 MSA, 可见 MSA 加载日志)"
    )
    ap.add_argument("--seq", required=True, help="RNA 序列 (或 fasta 文件)")
    ap.add_argument("--ss", default="", help="dot-bracket 二级结构 (仅记录)")
    ap.add_argument("--msa", default="", help="MSA fasta 路径 (可选, 不传则单序列)")
    ap.add_argument("--outdir", default="rho_cli_out", help="输出目录")
    ap.add_argument("--name", default="cli_chunk", help="chunk 名称")
    args = ap.parse_args()

    # 支持 --seq 传文件或纯序列
    # 若传 .2d 文件 (序列行 + dot-bracket 行), 只取第一行序列
    seq = args.seq
    if Path(args.seq).exists():
        with open(args.seq) as f:
            lines = [line.strip() for line in f if line.strip()]
        seq = lines[0]  # 第一行 = 序列 (若 .2d)
        # 若是 fasta, 跳过 > 行
        if lines[0].startswith(">"):
            seq = "".join(l for l in lines[1:] if not l.startswith(">"))

    print(f"[CLI] RhoFold+ 预测 {args.name} (len={len(seq)})")
    print(f"[CLI] MSA 模式: {'有' if args.msa else '单序列'}")
    coords = rhofold_predict_chunk(
        seq, args.ss, args.outdir, name=args.name,
        msa_path=args.msa or None, verbose=True,
    )
    print(f"[CLI] 完成: {args.name}_rhofold_p.npy, shape={coords.shape}")
