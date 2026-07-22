"""bond_angle_length_scale.py - 测重建后磷酸桥键角畸变随序列长度放大效应。

只跑 1EHZ 重建 (不跑 amber, 本地 CPU 跑不动长序列 amber), 看重建后
C3'-O3'-P / O3'-P-O5' 等磷酸桥键角在不同长度 (20/100/500/1000nt) 的畸变。

用合成 A-form 环形序列 (不调 ViennaRNA, 避免配对计算慢)。CG P 用正多边形。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfold.scheme2.aform_from_template import reconstruct_all_atom as reconstruct_1ehz
from torusfold.scheme2.allatom_reconstruct import get_atom_xyzs
# 复用诊断脚本的键角函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bond_angle_diag import collect_angles, OL3_EQ  # noqa: E402


def make_ring_cg(L, R_factor=5.9):
    """正多边形 CG P 坐标 (Å), P-P 间距 = R_factor (A-form 5.9Å)。"""
    R = L * R_factor / (2 * np.pi)
    ang = np.linspace(0, 2 * np.pi, L, endpoint=False)
    return np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(L)], axis=1).astype(np.float32)


def run_length(L):
    """重建长度 L 的环形序列, 返回 (stats, n_atoms, elapsed)。"""
    import time
    seq = "AUGC" * (L // 4) + "A" * (L % 4)
    seq = seq[:L]
    cg = make_ring_cg(L)
    t0 = time.time()
    struct = reconstruct_1ehz(cg, seq)
    elapsed = time.time() - t0
    xyz = np.array([a.xyz for a in struct.atoms], dtype=np.float64)
    stats = collect_angles(xyz, struct)
    return stats, len(struct.atoms), elapsed


def main():
    lengths = [20, 50, 100, 200, 500, 1000]
    print(f"[bond_angle_length_scale] 测重建后磷酸桥键角随长度变化")
    print(f"长度: {lengths}")
    print(f"只跑 1EHZ 重建, 不跑 amber (amber 前, 根因层)\n")

    results = {}
    for L in lengths:
        try:
            stats, n_atoms, elapsed = run_length(L)
            results[L] = stats
            print(f"--- L={L} (n_atoms={n_atoms}, 重建 {elapsed:.2f}s) ---")
            for name, eq in OL3_EQ.items():
                m, s, mn, mx, n = stats[name]
                if n == 0:
                    continue
                delta = m - eq
                flag = " !!" if abs(delta) > 5.0 else ""
                print(f"  {name:<14} OL3={eq:>6.1f} mean={m:>7.2f} std={s:>6.2f} "
                      f"min={mn:>7.2f} max={mx:>7.2f} Δ={delta:>+7.2f}{flag} n={n}")
            print()
        except Exception as e:
            print(f"--- L={L} 失败: {e!r} ---\n")
            import traceback
            traceback.print_exc()

    # 汇总: 看磷酸桥键角随长度放大
    print("=== 磷酸桥键角 Δ(偏离 OL3) 随长度变化 ===")
    print(f"{'长度':>6}", end="")
    for name in ["C3'-O3'-P", "O3'-P-O5'", "C4'-C3'-O3'", "C5'-C4'-C3'"]:
        print(f" {name:>12}", end="")
    print()
    for L in lengths:
        if L not in results:
            continue
        print(f"{L:>6}", end="")
        for name in ["C3'-O3'-P", "O3'-P-O5'", "C4'-C3'-O3'", "C5'-C4'-C3'"]:
            m = results[L][name][0]
            d = m - OL3_EQ[name] if not np.isnan(m) else float("nan")
            print(f" {d:>+12.2f}", end="")
        print()
    print("\n解读: Δ 随长度增大 = 畸变放大 (长序列问题严重); Δ 平稳 = 长度无关。")


if __name__ == "__main__":
    main()
