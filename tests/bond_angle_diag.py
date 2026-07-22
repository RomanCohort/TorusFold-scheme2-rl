"""bond_angle_diag.py - RNA backbone 键角几何诊断 (baseline vs +RL)。

从 amber 精修后的重原子坐标算 6 个 backbone 关键键角度数, 跟 amber14 OL3
平衡值比, 看 RL 有没有把键角拉畸变 (总能量负但局部键角畸变的盲区)。

键角定义 (残基 i 内 + 跨残基桥):
  P-O5'-C5'      (alpha 桥)   OL3 平衡 ~119.5°
  O5'-C5'-C4'    (beta)      ~113.5°
  C5'-C4'-C3'    (gamma)     ~110.5°
  C4'-C3'-O3'    (epsilon)   ~110.5°
  C3'-O3'-P      (zeta 跨残基, i 的 O3' 接 i+1 的 P)  ~119.5°
  O3'-P-O5'      (上一残基 O3' 接本残基 P-O5')        ~104.1°

用法:
  python bond_angle_diag.py            # 默认 32nt 环形, baseline vs +RL
  python bond_angle_diag.py --smoke    # 更小更快
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfold.scheme2 import predict_3d_allatom  # noqa: E402

# amber14 OL3 backbone 键角平衡值 (度, 文献/力场参数表)
OL3_EQ = {
    "P-O5'-C5'": 119.5,
    "O5'-C5'-C4'": 113.5,
    "C5'-C4'-C3'": 110.5,
    "C4'-C3'-O3'": 110.5,
    "C3'-O3'-P": 119.5,    # 跨残基 zeta (本残基 O3' → 下一残基 P)
    "O3'-P-O5'": 104.1,    # 跨残基桥 (上一残基 O3' → 本残基 P → 本残基 O5')
}

# 残基内键角 (三个原子都在同一残基 i):
#   (name, atom_a, atom_b (顶点), atom_c)
INTRARESIDUE = [
    ("P-O5'-C5'",   "P",  "O5'", "C5'"),
    ("O5'-C5'-C4'", "O5'","C5'", "C4'"),
    ("C5'-C4'-C3'", "C5'","C4'", "C3'"),
    ("C4'-C3'-O3'", "C4'","C3'", "O3'"),
]

# 跨残基键角: 涉及残基 i 和 i+1
#   C3'-O3'-P:  顶点 O3' 在残基 i, C3' 也在 i, P 在 i+1
#   O3'-P-O5':  顶点 P 在残基 i+1, O3' 在 i, O5' 在 i+1
INTERRESIDUE = [
    ("C3'-O3'-P", "C3'", "O3'", "P"),    # i: C3',O3'; i+1: P
    ("O3'-P-O5'","O3'", "P",  "O5'"),   # i: O3'; i+1: P,O5'
]


def _angle_deg(a, b, c):
    """三点键角, b 为顶点。度。a/b/c 是 (3,) 坐标 (Å)。"""
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-9 or nc < 1e-9:
        return float("nan")
    cosang = np.dot(ba, bc) / (na * nc)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def collect_angles(coords_aa, structure) -> dict:
    """从精修后重原子坐标 + structure 取原子索引, 算所有 backbone 键角。

    coords_aa: (n_heavy, 3) Å, 按 structure.atoms 顺序 (amber_refine 返回)
               或 (n_atoms, 3) Å, 按 structure.atoms 顺序 (reconstruct 输出)
    structure: AllAtomStructure, residue_atom_index[res_idx][atom_name] = serial
    """
    rai = structure.residue_atom_index
    L = len(rai)
    results = {name: [] for name in OL3_EQ}

    def xyz(res_idx, atom_name):
        """取 (res_idx, atom_name) 原子坐标。"""
        idx = rai[res_idx].get(atom_name)
        if idx is None:
            return None
        return coords_aa[idx]

    for i in range(L):
        # 残基内键角
        for name, a, b, c in INTRARESIDUE:
            pa, pb, pc = xyz(i, a), xyz(i, b), xyz(i, c)
            if pa is None or pb is None or pc is None:
                continue
            results[name].append(_angle_deg(pa, pb, pc))

        # 跨残基键角 (i → i+1, 环形 mod L)
        j = (i + 1) % L
        for name, a, b, c in INTERRESIDUE:
            # a, b 在残基 i; c 在残基 i+1 (对 C3'-O3'-P)
            # O3'-P-O5': a 在 i, b,c 在 i+1
            # 顶点 b 决定哪个残基
            if name == "C3'-O3'-P":
                pa, pb, pc = xyz(i, a), xyz(i, b), xyz(j, c)
            else:  # O3'-P-O5'
                pa, pb, pc = xyz(i, a), xyz(j, b), xyz(j, c)
            if pa is None or pb is None or pc is None:
                continue
            results[name].append(_angle_deg(pa, pb, pc))

    # 每个键角: (mean, std, min, max, n)
    stats = {}
    for name, vals in results.items():
        arr = np.array(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            stats[name] = (float("nan"),) * 4 + (0,)
        else:
            stats[name] = (float(arr.mean()), float(arr.std()),
                           float(arr.min()), float(arr.max()), int(len(arr)))
    return stats


def run_one(sequence, use_rl, max_iter, mask=None):
    """跑一次 predict_3d_allatom, 返回 (amber前stats, amber后stats, e1_aa, n_atoms)。"""
    res = predict_3d_allatom(
        sequence, max_iterations=max_iter,
        use_rl=use_rl, coding_mask=mask,
    )
    structure = res["atoms"]
    # amber 前重建坐标 (structure.atoms.xyz, Å): 直接从重建结构取, 不跑 amber
    pre_xyz = np.array([a.xyz for a in structure.atoms], dtype=np.float64)
    pre_stats = collect_angles(pre_xyz, structure)
    post_stats = collect_angles(res["coords_aa"], structure)
    return pre_stats, post_stats, float(res["e1_aa"]), len(structure.atoms)


def print_stats(label, stats, e1):
    print(f"\n=== {label} (e1_aa={e1:.0f} kJ/mol) ===")
    print(f"{'键角':<14} {'OL3平衡':>8} {'mean':>8} {'std':>7} {'min':>8} {'max':>8} {'Δmean':>8} {'n':>5}")
    for name, eq in OL3_EQ.items():
        m, s, mn, mx, n = stats[name]
        if n == 0:
            print(f"{name:<14} {eq:>8.1f}   (无数据)")
            continue
        delta = m - eq
        flag = " !!" if abs(delta) > 5.0 else ""
        print(f"{name:<14} {eq:>8.1f} {m:>8.2f} {s:>7.2f} {mn:>8.2f} {mx:>8.2f} "
              f"{delta:>+8.2f}{flag} {n:>5}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC", help="测试序列")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seq = "AUGCAUGCAUGCAUGCAUGC"  # 20nt
        args.max_iter = 400

    seq = args.seq
    L = len(seq)
    print(f"[bond_angle_diag] L={L}, seq={seq}")
    print(f"  P0 根因验证: amber 前重建坐标 vs amber 后, 看畸变是 1EHZ 对齐引入还是 amber 引入")
    mask = np.zeros(L, dtype=bool); mask[:L // 2] = True

    print(f"\n[bond_angle_diag] 跑 baseline (无 RL)...")
    pre_b, post_b, e_b, n_b = run_one(seq, use_rl=False, max_iter=args.max_iter)
    print_stats("baseline amber 前 (1EHZ 重建)", pre_b, 0.0)
    print_stats(f"baseline amber 后", post_b, e_b)

    print(f"\n=== P0 根因: baseline amber 前 vs 后 (Δ = 后 - 前) ===")
    print(f"{'键角':<14} {'OL3':>6} {'前mean':>7} {'后mean':>7} {'Δ':>7} {'根因':>10}")
    for name, eq in OL3_EQ.items():
        mpre = pre_b[name][0]; mpost = post_b[name][0]
        if np.isnan(mpre) or np.isnan(mpost):
            print(f"{name:<14} {eq:>6.1f}   (无数据)")
            continue
        d = mpost - mpre
        # 畸变在 amber 前就有 -> 1EHZ 对齐根因; amber 后才出现 -> amber 引入
        src = "1EHZ对齐" if abs(mpre - eq) > 5.0 else ("amber引入" if abs(mpost - eq) > 5.0 else "正常")
        print(f"{name:<14} {eq:>6.1f} {mpre:>7.2f} {mpost:>7.2f} {d:>+7.2f} {src:>10}")

    print(f"\n[bond_angle_diag] 跑 +RL...")
    pre_r, post_r, e_r, n_r = run_one(seq, use_rl=True, max_iter=args.max_iter, mask=mask)
    print_stats("+RL amber 后", post_r, e_r)

    print(f"\n=== baseline vs +RL (amber 后) ===")
    print(f"{'键角':<14} {'OL3':>6} {'base':>7} {'RL':>7} {'Δ':>7} {'RL畸变?':>8}")
    any_distortion = False
    for name, eq in OL3_EQ.items():
        mb = post_b[name][0]; mr = post_r[name][0]
        if np.isnan(mb) or np.isnan(mr):
            print(f"{name:<14} {eq:>6.1f}   (无数据)")
            continue
        d = mr - mb
        distort = abs(mr - eq) > 5.0
        any_distortion = any_distortion or distort
        print(f"{name:<14} {eq:>6.1f} {mb:>7.2f} {mr:>7.2f} {d:>+7.2f} {'是' if distort else '否':>8}")

    print(f"\n[结论] {'发现键角畸变 (>5°)' if any_distortion else '键角均在 OL3 平衡值 ±5° 内'}")
    print(f"       P0 看 'amber 前 vs 后' 那段: 畸变标 '1EHZ对齐' = 根因在重建对齐 (P1 修 Kabsch);")
    print(f"       标 'amber引入' = 根因在 amber 最小化 (改 amber 约束)。")


if __name__ == "__main__":
    main()
