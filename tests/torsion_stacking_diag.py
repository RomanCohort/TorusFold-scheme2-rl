"""torsion_stacking_diag.py - 二面角 + 碱基堆积几何诊断。

P1 干净版 amber 后 (默认状态), 测 RNA backbone 二面角 + 碱基堆积距离, 跟
A-form 真值对比。补全键角诊断 (bond_angle_diag.py) 的盲区。

A-form RNA 真值 (Aronovick 1983, 文献 + OL3 平衡):
  二面角: alpha=-60, gamma=60, delta=84, zeta=-90 (deg, ±20)
  堆积距离: A-form 相邻碱基平面间距 3.4Å (3.2-3.6Å), 垂直距离 2.8Å
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfold.scheme2 import predict_3d_allatom  # noqa: E402

# A-form RNA 二面角定义 (同 amber_refine._AFORM_TORSIONS)
# 跨残基: alpha (O3'[i-1]-P-O5'-C5'), zeta (C4'-C3'-O3'-P[i+1])
# 残基内: gamma, delta
AFORM_TORSIONS = {
    "alpha": (-60.0, "O3'", "P",   "O5'", "C5'"),
    "gamma": (60.0,  "O5'", "C5'", "C4'", "C3'"),
    "delta": (84.0,  "C5'", "C4'", "C3'", "O3'"),
    "zeta":  (-90.0, "C4'", "C3'", "O3'", "P"),
    # beta/epsilon: amber _AFORM_TORSIONS 表里没有 (amber 没约束), 诊断暴露的盲区
    "beta":  (-180.0, "P", "O5'", "C5'", "C4'"),   # 残基内
    "epsilon": (-155.0, "C3'", "O3'", "P", "O5'"),  # 跨残基 (P/O5' 来自下一残基)
}

# 碱基原子名白名单 (嘌呤 A/G: N9,C8,N7,C5,C6,N1,C2,N3,C4; 嘧啶 C/U 取公共子集)
# 用 C1'/N1/N3/C2/C4/C5/C6/C8 这些碱基骨架原子 (跨碱基类型都有的)
BASE_ATOMS = {"N1", "N3", "C2", "C4", "C5", "C6", "C8"}  # 7 个公共原子


def dihedral_deg(p1, p2, p3, p4):
    """四点二面角 (度), p2-p3 是中间键。右手规则。"""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-9 or n2_norm < 1e-9:
        return float("nan")
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return -float(np.degrees(np.arctan2(y, x)))


def collect_torsions(coords_aa, structure):
    """从精修后坐标 + structure 取原子索引, 算 4 个二面角。
    alpha 和 zeta 跨残基, gamma 和 delta 残基内。
    """
    rai = structure.residue_atom_index
    L = len(rai)
    results = {name: [] for name in AFORM_TORSIONS}

    def xyz(res_idx, atom_name):
        idx = rai[res_idx].get(atom_name)
        return None if idx is None else coords_aa[idx]

    def all_present(*pts):
        """所有点都非 None (坐标数组不能直接 in)。"""
        return all(p is not None for p in pts)

    for i in range(L):
        prev = (i - 1) % L
        nxt = (i + 1) % L
        # 通用: 每个二面角的 4 个原子位置 (prev/i/nxt 哪个残基)
        # 格式: (name, [(res_offset, atom_name) x4])
        tors_specs = [
            ("alpha",   [(prev, "O3'"), (i, "P"),    (i, "O5'"), (i, "C5'")]),
            ("beta",    [(i, "P"),    (i, "O5'"), (i, "C5'"),  (i, "C4'")]),
            ("gamma",   [(i, "O5'"), (i, "C5'"), (i, "C4'"),  (i, "C3'")]),
            ("delta",   [(i, "C5'"), (i, "C4'"), (i, "C3'"),  (i, "O3'")]),
            ("epsilon", [(i, "C3'"), (i, "O3'"), (nxt, "P"), (nxt, "O5'")]),
            ("zeta",    [(i, "C4'"), (i, "C3'"), (i, "O3'"),  (nxt, "P")]),
        ]
        for name, specs in tors_specs:
            pts = [xyz(roff, an) for (roff, an) in specs]
            if all_present(*pts):
                results[name].append(dihedral_deg(*pts))

    stats = {}
    for name, vals in results.items():
        arr = np.array([v for v in vals if not np.isnan(v)], dtype=np.float64)
        if len(arr) == 0:
            stats[name] = (float("nan"),) * 5
            continue
        # 角度的 mean 要处理环绕 (例如 alpha=-60 平均到 300 vs -60 要正确)
        # 用 circular mean: mean = atan2(mean_sin, mean_cos)
        rad = np.radians(arr)
        mean_circ = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
        # std 用角度差的方差 (环绕处理)
        diffs = np.angle(np.exp(1j * (rad - np.radians(mean_circ))))
        std_circ = float(np.degrees(np.std(diffs)))
        stats[name] = (float(mean_circ), std_circ, float(arr.min()), float(arr.max()), int(len(arr)))
    return stats


# backbone 键长定义 + A-form 真值 (Å)
# 格式: (name, atom_a, atom_b, a_res_offset, b_res_offset)
# offset 0 = 本残基, +1 = 下一残基 (跨残基 O3'-P[i+1])
BOND_LENGTHS = [
    ("P-O5'",     "P",   "O5'", 0, 0),   # 1.61
    ("O5'-C5'",   "O5'", "C5'", 0, 0),   # 1.43
    ("C5'-C4'",   "C5'", "C4'", 0, 0),   # 1.52
    ("C4'-C3'",   "C4'", "C3'", 0, 0),   # 1.52
    ("C4'-O4'",   "C4'", "O4'", 0, 0),   # 1.41
    ("C3'-C2'",   "C3'", "C2'", 0, 0),   # 1.52
    ("C3'-O3'",   "C3'", "O3'", 0, 0),   # 1.41
    ("C1'-N9/N1", "C1'", "N9",  0, 0),   # 1.47 (嘌呤用 N9, 嘧啶 N1, 取公共)
    ("O3'-P",     "O3'", "P",   0, +1),  # 1.61 跨残基磷酸桥
]
BOND_EQ = {  # A-form 真值 (Å)
    "P-O5'": 1.61, "O5'-C5'": 1.43, "C5'-C4'": 1.52, "C4'-C3'": 1.52,
    "C4'-O4'": 1.41, "C3'-C2'": 1.52, "C3'-O3'": 1.41,
    "C1'-N9/N1": 1.47, "O3'-P": 1.61,
}


def collect_bondlengths(coords_aa, structure):
    """全 backbone 键长分布。返回 {name: (mean, std, min, max, n)}。"""
    rai = structure.residue_atom_index
    L = len(rai)
    results = {b[0]: [] for b in BOND_LENGTHS}

    def xyz(res_idx, atom_name):
        idx = rai[res_idx].get(atom_name)
        return None if idx is None else coords_aa[idx]

    for i in range(L):
        for (name, a, b, oa, ob) in BOND_LENGTHS:
            pa = xyz(i, a)
            pb = xyz((i + ob) % L, b) if ob != 0 else xyz(i, b)
            # C1'-N9/N1: 嘌呤 N9, 嘧啶 N1 — 优先 N9, 没有试 N1
            if name == "C1'-N9/N1":
                pa = xyz(i, "C1'")
                pb = xyz(i, "N9")
                if pb is None:
                    pb = xyz(i, "N1")
            if pa is not None and pb is not None:
                results[name].append(float(np.linalg.norm(pa - pb)))
    stats = {}
    for name, vals in results.items():
        arr = np.array(vals, dtype=np.float64)
        if len(arr) == 0:
            stats[name] = (float("nan"),) * 4 + (0,)
        else:
            stats[name] = (float(arr.mean()), float(arr.std()),
                           float(arr.min()), float(arr.max()), int(len(arr)))
    return stats


def collect_stacking(coords_aa, structure):
    """碱基堆积距离: 相邻残基碱基质心距离。A-form 真值 ~3.4Å。"""
    rai = structure.residue_atom_index
    L = len(rai)

    def base_centroid(res_idx):
        pts = []
        for name in BASE_ATOMS:
            idx = rai[res_idx].get(name)
            if idx is not None:
                pts.append(coords_aa[idx])
        if not pts:
            return None
        return np.mean(pts, axis=0)

    dists = []
    for i in range(L):
        j = (i + 1) % L
        ci = base_centroid(i)
        cj = base_centroid(j)
        if ci is not None and cj is not None:
            dists.append(float(np.linalg.norm(ci - cj)))
    if not dists:
        return (float("nan"),) * 5
    arr = np.array(dists)
    return (float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), int(len(arr)))


def print_torsion_stats(label, stats):
    print(f"\n=== {label}: 二面角 ===")
    print(f"{'名称':<8} {'A-form':>8} {'mean':>8} {'std':>7} {'min':>8} {'max':>8} {'Δ':>8} {'n':>5}")
    for name, val in AFORM_TORSIONS.items():
        eq = val[0]
        m, s, mn, mx, n = stats[name]
        if n == 0:
            print(f"{name:<8} {eq:>8.1f}   (无数据)")
            continue
        # 二面角环绕: Δ 取最小环绕差 (e.g. -175° vs +175° Δ 是 10°, 不是 350°)
        d = m - eq
        if d > 180:
            d -= 360
        elif d < -180:
            d += 360
        flag = " !!" if abs(d) > 20.0 else ""
        print(f"{name:<8} {eq:>8.1f} {m:>8.2f} {s:>7.2f} {mn:>8.2f} {mx:>8.2f} "
              f"{d:>+8.2f}{flag} {n:>5}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC", help="测试序列")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seq = "AUGCAUGCAUGCAUGCAUGC"
        args.max_iter = 400

    seq = args.seq
    L = len(seq)
    print(f"[torsion_stacking_diag] L={L}, seq={seq}")

    print(f"\n[torsion_stacking_diag] 跑 baseline (P1 干净版, 无 RL, 无 P2/P2.5 约束)...")
    res = predict_3d_allatom(seq, max_iterations=args.max_iter)
    e1 = float(res["e1_aa"])
    coords_aa = res["coords_aa"]
    structure = res["atoms"]

    tors = collect_torsions(coords_aa, structure)
    print_torsion_stats("amber 后 baseline", tors)

    stack = collect_stacking(coords_aa, structure)
    print(f"\n=== amber 后 baseline: 碱基堆积距离 ===")
    m, s, mn, mx, n = stack
    print(f"  质心距离: mean={m:.2f}Å, std={s:.2f}, min={mn:.2f}, max={mx:.2f}, n={n}")
    print(f"  A-form 真值: 3.4Å (范围 3.2-3.6Å)")
    d = m - 3.4 if not np.isnan(m) else float("nan")
    print(f"  Δ = {d:+.2f}Å  {'!! 偏离' if abs(d) > 0.4 else 'OK'}")

    # 键长分布
    bl = collect_bondlengths(coords_aa, structure)
    print(f"\n=== amber 后 baseline: 键长分布 ===")
    print(f"{'键':<12} {'A-form':>7} {'mean':>7} {'std':>6} {'min':>6} {'max':>6} {'Δ':>6} {'n':>5}")
    for (name, _a, _b, _oa, _ob) in BOND_LENGTHS:
        eq = BOND_EQ[name]
        m, s, mn, mx, n = bl[name]
        if n == 0:
            print(f"{name:<12} {eq:>7.2f}   (无数据)")
            continue
        delta = m - eq
        flag = " !!" if abs(delta) > 0.05 else ""
        print(f"{name:<12} {eq:>7.2f} {m:>7.3f} {s:>6.3f} {mn:>6.3f} {mx:>6.3f} "
              f"{delta:>+6.3f}{flag} {n:>5}")

    # 二面角环绕分布 (直方图感, 找离群)
    print(f"\n=== 二面角分布 (每个残基, 找离群) ===")
    tors_per_res = {}
    rai = structure.residue_atom_index
    for name in AFORM_TORSIONS:
        tors_per_res[name] = []
    def all_present(*pts):
        return all(p is not None for p in pts)
    for i in range(L):
        prev = (i - 1) % L
        def xyz(r, a):
            idx = rai[r].get(a)
            return None if idx is None else coords_aa[idx]
        p_i = xyz(i, "P"); o5_i = xyz(i, "O5'"); c5_i = xyz(i, "C5'")
        c4_i = xyz(i, "C4'"); c3_i = xyz(i, "C3'"); o3_i = xyz(i, "O3'")
        p_nxt = xyz((i + 1) % L, "P")
        o3_prev = xyz(prev, "O3'")
        if all_present(o3_prev, p_i, o5_i, c5_i):
            tors_per_res["alpha"].append(dihedral_deg(o3_prev, p_i, o5_i, c5_i))
        if all_present(o5_i, c5_i, c4_i, c3_i):
            tors_per_res["gamma"].append(dihedral_deg(o5_i, c5_i, c4_i, c3_i))
        if all_present(c5_i, c4_i, c3_i, o3_i):
            tors_per_res["delta"].append(dihedral_deg(c5_i, c4_i, c3_i, o3_i))
        if all_present(c4_i, c3_i, o3_i, p_nxt):
            tors_per_res["zeta"].append(dihedral_deg(c4_i, c3_i, o3_i, p_nxt))

    for name, val in AFORM_TORSIONS.items():
        eq = val[0]
        arr = [v for v in tors_per_res[name] if not np.isnan(v)]
        if not arr:
            continue
        # 找离群: 环绕差 > 30°
        arr = np.array(arr)
        rad = np.radians(arr - eq)
        diffs = np.degrees(np.angle(np.exp(1j * rad)))
        outliers = np.where(np.abs(diffs) > 30)[0]
        print(f"  {name} (A-form={eq:.0f}°): {len(outliers)}/{len(arr)} 离群 (>30°)")
        if len(outliers) > 0 and len(outliers) <= 5:
            for idx in outliers:
                print(f"    res {idx}: {arr[idx]:.1f}° (Δ={diffs[idx]:+.1f}°)")


if __name__ == "__main__":
    main()
