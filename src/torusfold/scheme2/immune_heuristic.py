"""immune_heuristic.py - scheme2 几何启发式免疫指纹 (10 指标)。

scheme10 ImmuneFingerprintHeads 是训练好的神经网络头, 云端权重没下来前
用全原子坐标 + 序列 + ViennaRNA 配对几何启发式算 10 个免疫指标, 让前端
Coloring Scheme 下拉恢复 pkr/m6a/tlr7/rigi/nlrp3/sponge 等。指标对齐
export.py 的 default_per_res / default_scalar 字段名, 权重下来后替换。

不依赖 amber 精修是否跑通 - 用全原子重建坐标 (理想 A-form 几何) 算就行,
精度低但够配图着色。
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def compute_immune_fingerprints(
    coords_aa: np.ndarray,
    structure,
    pairs: List[Tuple[int, int, float]],
    sequence: str,
) -> Dict[str, np.ndarray]:
    """几何启发式算 10 个免疫指标。

    Args:
        coords_aa: (N, 3) 全原子坐标 (含 H 或不含都行, 只用 C1' 索引)
        structure: AllAtomStructure (用来按残基查 C1' 索引)
        pairs: ViennaRNA 配对 [(i, j, w), ...] 0-based
        sequence: ACGU 字符串

    Returns:
        dict, per-residue 指纹 (长度 L) + scalar 指纹 (长度 1)。
        字段对齐 export.py default_per_res / default_scalar。
    """
    L = len(sequence)

    # 每残基的 C1' 坐标 (Å)
    c1_xyzs = np.array([
        structure.atoms[structure.residue_atom_index[i]["C1'"]].xyz
        for i in range(L)
    ], dtype=np.float64)

    # --- stem/loop (从 ViennaRNA pairs 推导) ---
    paired = set()
    for (i, j, w) in pairs:
        if 0 <= i < L:
            paired.add(i)
        if 0 <= j < L:
            paired.add(j)
    is_stem = np.array([1.0 if i in paired else 0.0 for i in range(L)],
                       dtype=np.float32)

    # --- PKR SASA exposure (C1'-C1' 最近邻距离 / 15 归一, 暴露越大值越大) ---
    if L > 1:
        d = np.linalg.norm(c1_xyzs[:, None] - c1_xyzs[None], axis=2)
        np.fill_diagonal(d, 1e9)
        pkr_sasa = np.clip(d.min(axis=1) / 15.0, 0.0, 1.0).astype(np.float32)
    else:
        pkr_sasa = np.ones(L, dtype=np.float32)

    pkr_stem_logit = is_stem.copy()

    # --- m6A: DRACH motif 检测 ---
    drach_is = np.zeros(L, dtype=np.float32)
    drach_in_loop = np.zeros(L, dtype=np.float32)
    m6a_prob = np.zeros(L, dtype=np.float32)
    for i in range(L):
        if sequence[i] != "A" or i < 2 or i > L - 3:
            continue
        mer = sequence[i - 2:i + 3]
        # DRACH: D=A/G/U, R=A/G, A, C, H=A/C/U
        if (mer[0] in "AGU" and mer[1] in "AG" and mer[2] == "A"
                and mer[3] == "C" and mer[4] in "ACU"):
            drach_is[i] = 1.0
            if i not in paired:
                drach_in_loop[i] = 1.0
            # 暴露 + loop 区 → 高写入概率 (启发式)
            m6a_prob[i] = (0.5
                           + 0.3 * drach_in_loop[i]
                           + 0.2 * pkr_sasa[i])

    # --- TLR7: GU-rich density (5nt 窗口 G+U 占比) ---
    tlr7 = np.zeros(L, dtype=np.float32)
    for i in range(L):
        lo, hi = max(0, i - 2), min(L, i + 3)
        window = sequence[lo:hi]
        tlr7[i] = sum(1 for c in window if c in "GU") / len(window)

    # --- RIG-I: 负对照, 只在首末残基给低值 (RIG-I 识双链, scheme2 单链无源) ---
    rigi_per = np.zeros(L, dtype=np.float32)
    if L > 0:
        rigi_per[0] = 0.1
        rigi_per[-1] = 0.1

    # --- NLRP3: persistence length (回转半径 / 10, 启发式) ---
    if L > 2:
        rg = float(np.sqrt(
            ((c1_xyzs - c1_xyzs.mean(axis=0)) ** 2).sum(axis=1).mean()
        ))
        nlrp3_persist = float(np.clip(rg / 10.0, 0.0, 100.0))
    else:
        nlrp3_persist = 0.0

    # --- miRNA sponge: GU 含量 × 长度因子 ---
    gu_frac = sum(1 for c in sequence if c in "GU") / max(1, L)
    sponge_score = float(gu_frac * min(L / 200.0, 1.0))

    rigi_score = float(rigi_per.sum() / max(1, L))

    return {
        # per-residue (长度 L)
        "pkr_sasa": pkr_sasa,
        "pkr_stem_logit": pkr_stem_logit,
        "drach_is_drach": drach_is,
        "drach_in_loop": drach_in_loop,
        "m6a_write_prob": m6a_prob,
        "tlr7_gu_density": tlr7,
        "rigi_per_pos": rigi_per,
        # scalar (长度 1)
        "nlrp3_persistence_length": np.array([nlrp3_persist],
                                              dtype=np.float32),
        "sponge_score": np.array([sponge_score], dtype=np.float32),
        "rigi_score": np.array([rigi_score], dtype=np.float32),
    }


def compute_structure_signals(
    coords_aa: np.ndarray,
    structure,
    pairs: List[Tuple[int, int, float]],
    bpp: np.ndarray,
    sequence: str,
    e1_aa: float,
    bsj_dist: float,
    cg_coords: np.ndarray,
) -> Dict[str, float]:
    """纯计算结构信号 (对齐原生 TorusFold TorusFoldSignals + ImmuneSensingResultV3 透传字段)。

    覆盖组 A (序列/配对派生) + 组 B (几何派生) + G 物理指纹, 全部不依赖 DL。
    字段名与原生契约严格对齐, 下游 ImmuneSensingResultV3 不再走 heuristic fallback。

    Args:
        coords_aa: (N, 3) 全原子坐标 (精修后)
        structure: AllAtomStructure (查 C1'/P 索引)
        pairs: ViennaRNA 配对 [(i, j, w), ...] 0-based
        bpp: (L, L) ViennaRNA 配对概率矩阵
        sequence: ACGU 字符串
        e1_aa: amber 精修后能量 (kJ/mol)
        bsj_dist: BSJ 距离 (Å), ||coords[0]-coords[-1]||
        cg_coords: (L, 3) CG P 原子坐标 (精修后), 算几何指标用

    Returns:
        dict[str, float], 全部标量, 字段名对齐原生契约。
    """
    L = len(sequence)
    bond_length = 5.9
    pair_dist_target = 10.6
    clash_dist = 3.0

    # --- 组 A: 序列/配对派生 (ViennaRNA bpp) ---
    upper = np.triu(bpp, k=1)
    pair_mask = upper > 0.5
    dsrna_fraction = float(pair_mask.sum() * 2 / max(1, L))
    mean_pair_prob = float(upper.sum() * 2 / max(1, L * (L - 1)))

    # long_range_pair_fraction: circ_dist > L/4 的配对比例
    pair_count = int(pair_mask.sum())
    if pair_count > 0:
        long_range = 0
        for i, j in zip(*np.where(pair_mask)):
            circ = min(abs(i - j), L - abs(i - j))
            if circ > L / 4:
                long_range += 1
        long_range_frac = float(long_range / pair_count)
    else:
        long_range_frac = 0.0

    # pairing_stability: 1 - pair_probs 归一化熵 (熵低=配对确定=稳定)
    probs = upper[upper > 1e-6]
    if probs.size > 0:
        p = probs / probs.sum()
        entropy = float(-(p * np.log(p)).sum())
        max_entropy = float(np.log(probs.size))
        pairing_stability = float(1.0 - entropy / max_entropy) if max_entropy > 0 else 0.5
    else:
        pairing_stability = 0.5

    # --- 组 B: 几何派生 (从 cg_coords + coords_aa) ---
    closure_err = abs(bsj_dist - bond_length)
    closure_score = float(max(0.0, 1.0 - closure_err / 2.0))

    # bond_rmsd: 相邻 P-P 偏差
    bb = np.linalg.norm(np.diff(cg_coords, axis=0), axis=1)
    bond_rmsd = float(np.sqrt(((bb - bond_length) ** 2).mean()))

    # pair_satisfaction: 配对 P-P 距离达标率 (10.6 ± 1.5)
    if pairs:
        pair_hits = 0
        for i, j, _w in pairs:
            if 0 <= i < L and 0 <= j < L:
                d = float(np.linalg.norm(cg_coords[i] - cg_coords[j]))
                if abs(d - pair_dist_target) < 1.5:
                    pair_hits += 1
        pair_satisfaction = float(pair_hits / len(pairs))
    else:
        pair_satisfaction = 0.0

    # clash_count: 非相邻 P-P < 3.0
    clash_count = 0
    for a in range(L):
        for b in range(a + 2, L):
            if (a, b) == (0, L - 1):
                continue
            if float(np.linalg.norm(cg_coords[a] - cg_coords[b])) < clash_dist:
                clash_count += 1

    # SASA: 用 C1' 最近邻距离代理 (暴露=最近邻远)
    c1_xyzs = np.array([
        structure.atoms[structure.residue_atom_index[i]["C1'"]].xyz
        for i in range(L)
    ], dtype=np.float64)
    if L > 1:
        d = np.linalg.norm(c1_xyzs[:, None] - c1_xyzs[None], axis=2)
        np.fill_diagonal(d, 1e9)
        nn = d.min(axis=1)
        sasa_per = np.clip(nn / 15.0, 0.0, 1.0)
    else:
        sasa_per = np.ones(L)
    sasa_mean = float(sasa_per.mean())
    # BSJ 区域 SASA (首末各 3 残基)
    bsj_idx = list(range(min(3, L))) + list(range(max(0, L - 3), L))
    sasa_bsj = float(sasa_per[bsj_idx].mean()) if bsj_idx else sasa_mean
    surface_exposed_fraction = float((sasa_per > 0.5).mean())

    # bsj_stability / bsj_confidence: 从 closure + energy 派生
    bsj_stability = float(0.5 + 0.3 * closure_score + 0.2 * (1.0 if e1_aa < 0 else 0.0))
    bsj_stability = min(1.0, bsj_stability)
    bsj_confidence = float(closure_score * 0.7 + min(1.0, max(0.0, -e1_aa / 10000.0)) * 0.3)

    # bsj_3d_closure_tightness: BSJ 区域紧凑度 (首末 3 残基回转半径归一)
    if len(bsj_idx) > 1:
        bsj_pts = c1_xyzs[bsj_idx]
        rg_bsj = float(np.sqrt(((bsj_pts - bsj_pts.mean(axis=0)) ** 2).sum(axis=1).mean()))
        bsj_3d_closure_tightness = float(np.clip(1.0 - rg_bsj / 20.0, 0.0, 1.0))
    else:
        bsj_3d_closure_tightness = 0.5

    # --- G 物理指纹 (记忆 torusfold-immune-fingerprints: 纯计算) ---
    # mechanical_stiffness = f(closure, pair_satisfaction, energy)
    energy_norm = float(min(1.0, max(0.0, -e1_aa / 50000.0))) if e1_aa < 0 else 0.0
    mechanical_stiffness = float(0.4 * closure_score + 0.3 * pair_satisfaction + 0.3 * energy_norm)
    mechanical_stiffness = min(1.0, mechanical_stiffness)

    # solvent_response = f(SASA, closure, IRES) - IRES 用 mean_pair_prob 代理
    solvent_response = float(0.4 * sasa_mean + 0.3 * bsj_3d_closure_tightness + 0.3 * (1.0 - mean_pair_prob))
    solvent_response = min(1.0, solvent_response)

    return {
        # 组 A
        "dsRNA_fraction": dsrna_fraction,
        "mean_pair_prob": mean_pair_prob,
        "long_range_pair_fraction": long_range_frac,
        "pairing_stability": pairing_stability,
        # 组 B
        "closure_distance": float(bsj_dist),
        "closure_score": closure_score,
        "bond_rmsd": bond_rmsd,
        "pair_satisfaction": pair_satisfaction,
        "clash_count": clash_count,
        "sasa_mean": sasa_mean,
        "sasa_bsj": sasa_bsj,
        "surface_exposed_fraction": surface_exposed_fraction,
        "bsj_stability": bsj_stability,
        "bsj_confidence": bsj_confidence,
        "bsj_3d_closure_tightness": bsj_3d_closure_tightness,
        "energy_score": float(e1_aa),
        # G 物理指纹
        "mechanical_stiffness": mechanical_stiffness,
        "solvent_response": solvent_response,
    }


if __name__ == "__main__":
    # 自测: 用正多边形 P 坐标重建后算指标
    from .allatom_reconstruct import reconstruct_all_atom
    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"  # 32nt
    L = len(seq)
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    ps = np.stack([R * np.cos(angles), R * np.sin(angles),
                   np.zeros(L)], axis=1)
    s = reconstruct_all_atom(ps, seq)
    coords_aa = np.array([a.xyz for a in s.atoms], dtype=np.float64)
    pairs = [(i, L - 1 - i, 1.0) for i in range(L // 2)]
    fp = compute_immune_fingerprints(coords_aa, s, pairs, seq)
    print(f"L={L} 指标数={len(fp)}")
    for k, v in fp.items():
        print(f"  {k}: shape={v.shape} min={v.min():.3f} max={v.max():.3f}")
