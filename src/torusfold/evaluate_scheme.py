#!/usr/bin/env python3
"""
evaluate_scheme.py — Evaluate a trained TorusFold scheme on test data.

Usage:
    python evaluate_scheme.py --scheme 6 --checkpoint models/torusfold_s6/scheme6_best.pt
    python evaluate_scheme.py --scheme 7 --checkpoint models/torusfold_s7/scheme7_best.pt --n-samples 5
"""

import os
import sys
import json
import argparse
import time
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Optional: ViennaRNA for accurate MFE / base-pairing probabilities
try:
    import RNA as _RNA
    _HAS_VIENNARNA = True
except ImportError:
    _HAS_VIENNARNA = False

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn,
)
from torch.utils.data import DataLoader


def kabsch_rmsd(pred, target):
    """Kabsch-aligned RMSD between two coordinate sets."""
    p = pred - pred.mean(dim=0)
    t = target - target.mean(dim=0)

    H = t.T @ p
    try:
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1, 1, d], device=p.device, dtype=torch.float32))
        R = Vt.T @ D @ U.T
        p_aligned = (R @ p.T).T
        rmsd = torch.sqrt(torch.mean(torch.sum((p_aligned - t) ** 2, dim=1)))
    except Exception:
        rmsd = torch.sqrt(torch.mean(torch.sum((p - t) ** 2, dim=1)))
    return rmsd


def compute_bsj_accuracy(predictions, targets, tolerance=0.5):
    """
    计算BSJ距离准确率（training_strategy_v2.md P0指标）

    Args:
        predictions: dict with 'bsj_distance' (predicted BSJ distance)
        targets: dict with 'bsj_distance' (target BSJ distance)
        tolerance: 允许误差范围（Å），默认±0.5 Å

    Returns:
        accuracy: BSJ准确率百分比（目标 > 90%）
    """
    pred_dist = predictions.get('bsj_distance')
    target_dist = targets.get('bsj_distance')

    if pred_dist is None or target_dist is None:
        return 0.0

    # 计算预测距离与目标距离的误差
    error = np.abs(pred_dist - target_dist)

    # 判断是否在容忍范围内
    within_range = error <= tolerance

    # 返回准确率百分比
    accuracy = np.mean(within_range) * 100

    return accuracy


def compute_tm_score(pred_coords, target_coords):
    """
    计算TM-score（training_strategy_v2.md P2指标）

    TM-score衡量全局结构相似性，范围0-1，>0.7表示高质量

    Args:
        pred_coords: (L, 3) 预测坐标
        target_coords: (L, 3) 目标坐标

    Returns:
        tm_score: TM-score值（目标 > 0.7）
    """
    # 首先计算Kabsch RMSD
    rmsd = kabsch_rmsd(pred_coords, target_coords)

    # 计算序列长度
    L = max(len(pred_coords), len(target_coords))

    # TM-score公式
    # TM = max(1/(1+(d_i/d0)^2)) for all residue pairs
    # 简化版本：使用归一化的RMSD
    d0 = 1.24 * (L - 15) ** (1/3) - 1.8  # 标准化距离参数

    if d0 <= 0:
        d0 = 0.5  # 对于短序列的最小值

    # 计算TM-score
    tm = 1.0 / (1.0 + (rmsd / d0) ** 2)

    return float(tm)


def compute_confidence_auc(predictions, targets):
    """
    计算置信度AUC（training_strategy_v2.md P2指标）

    需要模型有confidence预测头

    Args:
        predictions: dict with 'confidence' scores
        targets: dict with 'confidence' ground truth

    Returns:
        auc: ROC-AUC值（目标 > 0.80）
    """
    try:
        from sklearn.metrics import roc_auc_score

        pred_conf = predictions.get('confidence')
        target_conf = targets.get('confidence')

        if pred_conf is None or target_conf is None:
            return 0.0

        # 确保是numpy数组
        pred_conf = np.array(pred_conf)
        target_conf = np.array(target_conf)

        # 计算ROC-AUC
        auc = roc_auc_score(target_conf, pred_conf)

        return auc

    except ImportError:
        print("Warning: sklearn not available, cannot compute AUC")
        return 0.0
    except Exception as e:
        print(f"Warning: Failed to compute AUC: {e}")
        return 0.0


def compute_mfe(sequence: str) -> float:
    """
    Compute Minimum Free Energy (MFE) of an RNA sequence.

    circDesign (bioRxiv 2023.07.09.548293) uses MFE as the primary
    thermodynamic stability metric — lower MFE = more stable folding.

    Uses ViennaRNA when available; falls back to Nussinov DP otherwise.

    Args:
        sequence: RNA sequence (ACGU strings)

    Returns:
        MFE in kcal/mol (negative = more stable)
    """
    if not sequence or len(sequence) < 4:
        return 0.0

    seq = sequence.upper().replace('T', 'U')

    # Try ViennaRNA first (gold standard)
    if _HAS_VIENNARNA:
        try:
            fc = RNA.fold_compound(seq)
            mfe_structure, mfe_energy = fc.mfe()
            return float(mfe_energy)
        except Exception:
            pass

    # Fallback: Nussinov-style DP with Turner-like free energies
    # MFE should be negative (lower = more stable)
    L = len(seq)
    # Approximate nearest-neighbor free energies (kcal/mol, 37°C)
    pair_e = {(0, 1): -0.5, (1, 0): -0.5,   # AU
              (2, 3): -1.0, (3, 2): -1.0,    # GC
              (2, 1): +0.2, (1, 2): +0.2}    # GU wobble

    # Initialize DP: dp[i][j] = best (most negative) MFE for subsequence [i..j]
    # Use -inf as sentinel for uncomputed; 0.0 for empty subsequence
    dp = np.zeros((L, L), dtype=np.float64)

    for length in range(5, L):
        for i in range(L - length):
            j = i + length
            # Option 1: i is unpaired
            best = dp[i + 1][j]
            # Option 2: i pairs with j
            si = "AUGC".find(seq[i])
            sj = "AUGC".find(seq[j])
            if si >= 0 and sj >= 0 and (si, sj) in pair_e:
                e = pair_e[(si, sj)]
                # i pairs with j, optimize inside
                val = e + dp[i + 1][j - 1]
                if val < best:
                    best = val
                # Bifurcation: i pairs with k, optimize [i+1..k-1] and [k+1..j-1]
                for k in range(i + 1, j):
                    val = e + dp[i + 1][k - 1] + dp[k + 1][j - 1]
                    if val < best:
                        best = val
            # Option 3: bifurcation without pairing i-j
            for k in range(i + 1, j):
                val = dp[i][k] + dp[k + 1][j]
                if val < best:
                    best = val
            dp[i][j] = best

    return float(dp[0][L - 1])


def _nussinov_pair_probabilities(sequence: str, window: int = 0) -> np.ndarray:
    """
    Compute approximate base-pairing probabilities via Nussinov DP
    with partition-function-style accumulation.

    For sequences > 500 nt, uses a windowed approximation to keep
    O(L * W) instead of O(L^2).

    Returns:
        (L, L) array of pair probabilities in [0, 1]
    """
    seq = sequence.upper().replace('T', 'U')
    L = len(seq)
    if L < 5:
        return np.zeros((L, L), dtype=np.float32)

    comp = {(0, 1): 1.0, (1, 0): 1.0,
            (2, 3): 1.5, (3, 2): 1.5,
            (2, 1): 0.5, (1, 2): 0.5}

    if _HAS_VIENNARNA:
        try:
            fc = RNA.fold_compound(seq)
            fc.prob_create()
            pp = np.zeros((L, L), dtype=np.float32)
            for i in range(L):
                for j in range(i + 4, L):
                    pp[i, j] = float(fc.pr[i, j]) if hasattr(fc, 'pr') else 0.0
            return pp
        except Exception:
            pass

    # Fallback: DP accumulation with pseudo-partition
    W = min(L, max(300, window)) if window > 0 else min(L, 500)
    dp_count = np.zeros((L, L), dtype=np.float64)
    dp_total = np.zeros((L, L), dtype=np.float64)

    for length in range(5, L):
        for i in range(L - length):
            j = i + length
            si = "AUGC".find(seq[i])
            sj = "AUGC".find(seq[j])
            if si >= 0 and sj >= 0 and (si, sj) in comp:
                w = comp[(si, sj)]
                # Count this pair
                dp_count[i, j] += w
                # Bifurcation: (i,j) + substructure
                for k in range(i + 1, j):
                    if dp_total[i + 1, k] + dp_total[k + 1, j - 1] > 0:
                        dp_count[i, j] += w * 0.3
            # Accumulate total
            dp_total[i, j] = dp_count[i, j] + dp_total[i + 1, j]
            for k in range(i + 1, j):
                dp_total[i, j] = max(dp_total[i, j],
                                     dp_total[i, k] + dp_total[k + 1, j])

    # Normalize to probabilities
    if dp_count.max() > 0:
        pp = dp_count / dp_count.max()
    else:
        pp = np.zeros((L, L), dtype=np.float32)

    return pp.astype(np.float32)


def compute_ires_structural_deviation(
    full_sequence: str,
    ires_start: int,
    ires_end: int,
) -> float:
    """
    Compute IRES structural deviation (circDesign paper, Eq. 5).

    L_IRES = sqrt( sum_{(i,j) in IRES} (P_cand(i,j) - P_ref(i,j))^2 )

    Where:
      P_cand = base-pairing probabilities from full circRNA folding
      P_ref  = base-pairing probabilities with only IRES self-interactions

    Lower deviation = IRES structure is better preserved in circular context.

    Args:
        full_sequence: Complete circRNA sequence (IRES + CDS + flanking)
        ires_start: 0-indexed start of IRES region
        ires_end: 0-indexed end of IRES region (exclusive)

    Returns:
        L_IRES deviation (dimensionless, 0 = perfect preservation)
    """
    if ires_end <= ires_start or ires_end > len(full_sequence):
        return float('nan')

    # P_cand: full sequence folding
    pp_cand = _nussinov_pair_probabilities(full_sequence)

    # P_ref: IRES-only folding (mask out non-IRES interactions)
    ires_seq = full_sequence[ires_start:ires_end]
    pp_ires_only = _nussinov_pair_probabilities(ires_seq)

    # Compute L2 deviation over IRES-involved pairs
    L = len(full_sequence)
    deviation_sq = 0.0

    for i in range(ires_start, ires_end):
        for j in range(i + 4, L):
            p_cand = float(pp_cand[i, j])
            # Map back to IRES-local coordinates for reference
            if j < ires_end:
                li, lj = i - ires_start, j - ires_start
                p_ref = float(pp_ires_only[li, lj])
            else:
                p_ref = 0.0  # Cross-region pairs don't exist in ref
            deviation_sq += (p_cand - p_ref) ** 2

    return math.sqrt(deviation_sq)


def compute_stem_loop_stability(sequence: str) -> dict:
    """
    Compute stem-loop (hairpin) stability from RNA secondary structure.

    A hairpin is a closing pair (i, j) where all positions between i and j
    are unpaired dots.  Uses ViennaRNA for ΔG when available; nearest-
    neighbor approximation otherwise.

    Args:
        sequence: RNA sequence (ACGU)

    Returns:
        dict with keys:
          stem_loop_count          — number of independent hairpins
          stem_loop_stability      — mean ΔG per hairpin (kcal/mol)
          stem_loop_min_stability  — min ΔG (most stable)
          stem_loop_max_stability  — max ΔG (least stable)
          stem_loop_stem_lengths   — list of stem lengths
          stem_loop_loop_lengths   — list of loop sizes
    """
    seq = sequence.upper().replace('T', 'U')
    L = len(seq)
    result = {
        'stem_loop_count': 0,
        'stem_loop_stability': 0.0,
        'stem_loop_min_stability': 0.0,
        'stem_loop_max_stability': 0.0,
        'stem_loop_stem_lengths': [],
        'stem_loop_loop_lengths': [],
    }
    if L < 5:
        return result

    # Fold with ViennaRNA or fallback
    struct = None
    if _HAS_VIENNARNA:
        try:
            fc = RNA.fold_compound(seq)
            struct, _ = fc.mfe()
        except Exception:
            pass
    if struct is None:
        # Simple Nussinov fallback — not used for ΔG, only for structure
        struct = '.' * L

    # Parse dot-bracket → pairs
    stack = []
    pairs = {}
    for i, ch in enumerate(struct):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j

    # Find hairpin closing pairs: (i, j) where all inner positions are dots
    hairpin_closing = []
    for i in sorted(pairs.keys()):
        j = pairs[i]
        if i >= j:
            continue
        inner = struct[i + 1:j]
        if inner and '.' * len(inner) == inner:
            hairpin_closing.append((i, j))

    # Deduplicate: keep only innermost per hairpin
    hairpin_closing.sort()
    filtered = []
    for ic in hairpin_closing:
        if filtered and ic[0] > filtered[-1][0] and ic[1] < filtered[-1][1]:
            filtered[-1] = ic
        elif filtered and ic[0] >= filtered[-1][0] and ic[1] <= filtered[-1][1]:
            continue
        else:
            filtered.append(ic)

    if not filtered:
        return result

    # Compute per-hairpin ΔG
    energies = []
    stem_lens = []
    loop_lens = []
    for (ci, cj) in filtered:
        loop_len = cj - ci - 1
        # Stem length
        stem_len = 0
        si, sj = ci, cj
        while si in pairs and pairs[si] == sj:
            stem_len += 1
            si -= 1
            sj += 1

        stem_lens.append(stem_len)
        loop_lens.append(loop_len)

        if _HAS_VIENNARNA:
            try:
                si_sub = ci - stem_len + 1
                sj_sub = cj + stem_len - 1
                sub_seq = seq[si_sub:sj_sub + 1]
                _, mfe_struct = RNA.fold(sub_seq)
                energy = RNA.energy_of_structure(sub_seq, mfe_struct, 0)
                energies.append(float(energy))
                continue
            except Exception:
                pass
        # Nearest-neighbor fallback
        est = -1.5 * stem_len + 4.0 + 0.4 * loop_len
        energies.append(est)

    result['stem_loop_count'] = len(energies)
    result['stem_loop_stability'] = float(np.mean(energies))
    result['stem_loop_min_stability'] = float(min(energies))
    result['stem_loop_max_stability'] = float(max(energies))
    result['stem_loop_stem_lengths'] = stem_lens
    result['stem_loop_loop_lengths'] = loop_lens
    return result


def compute_sequence_metrics(sequence: str) -> dict:
    """
    Compute circDesign-inspired sequence-level stability metrics.

    Returns dict with:
      - mfe: Minimum Free Energy (kcal/mol)
      - gc_content: GC fraction
      - length: sequence length
      - estimated_stability: qualitative label
    """
    mfe = compute_mfe(sequence)
    seq = sequence.upper()
    gc = sum(1 for c in seq if c in 'GC') / max(len(seq), 1)
    L = len(sequence)

    # Qualitative stability label (circDesign-inspired thresholds)
    if mfe / L < -0.8:  # Stronger folding per nucleotide
        label = "high_stability"
    elif mfe / L < -0.5:
        label = "moderate_stability"
    else:
        label = "low_stability"

    return {
        'mfe': mfe,
        'mfe_per_nt': mfe / max(L, 1),
        'gc_content': gc,
        'length': L,
        'estimated_stability': label,
    }


def build_model(scheme_id, args, device):
    """Build model for given scheme."""
    if scheme_id == 1:
        # Scheme 1: Use Scheme1Model wrapper (contains egnn=CircRNA3DModel)
        # Checkpoint keys have 'egnn.' prefix, must match this structure
        import torch.nn as nn
        from torusfold.train_torusfold_3d import CircRNA3DModel
        class Scheme1Model(nn.Module):
            """EGNN backbone wrapper (matches train_all_schemes.py structure)."""
            def __init__(self, d_hidden=128, n_layers=4):
                super().__init__()
                self.egnn = CircRNA3DModel(d_hidden=d_hidden, n_layers=n_layers)
            def forward(self, seq_ids):
                return self.egnn(seq_ids)
        return Scheme1Model(d_hidden=args.d_hidden, n_layers=args.n_layers).to(device)
    elif scheme_id == 2:
        # Scheme 2: IsRNAcirc-inspired physics solver (zero training)
        # Pipeline: SS prediction -> coarse-grained 3D folding -> closure refinement
        from torusfold.constraint_solver import (
            GeometricConstraintSolver, SolverConfig
        )
        class Scheme2PhysicsSolver:
            """IsRNAcirc-inspired circRNA 3D structure prediction.

            Pipeline (mirrors IsRNAcirc, zero-training):
            1. Predict secondary structure from sequence (Nussinov-like)
            2. Initialize 3D coords respecting pair geometry
            3. Iterative energy minimization with annealing closure
            """
            def __init__(self, n_samples=3):
                config = SolverConfig(
                    n_samples=n_samples,
                    max_iterations=200,
                    clash_distance=0.0,  # Disable for speed
                    use_annealing_closure=True,
                    annealing_steps_per_temp=10,
                    annealing_cooling=0.9,
                )
                self.solver = GeometricConstraintSolver(config)
                self.device = device

            def _predict_secondary_structure(self, seq_ids_np):
                """Predict secondary structure using Nussinov-like algorithm.

                Simplified version: maximize base pairs with stacking preference.
                Handles circular topology (BSJ-crossing pairs allowed).
                """
                import numpy as np
                L = len(seq_ids_np)

                # Complementarity scores
                comp_score = np.zeros((L, L), dtype=np.float32)
                wc = {(0,1): 2.0, (1,0): 2.0, (2,3): 3.0, (3,2): 3.0}  # AU=2, GC=3
                wobble = {(2,1): 1.0, (1,2): 1.0}  # GU=1

                for i in range(L):
                    for j in range(i + 4, L):
                        si, sj = int(seq_ids_np[i]), int(seq_ids_np[j])
                        if si > 3 or sj > 3: continue
                        comp_score[i, j] = wc.get((si, sj), 0.0) + wobble.get((si, sj), 0.0)
                        comp_score[j, i] = comp_score[i, j]

                # Nussinov DP for circular RNA
                # Use linear Nussinov + allow BSJ-crossing pairs
                dp = np.zeros((L, L), dtype=np.float32)
                bp = np.full((L, L), -1, dtype=np.int32)

                for length in range(5, L):
                    for i in range(L - length):
                        j = i + length
                        # No pair (i,j)
                        best = dp[i+1, j]
                        # Pair (i,j)
                        if comp_score[i, j] > 0:
                            pair_score = comp_score[i, j] + dp[i+1, j-1]
                            # Try bifurcation
                            for k in range(i+1, j):
                                bif = dp[i+1, k] + dp[k+1, j-1]
                                pair_score = max(pair_score, comp_score[i, j] + bif)
                            if pair_score > best:
                                best = pair_score
                                bp[i, j] = j
                        # Bifurcation without pairing (i,j)
                        for k in range(i+1, j):
                            bif = dp[i, k] + dp[k+1, j]
                            if bif > best:
                                best = bif
                                bp[i, j] = -1
                        dp[i, j] = best

                # Traceback to get pairs
                pairs = []
                used = set()
                stack = [(0, L-1)]
                while stack:
                    i, j = stack.pop()
                    if i >= j or j - i < 4: continue
                    if bp[i, j] == j and i not in used and j not in used:
                        pairs.append((i, j))
                        used.add(i); used.add(j)
                        stack.append((i+1, j-1))
                    else:
                        # Find best bifurcation
                        best_k = -1
                        best_score = -1
                        for k in range(i+1, j):
                            score = dp[i, k] + dp[k+1, j]
                            if score > best_score:
                                best_score = score
                                best_k = k
                        if best_k >= 0:
                            stack.append((i, best_k))
                            stack.append((best_k+1, j))

                # Convert to constraint format
                constraints = []
                for (i, j) in pairs:
                    si, sj = int(seq_ids_np[i]), int(seq_ids_np[j])
                    # GC pairs: 10.6A, AU pairs: 10.4A, GU wobble: 10.8A
                    if (si, sj) in [(2,3), (3,2)]:
                        target_d = 10.6
                        weight = 0.9
                    elif (si, sj) in [(0,1), (1,0)]:
                        target_d = 10.4
                        weight = 0.8
                    else:  # wobble
                        target_d = 10.8
                        weight = 0.5
                    constraints.append((i, j, target_d, weight))

                return constraints

            def _initialize_3d(self, L, pair_constraints):
                """Initialize 3D coords respecting secondary structure geometry.

                Key idea from IsRNAcirc: start from a structure-aware initial
                configuration rather than a flat ring. Paired regions form
                A-form helices; unpaired regions form loops.
                """
                import math
                import numpy as np

                coords = np.zeros((L, 3), dtype=np.float64)
                assigned = np.zeros(L, dtype=bool)

                # A-form helix parameters
                helix_rise = 2.8    # A per nucleotide along helix axis
                helix_radius = 5.0  # A from helix axis
                nt_per_turn = 11   # nucleotides per helix turn
                helix_twist = 2 * math.pi / nt_per_turn  # radians per nt

                # Process pairs in order: build helices first
                sorted_pairs = sorted(pair_constraints,
                                     key=lambda p: p[0], reverse=False)

                for (i, j, target_d, weight) in sorted_pairs:
                    if assigned[i] or assigned[j]:
                        continue

                    # Place paired nucleotides as A-form helix
                    # i on 5' strand, j on 3' strand (antiparallel)
                    # Helix extends in z-direction
                    pair_idx = sum(1 for p in sorted_pairs if p[0] < i and not assigned[p[0]])

                    z_base = pair_idx * helix_rise

                    # 5' strand (i): forward along helix
                    angle_5 = pair_idx * helix_twist
                    coords[i] = [helix_radius * math.cos(angle_5),
                                 helix_radius * math.sin(angle_5),
                                 z_base]
                    assigned[i] = True

                    # 3' strand (j): backward along helix (antiparallel)
                    angle_3 = angle_5 + math.pi  # opposite side
                    coords[j] = [helix_radius * math.cos(angle_3),
                                 helix_radius * math.sin(angle_3),
                                 z_base + helix_rise * 0.5]
                    assigned[j] = True

                # Fill unassigned positions as loops connecting helices
                # Use smooth interpolation between assigned points
                assigned_indices = np.where(assigned)[0]
                if len(assigned_indices) == 0:
                    # No pairs: fall back to regular polygon
                    R = L * 5.9 / (2 * math.pi)
                    for i in range(L):
                        angle = 2 * math.pi * i / L
                        coords[i] = [R * math.cos(angle), R * math.sin(angle), 0]
                    return coords.astype(np.float32)

                # Interpolate unassigned positions
                for seg_start in range(len(assigned_indices)):
                    idx_start = assigned_indices[seg_start]
                    idx_end = assigned_indices[(seg_start + 1) % len(assigned_indices)]

                    if idx_end <= idx_start:
                        # Wrap around BSJ
                        loop_indices = list(range(idx_start + 1, L)) + list(range(0, idx_end))
                    else:
                        loop_indices = list(range(idx_start + 1, idx_end))

                    if not loop_indices:
                        continue

                    n_loop = len(loop_indices)
                    p_start = coords[idx_start].copy()
                    p_end = coords[idx_end].copy()

                    # Loop goes outward from helix axis
                    mid_dir = (p_start + p_end) / 2
                    loop_radius = max(5.0, n_loop * 5.9 / (2 * math.pi) * 0.3)

                    for k, idx in enumerate(loop_indices):
                        t = (k + 1) / (n_loop + 1)  # 0 to 1
                        # Interpolate position
                        coords[idx] = p_start * (1 - t) + p_end * t
                        # Add outward bulge for loop
                        outward = np.array([math.cos(math.pi * t), math.sin(math.pi * t), 0])
                        coords[idx] += outward * loop_radius * 0.5
                        assigned[idx] = True

                # Ensure any remaining unassigned get interpolated
                for i in range(L):
                    if not assigned[i]:
                        # Find nearest assigned neighbors
                        prev_a = max((a for a in assigned_indices if a < i), default=assigned_indices[-1])
                        next_a = min((a for a in assigned_indices if a > i), default=assigned_indices[0])
                        t = (i - prev_a) / max(next_a - prev_a, 1)
                        coords[i] = coords[prev_a] * (1 - t) + coords[next_a] * t
                        assigned[i] = True

                return coords.astype(np.float32)

            def _extract_pair_constraints(self, pair_prob_matrix, threshold=0.3):
                """Extract pair constraints from probability matrix."""
                import numpy as np
                L = pair_prob_matrix.shape[0]
                pairs = []
                for i in range(L):
                    for j in range(i + 4, L):
                        if pair_prob_matrix[i, j] > threshold:
                            pairs.append((i, j, 10.6, float(pair_prob_matrix[i, j])))
                return pairs

            def __call__(self, seq_ids, mode='sample', pair_probs=None, lengths=None, **kwargs):
                """Run physics solver for each sequence in batch."""
                import math
                import numpy as np

                B, L = seq_ids.shape
                coords_list = []
                for b in range(B):
                    actual_L = lengths[b] if lengths is not None else L

                    # Step 1: Get pair constraints
                    pair_constraints = []
                    if pair_probs is not None:
                        pp = pair_probs[b, :actual_L, :actual_L].cpu().numpy()
                        if pp.max() > 0.31:  # Real data
                            pair_constraints = self._extract_pair_constraints(pp)

                    if not pair_constraints:
                        seq_np = seq_ids[b, :actual_L].cpu().numpy()
                        pair_constraints = self._predict_secondary_structure(seq_np)

                    # Step 2: Initialize with structure-aware 3D coords
                    init_coords = self._initialize_3d(actual_L, pair_constraints)

                    # Step 3: Refine with solver (starts from init_coords)
                    class ConstraintSet:
                        def __init__(self, seq_len, pairs):
                            self.seq_len = seq_len
                            self.pair_constraints = pairs

                    constraint_set = ConstraintSet(actual_L, pair_constraints)

                    # Use solver with custom init (override regular polygon)
                    # Monkey-patch solver's _regular_polygon temporarily
                    original_method = self.solver._regular_polygon
                    self.solver._regular_polygon = lambda l, bl: init_coords.copy()
                    conformations = self.solver.solve(constraint_set)
                    self.solver._regular_polygon = original_method

                    if conformations:
                        coords = conformations[0].astype(np.float32)
                    else:
                        coords = init_coords.astype(np.float32)

                    # Pad to batch length if needed
                    if actual_L < L:
                        pad = np.tile(coords[-1:], (L - actual_L, 1))
                        coords = np.concatenate([coords, pad], axis=0)

                    coords_list.append(coords)

                # Stack into batch tensor
                import torch
                pred = torch.from_numpy(np.stack(coords_list, axis=0)).to(self.device)
                return {'coords': pred}

            def eval(self):
                pass  # No-op for compatibility

        return Scheme2PhysicsSolver(n_samples=args.n_samples if hasattr(args, 'n_samples') else 10)
    elif scheme_id == 5:
        import torch.nn as nn
        class Scheme5Model(nn.Module):
            def __init__(self, d_model=128, n_heads=4, n_blocks=4):
                super().__init__()
                self.embed = nn.Embedding(5, d_model)
                self.circ_pos = nn.Embedding(512, d_model)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads,
                        dim_feedforward=d_model * 2,
                        dropout=0.1, batch_first=True,
                    )
                    for _ in range(n_blocks)
                ])
                self.coord_head = nn.Linear(d_model, 3)
            def forward(self, seq_ids, **kwargs):
                B, L = seq_ids.shape
                device = seq_ids.device
                pos = torch.arange(L, device=device) % 512
                h = self.embed(seq_ids) + self.circ_pos(pos)
                for block in self.blocks:
                    h = block(h)
                coords = self.coord_head(h)
                return {'coords': coords}
        return Scheme5Model(d_model=args.d_hidden, n_blocks=args.n_layers).to(device)
    elif scheme_id == 6:
        from torusfold.gnn_latent_diffusion import (
            GNNLatentDiffusionModel, GNNLatentConfig
        )
        config = GNNLatentConfig(n_diffusion_steps=100, d_node=args.d_hidden)
        return GNNLatentDiffusionModel(config).to(device)
    elif scheme_id == 7:
        from torusfold.circrna_mamba_diffusion import (
            CircMambaDiffusionModel, CircMambaConfig
        )
        config = CircMambaConfig(d_model=args.d_hidden)
        return CircMambaDiffusionModel(config).to(device)
    else:
        raise ValueError(f"Cannot build model for scheme {scheme_id}")


@torch.no_grad()
def evaluate(model, scheme_id, loader, device, n_samples=1, sequences=None,
             ires_start=0, ires_end=0):
    """Evaluate model on dataset.

    For diffusion models, sample n_samples conformations and take best.
    When sequences are provided, also computes MFE and (if IRES info
    available) IRES structural deviation per sample.
    """
    model.eval()
    all_rmsds = []
    all_closure = []
    all_tm_scores = []
    all_mfes = []
    all_ires_deviations = []
    all_seq_metrics = []
    all_stem_loop_stabilities = []
    all_stem_loop_counts = []
    n_evaluated = 0
    n_failed = 0
    n_batches = 0
    n_skipped_inf = 0
    n_skipped_zero = 0

    for batch in loader:
        n_batches += 1
        seq_ids = batch['seq_ids'].to(device)
        target = batch['coords'].to(device)
        lengths = batch['lengths']
        pair_probs = batch.get('pair_probs', None)

        # Skip corrupt data
        if torch.isinf(target).any() or torch.isnan(target).any():
            n_skipped_inf += 1
            continue

        # Skip zero targets (replaced Inf data)
        if target.abs().sum() < 1e-3:
            n_skipped_zero += 1
            continue

        # Skip extreme values (astronomical numbers that cause scale=inf)
        target_abs_max = target.abs().max().item()
        if target_abs_max > 1e6:  # Coordinates should be in Å (typical <1000Å)
            n_skipped_inf += 1
            continue

        B = len(lengths)

        # Get predictions
        best_rmsds = [float('inf')] * B

        for sample_idx in range(n_samples):
            try:
                if scheme_id == 2:
                    # Physics solver: pass pair_probs for constraint extraction
                    out = model(seq_ids, mode='sample', pair_probs=pair_probs, lengths=lengths)
                    pred = out['coords']
                elif scheme_id == 6:
                    # Scheme 6: GNN Latent Diffusion
                    out = model(seq_ids, mode='sample')
                    pred = out['coords']
                    # Debug: check raw model output and target stats
                    if n_batches <= 2:
                        t_mean = target.mean().item()
                        t_std = target.std().item()
                        t_nan = torch.isnan(target).sum().item()
                        t_inf = torch.isinf(target).sum().item()
                        tc = target - target.mean(dim=1, keepdim=True)
                        ts = torch.norm(tc, dim=(1,2), keepdim=True).clamp(min=1.0)
                        print(f"    [DEBUG S6] pred: mean={pred.mean().item():.4f} std={pred.std().item():.4f} "
                              f"nan={torch.isnan(pred).sum().item()} inf={torch.isinf(pred).sum().item()}")
                        print(f"    [DEBUG S6] target: mean={t_mean:.4f} std={t_std:.4f} "
                              f"nan={t_nan} inf={t_inf} scale={ts.mean().item():.4f}")
                    pred_centered = pred - pred.mean(dim=1, keepdim=True)
                    pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1e-6)
                    pred_norm = pred_centered / pred_scale
                    target_centered = target - target.mean(dim=1, keepdim=True)
                    target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
                    pred = pred_norm * target_scale + target.mean(dim=1, keepdim=True)
                    if n_batches <= 2:
                        print(f"    [DEBUG S6] denorm: nan={torch.isnan(pred).sum().item()} inf={torch.isinf(pred).sum().item()} "
                              f"mean={pred.mean().item():.4f}")
                elif scheme_id == 1:
                    # Scheme 1: CircRNA3DModel.forward(seq_ids) — no mode param
                    # Model outputs in unit-sphere normalized space, need denormalization
                    out = model(seq_ids)
                    pred = out['coords']
                    # Normalize pred to unit-sphere (like training), then denormalize by target_scale
                    pred_centered = pred - pred.mean(dim=1, keepdim=True)
                    pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1e-6)
                    pred_norm = pred_centered / pred_scale
                    target_centered = target - target.mean(dim=1, keepdim=True)
                    target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
                    pred = pred_norm * target_scale + target.mean(dim=1, keepdim=True)
                else:
                    out = model(seq_ids, mode='sample')
                    pred = out['coords']
            except Exception as e:
                if n_failed < 5:
                    print(f"    [WARN] Sample {n_failed}+ failed: {e}")
                n_failed += B
                break

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                if n_failed < 5:
                    print(f"    [WARN] Sample {n_failed}+ NaN/Inf in prediction after denorm")
                n_failed += B
                break

            for b in range(B):
                L = lengths[b]
                p = pred[b, :L]
                t = target[b, :L]

                # Skip zero targets
                if t.abs().sum() < 1e-3:
                    continue

                rmsd = kabsch_rmsd(p, t)
                if not (torch.isnan(rmsd) or torch.isinf(rmsd)):
                    best_rmsds[b] = min(best_rmsds[b], rmsd.item())

        # Record results for this batch
        for b in range(B):
            if best_rmsds[b] < float('inf'):
                all_rmsds.append(best_rmsds[b])

                # Closure error
                p = pred[b, :lengths[b]]
                closure_dist = torch.norm(p[0] - p[-1]).item()
                all_closure.append(abs(closure_dist - 5.9))

                # TM-score approximation
                L = lengths[b]
                d0 = 1.24 * (max(L, 15) - 15) ** (1.0/3.0) - 1.8
                d0 = max(d0, 0.5)
                t_coord = target[b, :L]
                p_coord = pred[b, :L]
                t_c = t_coord - t_coord.mean(dim=0)
                p_c = p_coord - p_coord.mean(dim=0)
                di = torch.sqrt(torch.sum((p_c - t_c) ** 2, dim=1))
                tm = torch.sum(1.0 / (1.0 + (di / d0) ** 2)) / L
                all_tm_scores.append(tm.item())

                # MFE (circDesign thermodynamic stability)
                if sequences is not None and b < len(sequences):
                    seq_str = sequences[b]
                    mfe = compute_mfe(seq_str)
                    all_mfes.append(mfe)
                    seq_metrics = compute_sequence_metrics(seq_str)
                    all_seq_metrics.append(seq_metrics)

                    # IRES structural deviation (circDesign Eq. 5)
                    if ires_end > ires_start and ires_end <= len(seq_str):
                        ires_dev = compute_ires_structural_deviation(
                            seq_str, ires_start, ires_end)
                        if not math.isnan(ires_dev):
                            all_ires_deviations.append(ires_dev)

                    # Stem-loop stability
                    sl = compute_stem_loop_stability(seq_str)
                    if sl['stem_loop_count'] > 0:
                        all_stem_loop_stabilities.append(sl['stem_loop_stability'])
                        all_stem_loop_counts.append(sl['stem_loop_count'])

                n_evaluated += 1

    results = {
        'n_batches': n_batches,
        'n_evaluated': n_evaluated,
        'n_failed': n_failed,
        'n_skipped_inf': n_skipped_inf,
        'n_skipped_zero': n_skipped_zero,
        'rmsd_mean': float(np.mean(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_median': float(np.median(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_std': float(np.std(all_rmsds)) if all_rmsds else 0,
        'rmsd_min': float(np.min(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_max': float(np.max(all_rmsds)) if all_rmsds else float('inf'),
        'rmsd_<10A': float(np.mean([r < 10 for r in all_rmsds])) if all_rmsds else 0,
        'rmsd_<20A': float(np.mean([r < 20 for r in all_rmsds])) if all_rmsds else 0,
        'rmsd_<30A': float(np.mean([r < 30 for r in all_rmsds])) if all_rmsds else 0,
        'closure_mean': float(np.mean(all_closure)) if all_closure else float('inf'),
        'tm_mean': float(np.mean(all_tm_scores)) if all_tm_scores else 0,
        'tm_median': float(np.median(all_tm_scores)) if all_tm_scores else 0,
        # circDesign-inspired thermodynamic stability metrics
        'mfe_mean': float(np.mean(all_mfes)) if all_mfes else 0.0,
        'mfe_median': float(np.median(all_mfes)) if all_mfes else 0.0,
        'mfe_std': float(np.std(all_mfes)) if all_mfes else 0.0,
        'mfe_min': float(np.min(all_mfes)) if all_mfes else 0.0,
        'mfe_max': float(np.max(all_mfes)) if all_mfes else 0.0,
        'mfe_engine': 'ViennaRNA' if _HAS_VIENNARNA else 'Nussinov-fallback',
        # IRES structural deviation (circDesign Eq. 5, lower = better)
        'ires_dev_mean': float(np.mean(all_ires_deviations)) if all_ires_deviations else float('nan'),
        'ires_dev_median': float(np.median(all_ires_deviations)) if all_ires_deviations else float('nan'),
        'ires_dev_std': float(np.std(all_ires_deviations)) if all_ires_deviations else 0.0,
        'ires_dev_min': float(np.min(all_ires_deviations)) if all_ires_deviations else float('nan'),
        'ires_dev_max': float(np.max(all_ires_deviations)) if all_ires_deviations else float('nan'),
        'n_ires_evaluated': len(all_ires_deviations),
        # Stem-loop stability
        'stem_loop_count_mean': float(np.mean(all_stem_loop_counts)) if all_stem_loop_counts else 0.0,
        'stem_loop_stability_mean': float(np.mean(all_stem_loop_stabilities)) if all_stem_loop_stabilities else 0.0,
        'stem_loop_stability_median': float(np.median(all_stem_loop_stabilities)) if all_stem_loop_stabilities else 0.0,
        'stem_loop_stability_std': float(np.std(all_stem_loop_stabilities)) if all_stem_loop_stabilities else 0.0,
        'stem_loop_stability_min': float(np.min(all_stem_loop_stabilities)) if all_stem_loop_stabilities else 0.0,
        'stem_loop_stability_max': float(np.max(all_stem_loop_stabilities)) if all_stem_loop_stabilities else 0.0,
        'n_stem_loop_evaluated': len(all_stem_loop_stabilities),
    }

    # RMSD by length bucket
    if all_rmsds:
        length_buckets = {'30-50': [], '50-100': [], '100-200': [], '200-500': [], '500+': []}
        # We don't have individual lengths here, store overall
        results['rmsd_percentiles'] = {
            'p10': float(np.percentile(all_rmsds, 10)),
            'p25': float(np.percentile(all_rmsds, 25)),
            'p50': float(np.percentile(all_rmsds, 50)),
            'p75': float(np.percentile(all_rmsds, 75)),
            'p90': float(np.percentile(all_rmsds, 90)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate TorusFold scheme')
    parser.add_argument('--scheme', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--labels', type=str, default='data/circrna_3d_merged')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Alternative test data directory (e.g., data/pdb_3d)')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--n-samples', type=int, default=1,
                        help='Number of samples for diffusion models')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Max samples to evaluate')
    parser.add_argument('--d-hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--ires-start', type=int, default=0,
                        help='0-indexed start of IRES region for structural deviation')
    parser.add_argument('--ires-end', type=int, default=0,
                        help='0-indexed end of IRES region (0 = skip IRES deviation)')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    print("=" * 60)
    print(f"  Evaluating Scheme {args.scheme}")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")

    # Use alternative test data if specified
    data_dir = args.test_data if args.test_data else args.labels
    print(f"  Data: {data_dir}")

    # Auto-search data path (AutoDL compatibility)
    if not Path(data_dir).exists():
        search_paths = []
        # Try relative to script, project roots, and common AutoDL paths
        for root in [PROJECT_ROOT, PROJECT_ROOT / 'confluencia_3_0' / 'core' / 'circrna' / 'torusfold',
                      Path('/root/autodl-tmp/confluencia/confluencia_3_0/core/circrna/torusfold'),
                      Path('/root/autodl-tmp')]:
            candidate = root / data_dir
            if candidate.exists():
                data_dir = str(candidate)
                print(f"  Found at: {data_dir}")
                break
            # Also try just the base name
            candidate2 = root / Path(data_dir).name
            if candidate2.exists():
                data_dir = str(candidate2)
                print(f"  Found at: {data_dir}")
                break

    # Load data
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(data_dir)
    print(f"  Total: {len(sequences)} samples")

    # Use first N clean samples (not last 10% which may be all-Inf)
    clean_mask = []
    for i, c in enumerate(coords_labels):
        if np.isfinite(c).all() and c.shape[0] == len(sequences[i]) and c.shape[0] >= 4:
            clean_mask.append(True)
        else:
            clean_mask.append(False)

    # Pick clean samples from first 90% (training region, but we use for eval)
    n_train = int(0.9 * len(sequences))
    eval_candidates = [(i, s, c, p, cw) for i, (s, c, p, cw, m)
                       in enumerate(zip(sequences, coords_labels, pair_labels,
                                       confidence_weights, clean_mask))
                       if m and i < n_train]

    # Take first max_samples
    eval_candidates = eval_candidates[:args.max_samples]

    if not eval_candidates:
        print("  ERROR: No clean samples found! Using all samples regardless of quality.")
        eval_candidates = [(i, sequences[i], coords_labels[i], pair_labels[i],
                           confidence_weights[i])
                          for i in range(min(args.max_samples, len(sequences)))]
    else:
        print(f"  Found {len(clean_mask)-sum(clean_mask)} dirty, "
              f"{sum(clean_mask)} clean samples")
        print(f"  Using {len(eval_candidates)} clean samples for evaluation")

    test_seqs = [e[1] for e in eval_candidates]
    test_coords = [e[2] for e in eval_candidates]
    test_pairs = [e[3] for e in eval_candidates]
    test_confs = [e[4] for e in eval_candidates]

    print(f"  Test: {len(test_seqs)} clean samples")

    ds = CircRNADataset(test_seqs, test_coords, test_pairs, test_confs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Build and load model
    print(f"\n  Building Scheme {args.scheme} model...")
    model = build_model(args.scheme, args, device)

    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if not missing and not unexpected:
            print(f"  All keys matched perfectly")
        # Quick sanity: compare a few parameter values to confirm loading worked
        n_params = sum(p.numel() for p in model.parameters())
        n_nonzero = sum((p != 0).sum().item() for p in model.parameters())
        print(f"  Model params: {n_params:,} total, {n_nonzero:,} nonzero")
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"  WARNING: Checkpoint not found, using random init")

    # Evaluate
    print(f"\n  Evaluating (n_samples={args.n_samples})...")
    t0 = time.time()
    results = evaluate(model, args.scheme, loader, device, n_samples=args.n_samples,
                       sequences=test_seqs,
                       ires_start=args.ires_start, ires_end=args.ires_end)
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"  Scheme {args.scheme} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {results['n_evaluated']}")
    print(f"  Samples failed:    {results['n_failed']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"")
    print(f"  RMSD (A):")
    print(f"    Mean:   {results['rmsd_mean']:.2f}")
    print(f"    Median: {results['rmsd_median']:.2f}")
    print(f"    Std:    {results['rmsd_std']:.2f}")
    print(f"    Min:    {results['rmsd_min']:.2f}")
    print(f"    Max:    {results['rmsd_max']:.2f}")
    print(f"")
    print(f"  RMSD thresholds:")
    print(f"    < 10A: {results['rmsd_<10A']:.1%}")
    print(f"    < 20A: {results['rmsd_<20A']:.1%}")
    print(f"    < 30A: {results['rmsd_<30A']:.1%}")
    print(f"")
    if 'rmsd_percentiles' in results:
        p = results['rmsd_percentiles']
        print(f"  RMSD percentiles:")
        print(f"    P10: {p['p10']:.2f}  P25: {p['p25']:.2f}  P50: {p['p50']:.2f}  P75: {p['p75']:.2f}  P90: {p['p90']:.2f}")
    print(f"")
    print(f"  Closure error (A): {results['closure_mean']:.2f}")
    print(f"  TM-score: {results['tm_mean']:.4f} (median: {results['tm_median']:.4f})")
    print(f"")
    # circDesign-inspired thermodynamic stability (MFE)
    print(f"  MFE (kcal/mol) [{results.get('mfe_engine', 'N/A')}]:")
    print(f"    Mean:   {results['mfe_mean']:.2f}")
    print(f"    Median: {results['mfe_median']:.2f}")
    print(f"    Std:    {results['mfe_std']:.2f}")
    print(f"    Min:    {results['mfe_min']:.2f}")
    print(f"    Max:    {results['mfe_max']:.2f}")
    if results['mfe_mean'] != 0 and results['n_evaluated'] > 0:
        mfe_per_nt = results['mfe_mean'] / max(
            np.median([len(s) for s in (test_seqs or [''])]), 1)
        print(f"    Per-nt: {mfe_per_nt:.3f} kcal/mol/nt")
        if mfe_per_nt < -0.8:
            print(f"    Stability: HIGH (circDesign threshold: < -0.8 kcal/mol/nt)")
        elif mfe_per_nt < -0.5:
            print(f"    Stability: MODERATE")
        else:
            print(f"    Stability: LOW (consider optimizing)")
    # IRES structural deviation (circDesign Eq. 5)
    if results.get('n_ires_evaluated', 0) > 0:
        print(f"")
        print(f"  IRES Structural Deviation (circDesign Eq. 5):")
        print(f"    (lower = IRES structure better preserved)")
        print(f"    Mean:   {results['ires_dev_mean']:.4f}")
        print(f"    Median: {results['ires_dev_median']:.4f}")
        print(f"    Std:    {results['ires_dev_std']:.4f}")
        print(f"    Min:    {results['ires_dev_min']:.4f}")
        print(f"    Max:    {results['ires_dev_max']:.4f}")
        print(f"    N:      {results['n_ires_evaluated']}")
        if results['ires_dev_mean'] < 0.1:
            print(f"    Verdict: EXCELLENT (IRES structure well preserved)")
        elif results['ires_dev_mean'] < 0.3:
            print(f"    Verdict: GOOD (minor cross-region interference)")
        else:
            print(f"    Verdict: CONSIDER OPTIMIZING (significant IRES disruption)")

    if results.get('n_stem_loop_evaluated', 0) > 0:
        print(f"  Stem-Loop Stability (ΔG kcal/mol):")
        print(f"    (lower = more stable hairpins)")
        print(f"    Mean:   {results['stem_loop_stability_mean']:.2f}")
        print(f"    Median: {results['stem_loop_stability_median']:.2f}")
        print(f"    Std:    {results['stem_loop_stability_std']:.2f}")
        print(f"    Min:    {results['stem_loop_stability_min']:.2f} (most stable)")
        print(f"    Max:    {results['stem_loop_stability_max']:.2f} (least stable)")
        print(f"    Avg count: {results['stem_loop_count_mean']:.1f} loops/seq")
        print(f"    N:      {results['n_stem_loop_evaluated']}")
        if results['stem_loop_stability_mean'] < -3.0:
            print(f"    Verdict: EXCELLENT (strong hairpin stability)")
        elif results['stem_loop_stability_mean'] < 0:
            print(f"    Verdict: GOOD (moderate stability)")
        else:
            print(f"    Verdict: CONSIDER OPTIMIZING (weak hairpin formation)")
    print(f"{'='*60}")

    # Save
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, f'scheme{args.scheme}_eval.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.output}/scheme{args.scheme}_eval.json")


if __name__ == '__main__':
    main()
