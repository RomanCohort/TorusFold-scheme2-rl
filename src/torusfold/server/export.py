"""
structure_export.py — TorusFold 输出 → PDB 字符串 + 指纹旁路 JSON。

A1 档（coarse-grained，每核苷酸一个 P 原子）：
    TorusFold 的 coords(B,L,3) 是 C1'/P proxy 的单点坐标。
    本模块把每个点当作磷酸 P 原子，写成 PDB ATOM 记录，
    Mol* 读入后自动连 backbone、渲染为 cartoon tube。

    闭环（circRNA 的灵魂）：
        PDB 格式本身不支持"环"，线性 backbone 在末端会断开。
        这里用 CONECT 记录显式把最后一个残基的 P 连回第一个，
        Mol* 会画出闭合的 circRNA 环。BSJ 处的残基用特殊
        residue name (BSJ) 标记，前端可单独上色高亮。

A2 档（全原子，v2）：
    用 biotite + 理想化核苷酸模板补全碱基/糖环原子。
    接口不变，只是 _build_atom_records 从"只写 P"换成"拼模板"。

输入契约（来自 torus_coord_head.py 的 forward 返回 + torusfold.py 的 result）：
    coords:          (B, L, 3)  Cartesian, Å
    sequence:        str        长度 L，字母 ACGU
    confidence:      (B, L)     [0,100]，可选
    immune_fingerprints: dict   来自 ImmuneFingerprintHeads.forward
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# RNA 标准残基名（PDB 三字母码）
RES_NAME = {"A": "A", "U": "U", "G": "G", "C": "C"}
# BSJ 标记残基名（前端高亮用）
BSJ_RES_NAME = "BSJ"


def _to_numpy(x: Any) -> np.ndarray:
    """torch.Tensor / list / np.ndarray → np.ndarray。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _take_batch0(x: Any) -> np.ndarray:
    """取 batch=0，去 batch 维。输入 (B, L, ...) → (L, ...)。"""
    arr = _to_numpy(x)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _validate(seq: str, coords: np.ndarray) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords 期望 (L, 3)，实际 {coords.shape}"
        )
    if len(seq) != coords.shape[0]:
        raise ValueError(
            f"sequence 长度 {len(seq)} != coords 残基数 {coords.shape[0]}"
        )
    bad = [c for c in seq if c not in RES_NAME]
    if bad:
        raise ValueError(f"sequence 含非法字母 {set(bad)}，只允许 ACGU")


def _atom_record(
    serial: int,
    res_seq: int,
    res_name: str,
    x: float,
    y: float,
    z: float,
    b_factor: float = 0.0,
) -> str:
    """
    PDB ATOM 记录（固定列格式，列号从 1 起）：
        1-6   "ATOM  "
        7-11  serial        右对齐
        13-16 atom name     左对齐（P 是单字符，第 14 列起）
        17    altLoc        空格
        18-20 resName       右对齐
        22    chainID       A
        23-26 resSeq        右对齐
        27    iCode         空格
        31-38 x             %8.3f
        39-46 y             %8.3f
        47-54 z             %8.3f
        55-60 occupancy     1.00
        61-66 tempFactor    %6.2f（这里放 confidence 当 B-factor）
        73-76 element       " P"
    """
    return (
        f"ATOM  "
        f"{serial:>5d} "
        f" P  "           # atom name（注意 PDB 对齐规则：单字符名从 14 列）
        f" "               # altLoc
        f"{res_name:>3s} "
        f"A"               # chainID
        f"{res_seq:>4d}"
        f"    "            # iCode + 3 spaces
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}"
        f"{b_factor:6.2f}"
        f"          "
        f" P"
    )


def coords_to_pdb(
    coords: Any,
    sequence: str,
    confidence: Optional[Any] = None,
    circular: bool = True,
    bsj_residue_names: bool = False,
) -> str:
    """
    A1 档主函数：coords + sequence → PDB 字符串。

    Args:
        coords: (B, L, 3) 或 (L, 3)，Å
        sequence: 长度 L 的 ACGU 字符串
        confidence: (B, L) 或 (L,)，可选；写入 B-factor 列（Mol* 可按它上色）
        circular: True=显式 CONECT 闭合最后一个残基回第一个（circRNA 闭环）
        bsj_residue_names: True=首尾残基 resName 改为 BSJ（前端高亮 back-splice junction）。
            默认 False —— Mol* 5.x 不认识 BSJ 这个非标准残基名，建 representation
            时会报 "Cannot read properties of undefined (reading 'data')"。
            BSJ 高亮改由前端按残基序号 (1 和 L) 单独着色，不再靠 resName。
    """
    coords_arr = _take_batch0(coords)
    _validate(sequence, coords_arr)

    conf_arr = None
    if confidence is not None:
        conf_arr = _take_batch0(confidence)
        if conf_arr.ndim == 0 or conf_arr.shape[0] != coords_arr.shape[0]:
            conf_arr = None  # 形状对不上就别写 B-factor

    L = len(sequence)
    lines: List[str] = []
    lines.append(
        f"REMARK   1 TorusFold circRNA structure (coarse-grained, P-only)"
    )
    lines.append(f"REMARK   2 length={L} circular={circular}")
    lines.append(f"HEADER    circRNA 3D STRUCTURE")

    atom_serial = 0
    res_indices: List[int] = []  # 记录每个残基的 P 原子 serial，给 CONECT 用

    for i in range(L):
        atom_serial += 1
        x, y, z = coords_arr[i]
        base = sequence[i]

        if bsj_residue_names and (i == 0 or i == L - 1):
            res_name = BSJ_RES_NAME  # 首尾标 BSJ
        else:
            res_name = RES_NAME[base]

        b_factor = float(conf_arr[i]) if conf_arr is not None else 0.0
        lines.append(
            _atom_record(atom_serial, i + 1, res_name, float(x), float(y), float(z), b_factor)
        )
        res_indices.append(atom_serial)

    # 闭环：显式 CONECT 最后一个 P → 第一个 P
    # Mol* 读 CONECT 会在两残基间画 bond，视觉上闭合 circRNA 环
    if circular and L >= 2:
        last_serial = res_indices[-1]
        first_serial = res_indices[0]
        lines.append(f"CONECT{last_serial:>5d}{first_serial:>5d}")

    # 相邻 backbone bond（可选，让 Mol* 显式连线，不依赖自动推断）
    # 只在 circular=False 或想要完整骨架时加；circRNA 时自动推断已够
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _clamp01(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(v)))


def coords_to_pdb_allatom(
    atom_records: Sequence[Dict[str, Any]],
    sequence: str,
    confidence: Optional[Sequence[float]] = None,
    circular: bool = True,
) -> str:
    """A2 档全原子 PDB (amber14 RNA.OL3 命名)。

    Args:
        atom_records: 每原子 dict {serial, res_seq, res_name, atom_name,
            element, xyz}。来自 allatom_reconstruct + amber_refine。
        sequence: ACGU 字符串, 长度 = 残基数。
        confidence: 每残基 confidence (0~100), 写入 B-factor, 残基内所有
            原子共享该残基 confidence。
        circular: True=显式 CONECT O3'[L-1]↔P[0] 闭环 (真实化学键)。

    残基名 = A/U/G/C (amber14 RNA 模板, Mol* 可按残基类型着色)。
    """
    L = len(sequence)
    lines: List[str] = []
    lines.append("REMARK   1 TorusFold circRNA all-atom structure (amber14 RNA.OL3)")
    lines.append(f"REMARK   2 length={L} circular={circular} atoms={len(atom_records)}")
    lines.append("HEADER    circRNA 3D STRUCTURE (ALL-ATOM)")

    # 残基序号 → confidence (per-residue, 0~100)
    conf_by_res: Dict[int, float] = {}
    if confidence is not None:
        for i, v in enumerate(confidence):
            if i < L:
                conf_by_res[i + 1] = float(v) * 100.0 if float(v) <= 1.0 else float(v)

    # P 原子 serial (给 CONECT BSJ 用), 以及每残基的 O3' serial
    first_p_serial: Optional[int] = None
    last_o3_serial: Optional[int] = None

    serial = 0
    for rec in atom_records:
        serial += 1
        res_seq = rec["res_seq"]
        res_name = rec["res_name"]
        atom_name = rec["atom_name"]
        element = rec["element"]
        x, y, z = rec["xyz"]
        b_factor = conf_by_res.get(res_seq, 0.0)

        # 记录 BSJ 闭环需要的原子
        if atom_name == "P" and res_seq == 1:
            first_p_serial = serial
        if atom_name == "O3'" and res_seq == L:
            last_o3_serial = serial

        lines.append(_atom_record_allatom(
            serial, res_seq, res_name, atom_name, element,
            float(x), float(y), float(z), b_factor
        ))

    # 闭环: O3'[L-1] ↔ P[0] (真实 P-O3' 化学键, Mol* 画 BSJ 闭合)
    if circular and first_p_serial is not None and last_o3_serial is not None:
        lines.append(f"CONECT{last_o3_serial:>5d}{first_p_serial:>5d}")

    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _atom_record_allatom(
    serial: int, res_seq: int, res_name: str, atom_name: str,
    element: str, x: float, y: float, z: float, b_factor: float = 0.0,
) -> str:
    """全原子 PDB ATOM 记录 (amber14 命名, 含撇号糖环)。

    atom_name 对齐规则: ≤3 字符右对齐到 13-16 列; 4 字符左对齐到 13-16 列。
    element 列 (77-78)。
    """
    # atom_name 格式化: PDB 列 13-16 (4 字符宽)
    if len(atom_name) >= 4:
        name_field = atom_name[:4]
    elif len(atom_name) == 3:
        name_field = " " + atom_name
    elif len(atom_name) == 2:
        # 撇号原子 (O5' C3' 等): 原子符号首字母 + 元素对齐
        # PDB 约定: 单元素+撇号, 第 14 列起 (右空格)
        name_field = " " + atom_name + " "
    else:
        name_field = " " + atom_name + "  "
    name_field = name_field[:4]

    return (
        f"ATOM  "
        f"{serial:>5d} "
        f"{name_field}"
        f" "               # altLoc
        f"{res_name:>3s} "
        f"A"               # chainID
        f"{res_seq:>4d}"
        f"    "            # iCode + 3 spaces
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}"
        f"{b_factor:6.2f}"
        f"          "
        f"{element:>2s}"
    )


def _add_signal_coloring(
    per_residue: dict,
    schemes: list,
    coords: Any,
    sequence: str,
    signals: dict,
) -> None:
    """Generate per-residue coloring arrays derived from 3D coordinates + signals.

    Adds entries to per_residue dict and corresponding coloring_schemes list.
    These give the user visual feedback on structural quality at each residue.
    """
    coords_arr = _take_batch0(coords)
    L = len(sequence)

    # 1. Per-residue clash density: count atoms within 3.0Å of each P atom
    clash_density = np.zeros(L, dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            d = np.linalg.norm(coords_arr[i] - coords_arr[j])
            if d < 3.0:
                clash_density[i] += 1
    per_residue["clash_density"] = [_clamp01(float(v / 10.0)) for v in clash_density]
    schemes.append({"key": "clash_density", "label": "Clash density (3Å)", "type": "per_residue"})

    # 2. Per-residue bond strain: |bond_length - 5.9| / 5.9 for each backbone bond
    bond_strain = np.zeros(L, dtype=np.float32)
    for i in range(L):
        j = (i + 1) % L
        d = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
        bond_strain[i] = abs(d - 5.9) / 5.9
    per_residue["bond_strain"] = [_clamp01(float(v)) for v in bond_strain]
    schemes.append({"key": "bond_strain", "label": "Bond strain (vs 5.9Å)", "type": "per_residue"})

    # 3. Per-residue local curvature: angle deviation from 180° at each residue
    curvature = np.zeros(L, dtype=np.float32)
    for i in range(L):
        prev_i = (i - 1) % L
        next_i = (i + 1) % L
        v1 = coords_arr[prev_i] - coords_arr[i]
        v2 = coords_arr[next_i] - coords_arr[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
            curvature[i] = abs(angle_deg - 180.0) / 180.0  # 0=straight, 1=sharp turn
    per_residue["local_curvature"] = [_clamp01(float(v)) for v in curvature]
    schemes.append({"key": "local_curvature", "label": "Local curvature", "type": "per_residue"})

    # 4. Per-residue distance to BSJ (first residue): normalised
    if L > 1:
        bsj_dist = np.zeros(L, dtype=np.float32)
        ref = coords_arr[0]
        max_d = 1.0
        for i in range(L):
            bsj_dist[i] = float(np.linalg.norm(coords_arr[i] - ref))
        max_d = max(float(bsj_dist.max()), 1.0)
        per_residue["distance_to_bsj"] = [_clamp01(float(v / max_d)) for v in bsj_dist]
        schemes.append({"key": "distance_to_bsj", "label": "Distance to BSJ", "type": "per_residue"})

    # 5. Per-residue contact density (8Å): how many neighbors within 8Å
    contact_density = np.zeros(L, dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if i != j and float(np.linalg.norm(coords_arr[i] - coords_arr[j])) < 8.0:
                contact_density[i] += 1
    max_cd = max(float(contact_density.max()), 1.0)
    per_residue["contact_density_8a"] = [_clamp01(float(v / max_cd)) for v in contact_density]
    schemes.append({"key": "contact_density_8a", "label": "Contact density (8Å)", "type": "per_residue"})

    # 6. Scalar signals as uniform-colored options
    scalar_signals = [
        ("closure_distance", "Closure distance"),
        ("bond_rmsd", "Bond RMSD"),
        ("circdesign_mfe", "MFE (kcal/mol)"),
        ("circdesign_cai", "CAI"),
        ("circdesign_ires_deviation", "IRES deviation"),
        ("stem_loop_stability", "Stem-loop ΔG"),
        ("stem_loop_count", "Stem-loop count"),
    ]
    for key, label in scalar_signals:
        if key in signals and signals[key] is not None:
            per_residue[f"__scalar_{key}"] = [float(signals[key])] * L
            schemes.append({"key": f"__scalar_{key}", "label": f"{label} (scalar)", "type": "scalar"})


def fingerprints_to_json(
    coords: Any,
    sequence: str,
    immune_fingerprints: Optional[Dict[str, Any]] = None,
    confidence: Optional[Any] = None,
    per_residue_keys: Optional[Sequence[str]] = None,
    scalar_keys: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[tuple]] = None,
    signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    把免疫指纹 + confidence 整理成前端 coloring 用的 JSON 结构。

    返回 dict（序列化前），结构：
        {
          "sequence": "ACGU...",
          "length": L,
          "per_residue": {              # 每残基，长度 L 的数组
            "confidence": [...],
            "pkr_sasa": [...],
            "m6a_write_prob": [...],
            "tlr7_gu_density": [...],
            ...
          },
          "scalar": {                    # 整分子标量
            "nlrp3_persistence_length": ...,
            "sponge_score": ...,
            ...
          },
          "coloring_schemes": [         # 前端下拉框可选项
            {"key": "confidence", "label": "Confidence (pLDDT)", "type": "per_residue"},
            {"key": "pkr_sasa", "label": "PKR / SASA", "type": "per_residue"},
            ...
          ]
        }

    Args:
        per_residue_keys: 显式指定哪些 key 是 per-residue（默认按 shape 自动判断）
        scalar_keys: 显式指定哪些 key 是整分子标量
    """
    coords_arr = _take_batch0(coords)
    _validate(sequence, coords_arr)
    L = len(sequence)

    immune = immune_fingerprints or {}

    # 默认 per-residue / scalar 分类（基于 ImmuneFingerprintHeads.forward 的输出契约）
    default_per_res = {
        "pkr_stem_logit", "pkr_sasa",
        "drach_is_drach", "drach_in_loop", "m6a_write_prob",
        "tlr7_gu_density", "rigi_per_pos",
    }
    default_scalar = {
        "nlrp3_persistence_length", "sponge_score", "rigi_score",
    }

    per_res_keys = set(per_residue_keys) if per_residue_keys else default_per_res
    scal_keys = set(scalar_keys) if scalar_keys else default_scalar

    per_residue: Dict[str, List[float]] = {}

    # confidence 永远是 per-residue。原始值是 0~100（pLDDT 风格），
    # 归一到 [0,1] 跟其它 per-residue 指纹统一区间。前端会再 normalize 一次，
    # 但 JSON 里值域一致，将来读 JSON 不会困惑。
    if confidence is not None:
        conf = _take_batch0(confidence)
        if conf.ndim >= 1 and conf.shape[0] == L:
            per_residue["confidence"] = [_clamp01(float(v) / 100.0) for v in conf]
    # 没给 confidence 就合成一个中位值，前端总有这档可切
    if "confidence" not in per_residue:
        per_residue["confidence"] = [0.5] * L

    # --- categorical: 碱基类型 (A=0 U=1 G=2 C=3) ---
    base_map = {"A": 0, "U": 1, "G": 2, "C": 3}
    per_residue["base_type"] = [base_map.get(b, 0) for b in sequence]

    # --- categorical: 二级结构 (stem=1, loop=0) ---
    # 从 ViennaRNA pairs 推导: 出现在任何配对中的残基 = stem
    if pairs is not None:
        paired = set()
        for p in pairs:
            i, j = p[0], p[1]
            if 0 <= i < L:
                paired.add(i)
            if 0 <= j < L:
                paired.add(j)
        per_residue["secondary_structure"] = [
            1 if i in paired else 0 for i in range(L)
        ]

    # 免疫指纹 per-residue
    # confidence 来自 build_synthetic_data / TorusFold 是 0~100（pLDDT 风格），
    # 单独按 /100 归一到 [0,1]；其它 per-residue 指纹本就是 [0,1] 概率/暴露度，
    # 直接 clamp01。原代码用嵌套三元，confidence 落在 0.7~0.95 区间梯度拉不开。
    for k in per_res_keys:
        if k in immune:
            arr = _take_batch0(immune[k])
            if arr.ndim >= 1 and arr.shape[0] == L:
                if k == "confidence":
                    per_residue[k] = [_clamp01(float(v) / 100.0) for v in arr]
                else:
                    per_residue[k] = [_clamp01(float(v)) for v in arr]
            # shape 对不上就跳过，不硬塞

    scalar: Dict[str, float] = {}
    for k in scal_keys:
        if k in immune:
            arr = _take_batch0(immune[k])
            try:
                scalar[k] = float(arr.item() if arr.size == 1 else arr.flat[0])
            except (ValueError, IndexError):
                pass

    # 构造前端下拉选项（只列实际存在的数据）
    schemes: List[Dict[str, str]] = []
    label_map = {
        "confidence": "Confidence (pLDDT)",
        "base_type": "Base type (A/U/G/C)",
        "secondary_structure": "Secondary structure (stem/loop)",
        "pkr_sasa": "PKR / SASA exposure",
        "pkr_stem_logit": "PKR stem",
        "m6a_write_prob": "m6A write probability",
        "drach_is_drach": "DRACH motif",
        "drach_in_loop": "in-loop",
        "tlr7_gu_density": "TLR7 GU density",
        "rigi_per_pos": "RIG-I (neg ctrl)",
        "nlrp3_persistence_length": "NLRP3 persistence length",
        "sponge_score": "miRNA sponge",
        "rigi_score": "RIG-I score (neg ctrl)",
    }
    categorical_keys = {"base_type", "secondary_structure"}
    for k in per_residue:
        schemes.append({
            "key": k,
            "label": label_map.get(k, k),
            "type": "categorical" if k in categorical_keys else "per_residue",
        })
    for k in scalar:
        schemes.append({
            "key": k,
            "label": label_map.get(k, k) + " (scalar)",
            "type": "scalar",
        })

    # --- Signals-based coloring: derive per-residue arrays from 3D coords ---
    if signals is not None and coords_arr is not None and L > 1:
        try:
            _add_signal_coloring(per_residue, schemes, coords_arr, sequence, signals)
        except Exception as exc:
            import traceback
            print(f"[export] _add_signal_coloring FAILED: {exc}")
            traceback.print_exc()

    return {
        "sequence": sequence,
        "length": L,
        "per_residue": per_residue,
        "scalar": scalar,
        "coloring_schemes": schemes,
    }


def export_circrna_structure(
    coords: Any,
    sequence: str,
    immune_fingerprints: Optional[Dict[str, Any]] = None,
    confidence: Optional[Any] = None,
    circular: bool = True,
) -> Dict[str, str]:
    """
    一站式：输入 TorusFold 输出，返回 {pdb, fingerprint_json}。
    供 html_renderer 直接注入 HTML 模板。

    Returns:
        {
          "pdb": "ATOM ...",
          "fingerprint_json": "{...}",   # 已 json.dumps 的字符串
        }
    """
    pdb = coords_to_pdb(coords, sequence, confidence=confidence, circular=circular)
    fp_dict = fingerprints_to_json(
        coords, sequence, immune_fingerprints, confidence=confidence
    )
    return {
        "pdb": pdb,
        "fingerprint_json": json.dumps(fp_dict, ensure_ascii=False),
    }
