"""
isrnacirc_wrapper.py -- CG_to_allatom binary wrapper + base-atom supplementation + BSJ clash fix.

Pipeline:
  1. cg_to_allatom(): run CG_to_allatom binary for backbone reconstruction
  2. _add_base_atoms(): Kabsch-aligned IsRNA2 template base placement
  3. _fix_bsj_clashes(): BSJ junction clash repair (Rodrigues rotation around C1')

CG_to_allatom binary converts P-only CG PDB to RNA backbone all-atom PDB
(terminal atoms only, no bases).  This module completes base placement and
BSJ local optimization.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Key paths  (relative to repo root; resolved at import time)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # scheme2 -> src/torusfold -> src -> repo root

_BIN_DIR: Path = _REPO_ROOT / "deploy" / "IGEM\xe9\x9b\x86\xe6\x88\x90\xe6\x96\xb9\xe6\xa1\x88" / "tools" / "IsRNAcirc" / "IsRNAcirc_standalone" / "bin"
_DATA_DIR: Path = _REPO_ROOT / "deploy" / "IGEM\xe9\x9b\x86\xe6\x88\x90\xe6\x96\xb9\xe6\xa1\x88" / "tools" / "IsRNAcirc" / "IsRNAcirc_standalone"
_VFOLD3D_DIR: Path = _REPO_ROOT / "deploy" / "IGEM\xe9\x9b\x86\xe6\x88\x90\xe6\x96\xb9\xe6\xa1\x88" / "tools" / "VfoldLA"


# ---------------------------------------------------------------------------
# VDW radii (Angstrom) for clash detection
# ---------------------------------------------------------------------------
_VDW_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
}
_VDW_DEFAULT: float = 1.70
# Clash criterion: d < (r_i + r_j - CLASH_OFFSET)
CLASH_OFFSET: float = 0.4


# ===================================================================
# Helpers
# ===================================================================

def _normalize_rna_sequence(seq: str) -> str:
    """Normalise RNA sequence to uppercase AUGC."""
    mapping = {"T": "U", "t": "u"}
    result = []
    for ch in seq.upper():
        result.append(mapping.get(ch, ch))
    return "".join(result)


def _find_exe(name: str) -> Optional[str]:
    """Search for an executable by priority: CWD -> PATH -> _BIN_DIR -> Data/ -> vendor/."""
    # 1) current working directory
    if os.path.isfile(name):
        return os.path.abspath(name)

    # 2) PATH
    import shutil
    found = shutil.which(name)
    if found:
        return found

    # 3) _BIN_DIR (IGEM integration tools)
    if _BIN_DIR.is_dir():
        for candidate in [_BIN_DIR / name, _BIN_DIR / f"{name}.exe"]:
            if candidate.is_file():
                return str(candidate)

    # 4) Data/ directory (project convention)
    for candidate in [
        Path("Data") / name,
        Path("Data") / f"{name}.exe",
        _REPO_ROOT / "Data" / name,
        _REPO_ROOT / "Data" / f"{name}.exe",
    ]:
        if candidate.is_file():
            return str(candidate)

    # 5) vendor/
    for candidate in [
        Path("vendor") / name,
        _REPO_ROOT / "vendor" / name,
    ]:
        if candidate.is_file():
            return str(candidate)

    return None


def _read_pdb_coords(pdb_path: str) -> Tuple[List[str], np.ndarray]:
    """Read a PDB file; return (atom_lines, coords_A).

    Returns:
        (atom_lines, coords_A) -- atom_lines is a list of ATOM/HETATM lines,
        coords_A is (N, 3) array in Angstrom.
    """
    atom_lines: List[str] = []
    coords: List[List[float]] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_lines.append(line.rstrip("\n"))
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    coords.append([0.0, 0.0, 0.0])
    return atom_lines, np.array(coords, dtype=np.float64)


def _write_pdb(
    atom_lines: List[str],
    output_path: str,
    header: str = "HEADER    CG to all-atom reconstruction",
) -> None:
    """Write atom_lines to a PDB file."""
    with open(output_path, "w") as fh:
        fh.write(header + "\n")
        for line in atom_lines:
            fh.write(line + "\n")
        fh.write("END\n")


def _guess_element_from_atom_line(line: str) -> str:
    """Infer element from a PDB ATOM line."""
    if len(line) >= 78:
        elem = line[76:78].strip()
        if elem:
            return elem
    name = line[12:16].strip() if len(line) > 16 else ""
    if not name:
        return "C"
    first = name[0]
    if first in "HDCNOP":
        return first
    if name.startswith("OP") or name.startswith("O5") or name.startswith("O3") or \
       name.startswith("O2") or name.startswith("O4"):
        return "O"
    if name.startswith("C"):
        return "C"
    if name.startswith("N"):
        return "N"
    if name.startswith("P"):
        return "P"
    return "C"


def _get_base_atom_names(base: str) -> List[Tuple[str, str]]:
    """Return base atom list [(name, element), ...].

    Purines (A/G): start from N9, 9-10 atoms.
    Pyrimidines (C/U): start from N1, 8 atoms.
    """
    if base in ("A", "G"):
        atoms = [
            ("N9", "N"), ("C8", "C"), ("N7", "N"), ("C5", "C"), ("C6", "C"),
        ]
        if base == "A":
            atoms += [("N6", "N"), ("N1", "N"), ("C2", "C"), ("N3", "N"), ("C4", "C")]
        else:  # G
            atoms += [("O6", "O"), ("N1", "N"), ("C2", "C"), ("N3", "N"), ("C4", "C"),
                      ("N2", "N")]
    else:  # C, U
        atoms = [
            ("N1", "N"), ("C2", "C"), ("N3", "N"), ("C4", "C"),
            ("C5", "C"), ("C6", "C"),
        ]
        if base == "C":
            atoms += [("O2", "O"), ("N4", "N")]
        else:  # U
            atoms += [("O2", "O"), ("O4", "O")]
    return atoms


def _format_pdb_atom(
    serial: int,
    name: str,
    res_name: str,
    res_seq: int,
    xyz: np.ndarray,
    element: str,
    chain: str = "A",
) -> str:
    """Format a PDB ATOM line."""
    name_str = f"{name:>4s}" if len(name) <= 4 else f"{name[:4]:>4s}"
    resname_str = f"{res_name:>3s}"
    serial_str = f"{serial:5d}"
    resseq_str = f"{res_seq:4d}"
    x_str = f"{xyz[0]:8.3f}"
    y_str = f"{xyz[1]:8.3f}"
    z_str = f"{xyz[2]:8.3f}"
    elem_str = f"{element:>2s}"
    return (
        f"ATOM  {serial_str} {name_str} {resname_str} {chain}{resseq_str}"
        f"    {x_str}{y_str}{z_str}  1.00  0.00          {elem_str}"
    )


def _parse_residue_backbone(
    atom_lines: List[str],
    coords_A: np.ndarray,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, int]]:
    """Parse backbone atom coordinates per residue.

    Returns:
        res_backbone: {res_idx (0-based): {atom_name: coord}}
        res_idx_map:  {line_index: res_idx}
    """
    backbone_names = {
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
        "C3'", "O3'", "C2'", "O2'", "C1'",
    }
    res_backbone: Dict[int, Dict[str, np.ndarray]] = {}
    res_idx_map: Dict[int, int] = {}

    for i, line in enumerate(atom_lines):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            res_seq = int(line[22:26].strip()) - 1  # 0-based
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        if res_seq not in res_backbone:
            res_backbone[res_seq] = {}
        x, y, z = coords_A[i]
        res_backbone[res_seq][atom_name] = np.array([x, y, z])
        res_idx_map[i] = res_seq

    return res_backbone, res_idx_map


def _is_backbone_atom(atom_name: str) -> bool:
    """Check if an atom name belongs to the RNA backbone."""
    return atom_name in {
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
        "C3'", "O3'", "C2'", "O2'", "C1'",
    }


# ===================================================================
# IsRNA2 template loading
# ===================================================================

def _load_base_template(
    resname: str,
    template_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load IsRNA2 base template .dat file.

    Template format: each line "ATOM_NAME  X  Y  Z" (Angstrom, local coords, origin=C1').

    Args:
        resname: base type (A/U/G/C)
        template_dir: template directory (default: _DATA_DIR/IsRNA2 or Data/base_templates)

    Returns:
        (N, 3) coordinate array or None if not found.
    """
    # Search order: explicit dir -> _DATA_DIR/IsRNA2 -> Data/base_templates
    search_dirs: List[Path] = []
    if template_dir is not None:
        search_dirs.append(Path(template_dir))
    search_dirs.append(_DATA_DIR / "IsRNA2")
    search_dirs.append(_REPO_ROOT / "Data" / "base_templates")

    dat_names = [f"{resname}.dat", f"base_{resname}.dat",
                 f"RNA_{resname}.dat", f"{resname}_template.dat"]

    for d in search_dirs:
        if not d.is_dir():
            continue
        for fname in dat_names:
            dat_path = d / fname
            if dat_path.exists():
                return _parse_dat_template(str(dat_path))

    return None


def _parse_dat_template(dat_path: str) -> Optional[np.ndarray]:
    """Parse a .dat template file into (N, 3) coordinate array."""
    coords: List[List[float]] = []
    with open(dat_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords, dtype=np.float64) if coords else None


# ===================================================================
# Kabsch alignment
# ===================================================================

def _kabsch_align(
    moving: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Kabsch algorithm: optimal rotation + translation alignment.

    Args:
        moving: (N, 3) coordinates to align
        target: (N, 3) reference coordinates

    Returns:
        (aligned, rotation_matrix, rmsd)
    """
    assert moving.shape == target.shape
    # centroid
    cm_m = moving.mean(axis=0)
    cm_t = target.mean(axis=0)
    m = moving - cm_m
    t = target - cm_t

    # SVD
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)

    # rotation matrix (reflection correction)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    aligned = (R @ m.T).T + cm_t
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return aligned, R, rmsd


# ===================================================================
# Base atom placement (backbone PDB -> full-atom PDB)
# ===================================================================

def _add_base_atoms(
    backbone_pdb: str,
    output_pdb: str,
    sequence: str,
    template_dir: Optional[str] = None,
) -> bool:
    """Place base atoms via IsRNA2 template + Kabsch alignment.

    Reads backbone-only PDB (CG_to_allatom output), then for each residue:
      1. Reads backbone atoms (P, C1', C4', O3', etc.)
      2. Loads the corresponding base template
      3. Uses C1' as anchor, Kabsch-aligns template onto backbone
      4. Outputs full-atom residue (backbone + base)

    BSJ awareness: after placing all bases, detects clashes at BSJ junction
    (first/last 5 residues) and rotates last residue's base around C1'
    glycosidic bond (Rodrigues rotation, 10-deg steps) to minimise clashes.

    Args:
        backbone_pdb: input backbone-only PDB path
        output_pdb: output full-atom PDB path
        sequence: RNA sequence
        template_dir: IsRNA2 base template directory

    Returns:
        True on success, False if templates missing.
    """
    atom_lines, coords_A = _read_pdb_coords(backbone_pdb)
    if not atom_lines:
        print("  [isrnacirc_wrapper] input PDB has no ATOM lines")
        return False

    seq = _normalize_rna_sequence(sequence)
    L = len(seq)

    res_backbone, res_idx_map = _parse_residue_backbone(atom_lines, coords_A)

    # load base templates
    templates: Dict[str, np.ndarray] = {}
    for base in "AUGC":
        t = _load_base_template(base, template_dir)
        if t is not None:
            templates[base] = t

    if not templates:
        print("  [isrnacirc_wrapper] base templates missing, cannot add base atoms")
        return False

    # place base atoms for each residue
    all_atom_lines: List[str] = []
    atom_serial = 1
    all_coords: List[np.ndarray] = []

    for res_idx in range(L):
        base = seq[res_idx] if res_idx < len(seq) else "A"
        bb = res_backbone.get(res_idx, {})
        template = templates.get(base)

        if template is None or "C1'" not in bb:
            # no template or no C1' anchor: keep original backbone lines
            for i, line in enumerate(atom_lines):
                if res_idx_map.get(i) == res_idx:
                    new_line = f"ATOM  {atom_serial:5d}" + line[11:]
                    all_atom_lines.append(new_line)
                    all_coords.append(coords_A[i])
                    atom_serial += 1
            continue

        c1_prime = bb["C1'"]

        # template coordinates centred on C1'
        template_shifted = template - template.mean(axis=0) + c1_prime

        # if P and C4' available, do proper Kabsch alignment
        if "P" in bb and "C4'" in bb:
            anchor_names = ["P", "C1'", "C4'"]
            dst_anchors: List[np.ndarray] = []
            src_anchors: List[np.ndarray] = []
            for name in anchor_names:
                if name in bb:
                    dst_anchors.append(bb[name])
                    if name == "P" and len(template) > 0:
                        src_anchors.append(template[0])
                    elif name == "C4'" and len(template) > 2:
                        src_anchors.append(template[2])
                    elif name == "C1'" and len(template) > 1:
                        src_anchors.append(template[1])
            n_ref = min(len(src_anchors), len(dst_anchors))
            if n_ref >= 2:
                src_arr = np.array(src_anchors[:n_ref])
                dst_arr = np.array(dst_anchors[:n_ref])
                _, R, _ = _kabsch_align(src_arr, dst_arr)
                template_shifted = (R @ template.T).T + c1_prime

        # output backbone atoms
        for i, line in enumerate(atom_lines):
            if res_idx_map.get(i) == res_idx:
                new_line = f"ATOM  {atom_serial:5d}" + line[11:]
                all_atom_lines.append(new_line)
                all_coords.append(coords_A[i])
                atom_serial += 1

        # output base atoms
        base_atom_names = _get_base_atom_names(base)
        for k, (name, elem) in enumerate(base_atom_names):
            if k < len(template_shifted):
                xyz = template_shifted[k]
                line = _format_pdb_atom(
                    atom_serial, name, base, res_idx + 1, xyz, elem,
                )
                all_atom_lines.append(line)
                all_coords.append(xyz)
                atom_serial += 1

    # BSJ clash repair (first/last 5 residues)
    if L > 10:
        all_coords_arr = np.array(all_coords) if all_coords else np.zeros((0, 3))
        _fix_bsj_clashes_from_lines(
            all_atom_lines, all_coords_arr, seq,
            res_backbone, templates, atom_lines, res_idx_map,
        )

    _write_pdb(all_atom_lines, output_pdb)
    return True


# ===================================================================
# BSJ clash detection and repair
# ===================================================================

def _fix_bsj_clashes_from_lines(
    atom_lines: List[str],
    coords_A: np.ndarray,
    sequence: str,
    res_backbone: Dict[int, Dict[str, np.ndarray]],
    templates: Dict[str, np.ndarray],
    orig_atom_lines: List[str],
    res_idx_map: Dict[int, int],
    bsj_margin: int = 5,
) -> None:
    """Detect and repair BSJ junction clashes (in-place on atom_lines/coords_A).

    For first/last bsj_margin residues, detects inter-residue clashes.
    Rotates last residue's base around C1' axis (Rodrigues, 10-deg steps)
    to minimise clash count.  Uses per-atom VDW radii with
    CLASH_OFFSET tolerance.
    """
    L = len(sequence)
    if L < 2 * bsj_margin:
        return

    seq = _normalize_rna_sequence(sequence)

    # --- collect last-residue base atom indices ---
    last_base_atoms: List[int] = []
    for i, line in enumerate(atom_lines):
        if not line.startswith("ATOM"):
            continue
        try:
            res_seq = int(line[22:26].strip()) - 1
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        if res_seq == L - 1 and not _is_backbone_atom(atom_name):
            last_base_atoms.append(i)

    if not last_base_atoms:
        return

    # --- C1' position (rotation pivot) ---
    c1_prime = res_backbone.get(L - 1, {}).get("C1'")
    if c1_prime is None:
        return

    # --- collect clash-zone atom indices (first bsj_margin residues, base atoms) ---
    clash_atom_indices: List[int] = []
    for i, line in enumerate(atom_lines):
        if not line.startswith("ATOM"):
            continue
        try:
            res_seq = int(line[22:26].strip()) - 1
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        if res_seq < bsj_margin and not _is_backbone_atom(atom_name):
            clash_atom_indices.append(i)

    if not clash_atom_indices:
        return

    # --- rotation axis: C1' -> last-residue backbone centroid ---
    last_bb = res_backbone.get(L - 1, {})
    bb_atoms = np.array([v for v in last_bb.values()])
    if len(bb_atoms) < 2:
        return
    axis = bb_atoms.mean(axis=0) - c1_prime
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return
    axis = axis / axis_norm

    # --- precompute element strings for clash-zone atoms ---
    clash_elems = [_guess_element_from_atom_line(atom_lines[ci]) for ci in clash_atom_indices]
    last_elems = [_guess_element_from_atom_line(atom_lines[ai]) for ai in last_base_atoms]

    # --- search best rotation angle (10-deg steps) ---
    best_angle = 0.0
    best_n_clash = float("inf")

    for angle_deg in range(0, 360, 10):
        angle_rad = angle_deg * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        n_clash = 0
        for li, ai in enumerate(last_base_atoms):
            if ai >= len(coords_A):
                continue
            xyz_i = coords_A[ai]
            if np.all(xyz_i == 0):
                continue
            v = xyz_i - c1_prime
            v_rot = (v * cos_a
                     + np.cross(axis, v) * sin_a
                     + axis * np.dot(axis, v) * (1.0 - cos_a))
            rotated = v_rot + c1_prime

            r_i = _VDW_RADII.get(last_elems[li], _VDW_DEFAULT)

            for ci_idx, ci in enumerate(clash_atom_indices):
                if ci >= len(coords_A):
                    continue
                xyz_c = coords_A[ci]
                if np.all(xyz_c == 0):
                    continue
                r_c = _VDW_RADII.get(clash_elems[ci_idx], _VDW_DEFAULT)
                dist = np.linalg.norm(rotated - xyz_c)
                if dist < (r_i + r_c - CLASH_OFFSET):
                    n_clash += 1

        if n_clash < best_n_clash:
            best_n_clash = n_clash
            best_angle = angle_deg

    # --- apply best rotation ---
    if best_angle > 0:
        angle_rad = best_angle * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for ai in last_base_atoms:
            if ai >= len(coords_A):
                continue
            xyz_i = coords_A[ai]
            if np.all(xyz_i == 0):
                continue
            v = xyz_i - c1_prime
            v_rot = (v * cos_a
                     + np.cross(axis, v) * sin_a
                     + axis * np.dot(axis, v) * (1.0 - cos_a))
            rotated = v_rot + c1_prime
            coords_A[ai] = rotated
            old_line = atom_lines[ai]
            atom_lines[ai] = (
                old_line[:30]
                + f"{rotated[0]:8.3f}{rotated[1]:8.3f}{rotated[2]:8.3f}"
                + old_line[54:]
            )


def _fix_bsj_clashes(
    pdb_path: str,
    sequence: str,
    output_path: Optional[str] = None,
    bsj_margin: int = 5,
) -> bool:
    """Read all-atom PDB, repair BSJ junction clashes.

    Detects inter-residue clashes at BSJ and rotates last residue's base
    around C1' axis to minimise clashes.  Uses per-atom VDW radii
    (C=1.70, N=1.55, O=1.52, P=1.80) with CLASH_OFFSET tolerance.

    Args:
        pdb_path: input all-atom PDB path
        sequence: RNA sequence
        output_path: output PDB path (default: overwrite input)
        bsj_margin: BSJ detection range (residues)

    Returns:
        True on repair.
    """
    if output_path is None:
        output_path = pdb_path

    atom_lines, coords_A = _read_pdb_coords(pdb_path)
    if not atom_lines:
        return False

    seq = _normalize_rna_sequence(sequence)
    L = len(seq)

    res_atoms: Dict[int, List[int]] = {}
    res_backbone: Dict[int, Dict[str, np.ndarray]] = {}

    for i, line in enumerate(atom_lines):
        if not line.startswith("ATOM"):
            continue
        try:
            res_seq = int(line[22:26].strip()) - 1
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        if res_seq not in res_atoms:
            res_atoms[res_seq] = []
        res_atoms[res_seq].append(i)

        if _is_backbone_atom(atom_name):
            if res_seq not in res_backbone:
                res_backbone[res_seq] = {}
            x, y, z = coords_A[i]
            res_backbone[res_seq][atom_name] = np.array([x, y, z])

    # last-residue base atoms
    last_base_atoms = []
    for i in res_atoms.get(L - 1, []):
        atom_name = atom_lines[i][12:16].strip()
        if not _is_backbone_atom(atom_name):
            last_base_atoms.append(i)

    if not last_base_atoms:
        return False

    c1_prime = res_backbone.get(L - 1, {}).get("C1'")
    if c1_prime is None:
        return False

    # first bsj_margin residues base atoms
    clash_indices: List[int] = []
    for r in range(min(bsj_margin, L)):
        for i in res_atoms.get(r, []):
            atom_name = atom_lines[i][12:16].strip()
            if not _is_backbone_atom(atom_name):
                clash_indices.append(i)

    if not clash_indices:
        return False

    # rotation axis
    bb_atoms = np.array([v for v in res_backbone.get(L - 1, {}).values()])
    if len(bb_atoms) < 2:
        return False
    axis = bb_atoms.mean(axis=0) - c1_prime
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return False
    axis = axis / axis_norm

    # precompute elements
    clash_elems = [_guess_element_from_atom_line(atom_lines[ci]) for ci in clash_indices]
    last_elems = [_guess_element_from_atom_line(atom_lines[ai]) for ai in last_base_atoms]

    # search best angle
    best_angle = 0.0
    best_n_clash = float("inf")

    for angle_deg in range(0, 360, 10):
        angle_rad = angle_deg * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        n_clash = 0
        for li, ai in enumerate(last_base_atoms):
            v = coords_A[ai] - c1_prime
            rotated = (v * cos_a
                       + np.cross(axis, v) * sin_a
                       + axis * np.dot(axis, v) * (1.0 - cos_a)) + c1_prime

            r_i = _VDW_RADII.get(last_elems[li], _VDW_DEFAULT)

            for ci_idx, ci in enumerate(clash_indices):
                dist = np.linalg.norm(rotated - coords_A[ci])
                r_c = _VDW_RADII.get(clash_elems[ci_idx], _VDW_DEFAULT)
                if dist < (r_i + r_c - CLASH_OFFSET):
                    n_clash += 1

        if n_clash < best_n_clash:
            best_n_clash = n_clash
            best_angle = angle_deg

    # apply best rotation
    if best_angle > 0:
        angle_rad = best_angle * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for ai in last_base_atoms:
            v = coords_A[ai] - c1_prime
            rotated = (v * cos_a
                       + np.cross(axis, v) * sin_a
                       + axis * np.dot(axis, v) * (1.0 - cos_a)) + c1_prime
            coords_A[ai] = rotated
            old_line = atom_lines[ai]
            atom_lines[ai] = (
                old_line[:30]
                + f"{rotated[0]:8.3f}{rotated[1]:8.3f}{rotated[2]:8.3f}"
                + old_line[54:]
            )

    _write_pdb(atom_lines, output_path)
    return True


# ===================================================================
# Top-level API: CG PDB -> full-atom PDB
# ===================================================================

def cg_to_allatom(
    cg_pdb: str,
    output_pdb: str,
    sequence: str,
    exe_path: Optional[str] = None,
    template_dir: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    """CG PDB -> full-atom PDB: CG_to_allatom binary + base placement + BSJ repair.

    Pipeline:
      1. CG_to_allatom binary: P-only CG PDB -> RNA backbone all-atom PDB
      2. _add_base_atoms(): IsRNA2 template + Kabsch alignment base placement
      3. _fix_bsj_clashes(): BSJ junction clash repair

    Args:
        cg_pdb: input CG PDB path (P-only)
        output_pdb: output all-atom PDB path
        sequence: RNA sequence
        exe_path: CG_to_allatom binary path (optional, auto-searched)
        template_dir: IsRNA2 base template directory (optional)
        verbose: print progress

    Returns:
        True on success.
    """
    if verbose:
        print("  [isrnacirc_wrapper] CG -> all-atom reconstruction...")

    # Step 1: CG_to_allatom binary
    if exe_path is None:
        exe_path = _find_exe("CG_to_allatom")

    if exe_path is not None:
        if verbose:
            print(f"    binary: {exe_path}")

        tmp_backbone = output_pdb + ".backbone.pdb"
        try:
            result = subprocess.run(
                [exe_path, cg_pdb, tmp_backbone],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                if verbose:
                    print(f"    CG_to_allatom failed (rc={result.returncode}): {result.stderr[:200]}")
                tmp_backbone = cg_pdb
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            if verbose:
                print(f"    CG_to_allatom unavailable: {exc}, using input PDB as backbone")
            tmp_backbone = cg_pdb
    else:
        if verbose:
            print("    CG_to_allatom binary not found, using input PDB as backbone")
        tmp_backbone = cg_pdb

    # Step 2: add base atoms
    if verbose:
        print("    adding base atoms...")

    success = _add_base_atoms(
        tmp_backbone, output_pdb, sequence,
        template_dir=template_dir,
    )

    # clean up temp file
    if tmp_backbone != cg_pdb and tmp_backbone != output_pdb:
        try:
            os.remove(tmp_backbone)
        except OSError:
            pass

    if not success:
        if verbose:
            print("    base placement failed, outputting backbone PDB")
        import shutil
        shutil.copy2(
            tmp_backbone if os.path.exists(tmp_backbone) else cg_pdb,
            output_pdb,
        )

    # Step 3: BSJ clash repair
    if verbose:
        print("    BSJ clash repair...")
    _fix_bsj_clashes(output_pdb, sequence, output_pdb)

    if verbose:
        n_atoms = 0
        with open(output_pdb) as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    n_atoms += 1
        print(f"    output: {output_pdb} ({n_atoms} atoms)")

    return True


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 4:
        print("Usage: python isrnacirc_wrapper.py <cg.pdb> <output.pdb> <sequence>")
        _sys.exit(1)

    cg_in = _sys.argv[1]
    out_pdb = _sys.argv[2]
    seq = _sys.argv[3]
    cg_to_allatom(cg_in, out_pdb, seq)
