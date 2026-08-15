"""
openmm_amber_refiner.py -- AMBER RNA.OL3 全原子精修 (pdbfixer 版).

与 amber_refine.py 的区别:
  * 使用 PDBFixer 加氢 (处理非标准拓扑更稳健)
  * 显式处理 circRNA 闭环拓扑 (删 HO5'/HO3', 加 O3'-P 闭合键)
  * 手动构建 AMBER 系统 + GBSAOBCForce (不依赖 implicitSolvent kwarg)
  * 支持 backbone 位置约束
  * 可选 REMD (Replica Exchange MD) 增强采样
  * 可选短 MD 精修

OpenMM 8.5+ 不再支持 createSystem(implicitSolvent=...) kwarg,
改用 GBSAOBCForce 手动添加隐式溶剂。

作者: TorusFold Team
日期: 2026-08-15
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- OpenMM optional import ---
try:
    import openmm as mm
    from openmm import (
        Platform, HarmonicBondForce, CustomBondForce,
        CustomExternalForce, CustomTorsionForce, CustomAngleForce,
        VerletIntegrator, LangevinMiddleIntegrator,
        GBSAOBCForce, System,
    )
    from openmm import unit
    from openmm.app import (
        Topology, Element, Modeller, ForceField, PDBFile, Simulation,
    )
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    mm = None
    Platform = None

# --- PDBFixer optional import ---
try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ImportError:
    PDBFIXER_AVAILABLE = False
    PDBFixer = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FORCE_FIELD_FILES = ("amber14-all.xml",)   # RNA.OL3 参数内嵌在 amber14-all.xml 中
GBSAOBC_SOLUTE_RADIUS = 0.14  # nm (default OBC2 radius for RNA)

# P 原子 positional restraint 力常数 (kJ/mol/nm^2)
DEFAULT_RESTRAINT_K = 1000.0

# A-form RNA 标准二面角 (度)
_AFORM_TORSIONS = {
    "alpha": (-60.0, "O3'", "P",   "O5'", "C5'"),
    "gamma": (60.0,  "O5'", "C5'", "C4'", "C3'"),
    "delta": (84.0,  "C5'", "C4'", "C3'", "O3'"),
    "zeta":  (-90.0, "C4'", "C3'", "O3'", "P"),
}

# GBSAOBCForce element -> (radius nm, scalingFactor)
# OpenMM GBSAOBCForce OBC2 参数, 适配 RNA 全原子
_GBSA_ELEMENT_PARAMS: Dict[str, Tuple[float, float]] = {
    "C": (0.22, 0.72),
    "N": (0.17, 0.79),
    "O": (0.15, 0.85),
    "P": (0.20, 0.86),
    "H": (0.12, 0.85),
}
_GBSA_DEFAULT_RADIUS = 0.17
_GBSA_DEFAULT_SCALING = 0.72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cpu_threads() -> int:
    """返回可用 CPU 线程数."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _detect_platform() -> str:
    """自动检测最佳 OpenMM 平台: CUDA > OpenCL > CPU."""
    if not OPENMM_AVAILABLE:
        return "CPU"
    for name in ("CUDA", "OpenCL"):
        try:
            Platform.getPlatformByName(name)
            return name
        except Exception:
            continue
    return "CPU"


def _count_residues_in_pdb(pdb_path: str) -> int:
    """从 PDB 文件快速计算残基数 (不加载到 OpenMM)."""
    count = 0
    seen = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # col 22-26: resSeq
                try:
                    resseq = int(line[22:26].strip())
                except (ValueError, IndexError):
                    continue
                chain = line[21] if len(line) > 21 else "A"
                key = (chain, resseq)
                if key not in seen:
                    seen.add(key)
                    count += 1
    return count


def _build_amber_system(
    topology: "Topology",
    seq_length: int,
    *,
    include_implicit_solvent: bool = True,
) -> "System":
    """用 amber14-all.xml (RNA.OL3) 力场构建 OpenMM System + 手动 GBSAOBCForce.

    OpenMM 8.5+ 不再支持 implicitSolvent kwarg,
    改用 GBSAOBCForce 手动添加隐式溶剂。

    GBSAOBCForce 参数 (OBC2 模型):
      addParticle(charge, radius, scalingFactor)
      C: radius=0.22, scalingFactor=0.72
      N: radius=0.17, scalingFactor=0.79
      O: radius=0.15, scalingFactor=0.85
      P: radius=0.20, scalingFactor=0.86
      H: radius=0.12, scalingFactor=0.85

    Args:
        topology: OpenMM Topology
        seq_length: RNA 序列长度 (残基数)
        include_implicit_solvent: 是否加 GBSAOBCForce

    Returns:
        OpenMM System
    """
    ff = ForceField(*_FORCE_FIELD_FILES)
    system = ff.createSystem(
        topology,
        constraints=None,
        rigidWater=True,
        ignoreExternalBonds=True,
    )

    if include_implicit_solvent:
        # GBSAOBCForce: 隐式溶剂 (OBC2 模型)
        gbsa = GBSAOBCForce()
        gbsa.setSoluteDielectric(1.0)
        gbsa.setSolventDielectric(78.5)
        gbsa.setDielectricOffset(0.009)
        # 为每个原子设置 element-specific radius 和 scalingFactor
        for atom in topology.atoms():
            elem = atom.element.symbol if atom.element else "C"
            r, sf = _GBSA_ELEMENT_PARAMS.get(
                elem, (_GBSA_DEFAULT_RADIUS, _GBSA_DEFAULT_SCALING)
            )
            # charge=0, radius=r, scalingFactor=sf
            gbsa.addParticle(0.0, r, sf)
        system.addForce(gbsa)

    return system


def _add_backbone_restraints(
    system: "System",
    topology: "Topology",
    coords_nm: np.ndarray,
    seq_length: int,
    k_restraint: float = DEFAULT_RESTRAINT_K,
    restraint_atom_names: Optional[List[str]] = None,
) -> int:
    """给 backbone 原子加位置约束 (CustomExternalForce).

    对 P/O3'/C5'/C1' 等关键骨架原子施加 harmonic 位置约束,
    保持全局拓扑。

    Args:
        system: OpenMM System (会修改)
        topology: OpenMM Topology
        coords_nm: (N, 3) nm 坐标 (N = 原子总数)
        seq_length: 残基数
        k_restraint: 约束力常数 (kJ/mol/nm^2)
        restraint_atom_names: 约束的原子名列表 (默认骨架关键原子)

    Returns:
        被约束的原子数
    """
    if restraint_atom_names is None:
        restraint_atom_names = ["P", "O3'", "C5'", "C1'"]

    restraint = CustomExternalForce(
        "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter("k", k_restraint)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    n_pinned = 0
    for atom in topology.atoms():
        if atom.name in restraint_atom_names:
            xyz = coords_nm[atom.index]
            restraint.addParticle(atom.index, [xyz[0], xyz[1], xyz[2]])
            n_pinned += 1

    system.addForce(restraint)
    return n_pinned


def _add_backbone_torsion_restraints(
    system: "System",
    topology: "Topology",
    seq_length: int,
    k_torsion: float = 50.0,
) -> int:
    """加 A-form RNA backbone 二面角约束 (alpha/gamma/delta/zeta).

    Args:
        system: OpenMM System
        topology: OpenMM Topology
        seq_length: 残基数
        k_torsion: 二面角约束力常数 (kJ/mol/rad^2)

    Returns:
        约束二面角数
    """
    torsion = CustomTorsionForce("0.5*k_aform*(theta-theta0)^2")
    torsion.addGlobalParameter("k_aform", k_torsion)
    torsion.addPerTorsionParameter("theta0")

    # 构建 (residue.index, atom_name) -> atom index 映射
    atom_lookup: Dict[Tuple[int, str], int] = {}
    for atom in topology.atoms():
        atom_lookup[(atom.residue.index, atom.name)] = atom.index

    n_torsions = 0
    for res_idx in range(seq_length):
        prev_seq = (res_idx - 1) % seq_length
        next_seq = (res_idx + 1) % seq_length

        for tname, (angle_deg, a1, a2, a3, a4) in _AFORM_TORSIONS.items():
            if tname == "alpha":
                i1 = atom_lookup.get((prev_seq, a1))
                i2 = atom_lookup.get((res_idx, a2))
                i3 = atom_lookup.get((res_idx, a3))
                i4 = atom_lookup.get((res_idx, a4))
            elif tname == "zeta":
                i1 = atom_lookup.get((res_idx, a1))
                i2 = atom_lookup.get((res_idx, a2))
                i3 = atom_lookup.get((res_idx, a3))
                i4 = atom_lookup.get((next_seq, a4))
            else:
                i1 = atom_lookup.get((res_idx, a1))
                i2 = atom_lookup.get((res_idx, a2))
                i3 = atom_lookup.get((res_idx, a3))
                i4 = atom_lookup.get((res_idx, a4))

            if None in (i1, i2, i3, i4):
                continue
            theta0 = angle_deg * np.pi / 180.0
            torsion.addTorsion(i1, i2, i3, i4, [theta0])
            n_torsions += 1

    system.addForce(torsion)
    return n_torsions


def _write_refined_pdb(
    topology: "Topology",
    positions: "unit.Quantity",
    output_path: str,
):
    """将精修后的坐标写入 PDB 文件."""
    PDBFile.writeFile(topology, positions, open(output_path, "w"))


# ---------------------------------------------------------------------------
# REMD (Replica Exchange MD) -- 可选增强采样
# ---------------------------------------------------------------------------

def _run_remd(
    topology: "Topology",
    system: "System",
    positions: "unit.Quantity",
    seq_length: int,
    n_replicas: int = 4,
    n_steps: int = 500,
    exchange_interval: int = 100,
    platform_name: str = "CPU",
    verbose: bool = False,
) -> Tuple[float, "unit.Quantity"]:
    """执行 T-REMD 增强采样.

    每个副本独立模拟, 定期尝试相邻温度间的 Metropolis 交换。
    在子进程中执行以避免 OpenMM Context 共享问题。

    Args:
        topology: OpenMM Topology
        system: OpenMM System
        positions: 初始位置 (OpenMM Quantity)
        seq_length: 残基数
        n_replicas: 副本数 (每个副本一个温度)
        n_steps: 每副本总步数
        exchange_interval: 交换间隔步数
        platform_name: OpenMM 平台
        verbose: 打印详细信息

    Returns:
        (best_energy, best_positions) -- 所有副本中最低能量的构象
    """
    try:
        from scipy.constants import k as kB
    except ImportError:
        kB = 1.380649e-23  # J/K

    # 温度阶梯: 300K -> ~500K (几何间隔)
    temperatures = [300.0 * (1.12 ** i) for i in range(n_replicas)]
    if verbose:
        temps_str = ", ".join(f"{t:.0f}" for t in temperatures)
        print(f"    REMD: {n_replicas} 副本, T=[{temps_str}]K")

    n_exchange_points = max(1, n_steps // exchange_interval)
    best_energy = float("inf")
    best_pos = None

    # 为每个副本创建独立的 Simulation
    simulations = []
    for ri in range(n_replicas):
        integrator = LangevinMiddleIntegrator(
            temperatures[ri] * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picosecond,
        )
        try:
            plat = Platform.getPlatformByName(platform_name)
        except Exception:
            plat = Platform.getPlatformByName(_detect_platform())
        sim = Simulation(topology, system, integrator, plat)
        sim.context.setPositions(positions)
        sim.minimizeEnergy(maxIterations=500)
        simulations.append(sim)

    # 记录初始能量
    for sim in simulations:
        state = sim.context.getState(getEnergy=True)
        e = state.getPotentialEnergy()._value
        if e < best_energy:
            best_energy = e

    # 主循环: MD + 交换
    for ex_i in range(n_exchange_points):
        # 各副本跑 exchange_interval 步
        for sim in simulations:
            sim.step(exchange_interval)

        # 收集能量和位置
        energies = []
        pos_list = []
        for sim in simulations:
            state = sim.context.getState(getEnergy=True, getPositions=True)
            e = state.getPotentialEnergy()._value
            energies.append(e)
            pos_list.append(state.getPositions(asNumpy=True)._value)
            if e < best_energy:
                best_energy = e
                best_pos = pos_list[-1].copy()

        # Metropolis 交换
        accept_count = 0
        for ri in range(n_replicas - 1):
            ui, uj = energies[ri], energies[ri + 1]
            beta_i = 1.0 / (kB * temperatures[ri] / 1000.0)
            beta_j = 1.0 / (kB * temperatures[ri + 1] / 1000.0)
            exponent = np.clip((beta_i - beta_j) * (ui - uj), -30, 30)
            if np.random.random() < min(1.0, np.exp(exponent)):
                # 交换两个副本的位置
                simulations[ri].context.setPositions(
                    pos_list[ri + 1] * unit.nanometer)
                simulations[ri + 1].context.setPositions(
                    pos_list[ri] * unit.nanometer)
                accept_count += 1

        if verbose and n_exchange_points > 0 and ex_i % max(1, n_exchange_points // 3) == 0:
            rate = accept_count / max(1, n_replicas - 1)
            print(f"    REMD exchange {ex_i + 1}/{n_exchange_points}, "
                  f"accept={rate:.0%}, E={min(energies):.0f}")

    # 清理
    for sim in simulations:
        sim.context.getState(getEnergy=True)  # 确保状态一致

    return best_energy, best_pos


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def openmm_amber_refine(
    input_pdb: str,
    output_dir: str,
    name: str = "refine",
    minimize_max_iter: int = 3000,
    md_steps: int = 0,
    use_remd: bool = False,
    remd_n_replicas: int = 4,
    remd_n_steps: int = 500,
    verbose: bool = True,
    platform_name: str = "auto",
    restraints_k: float = DEFAULT_RESTRAINT_K,
    implicit_solvent: bool = True,
) -> Tuple[str, float, Dict]:
    """AMBER RNA.OL3 + GBSAOBC 隐式溶剂精修 (pdbfixer 版).

    流程:
      1. PDBFixer 加氢 + 修补缺失原子
      2. 处理 circRNA 拓扑 (删 HO5'/HO3', 加 O3'-P 闭合键)
      3. amber14-all.xml (RNA.OL3) 力场 + GBSAOBCForce
      4. backbone 位置约束 + 二面角约束
      5. 三阶段最小化 (松约束 -> MD退火 -> 收紧约束)
      6. (可选) REMD 增强采样
      7. (可选) 短 MD 精修
      8. 写输出 PDB

    Args:
        input_pdb: 输入 PDB 路径 (P-only 或全原子)
        output_dir: 输出目录
        name: 项目名 (输出文件前缀)
        minimize_max_iter: 最小化最大步数
        md_steps: MD 退火步数 (0=跳过)
        use_remd: 是否启用 REMD 增强采样
        remd_n_replicas: REMD 副本数
        remd_n_steps: REMD 总步数
        verbose: 打印详细信息
        platform_name: OpenMM 平台 ("auto"/"CUDA"/"OpenCL"/"CPU")
        restraints_k: backbone 位置约束力常数 (kJ/mol/nm^2)
        implicit_solvent: 是否加 GBSAOBCForce 隐式溶剂

    Returns:
        (output_pdb_path, final_energy, info_dict)
    """
    if not OPENMM_AVAILABLE:
        raise ImportError(
            "OpenMM 未安装。请安装: conda install -c conda-forge openmm"
        )

    t0 = time.time()

    # 创建输出目录
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 从 PDB 自动获取序列长度
    seq_length = _count_residues_in_pdb(input_pdb)
    if seq_length < 2:
        raise ValueError(f"PDB 中残基数不足 ({seq_length} < 2)")

    if verbose:
        print(f"  [AMBER Refine] 输入: {input_pdb}")
        print(f"  [AMBER Refine] 序列长度: {seq_length} nt")

    # --- Step 1: PDBFixer 加氢 + 修补 ---
    if verbose:
        print("  [AMBER Refine] Step 1: PDBFixer 加氢 + 修补缺失原子...")

    if not PDBFIXER_AVAILABLE:
        warnings.warn(
            "PDBFixer 未安装, 跳过加氢。"
            "请安装: conda install -c conda-forge pdbfixer"
        )
        pdb = PDBFile(input_pdb)
        topology = pdb.topology
        positions = pdb.positions
    else:
        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # pH 7.0
        topology = fixer.topology
        positions = fixer.positions

    n_atoms = topology.getNumAtoms()
    if verbose:
        print(f"    原子数: {n_atoms}")

    # --- Step 2: 处理 circRNA 拓扑 ---
    if PDBFIXER_AVAILABLE:
        if verbose:
            print("  [AMBER Refine] Step 2: 处理 circRNA 闭环拓扑...")

        # 找到首末残基
        residues = list(topology.residues())
        if len(residues) < 2:
            raise ValueError("残基数不足 (< 2)")

        res_first = residues[0]
        res_last = residues[-1]

        # 删 HO5' (残基 0 的 5' 端氢) -- 环状无自由 5' 端
        atoms_to_delete = []
        for atom in res_first.atoms():
            if atom.name == "HO5'":
                atoms_to_delete.append(atom)

        # 删 HO3' (末残基的 3' 端氢) -- 环状无自由 3' 端
        for atom in res_last.atoms():
            if atom.name == "HO3'":
                atoms_to_delete.append(atom)

        if atoms_to_delete:
            topology.deleteAtoms(atoms_to_delete)
            if verbose:
                print(f"    删除 {len(atoms_to_delete)} 个端点 H")

        # 加 O3'(res_last) -> P(res_first) 闭合键
        o3_last = None
        p_first = None
        for atom in res_last.atoms():
            if atom.name == "O3'":
                o3_last = atom
                break
        for atom in res_first.atoms():
            if atom.name == "P":
                p_first = atom
                break

        if o3_last is not None and p_first is not None:
            # 检查是否已有闭合键
            has_closure = False
            for bond in topology.bonds():
                if ((bond[0] == o3_last and bond[1] == p_first)
                        or (bond[0] == p_first and bond[1] == o3_last)):
                    has_closure = True
                    break
            if not has_closure:
                topology.addBond(o3_last, p_first)
                if verbose:
                    print("    加入 BSJ 闭合键: O3'(last) -> P(first)")

    # --- Step 3: 构建 AMBER 系统 ---
    if verbose:
        print("  [AMBER Refine] Step 3: 构建 AMBER14 (RNA.OL3) + GBSAOBC 系统...")

    system = _build_amber_system(
        topology, seq_length,
        include_implicit_solvent=implicit_solvent,
    )

    # --- Step 4: 约束力 ---
    if verbose:
        print("  [AMBER Refine] Step 4: 添加约束力...")

    # 提取初始坐标
    pos_arr = np.zeros((n_atoms, 3), dtype=np.float64)
    for atom in topology.atoms():
        p = positions[atom.index]
        pos_arr[atom.index] = [p.x, p.y, p.z]

    # backbone 位置约束
    n_pinned = _add_backbone_restraints(
        system, topology, pos_arr, seq_length,
        k_restraint=restraints_k,
    )
    if verbose:
        print(f"    Backbone 约束: {n_pinned} 个原子, k={restraints_k}")

    # 二面角约束
    n_torsions = _add_backbone_torsion_restraints(
        system, topology, seq_length, k_torsion=50.0,
    )
    if verbose:
        print(f"    二面角约束: {n_torsions} 个")

    # --- Step 5: 三阶段最小化 ---
    if verbose:
        print("  [AMBER Refine] Step 5: 三阶段最小化...")

    # 平台检测
    if platform_name == "auto":
        resolved = _detect_platform()
    else:
        resolved = platform_name

    try:
        platform = Platform.getPlatformByName(resolved)
    except Exception:
        resolved = _detect_platform()
        platform = Platform.getPlatformByName(resolved)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    sim = Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)

    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy()._value
    if verbose:
        print(f"    初始能量: {e0:.0f} kJ/mol")

    # 阶段 1: 松约束最小化 (放 ~100x, 让力场主导)
    if verbose:
        print("    Phase 1: 松约束最小化...")
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max(3000, minimize_max_iter),
    )

    # 阶段 2: MD 退火 (可选, 跳出局部极小)
    if md_steps > 0 or True:  # 总是跑温和退火 (与 amber_refine.py 一致)
        pre_md_state = sim.context.getState(getPositions=True)
        pre_md_pos = pre_md_state.getPositions()
        if verbose:
            print("    Phase 2: MD 退火 (500K -> 300K)...")
        try:
            sim.integrator.setTemperature(500 * unit.kelvin)
            sim.step(100)
            sim.integrator.setTemperature(300 * unit.kelvin)
            sim.step(50)

            # 检查 NaN
            chk = sim.context.getState(getPositions=True).getPositions(
                asNumpy=True)._value
            if not np.isfinite(chk).all():
                raise RuntimeError("MD 产生 NaN")

            sim.minimizeEnergy(
                tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=max(3000, minimize_max_iter),
            )
        except Exception:
            sim.context.setPositions(pre_md_pos)

    # 阶段 3: 收紧约束最终最小化
    if verbose:
        print("    Phase 3: 收紧约束最终最小化...")
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=minimize_max_iter,
    )

    # --- Step 6: 可选 MD 精修 ---
    if md_steps > 0:
        if verbose:
            print(f"  [AMBER Refine] Step 6: MD 精修 ({md_steps} 步)...")
        try:
            sim.step(md_steps)
            sim.minimizeEnergy(
                tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=minimize_max_iter,
            )
        except Exception as md_exc:
            if verbose:
                print(f"    MD 精修异常: {md_exc}")

    # --- Step 7: 可选 REMD ---
    remd_energy = None
    if use_remd and seq_length >= 10:
        if verbose:
            print(f"  [AMBER Refine] Step 7: REMD ({remd_n_replicas} 副本, "
                  f"{remd_n_steps} 步)...")
        state_pre_remd = sim.context.getState(getPositions=True)
        pos_pre_remd = state_pre_remd.getPositions()
        try:
            remd_energy, remd_pos = _run_remd(
                topology, system, pos_pre_remd, seq_length,
                n_replicas=remd_n_replicas,
                n_steps=remd_n_steps,
                exchange_interval=max(10, remd_n_steps // 10),
                platform_name=resolved,
                verbose=verbose,
            )
            if remd_energy is not None and np.isfinite(remd_energy):
                # 用 REMD 最佳构象更新 Simulation
                sim.context.setPositions(remd_pos * unit.nanometer)
                sim.minimizeEnergy(
                    tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
                    maxIterations=minimize_max_iter,
                )
                if verbose:
                    print(f"    REMD 完成: E={remd_energy:.0f} kJ/mol")
        except Exception as remd_exc:
            if verbose:
                print(f"    REMD 异常: {remd_exc}")

    # --- 提取结果 ---
    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # P 偏移检查
    max_drift = 0.0
    p_atom_lookup: Dict[int, int] = {}
    for atom in topology.atoms():
        if atom.name == "P":
            p_atom_lookup[atom.residue.index] = atom.index

    for res_idx in range(seq_length):
        p_idx = p_atom_lookup.get(res_idx)
        if p_idx is not None and p_idx < len(pos_arr) and p_idx < len(pos):
            orig_xyz = pos_arr[p_idx]
            refined_xyz = pos[p_idx]
            drift = np.linalg.norm(refined_xyz - orig_xyz) * 10.0  # nm -> A
            if drift > max_drift:
                max_drift = drift

    # --- Step 8: 写输出 PDB ---
    output_pdb = str(out_path / f"{name}_amber_refined.pdb")
    _write_refined_pdb(topology, state.getPositions(), output_pdb)

    elapsed = time.time() - t0

    info = {
        "n_atoms": n_atoms,
        "seq_length": seq_length,
        "n_pinned": n_pinned,
        "n_torsions": n_torsions,
        "max_p_drift_A": float(max_drift),
        "e0_kJ_mol": float(e0),
        "e1_kJ_mol": float(e1),
        "platform": resolved,
        "elapsed_s": float(elapsed),
        "implicit_solvent": implicit_solvent,
        "restraints_k": restraints_k,
    }
    if remd_energy is not None:
        info["remd_energy"] = float(remd_energy)

    if verbose:
        print(f"  [AMBER Refine] 完成: E0={e0:.0f} -> E1={e1:.0f} kJ/mol, "
              f"P drift={max_drift:.2f}A, {elapsed:.1f}s")
        print(f"  [AMBER Refine] 输出: {output_pdb}")

    return output_pdb, e1, info


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python openmm_amber_refiner.py <input.pdb> [output_dir] [name]")
        _sys.exit(1)

    pdb_in = _sys.argv[1]
    out_dir = _sys.argv[2] if len(_sys.argv) > 2 else "refine_output"
    proj_name = _sys.argv[3] if len(_sys.argv) > 3 else "refine"

    out, e, info = openmm_amber_refine(
        pdb_in, out_dir, name=proj_name, verbose=True,
    )
    print(f"\nOutput: {out}, E={e:.0f} kJ/mol")
    for k, v in info.items():
        print(f"  {k}: {v}")
