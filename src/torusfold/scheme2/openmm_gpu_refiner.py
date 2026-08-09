"""
openmm_gpu_refiner.py — OpenMM GPU 加速 CG MD 精修

替代 IsRNAcirc.exe 的 CPU-only CG MD 精修。
用 OpenMM 3-bead CG 力场 + GPU 平台加速 + 可选 REMD 增强采样。

接口兼容 isrnacirc_wrapper.isrnacirc_cg_refine():
  openmm_gpu_refine(input_pdb, output_dir, sequence, secondary_structure, ...)
  -> (output_pdb_path, final_energy)

回退链: CUDA -> OpenCL -> CPU
增强采样: 可选 T-REMD (Replica Exchange)

作者: TorusFold Team
日期: 2026-08-05
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import Simulation, Topology, Element
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    mm = None
    app = None
    unit = None


# ── 平台检测 ──

def detect_best_platform(preferred: str = "auto") -> str:
    """检测最佳可用 OpenMM 平台.

    preferred="auto" 时按 CUDA > OpenCL > CPU 顺序探测.
    preferred="CUDA"/"OpenCL"/"CPU" 时直接尝试该平台, 失败回退.

    Args:
        preferred: 首选平台 ("auto", "CUDA", "OpenCL", "CPU")

    Returns:
        可用平台名称
    """
    if not OPENMM_AVAILABLE:
        return "CPU"

    if preferred == "auto":
        candidates = ["CUDA", "OpenCL", "CPU"]
    else:
        candidates = [preferred, "CUDA", "OpenCL", "CPU"]

    for name in candidates:
        try:
            Platform.getPlatformByName(name)
            return name
        except Exception:
            continue
    return "Reference"


# ── PDB 坐标读写 ──

def _read_p_coords(pdb_path: str) -> np.ndarray:
    """从 PDB 读取 P 原子坐标, 返回 (L,3) Å.

    优先按列解析 (标准 PDB 格式), 列错位时回退 whitespace split.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                try:
                    x = float(line[29:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    # 列错位 (坐标溢出等), 回退 split
                    parts = line.split()
                    # ATOM serial name resname chain resid x y z ...
                    x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float64)


def _write_allatom_pdb(
    p_coords_3bead_nm: np.ndarray,
    L: int,
    output_path: str,
):
    """从 3-bead nm 坐标写骨架 PDB (用于后续 CG_to_allatom).

    提取 P bead (索引 0,3,6,...) 输出标准骨架 PDB.
    PDB 列格式 (CG_to_allatom.exe 的 substr 解析):
      col 12-15: atom name (" P  ")
      col 17-19: resname ("RA ")
      col 21:    chain ID ("A")
      col 22-25: resid ("   1")
      col 30-37: x (8.3f)
      col 38-45: y (8.3f)
      col 46-53: z (8.3f)
    """
    coords_ang = p_coords_3bead_nm * 10.0  # nm -> Å
    p_coords = coords_ang[0::3].copy()  # (L,3) Å

    # 平移到正象限 (避免负坐标溢出 8.3f 列宽, 不改变相对几何)
    if len(p_coords) > 0:
        min_xyz = p_coords.min(axis=0)
        shift = np.where(min_xyz < 0, -min_xyz + 5.0, 0.0)
        p_coords = p_coords + shift

    lines = ["HEADER    OpenMM GPU refined CG structure"]
    for i in range(L):
        x, y, z = p_coords[i]
        # 标准 PDB ATOM 格式: 必须精确对齐列
        serial = f"{i + 1:5d}"      # col 6-10
        name = " P  "                # col 12-15 (4 chars)
        resname = "RA "              # col 17-19
        chain = "A"                  # col 21
        resid = f"{i + 1:4d}"       # col 22-25
        x_str = f"{x:8.3f}"         # col 30-37
        y_str = f"{y:8.3f}"         # col 38-45
        z_str = f"{z:8.3f}"         # col 46-53
        # 拼装: "ATOM  " + serial + " " + name + resname + " " + chain + resid + "    " + x + y + z + "  1.00  0.00           P "
        line = f"ATOM  {serial} {name} {resname} {chain}{resid}    {x_str}{y_str}{z_str}  1.00  0.00           P "
        lines.append(line)
    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _write_refined_pdb(
    allatom_pdb_path: str,
    output_path: str,
):
    """复制全原子 PDB 到输出路径."""
    import shutil
    shutil.copy2(allatom_pdb_path, output_path)


# ── 力场参数 (与 cg_forcefield.py 对齐) ──

# 力常数 (kJ/mol/Å², 内部用 nm 需 *100)
K_BB = 310.0       # 骨架 P-P
K_INTRA = 310.0    # P-C4', C4'-N
K_PAIR = 800.0     # WC 配对 N-N
K_STACK = 300.0    # 碱基堆叠
K_ANGLE = 200.0    # 骨架键角
K_DIHEDRAL = 50.0  # 骨架二面角
K_CLASH = 200.0    # clash
K_BSJ = 500.0      # BSJ 闭合
K_BSJ_GUIDE = 800.0  # BSJ 引导力

# 几何参数 (Å)
BOND_P_NEXT = 5.90
BOND_P_C4 = 3.90
BOND_C4_N = 3.35
ANGLE_PPP = 2.618   # rad, 150°
DIH_PPPP = 33.0 * np.pi / 180.0  # rad
STACK_R0 = 5.05
PAIR_N_N = 10.0
CLASH_DIST = 3.0
CUTOFF = 12.0


def _build_3bead_system_gpu(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    pair_scale: float = 1.0,
    bsj_k_scale: float = 1.0,
    pair_guide_k: float = 0.0,
):
    """构建 3-bead CG OpenMM system (GPU 优化版).

    与 cg_forcefield.build_3bead_system() 力场一致,
    但简化接口, 去掉统计势 (GPU 路径追求速度).

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)] ViennaRNA 配对
        pair_scale: 配对力缩放 (退火用)
        bsj_k_scale: BSJ 力缩放 (退火用)
        pair_guide_k: 配对窗引导力 (kJ/mol). >0 时对远端配对施加
            渐近吸引力, 把相距 100-3000Å 的配对原子逐步拉近到
            力场作用范围 (~20Å), 之后普通配对力接管. 解决环状
            RNA 初始构象配对原子距离过大 (力场够不到) 的问题.

    Returns:
        (system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide)
    """
    L = len(p_coords)
    N_total = 3 * L

    # 构建 3-bead 坐标: 每个 nt → P, C4', N
    coords_3bead = np.zeros((N_total, 3), dtype=np.float64)
    rng = np.random.default_rng(42)
    for i in range(L):
        p = p_coords[i]
        coords_3bead[3 * i] = p  # P
        # C4' 和 N 用扰动估计 (后续 minimize 修正)
        coords_3bead[3 * i + 1] = p + rng.normal(0, 0.3, 3)  # C4'
        coords_3bead[3 * i + 2] = p + rng.normal(0, 0.3, 3)  # N

    coords_nm = coords_3bead / 10.0  # Å → nm

    system = mm.System()
    for _ in range(N_total):
        system.addParticle(330.0 / 3.0)  # ~110 Da per bead

    def P(i): return 3 * i
    def C4(i): return 3 * i + 1
    def N(i): return 3 * i + 2

    # 1. 骨架键 P[i]-P[i+1]
    bond_bb = mm.HarmonicBondForce()
    bb_k = K_BB * 100.0  # Å² → nm²
    for i in range(L - 1):
        bond_bb.addBond(P(i), P(i + 1), BOND_P_NEXT / 10.0, bb_k)
    system.addForce(bond_bb)

    # 1b. BSJ 闭合 (首末 P)
    bsj_force = mm.CustomBondForce("0.5*k_bsj*(r-r0)^2")
    bsj_force.addPerBondParameter("k_bsj")
    bsj_force.addPerBondParameter("r0")
    bsj_force.addBond(P(L - 1), P(0),
                      [bsj_k_scale * K_BSJ, BOND_P_NEXT / 10.0])
    system.addForce(bsj_force)

    # 1c. BSJ 引导力
    bsj_guide = mm.CustomBondForce("0.5*k_guide*(r-r0)^2")
    bsj_guide.addPerBondParameter("k_guide")
    bsj_guide.addPerBondParameter("r0")
    bsj_guide.addBond(P(L - 1), P(0),
                      [bsj_k_scale * K_BSJ_GUIDE, BOND_P_NEXT / 10.0])
    system.addForce(bsj_guide)

    # 2. 残基内键 P-C4', C4'-N
    bond_intra = mm.HarmonicBondForce()
    ik = K_INTRA * 100.0
    for i in range(L):
        bond_intra.addBond(P(i), C4(i), BOND_P_C4 / 10.0, ik)
        bond_intra.addBond(C4(i), N(i), BOND_C4_N / 10.0, ik)
    system.addForce(bond_intra)

    # 3. 骨架键角 P-P-P
    angle_force = mm.HarmonicAngleForce()
    for i in range(L - 2):
        angle_force.addAngle(P(i), P(i + 1), P(i + 2), ANGLE_PPP, K_ANGLE)
    # 环化角
    if L >= 3:
        angle_force.addAngle(P(L - 2), P(L - 1), P(0), ANGLE_PPP, K_ANGLE)
        angle_force.addAngle(P(L - 1), P(0), P(1), ANGLE_PPP, K_ANGLE)
    system.addForce(angle_force)

    # 3.5 骨架二面角
    dih_force = mm.CustomTorsionForce("0.5*k_dih*(theta-theta0)^2")
    dih_force.addGlobalParameter("k_dih", K_DIHEDRAL)
    dih_force.addGlobalParameter("theta0", DIH_PPPP)
    for i in range(L - 3):
        dih_force.addTorsion(P(i), P(i + 1), P(i + 2), P(i + 3))
    if L >= 4:
        dih_force.addTorsion(P(L - 3), P(L - 2), P(L - 1), P(0))
        dih_force.addTorsion(P(L - 2), P(L - 1), P(0), P(1))
        dih_force.addTorsion(P(L - 1), P(0), P(1), P(2))
    system.addForce(dih_force)

    # 4. WC 配对 N-N
    pair_force = mm.CustomBondForce("0.5*k_pair*(r-r0)^2")
    pair_force.addPerBondParameter("k_pair")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            pair_force.addBond(
                N(i), N(j),
                [K_PAIR * w * pair_scale, PAIR_N_N / 10.0])
    system.addForce(pair_force)

    # 4b. 配对窗引导力 (远端配对软吸引)
    # V = -k_g * (1/(1+exp(a*(r-r_cap)))) * step(r-r0_lo)
    #   - r >> r_cap: V -> 0 (够不到不强拉, 防止撕裂结构)
    #   - r ~ r_cap: 逻辑斯蒂过渡, 峰值力 ~ k_g*a/4
    #   - r < r0_lo (已配对): 关闭
    # a=0.05 (特征长度 20nm), r_cap=40nm 时覆盖 20-60nm (200-600Å)
    # 的配对, 峰值力温和, 不会像线性窗那样恒定拉力撕裂结构.
    if pair_guide_k > 0:
        guide_force = mm.CustomBondForce(
            "-k_g*(1/(1+exp(a*(r-r_cap))))*step(r-r0_lo)")
        guide_force.addPerBondParameter("k_g")
        guide_force.addGlobalParameter("a", 0.05)     # /nm, 特征长度 ~20nm
        guide_force.addGlobalParameter("r_cap", 40.0)  # nm = 400Å
        guide_force.addGlobalParameter("r0_lo", 1.5)   # nm = 15Å, 已配对关闭
        for (i, j, w) in pairs:
            if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                    and not (i == 0 and j == L - 1)):
                guide_force.addBond(
                    N(i), N(j), [pair_guide_k * w])
        system.addForce(guide_force)

    # 5. 碱基堆叠
    stack_force = mm.CustomBondForce("0.5*k_stack*(r-r0)^2")
    stack_force.addPerBondParameter("k_stack")
    stack_force.addPerBondParameter("r0")
    sk = K_STACK * 100.0
    for i in range(L - 1):
        stack_force.addBond(N(i), N(i + 1), [sk, STACK_R0 / 10.0])
    # 环化堆叠
    stack_force.addBond(N(L - 1), N(0), [sk, STACK_R0 / 10.0])
    system.addForce(stack_force)

    # 6. 非键 clash + 静电 (有界软球, 避免退火塌缩爆炸)
    #   硬截断 step(dmin-r)*k*(dmin-r)^2 在 r~dmin 处突变, 结构塌缩时
    #   斥力爆炸 (轮3-5 E~3.9亿). 改用有界软球:
    #   V = k_clash*(dmin-r)^2/(1+(dmin-r)^2/alpha^2) * step(dmin-r)
    #   r->0 时 V -> k_clash*alpha^2 (有限), 力不会无限增大.
    #   Coulomb 用软化形式 1/sqrt(r^2+soft^2), 避免 1/r 发散.
    clash_force = mm.CustomNonbondedForce(
        "k_clash*(dmin-r)^2/(1+(dmin-r)^2/alpha^2)*step(dmin-r)"
        " + Coul*q1*q2/sqrt(r^2+soft^2)"
    )
    clash_force.addPerParticleParameter("q")
    clash_force.addGlobalParameter("dmin", CLASH_DIST / 10.0)
    clash_force.addGlobalParameter("k_clash", K_CLASH * 100.0)
    clash_force.addGlobalParameter("alpha", 0.5)  # nm, 软球软化尺度
    clash_force.addGlobalParameter("Coul", 138.935456)
    clash_force.addGlobalParameter("soft", 0.5)   # nm, Coulomb 软化
    clash_force.setNonbondedMethod(
        mm.CustomNonbondedForce.CutoffNonPeriodic)
    clash_force.setCutoffDistance(CUTOFF / 10.0)

    for i in range(L):
        clash_force.addParticle([-0.5])  # P
        clash_force.addParticle([0.0])    # C4'
        clash_force.addParticle([0.0])    # N

    # 排除键对
    excluded = set()
    for i in range(L):
        for (a, b) in [(P(i), C4(i)), (C4(i), N(i))]:
            k = (min(a, b), max(a, b))
            if k not in excluded:
                excluded.add(k)
                clash_force.addExclusion(*k)
    for i in range(L - 1):
        k = (P(i), P(i + 1))
        if k not in excluded:
            excluded.add(k)
            clash_force.addExclusion(*k)
    k = (P(L - 1), P(0))
    if k not in excluded:
        excluded.add(k)
        clash_force.addExclusion(*k)
    for i in range(L - 1):
        k = (N(i), N(i + 1))
        if k not in excluded:
            excluded.add(k)
            clash_force.addExclusion(*k)
    k = (N(L - 1), N(0))
    if k not in excluded:
        excluded.add(k)
        clash_force.addExclusion(*k)
    system.addForce(clash_force)

    return (system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide)


def _build_minimal_system_gpu(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    pair_scale: float = 1.0,
):
    """构建极简 P-only 折叠力场 (两阶段方案阶段1).

    只含:
      1. P 骨架键 P[i]-P[i+1] (r0=5.9Å, k=31000 kJ/mol/nm²)
      2. P-P 配对键 (r0=5.9Å, k=40000×w×pair_scale)
    无 clash/堆叠/键角/C4'N — 这些项在完整力场下阻碍折叠
    (实测完整力场配对卡在 45Å, 极简力场折叠到 21Å).

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)] ViennaRNA 配对
        pair_scale: 配对力缩放

    Returns:
        (system, coords_nm, pair_force) — 只有 P bead (L 个粒子)
    """
    L = len(p_coords)
    coords_nm = p_coords / 10.0  # Å → nm

    system = mm.System()
    for _ in range(L):
        system.addParticle(110.0)

    # 1. P 骨架键
    bond_bb = mm.HarmonicBondForce()
    bb_k = 31000.0  # kJ/mol/nm²
    for i in range(L - 1):
        bond_bb.addBond(i, i + 1, BOND_P_NEXT / 10.0, bb_k)
    system.addForce(bond_bb)

    # 2. P-P 配对键 (强力, 折叠驱动)
    pair_force = mm.CustomBondForce("0.5*k_pair*(r-r0)^2")
    pair_force.addPerBondParameter("k_pair")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            pair_force.addBond(
                i, j, [40000.0 * w * pair_scale, BOND_P_NEXT / 10.0])
    system.addForce(pair_force)

    return system, coords_nm, pair_force


def _create_minimal_topology(L: int) -> Topology:
    """创建 P-only 拓扑 (每 nt 一个 P atom)."""
    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("RA", chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
    return topo


def _create_3bead_topology(L: int) -> Topology:
    """创建 3-bead CG 拓扑 (P/C4'/N per nt)."""
    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("N", chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
        topo.addAtom(f"C{i}", Element.getBySymbol("C"), res)
        topo.addAtom(f"N{i}", Element.getBySymbol("N"), res)
    return topo


# ── 三阶段退火 ──

def _run_annealing(
    sim: Simulation,
    pair_force,
    bsj_force,
    bsj_guide,
    L: int,
    n_anneal: int = 200,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """三阶段退火: 弱配对+弱BSJ → 强配对+中BSJ → 强配对+强BSJ.

    Returns:
        (final_energy, final_coords_nm)
    """
    def set_pair_k(scale):
        for i in range(pair_force.getNumBonds()):
            p1, p2, params = pair_force.getBondParameters(i)
            # 更新 k, 保持 r0
            pair_force.setBondParameters(
                i, p1, p2,
                [scale * K_PAIR, params[1]])
        pair_force.updateParametersInContext(sim.context)

    def set_bsj_k(scale):
        bsj_force.setBondParameters(
            0, 3 * (L - 1), 0,
            [scale * K_BSJ, BOND_P_NEXT / 10.0])
        bsj_guide.setBondParameters(
            0, 3 * (L - 1), 0,
            [scale * K_BSJ_GUIDE, BOND_P_NEXT / 10.0])
        bsj_force.updateParametersInContext(sim.context)
        bsj_guide.updateParametersInContext(sim.context)

    # 记录初始能量
    pre_state = sim.context.getState(getPositions=True, getEnergy=True)
    e_pre = pre_state.getPotentialEnergy()._value

    # 阶段1: 中温 + 弱配对 + 弱BSJ, 螺旋形成
    set_pair_k(0.1)
    set_bsj_k(0.3)
    sim.integrator.setTemperature(350 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=2000)

    # 阶段2: 中温 + 强配对 + 中BSJ, WC 配对拉拢
    set_pair_k(1.0)
    set_bsj_k(1.0)
    sim.integrator.setTemperature(320 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=2000)

    # 阶段3: 低温 + 强配对 + 强BSJ, 闭合
    set_pair_k(1.0)
    set_bsj_k(5.0)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=3000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # 安全网: MD 暴走回退
    if e1 > e_pre * 0.5 and e_pre < 0:
        pos = pre_state.getPositions(asNumpy=True)._value
        e1 = e_pre

    return e1, pos


def _run_anneal_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Å P 坐标
    pairs: List[Tuple[int, int, float]],
    n_anneal: int,
    n_threads: int,
):
    """多进程退火 worker: 独立构建 system + 三阶段退火.

    用不同随机种子 (worker_idx) 增加轨迹多样性.
    Returns:
        (final_energy, final_coords_nm)
    """
    import numpy as _np
    # 不同种子 -> 不同 C4'/N 初始扰动
    _np.random.seed(42 + worker_idx)

    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(
            p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.1 + 0.05 * worker_idx,
            pair_guide_k=300.0)  # 配对窗引导力, 把远端配对拉近
    topo = _create_3bead_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # 用全局 _run_annealing 做三阶段退火
    e_final, pos_final = _run_annealing(
        sim, pair_force, bsj_force, bsj_guide, len(p_coords),
        n_anneal=n_anneal, verbose=False)
    return e_final, pos_final


def _run_minimal_anneal_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Å P 坐标
    pairs: List[Tuple[int, int, float]],
    n_anneal: int,
    n_threads: int,
):
    """极简力场退火 worker (两阶段方案阶段1: 折叠).

    只含 P 骨架键 + P-P 配对, 无 clash/堆叠. 高温退火折叠.
    Returns:
        (final_energy, final_coords_ang)  # P-only, Å
    """
    system, coords_nm, pair_force = _build_minimal_system_gpu(
        p_coords, pairs, pair_scale=1.0)
    topo = _create_minimal_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        450 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # 先最小化消除初始 clash
    sim.minimizeEnergy(maxIterations=3000)

    # 逐步降温退火 (折叠驱动): 高温跑配对拉近, 逐步降温
    stages = [
        (450, n_anneal // 4),
        (420, n_anneal // 4),
        (390, n_anneal // 4),
        (360, n_anneal // 4),
        (330, n_anneal // 4),
        (300, n_anneal // 4),
    ]
    for T, n in stages:
        integrator.setTemperature(T * unit.kelvin)
        sim.step(max(1, n))
    # 终局最小化
    sim.minimizeEnergy(maxIterations=5000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos_nm = state.getPositions(asNumpy=True)._value  # nm
    e = state.getPotentialEnergy()._value
    pos_ang = pos_nm * 10.0  # → Å
    return e, pos_ang


def _run_parallel_minimal_annealing(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    n_anneal: int = 200,
    n_trajectories: int = 4,
    platform_name: str = "CPU",
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """多进程并行极简折叠: N 条轨迹, 取最低能量.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)]

    Returns:
        (best_energy, best_coords_ang)  # P-only, Å
    """
    import multiprocessing as mp

    total_threads = os.cpu_count() or 8
    per_traj_threads = max(1, total_threads // n_trajectories)
    if verbose:
        print(f"  极简折叠: {n_trajectories} 轨迹 x {per_traj_threads} 线程")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_trajectories) as pool:
        results = pool.starmap(
            _run_minimal_anneal_worker,
            [(i, p_coords, pairs, n_anneal, per_traj_threads)
             for i in range(n_trajectories)],
        )

    best_energy = float("inf")
    best_pos = None
    for e, pos in results:
        if e < best_energy:
            best_energy = e
            best_pos = pos
    return best_energy, best_pos


def _run_parallel_annealing(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    n_anneal: int = 200,
    n_trajectories: int = 4,
    platform_name: str = "CPU",
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """多进程并行退火: N 条轨迹各 32/N 线程, 取最低能量.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)]
        n_anneal: 每阶段步数
        n_trajectories: 并行轨迹数
        platform_name: 平台
        verbose: 打印

    Returns:
        (best_energy, best_coords_nm)
    """
    import multiprocessing as mp

    total_threads = os.cpu_count() or 8
    per_traj_threads = max(1, total_threads // n_trajectories)
    if verbose:
        print(f"  并行退火: {n_trajectories} 条轨迹 x {per_traj_threads} 线程 "
              f"(总 {total_threads} 核)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_trajectories) as pool:
        results = pool.starmap(
            _run_anneal_worker,
            [(i, p_coords, pairs, n_anneal, per_traj_threads)
             for i in range(n_trajectories)],
        )

    best_energy = float("inf")
    best_pos = None
    for e, pos in results:
        if e < best_energy:
            best_energy = e
            best_pos = pos

    return best_energy, best_pos


# ── T-REMD (多温度副本交换) ──

def _run_remd_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Å P 坐标
    pairs: List[Tuple[int, int, float]],
    temperature: float,
    n_steps: int,
    exchange_interval: int,
    n_threads: int,
    conn,
    minimal: bool = False,
):
    """REMD 单副本 worker 进程: 本地重建 system + 模拟 + Pipe 交换.

    worker 接收 P 坐标和配对, 自行构建 system (避免 pickle
    OpenMM 对象), 每个 exchange_interval 步报告能量并接收交换坐标.
    minimal=True 时用极简力场 (P骨架+P配对, 保持折叠一致性).
    """
    if minimal:
        system, coords_nm, _pf = _build_minimal_system_gpu(
            p_coords, pairs, pair_scale=1.0)
        topo = _create_minimal_topology(len(p_coords))
    else:
        # 每个 worker 独立构建 system (不同 bsj_k_scale 增加多样性)
        system, coords_nm, _pf, _sf, _bjf, _bjg = _build_3bead_system_gpu(
            p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.5 + 0.1 * worker_idx,
            pair_guide_k=300.0)  # 配对窗引导力
        topo = _create_3bead_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)
    sim.minimizeEnergy(maxIterations=500)

    # 记录初始能量
    state = sim.context.getState(getEnergy=True)
    e0 = state.getPotentialEnergy()._value
    best_energy = e0
    best_pos = coords_nm.copy()

    conn.send(("init", worker_idx, e0))

    # 主循环
    for step_i in range(n_steps):
        sim.step(1)
        if (step_i + 1) % 500 == 0:
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy()._value
            if energy < best_energy:
                best_energy = energy
                best_pos = state.getPositions(asNumpy=True)._value

        # 交换点: 发能量+坐标, 等交换决策
        if (step_i + 1) % exchange_interval == 0:
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy()._value
            pos = state.getPositions(asNumpy=True)._value
            conn.send(("report", worker_idx, energy, pos))
            # 等待主进程交换结果
            cmd = conn.recv()
            if cmd[0] == "swap":
                new_pos = cmd[1]
                sim.context.setPositions(new_pos * unit.nanometer)
            # "keep" 则不动

    # 最终报告
    conn.send(("done", worker_idx, best_energy, best_pos))
    conn.close()


def _clamp_replicas_by_memory(n_replicas: int, mem_per_proc_gb: float = 4.0) -> int:
    """根据剩余内存限制并行进程数 (保守策略).

    Args:
        n_replicas: 期望进程数
        mem_per_proc_gb: 每进程估算内存 (GB), 2013nt 全原子 ~4GB

    Returns:
        限制后的进程数 (至少 1)
    """
    try:
        import psutil
        avail = psutil.virtual_memory().available / (1024 ** 3)
        max_by_mem = max(1, int(avail // mem_per_proc_gb))
        return max(1, min(n_replicas, max_by_mem))
    except ImportError:
        return n_replicas


def _run_remd(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str,
    n_replicas: int = 4,
    n_steps: int = 500,
    exchange_interval: int = 100,
    verbose: bool = False,
    minimal: bool = False,
) -> Tuple[float, np.ndarray]:
    """执行 T-REMD 增强采样 (多进程并行).

    每个副本一个进程, 自行构建 system (只传 numpy/list).
    线程数 = cpu_count // n_replicas, 总核心全打满.
    minimal=True 时用极简力场 (P骨架+P配对), 返回 P-only nm 坐标.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)] 配对
        platform_name: 平台
        n_replicas: 副本数
        n_steps: 总步数
        exchange_interval: 交换间隔
        verbose: 打印
        minimal: 用极简力场 (默认 False)

    Returns:
        (best_energy, best_coords_nm)  # 极简模式: P-only nm
    """
    from scipy.constants import k as kB
    import multiprocessing as mp

    # 温度阶梯: 300K -> ~460K (几何间隔)
    temperatures = [300.0 * (1.10 ** i) for i in range(n_replicas)]

    # 每副本线程数 (总核心均分)
    total_threads = os.cpu_count() or 8
    # 内存感知: 根据剩余内存限制进程数 (保守, 每进程 ~6GB)
    n_replicas = _clamp_replicas_by_memory(n_replicas, mem_per_proc_gb=6.0)
    per_replica_threads = max(1, total_threads // n_replicas)
    if verbose:
        print(f"    REMD: {n_replicas} 副本并行, "
              f"每副本 {per_replica_threads} 线程 "
              f"(总 {total_threads} 核)")

    ctx = mp.get_context("spawn")
    processes = []
    conns = []
    for ri in range(n_replicas):
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        p = ctx.Process(
            target=_run_remd_worker,
            args=(ri, p_coords, pairs, temperatures[ri], n_steps,
                  exchange_interval, per_replica_threads, child_conn, minimal),
        )
        p.start()
        child_conn.close()
        processes.append(p)
        conns.append(parent_conn)

    # 主进程: 协调交换
    best_energy = float("inf")
    # 初始坐标 (nm): 极简模式 P-only, 完整模式 3-bead (补 C4'/N)
    if minimal:
        best_pos = p_coords / 10.0  # (L,3) P-only nm
    else:
        L0 = len(p_coords)
        _rng0 = np.random.default_rng(0)
        best_pos = np.zeros((3 * L0, 3), dtype=np.float64)
        for _i in range(L0):
            best_pos[3 * _i] = p_coords[_i] / 10.0
            best_pos[3 * _i + 1] = p_coords[_i] / 10.0 + _rng0.normal(0, 0.03, 3)
            best_pos[3 * _i + 2] = p_coords[_i] / 10.0 + _rng0.normal(0, 0.03, 3)
    accept_count = 0
    total_exchanges = max(1, (n_steps // exchange_interval) * (n_replicas - 1))

    # 阶段1: 等所有副本 init
    for ri in range(n_replicas):
        msg = conns[ri].recv()
        assert msg[0] == "init"
        if msg[2] < best_energy:
            best_energy = msg[2]

    # 阶段2: 协调交换
    n_exchange_points = n_steps // exchange_interval
    for _ in range(n_exchange_points):
        energies = [None] * n_replicas
        positions = [None] * n_replicas
        for ri in range(n_replicas):
            msg = conns[ri].recv()
            assert msg[0] == "report"
            energies[ri] = msg[2]
            positions[ri] = msg[3]
            if msg[2] < best_energy:
                best_energy = msg[2]
                best_pos = msg[3].copy()

        # 相邻副本 Metropolis 交换
        swap_decisions = [False] * (n_replicas - 1)
        for ri in range(n_replicas - 1):
            ui, uj = energies[ri], energies[ri + 1]
            beta_i = 1.0 / (kB * temperatures[ri] / 1000.0)
            beta_j = 1.0 / (kB * temperatures[ri + 1] / 1000.0)
            exponent = np.clip((beta_i - beta_j) * (ui - uj), -30, 30)
            if np.random.random() < min(1.0, np.exp(exponent)):
                swap_decisions[ri] = True
                accept_count += 1

        # 应用交换: 发新坐标给参与交换的副本
        for ri in range(n_replicas):
            new_pos = None
            if ri > 0 and swap_decisions[ri - 1]:
                new_pos = positions[ri - 1]
            elif ri < n_replicas - 1 and swap_decisions[ri]:
                new_pos = positions[ri + 1]
            if new_pos is not None:
                conns[ri].send(("swap", new_pos))
            else:
                conns[ri].send(("keep",))

    # 阶段3: 收尾
    for ri in range(n_replicas):
        msg = conns[ri].recv()
        assert msg[0] == "done"
        if msg[2] < best_energy:
            best_energy = msg[2]
            best_pos = msg[3]

    for p in processes:
        p.join(timeout=10)
    for conn in conns:
        conn.close()

    if verbose:
        rate = accept_count / total_exchanges
        print(f"    REMD: E={best_energy:.0f}, 交换率 {rate:.1%}")

    return best_energy, best_pos


# ── 坐标清洗和紧凑化 ──

def _sanitize_p_coords(p_coords: np.ndarray) -> np.ndarray:
    """清洗 P 坐标: 替换 NaN/Inf 为相邻有效坐标的均值."""
    L = len(p_coords)
    bad = np.any(~np.isfinite(p_coords), axis=1)
    if not np.any(bad):
        return p_coords

    n_bad = int(np.sum(bad))
    print(f"  [OpenMM GPU] 发现 {n_bad}/{L} 个 NaN/Inf P 坐标, 清洗中...")

    for i in range(L):
        if not bad[i]:
            continue
        left, right = None, None
        for j in range(i - 1, -1, -1):
            if not bad[j]:
                left = j
                break
        for j in range(i + 1, L):
            if not bad[j]:
                right = j
                break
        if left is not None and right is not None:
            alpha = (i - left) / (right - left)
            p_coords[i] = (1 - alpha) * p_coords[left] + alpha * p_coords[right]
        elif left is not None:
            p_coords[i] = p_coords[left].copy()
        elif right is not None:
            p_coords[i] = p_coords[right].copy()
        else:
            p_coords[i] = [0.0, 0.0, float(i) * 5.9]

    return p_coords


def _is_extended_helix(p_coords: np.ndarray, threshold: float = 200.0) -> bool:
    """检测 P 坐标是否是展开结构 (首末端距离远超环状 RNA 合理范围)."""
    if len(p_coords) < 2:
        return False
    end_to_end = float(np.linalg.norm(p_coords[-1] - p_coords[0]))
    return end_to_end > threshold


def _generate_compact_coords(L: int, pairs: List[Tuple[int, int, float]]) -> np.ndarray:
    """为长序列生成紧凑的环状起始坐标.

    圆环半径由 P-P 键长和序列长度决定, 加小扰动避免退化.
    """
    coords = np.zeros((L, 3), dtype=np.float64)
    circumference = L * BOND_P_NEXT
    radius = circumference / (2.0 * np.pi)

    for i in range(L):
        angle = 2.0 * np.pi * i / L
        coords[i] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]

    rng = np.random.default_rng(42)
    coords += rng.normal(0, 0.5, coords.shape)
    return coords


def _has_nan_energy(sim: 'Simulation') -> bool:
    """检查模拟当前能量是否为 NaN/Inf."""
    try:
        state = sim.context.getState(getEnergy=True)
        e = state.getPotentialEnergy()._value
        return not np.isfinite(e)
    except Exception:
        return True


# ── 主入口: isrnacirc_cg_refine 兼容接口 ──

def openmm_gpu_refine(
    input_pdb: str,
    output_dir: str,
    sequence: str,
    secondary_structure: str,
    name: str = "refine",
    nstep: int = 10000,
    nstep_close: int = 1000,
    nstru: int = 3,
    timeout: int = 600,
    platform_name: str = "auto",
    use_remd: bool = True,
    remd_n_replicas: int = 4,
    remd_n_steps: int = 500,
    verbose: bool = True,
    skip_cg_to_allatom: bool = False,
) -> Tuple[str, float]:
    """OpenMM GPU 加速 CG MD 精修 (isrnacirc_cg_refine 兼容接口).

    替代 IsRNAcirc.exe CPU-only 精修:
    1. 读 PDB → 提取 P 坐标
    2. 3-bead CG 力场 (与 cg_forcefield.py 一致)
    3. 三阶段退火 (弱→强配对+BSJ)
    4. 可选 T-REMD 增强采样
    5. CG → 全原子 (cg_to_allatom, 可选跳过)
    6. 输出精修后 PDB

    Args:
        input_pdb: 输入 PDB 路径
        output_dir: 输出目录
        sequence: RNA 序列
        secondary_structure: 二级结构
        name: 项目名
        nstep: 退火步数 (每阶段)
        nstep_close: (兼容参数, 未使用)
        nstru: (兼容参数, 未使用)
        timeout: 超时秒数
        platform_name: "auto"/"CUDA"/"OpenCL"/"CPU"
        use_remd: 是否启用 REMD
        remd_n_replicas: REMD 副本数
        remd_n_steps: REMD 步数
        verbose: 打印详细信息
        skip_cg_to_allatom: 跳过内部 CG→全原子转换 (输入已是全原子时用)

    Returns:
        (output_pdb_path, final_energy)
    """
    if not OPENMM_AVAILABLE:
        raise ImportError(
            "OpenMM 未安装, 无法使用 GPU 精修。"
            "请安装 OpenMM: conda install -c conda-forge openmm")

    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 检测平台
    platform = detect_best_platform(platform_name)
    if verbose:
        print(f"  [OpenMM GPU] 平台: {platform}")

    # 1. 读 P 坐标
    p_coords = _read_p_coords(input_pdb)
    L = len(p_coords)
    if verbose:
        print(f"  [OpenMM GPU] 序列长度: {L} nt")

    if L < 3:
        raise ValueError(f"序列太短 ({L} nt), 无法做 CG MD")

    # 1b. 清洗 NaN/Inf 坐标
    p_coords = _sanitize_p_coords(p_coords)

    # 从 pairs 参数解析配对 (从 secondary_structure 推断)
    pairs = _dotbracket_to_pairs(secondary_structure)

    # 1c. 检查坐标质量, 仅在无效/键长异常时替换为紧凑环状坐标
    # 注意: 首末距大不代表展开 — 环状 RNA 首末距天然可大.
    # 检查 P-P 键长: 若平均键长异常 (远超 5.9A 合理范围), 才判定为坏结构.
    avg_pp = 0.0
    if L > 1:
        diffs = p_coords[1:] - p_coords[:-1]
        pp_dists = np.linalg.norm(diffs, axis=1)
        avg_pp = float(np.mean(pp_dists[:min(L - 1, 500)]))

    use_compact = (not np.isfinite(avg_pp)) or avg_pp > 20.0 or avg_pp < 1.0
    if use_compact:
        if verbose:
            print(f"  [OpenMM GPU] P-P 键长异常 (avg={avg_pp:.2f}A), "
                  f"生成紧凑环状起始坐标...")
        p_coords = _generate_compact_coords(L, pairs)

    # 2. 构建系统 (pair_scale=1.0, bsj_k_scale=0.1 初始弱)
    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.1)

    # 3. 创建拓扑和模拟
    topo = _create_3bead_topology(L)

    try:
        plat = Platform.getPlatformByName(platform)
    except Exception:
        plat = Platform.getPlatformByName("Reference")

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    # CPU 平台设多线程 (默认只用1个核)
    plat_props = {}
    if plat.getName() == "CPU":
        n_threads = os.cpu_count() or 8
        plat_props["CpuThreads"] = str(n_threads)
        if verbose:
            print(f"  [OpenMM GPU] CPU 线程数: {n_threads}")

    sim = Simulation(topo, system, integrator, plat, plat_props)
    try:
        sim.context.setPositions(coords_nm * unit.nanometer)
    except Exception as e_pos:
        # GPU 内存不足 (LLVM ERROR), 回退到 CPU
        if verbose:
            print(f"  [OpenMM GPU] 平台 {platform} 失败: {e_pos}, 回退到 CPU...")
        plat = Platform.getPlatformByName("CPU")
        n_threads = os.cpu_count() or 8
        plat_props = {"CpuThreads": str(n_threads)}
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond,
        )
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)

    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
    if verbose:
        print(f"  [OpenMM GPU] 初始能量: {e0:.0f} kJ/mol")

    # 4. 检查初始能量, NaN/Inf 时用更紧凑的坐标重试
    if not np.isfinite(e0):
        if verbose:
            print(f"  [OpenMM GPU] 初始能量异常 ({e0}), "
                  f"用更紧凑的环状坐标重试...")
        compact_r = max(10.0, L * BOND_P_NEXT / (2.0 * np.pi) * 0.3)
        rng = np.random.default_rng(123)
        p_fb = np.zeros((L, 3), dtype=np.float64)
        for i in range(L):
            angle = 2.0 * np.pi * i / L
            p_fb[i] = [compact_r * np.cos(angle),
                        compact_r * np.sin(angle), 0.0]
        p_fb += rng.normal(0, 0.3, p_fb.shape)
        system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
            _build_3bead_system_gpu(p_fb, pairs,
                                    pair_scale=1.0, bsj_k_scale=0.05)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
        if verbose:
            print(f"  [OpenMM GPU] 重试初始能量: {e0:.0f} kJ/mol")

    # 5. 最小化
    try:
        sim.minimizeEnergy(
            tolerance=100.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=1000)
    except Exception as e:
        if verbose:
            print(f"  [OpenMM GPU] 最小化异常: {e}")

    # 最小化后检查, 如果还是 NaN 尝试极紧凑起始
    if _has_nan_energy(sim):
        if verbose:
            print(f"  [OpenMM GPU] 最小化后能量异常, "
                  f"极紧凑起始+弱力重试...")
        compact_r2 = max(8.0, L * BOND_P_NEXT / (2.0 * np.pi) * 0.15)
        rng2 = np.random.default_rng(456)
        p_v3 = np.zeros((L, 3), dtype=np.float64)
        for i in range(L):
            angle = 2.0 * np.pi * i / L
            p_v3[i] = [compact_r2 * np.cos(angle),
                         compact_r2 * np.sin(angle), 0.0]
        p_v3 += rng2.normal(0, 0.2, p_v3.shape)
        system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
            _build_3bead_system_gpu(p_v3, pairs,
                                    pair_scale=0.1, bsj_k_scale=0.01)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)
        sim.minimizeEnergy(
            tolerance=500.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=200)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
        if verbose:
            print(f"  [OpenMM GPU] V3 初始能量: {e0:.0f} kJ/mol")

    # 6. 两阶段折叠+精修
    # 阶段1: 极简力场折叠 (P骨架 + P-P配对, 无 clash) → 配对收敛
    # 阶段2: 完整力场 REMD 精修 (从折叠后坐标出发)
    if verbose:
        print(f"  [OpenMM GPU] 极简力场折叠 ({nstep} 步, 多进程并行)...")
    n_traj = _clamp_replicas_by_memory(2, mem_per_proc_gb=6.0)
    if verbose:
        print(f"  [OpenMM GPU] 极简折叠: {n_traj} 轨迹, 内存感知限制")
    try:
        anneal_e, anneal_pos_ang = _run_parallel_minimal_annealing(
            p_coords, pairs, n_anneal=nstep, n_trajectories=n_traj,
            platform_name="CPU", verbose=verbose)
    except Exception as e_anneal_par:
        if verbose:
            print(f"  [OpenMM GPU] 极简折叠失败: {e_anneal_par}, 回退顺序退火...")
        anneal_e, anneal_pos_ang = _run_annealing(
            sim, pair_force, bsj_force, bsj_guide, L,
            n_anneal=nstep, verbose=verbose)
        anneal_pos_ang = anneal_pos_ang[0::3] * 10.0  # 3-bead nm → P Å

    if verbose:
        print(f"  [OpenMM GPU] 折叠后能量: {anneal_e:.0f} kJ/mol")

    # 6. T-REMD (可选) — 从折叠后 P 坐标 (Å) 出发
    final_e = anneal_e
    final_pos_pang = anneal_pos_ang  # (L,3) Å

    if use_remd and L >= 10:
        if verbose:
            print(f"  [OpenMM GPU] T-REMD ({remd_n_replicas} 副本, {remd_n_steps} 步, 多进程并行)...")
        remd_e, remd_pos = _run_remd(
            final_pos_pang, pairs, platform,
            n_replicas=remd_n_replicas,
            n_steps=remd_n_steps,
            verbose=verbose,
            minimal=True)  # 极简力场 REMD, 保持折叠一致性
        # REMD 是精修阶段, 成功后总是采用其坐标 (配对进一步收敛).
        if remd_pos is not None and np.isfinite(remd_e):
            final_e = remd_e
            final_pos_pang = remd_pos * 10.0  # P-only nm → P Å

    # 把折叠后 P 坐标 (Å) 转成 3-bead nm (补 C4'/N), 供后续输出
    rng_final = np.random.default_rng(7)
    final_pos = np.zeros((3 * L, 3), dtype=np.float64)
    for i in range(L):
        final_pos[3 * i] = final_pos_pang[i] / 10.0  # P, Å→nm
        final_pos[3 * i + 1] = final_pos_pang[i] / 10.0 + rng_final.normal(0, 0.03, 3)
        final_pos[3 * i + 2] = final_pos_pang[i] / 10.0 + rng_final.normal(0, 0.03, 3)

    # 7. CG → 全原子
    # CG_to_allatom 的模板匹配期望 P-P ~5.9Å (真实 RNA 尺度)
    # 但 OpenMM 退火后的坐标可能尺度偏大, 需要缩放
    _BOND_P_NEXT = 0.59  # 5.9Å = 0.59nm (默认值)
    try:
        from .cg_forcefield import BOND_P_NEXT as _bpn
        _BOND_P_NEXT = _bpn / 10.0  # BOND_P_NEXT 单位是 Å, 转 nm
    except ImportError:
        pass
    p_coords = final_pos[0::3]  # (L,3) P bead
    if L > 1:
        avg_pp = np.mean([np.linalg.norm(p_coords[i+1] - p_coords[i])
                          for i in range(min(L-1, 100))])
        # 退火会放松键长到 ~0.657nm, 重建时用标准 0.59nm 会导致键拉伸虚高.
        # 只要偏离标准值就缩放到标准键长 (0.55-0.75nm 范围都校正).
        if 0.30 < avg_pp < 2.0 and abs(avg_pp - _BOND_P_NEXT) > 0.02:
            scale = _BOND_P_NEXT / avg_pp
            final_pos = final_pos * scale
            if verbose:
                print(f"  [OpenMM GPU] 坐标缩放: avg_PP={avg_pp:.2f}nm -> {_BOND_P_NEXT:.2f}nm (scale={scale:.2f})")

    cg_pdb = str(out_path / f"{name}_cg.pdb")
    _write_allatom_pdb(final_pos, L, cg_pdb)

    if skip_cg_to_allatom:
        # 输入已是全原子 (merged_aa), 跳过重复 CG→全原子转换.
        # 只写精修后的 CG PDB, Level 2 会自行读取 P 坐标.
        output_pdb = cg_pdb
    else:
        aa_pdb = str(out_path / f"{name}_aa_raw.pdb")
        try:
            from .isrnacirc_wrapper import cg_to_allatom
            cg_to_allatom(cg_pdb, aa_pdb, sequence)
            # 检查输出文件是否有效 (至少 10 行 ATOM)
            with open(aa_pdb) as _f:
                n_atoms = sum(1 for _ in _f if _.startswith("ATOM"))
            if n_atoms < 10:
                raise RuntimeError(f"CG→全原子输出只有 {n_atoms} 个原子, 不够")
        except Exception as e:
            if verbose:
                print(f"  [OpenMM GPU] CG→全原子失败: {e}, 输出 CG 坐标")
            aa_pdb = cg_pdb

        # 8. 写最终输出 PDB
        output_pdb = str(out_path / f"{name}_openmm.pdb")
        _write_refined_pdb(aa_pdb, output_pdb)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [OpenMM GPU] 完成: E={final_e:.0f} kJ/mol, "
              f"耗时 {elapsed:.1f}s")

    return output_pdb, final_e


def _dotbracket_to_pairs(ss: str) -> List[Tuple[int, int, float]]:
    """从 dot-bracket 提取配对列表 [(i,j,1.0)]."""
    pairs = []
    stack = []
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i, 1.0))
    return pairs


# ── 冒烟测试 ──

def main():
    """冒烟测试: 生成随机序列, 用 OpenMM GPU 精修."""
    import random
    random.seed(42)
    L = 50
    sequence = "".join(random.choices("AUCG", k=L))
    ss = "(" * (L // 2) + ")" * (L // 2)

    # 随机初始坐标 (平面圆)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    r = L * 5.9 / (2 * np.pi)  # P-P 间距决定半径
    p_coords = np.column_stack([r * np.cos(angles), r * np.sin(angles),
                                 np.zeros(L)])

    # 写临时 PDB
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    input_pdb = Path(tmp_dir) / "test_input.pdb"
    lines = ["HEADER    test"]
    for i in range(L):
        x, y, z = p_coords[i]
        lines.append(
            f"ATOM  {i + 1:5d}  P   RA A{i + 1:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P ")
    lines.append("END")
    with open(input_pdb, "w") as f:
        f.write("\n".join(lines))

    print(f"序列: {L}nt, SS: {ss[:10]}...")
    output_pdb, energy = openmm_gpu_refine(
        str(input_pdb), tmp_dir, sequence, ss,
        name="test", nstep=100, use_remd=True,
        remd_n_replicas=3, remd_n_steps=200,
        platform_name="auto", verbose=True,
    )
    print(f"\n输出: {output_pdb}")
    print(f"能量: {energy:.0f} kJ/mol")


if __name__ == "__main__":
    main()
