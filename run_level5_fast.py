"""
Level 5 AMBER 快速精修 — CPU 优化版
策略: 强骨架束缚 + 分阶段释放 + 最小化+短MD
预期: 1-2 hrs (vs 原版 6-12 hrs)
"""
import sys, time, numpy as np
sys.path.insert(0, "src")
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer
from openmm.app import Topology, ForceField

INPUT_PDB = "output_2013nt/isrnaclong_final_allatom_v2.pdb"
OUTPUT_PDB = "output_2013nt/level5_amber_fast.pdb"

t_start = time.time()

# Step 1: 加氢 + 环状修复
print("[1/4] PDBFixer 加氢 + 环状修复...")
fixer = PDBFixer(filename=INPUT_PDB)
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.addMissingHydrogens()

top = fixer.topology
pos = fixer.positions
res_list = list(top.residues())
res0, res_last = res_list[0], res_list[-1]
o3_last = [a for a in res_last.atoms() if a.name == "O3'"][0]
p0 = [a for a in res0.atoms() if a.name == "P"][0]

new_top = Topology()
new_chain = new_top.addChain()
old_to_new = {}
for res in res_list:
    new_res = new_top.addResidue(res.name, new_chain)
    for atom in res.atoms():
        if (res == res0 and atom.name == "HO5'") or (res == res_last and atom.name == "HO3'"):
            continue
        old_to_new[atom.index] = new_top.addAtom(atom.name, atom.element, new_res)
for bond in top.bonds():
    if bond.atom1.index in old_to_new and bond.atom2.index in old_to_new:
        new_top.addBond(old_to_new[bond.atom1.index], old_to_new[bond.atom2.index])
new_top.addBond(old_to_new[o3_last.index], old_to_new[p0.index])

new_pos = []
for atom in new_top.atoms():
    for res in res_list:
        if res.index == atom.residue.index:
            for orig_atom in res.atoms():
                if orig_atom.name == atom.name:
                    new_pos.append(pos[orig_atom.index])
                    break
            break
topology, positions = new_top, new_pos
print(f"  Atoms: {topology.getNumAtoms()}")

# Step 2: 构建 AMBER system
print("[2/4] 构建 AMBER system...")
ff = ForceField("amber14/RNA.OL3.xml")
system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
gb = mm.GBSAOBCForce()
gb.setSoluteDielectric(1.0)
gb.setSolventDielectric(78.5)
rm = {"C":(0.22,0.72),"N":(0.17,0.79),"O":(0.15,0.85),"P":(0.20,0.86),"H":(0.12,0.85)}
for atom in topology.atoms():
    elem = atom.element.symbol if atom.element else "C"
    r, s = rm.get(elem, (0.15, 0.85))
    gb.addParticle(0.0, r, s)
system.addForce(gb)

restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
restraint.addGlobalParameter("k", 500.0)
restraint.addPerParticleParameter("x0"); restraint.addPerParticleParameter("y0"); restraint.addPerParticleParameter("z0")
bb = {"P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"}
for atom in topology.atoms():
    if atom.name in bb:
        p = positions[atom.index]
        x0 = p[0]._value if hasattr(p[0],'_value') else float(p[0])
        y0 = p[1]._value if hasattr(p[1],'_value') else float(p[1])
        z0 = p[2]._value if hasattr(p[2],'_value') else float(p[2])
        restraint.addParticle(atom.index, [x0,y0,z0])
system.addForce(restraint)

# Step 3: 分阶段最小化
print("[3/4] 分阶段最小化...")
plat = app.Platform.getPlatformByName("CPU")
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
sim = app.Simulation(topology, system, integrator, plat, {"CpuThreads": "32"})
sim.context.setPositions(positions)
print(f"  Initial E: {sim.context.getState(getEnergy=True).getPotentialEnergy()._value:.0f}")

for phase, (k, iters) in enumerate([(500,5000),(50,5000),(5,5000),(0,5000)], 1):
    restraint.setGlobalParameterDefaultValue(0, k)
    sim.minimizeEnergy(maxIterations=iters)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
    print(f"  Phase {phase} (k={k}): E={e:.0f}")

# Phase 5: 短 MD
restraint.setGlobalParameterDefaultValue(0, 2.0)
print("  Phase 5: MD 5000 steps...")
sim.step(5000)
e = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
print(f"  E after MD: {e:.0f}")

# Step 4: 保存
print("[4/4] 保存...")
state = sim.context.getState(getPositions=True)
app.PDBFile.writeFile(topology, state.getPositions(), open(OUTPUT_PDB, "w"))
elapsed = (time.time() - t_start) / 60
print(f"\nDone! E: {e:.0f}, Time: {elapsed:.1f} min")
print(f"Saved: {OUTPUT_PDB}")
