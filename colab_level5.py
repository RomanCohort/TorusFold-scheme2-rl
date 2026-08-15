"""
Level 5 AMBER 全原子精修 — Colab GPU 版
Runtime > Change runtime type > T4 GPU > Run all
"""
# Cell 1: 安装
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "openmm", "pdbfixer"], check=True)
import openmm; print(f"OpenMM {openmm.__version__}")
for i in range(openmm.Platform.getNumPlatforms()):
    p = openmm.Platform.getPlatform(i); print(f"  {p.getName()}: speed={p.getSpeed()}")

# Cell 2: 上传 PDB
# from google.colab import files; uploaded = files.upload()
# PDB_PATH = list(uploaded.keys())[0]
PDB_PATH = "isrnaclong_final_allatom_v2.pdb"

# Cell 3: AMBER 精修
import time, numpy as np
import openmm.app as app, openmm.unit as unit, openmm
from pdbfixer import PDBFixer
from openmm.app import Topology, ForceField

t0 = time.time()
fixer = PDBFixer(filename=PDB_PATH)
fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues(); fixer.addMissingHydrogens()
top, pos = fixer.topology, fixer.positions
res_list = list(top.residues())
res0, res_last = res_list[0], res_list[-1]
o3_last = [a for a in res_last.atoms() if a.name == "O3'"][0]
p0 = [a for a in res0.atoms() if a.name == "P"][0]
new_top = Topology(); new_chain = new_top.addChain(); o2n = {}
for res in res_list:
    nr = new_top.addResidue(res.name, new_chain)
    for atom in res.atoms():
        if (res==res0 and atom.name=="HO5'") or (res==res_last and atom.name=="HO3'"): continue
        o2n[atom.index] = new_top.addAtom(atom.name, atom.element, nr)
for bond in top.bonds():
    if bond.atom1.index in o2n and bond.atom2.index in o2n:
        new_top.addBond(o2n[bond.atom1.index], o2n[bond.atom2.index])
new_top.addBond(o2n[o3_last.index], o2n[p0.index])
npos = []
for atom in new_top.atoms():
    for res in res_list:
        if res.index == atom.residue.index:
            for oa in res.atoms():
                if oa.name == atom.name: npos.append(pos[oa.index]); break
            break
topology, positions = new_top, npos
print(f"Atoms: {topology.getNumAtoms()}")

ff = ForceField("amber14/RNA.OL3.xml")
system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
gb = openmm.GBSAOBCForce(); gb.setSoluteDielectric(1.0); gb.setSolventDielectric(78.5)
rm = {"C":(0.22,0.72),"N":(0.17,0.79),"O":(0.15,0.85),"P":(0.20,0.86),"H":(0.12,0.85)}
for atom in topology.atoms():
    e = atom.element.symbol if atom.element else "C"; r,s = rm.get(e,(0.15,0.85)); gb.addParticle(0.0,r,s)
system.addForce(gb)

restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
restraint.addGlobalParameter("k", 500.0)
restraint.addPerParticleParameter("x0"); restraint.addPerParticleParameter("y0"); restraint.addPerParticleParameter("z0")
bb = {"P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"}
for atom in topology.atoms():
    if atom.name in bb:
        p=positions[atom.index]; x0=p[0]._value; y0=p[1]._value; z0=p[2]._value
        restraint.addParticle(atom.index,[x0,y0,z0])
system.addForce(restraint)

try: plat = openmm.Platform.getPlatformByName("CUDA"); print("GPU: CUDA")
except:
    try: plat = openmm.Platform.getPlatformByName("OpenCL"); print("GPU: OpenCL")
    except: plat = openmm.Platform.getPlatformByName("CPU"); print("CPU")

intg = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
sim = app.Simulation(topology, system, intg, plat)
sim.context.setPositions(positions)
print(f"Initial E: {sim.context.getState(getEnergy=True).getPotentialEnergy()._value:.0f}")

for k, iters in [(500,10000),(50,10000),(5,10000),(0,10000)]:
    restraint.setGlobalParameterDefaultValue(0, k)
    sim.minimizeEnergy(maxIterations=iters)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
    print(f"  k={k}: E={e:.0f}")

restraint.setGlobalParameterDefaultValue(0, 2.0)
sim.step(10000)
e = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
print(f"Final E: {e:.0f}")

state = sim.context.getState(getPositions=True)
app.PDBFile.writeFile(topology, state.getPositions(), open("level5_amber_gpu.pdb", "w"))
print(f"Done in {(time.time()-t0)/60:.1f} min")
