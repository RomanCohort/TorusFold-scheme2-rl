"""Level 5 AMBER RNA.OL3 全原子精修 — 独立脚本"""
import sys
sys.path.insert(0, "src")
from torusfold.scheme2.openmm_amber_refiner import openmm_amber_refine

out, e = openmm_amber_refine(
    "output_2013nt/isrnaclong_final_allatom_v2.pdb",
    "output_2013nt/level5_amber",
    name="level5",
    minimize_max_iter=20000,
    md_steps=30000,
    use_remd=True,
    remd_n_replicas=8,
    remd_n_steps=5000,
    verbose=True,
)
print(f"Level 5 DONE: E={e:.0f}, out={out}")
