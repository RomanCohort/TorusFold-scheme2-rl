# TorusFold

circRNA 3D structure prediction with 10 architecture schemes (S0–S10, S5 deprecated, S9 superseded by S10+Phase2), unified TrunkOutput protocol, and SO(2)×SO(2) equivariant immune fingerprint heads.

## Install

```bash
cd D:/TorusFold
pip install -e .
```

## Quick start

Run a smoke test (no training required):

```bash
# from repo root
python -m torusfold.tests.smoke_scheme1_trunk_output
python -m torusfold.tests.smoke_scheme10
python -m torusfold.tests.smoke_immune_equivariance
python -m torusfold.tests.smoke_steerable_kernel
```

## Schemes

| Scheme | Backbone | Notes |
|--------|----------|-------|
| S0 | — | pseudo-label / data generation |
| S1 | EGNN | baseline, lr ≤ 1e-4 with 5-epoch warmup |
| S3 | dual-engine distillation | deferred |
| S4 | DDPM | diffusion |
| S6 | latent GNN | fixed kabsch |
| S7 | Mamba | selective scan (needs `mamba_ssm` for O(L) memory) |
| S8 | sparse pair (dilated) | 12× VRAM reduction, covers 5000 nt |
| S10 | SO(2)×SO(2) equivariant | Phase 2 trunk + immune heads |

S5 deprecated. S9 absorbed into S10 (ring-(θ,φ,r) closure ≡ 0.0).

## TrunkOutput protocol (6 required fields)

All schemes emit a unified `TrunkOutput`:

1. `coords` — 3D backbone coordinates
2. `pair_probs` — base-pair probability matrix
3. `sasa_scalar` — 3D solvent-accessible surface area (scalar)
4. `ring_coords` — (θ, φ, r) torus coordinates (S9/S10)
5. `closure_dist` — torus closure residual (ideal ≡ 0.0)
6. `bsj` — back-splice junction score

## Immune fingerprint heads (5)

Equivariant heads attached to trunk output:

- PKR (dsRNA binding)
- m6A (N6-methyladenosine exposure; `enable_fingerprint_2d` switches exposure sub-quantity, not the whole head)
- TLR7 (endosomal ssRNA)
- NLRP3 (inflammasome)
- miRNA sponge

## Physical fields (8, hub-bridged)

Exposed via `ImmuneSensingResultV3` for CirculaPK and downstream:

`bsj`, `bsj_3d_closure`, `closure_score`, `sasa` ×3 (three definitions), `ires_3d`, `translation_eff`, `energy_score`.

Phase-2 steerable extras: `mech_stiff`, `solvent_resp`, `pair_stab`.

## Training data

- 130k PDB structures + synthetic samples (three-layer weighted), on cloud
- Held-out test set: IsRNAcirc + PDB (no leakage)

## Scheme 2 — CG geometric solve + all-atom amber refine + RL long-range pair optimization

`src/torusfold/scheme2/` is a zero-training pathway (no DL weights required):

```
ViennaRNA pairs → CG geometric solve → OpenMM CG refine → [RL far-pair opt]
                → 1EHZ template all-atom rebuild → Amber14 OL3 + OBC1 refine
```

- `constraint_solver.py` — polygon init + pair constraints + BSJ closure
- `refine.py` — ViennaRNA pairing (`vienna_pair_probs`), CG OpenMM refine
- `aform_from_template.py` — all-atom rebuild from 1EHZ crystal template (replaces hand-tuned geometry; amber_field drops from +71k to -10k kJ/mol)
- `amber_refine.py` — Amber14 OL3 + OBC1 constrained minimization
- `pair_graph.py` — complementarity scan (catches long-range pairs ViennaRNA misses) + topological distance (correct across-BSJ far-pair detection) + `parse_case_annotation` (case → coding mask)
- `rl_optimizer.py` — MCTS + PolicyNetwork (GNN message-passing) for far-pair block optimization
- `immune_heuristic.py` — 5 computable immune fingerprints + structure signals (pure compute, no DL)

### RL positioning (honesty-first)

RL does **not** replace ViennaRNA or amber. It only fills the gap they can't reach on long sequences (1000 nt+):

- **ViennaRNA**: near-range pairs (2D), but far-pair DP explodes on long sequences and misses some
- **amber refine**: local physics only (bond/angle/VdW/clash), can't pull "two segments hundreds of nt apart that should dock but are misplaced"
- **RL**: only supplements these two — pulls far-pair blocks into docked position. Reward is a **search signal, not an evaluation conclusion**; final structure is judged by independent amber energy (e1_aa), and RL output always goes back through amber.

```python
from torusfold.scheme2 import predict_3d_allatom

# baseline: ViennaRNA → CG → amber
res = predict_3d_allatom(seq, max_iterations=3000)

# +RL: ViennaRNA → CG → RL far-pair opt → amber (coding residues pinned to CG pre-RL coords)
res = predict_3d_allatom(seq, use_rl=True, rl_policy_path="models/rl/ppo_gnn_final.pth",
                         rl_n_simulations=50, coding_mask=mask)
```

### coding_mask scheme

RL action space is the full sequence (coding residues also move — needed to dock far pairs).
`coding_mask` is passed through to amber refine: coding-region P atoms are pinned with high k (10000) back to **pre-RL CG coordinates** (preserve real structure), non-coding P with low k (1000) accept RL optimization + physics convergence. RL moving coding is fine — amber pulls it back. Verified: coding P drift 0.5 Å (pulled back), non-coding 2.8 Å (accepted), amber energy stays negative.

Mask source: mixed-case carrier sequences parse by letter case (`parse_case_annotation`); CircBase lowercase-only → default non-coding (RL optimizes full sequence).

### Training RL (PPO + GAE)

```bash
# GNN variant (message passing, n_mp_layers=3) — 2.6x improvement vs MLP in ablation
python training/train_ppo.py --samples data/rl_samples --epochs 50 --batch 8 --variant gnn

# MLP ablation (n_mp_layers=0)
python training/train_ppo.py --variant mlp --epochs 50 --batch 8

# smoke (synthetic samples, 2 epochs)
python training/train_ppo.py --smoke
```

PPO fixes (all verified): true batch gradient aggregation (not single-sample), reward running-normalization (RunningMeanStd), gradient clip (MAX_GRAD_NORM=0.5). Outputs: `models/rl/ppo_<variant>_final.pth` + log JSON + rms stats.

**CPU training note**: amber all-atom refine is CPU-bound (≈10 min for 107 nt on a single core). RL training itself (MCTS + PPO, no amber) is CPU-light and parallelizes well — runs comfortably on a 32-core CPU machine. The end-to-end `compare_rl.py` comparison (which calls amber) is the slow part, not PPO training.

```bash
# comparison experiment (baseline vs +RL on CircBase long sequences)
python training/compare_rl.py --fa D:/IGEM集成方案/data/circrna/circbase_seqs.fa.gz \
    --min-len 1200 --max-len 2200 --limit 5 --policy models/rl/ppo_gnn_final.pth
```

`compare_rl.py` honesty: `rl_improvement` (RL reward gain) is a search signal, not a structure-quality verdict. Final structure evaluation uses `e1_aa` (amber physical energy). CSV + summary printed at end.

## License

MIT — see [LICENSE](./LICENSE).
