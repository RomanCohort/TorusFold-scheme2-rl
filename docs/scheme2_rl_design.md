"""
RL Far-end Pair Optimization Design for TorusFold Scheme2

Status: Design finalized (2026-07-21), pending implementation
Role: RL does NOT replace physics pipeline; only patches circRNA long-range pairing local-optimum blind spot

---

## 1. Problem Background

Scheme2 current pipeline: ViennaRNA pairing -> CG geometric solver -> 1EHZ template all-atom reconstruction -> amber14 OL3 force field refinement

Empirical 32nt sequence diagnosis (2026-07-21):
- Near-end pairing (pair 2-31) C1'-C1' = 10.79 Å, close to WC truth 10.5 Å ✓
- Far-end pairing (pair 4-29) C1'-C1' = 18.75 Å, severely deviated ✗
- BSJ closure 2.52 Å, deviated from truth 1.61 Å ✗
- amber_field = -10664 kJ/mol (negative, physically reasonable)

Root cause: CG geometric solver fails to pull in long-range pairings; amber all-atom refinement cannot rescue due to energy barriers.
L-BFGS minimization only rolls into nearest local minimum, cannot cross barrier to pull in far-end pairings.

**RL's role**: Use MCTS to explore and escape local optimum at CG granularity, pull far-end pairings to WC geometry,
then hand over to existing physics pipeline (1EHZ reconstruction + amber refinement) to refine local geometry.

---

## 2. Finalized Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RL paradigm | MCTS + policy network | Natural integration with existing amber minimization, exploration escapes local optimum |
| RL granularity | CG granularity (P coordinates) | 2013nt all-atom infeasible, CG runs in seconds |
| RL scope | Optimize far-end pairings only | Near-end pairings already OK from CG+amber, no redundant work |
| Far-end definition | Topological distance (graph distance) | |i-j| on circular RNA has cross-BSJ false positives |
| Pair graph source | ViennaRNA + complementarity scan | Scan supplements long-range pairings ViennaRNA misses (HA inverted repeats) |
| Action space | Far-end pairing block-level translation | Not residue-level, fewer steps for faster RL learning |
| reward | Only far-end pairing C1'-C1' deviation | Focused, excludes near-end/BSJ/energy |
| Training data | Synthetic + PDB real combined | Real for calibration, synthetic for scale |
| Pipeline | CG -> RL far-end optimization -> 1EHZ reconstruction -> amber refinement | RL inserted after CG |
| Deployment | Full training + server integration | Paper chapter level |

---

## 3. Pair Graph Construction (Key: Topological Distance)

### 3.1 Pair Source Merging

```
1. ViennaRNA pair_probs(thr=0.5) -> primary pair set P_vienna
2. Complementarity scan -> supplementary pair set P_scan
3. P_all = P_vienna ∪ P_scan (deduplicated)
```

### 3.2 Complementarity Scan Algorithm

Supplements long-range pairings ViennaRNA misses (e.g., HA inverted repeats):

- Sliding window W = 6 (minimum stable RNA stem length, scans internal pairings in poly-A inverted repeat segments)
- For each position i window seq[i:i+6], find complementary window seq[j:j+6] on the ring
- j traverses full ring, skips ring distance < 10 (avoids self-pairing/adjacent pairing)
- Complementarity criteria: Watson-Crick pairing rate ≥ 80% (allows 1 G·U wobble)
- Energy filter: simple nearest-neighbor free energy ΔG < -5 kcal/mol (removes false positives)

### 3.3 Topological Distance

Graph G = (V, E):
- V = {residue 0, 1, ..., L-1}
- E = skeleton adjacent edges {(i, (i+1) mod L)} ∪ pairing edges {(i, j) for (i,j) in P_all}

dist(i, j) = BFS shortest path length (edge weight = 1)

**Far-end pairing criteria**: dist(i, j) > 50 -> far-end

This way near-end pairings across BSJ (e.g., (5, 2010) at L=2013, ring distance 8) reach via skeleton in 8 steps,
dist=8, correctly identified as near-end. HA inverted repeat (441-464 ↔ corresponding segment) topologically far,
dist large, correctly identified as far-end.

---

## 4. RL Architecture

### 4.1 State: Far-end Pairing Block Small Graph

**Block extraction**:
- Scan P_all, find consecutive pairing ≥ 4nt segments -> one "stem block"
- Far-end stem blocks (intra-block pairing dist > 50) -> RL optimization candidates
- 2013nt estimated 5-20 far-end stem blocks

**Block GNN**:
- Nodes = far-end stem blocks
- Node features: [block length, block position (centroid P coordinates), current C1'-C1' average deviation, inter-block topological distance]
- Edges = block pairs with inter-block topological distance < 100 (sparse)
- 3-layer MessagePassing GNN, outputs block embedding e_b

### 4.2 Action Space

Discrete action = (block index, translation direction, step size)
- Block index: one of far-end stem blocks (5-20 choose 1)
- Translation direction: 6 basis vectors ±x ±y ±z
- Step size: 3 levels (0.5 Å, 2.0 Å, 5.0 Å)

Each step: select one block for whole translation -> run short CG OpenMM refinement (full L, ~1-3s) -> evaluate reward

### 4.3 Reward

```
R = Σ_{(i,j) in far-end pairings} exp(-|d_C1'C1'(i,j) - 10.5| / 2.0)
```

- Each far-end pairing C1'-C1' close to 10.5 Å scores 1, exponential decay for deviation
- Excludes near-end pairings (already OK)
- Excludes BSJ (closed at CG stage)
- Excludes energy (handed to amber)

### 4.4 MCTS Search

- Node = CG P coordinate state
- Expansion uses policy network π(a|s) for prior probability
- Each step expands 4 candidates (policy top-4), rollout runs CG refinement evaluation
- Total search steps T = 30-100 (few far-end blocks, fewer steps)
- Output state with highest rollout reward

### 4.5 Policy Network

```
Block GNN (3-layer MessagePassing)
  -> block embedding e_b
  -> action head (MLP)
     -> π_block(b): block selection probability (softmax over far-end blocks)
     -> π_dir(d): 6-direction probability
     -> π_step(s): 3-step size probability
```

Training: PPO + GAE, PyTorch.

---

## 5. Pipeline Integration

```
predictor._predict_scheme2:
  1. CG solver (existing: vienna_pair_probs + scheme2_initial_coords + openmm_refine)
  2. [NEW] RL far-end pairing optimization
     a. Build pair graph (ViennaRNA + complementarity scan)
     b. Calculate topological distance, mark far-end pairing blocks
     c. Load policy network, MCTS search optimizes far-end block P coordinates
     d. Output optimized CG P coordinates
  3. 1EHZ template reconstruction (existing: aform_from_template.reconstruct_all_atom)
  4. amber refinement (existing: amber_refine) -- relax interface stress from RL pull-in
```

New modules:
- `scheme2/pair_graph.py`: pair graph construction + complementarity scan + topological distance
- `scheme2/rl_optimizer.py`: policy network + MCTS + weight loading
- `models/scheme2_rl_policy.pt`: policy network weights
- Config adds `use_rl_optimization: bool = True` (turns off to fall back to pure CG)

---

## 6. Training Data

### 6.1 Synthetic Data (scale up)
- Randomly generate 1000 sequences of 50-2000nt
- ViennaRNA gives pairings + complementarity scan supplements
- CG solver outputs "near-end OK, far-end deviated" starting point
- Randomly perturb far-end block P coordinates (σ=5-15 Å) as RL starting point
- Target: RL learns to pull far-end blocks back to CG solver's "approximate target"

### 6.2 PDB Real Data (calibration)
- PDB RNA crystal structures (≤ 300nt, ~1000 entries)
- Extract pair graph, use real coordinates as ground truth
- Starting point: real coordinates with random perturbation
- Target: RL fine-tune to real physics

### 6.3 Training Flow
- Stage 1: Synthetic data PPO training, 5000 episodes
- Stage 2: PDB real fine-tune
- Stage 3: MCTS replay buffer expansion

---

## 7. Engineering Estimation

| Item | Estimate |
|------|----------|
| rl_optimizer.py | ~300 lines |
| pair_graph.py | ~200 lines |
| training/ | ~500 lines |
| predictor integration | ~50 lines |
| tests | ~300 lines |

Hardware:
- Training: GPU (RTX 3060+ or cloud), policy network small, 5000 episodes ≈ 10 hours
- Inference: CPU runnable, MCTS 30-100 steps × 1-3s per step = 5-10 minutes/sequence

Time: 2 weeks deployment (week 1 GNN+synthetic data+PPO, week 2 MCTS+PDB+integration+tests)

---

## 8. Paper Framing

- **Problem**: CG + physics minimization gets stuck in local optimum for circRNA long-range pairings
- **Method**: Focused RL -- MCTS search only on far-end pairing blocks, near-end handed to existing physics pipeline
- **Contributions**:
  1. First application of RL to circRNA 3D structure prediction long-range pairing problem
  2. Topological distance criteria (|i-j| on ring has cross-BSJ false positives, graph distance correct)
  3. ViennaRNA + complementarity scan complementary pair graph construction
  4. Lightweight pluggable, does not break validated physics pipeline
- **Experiments**: Ablation RL vs pure amber, far-end pairing C1'-C1' deviation comparison

---

## 9. Implementation Details to Confirm

- Complementarity scan nearest-neighbor energy parameter table (RNA NN model)
- Policy network specific layers/hidden dimensions (start with 3-layer GNN + 128-dim, tune empirically)
- MCTS c_puct exploration constant
- PPO clip ratio / batch size
- 2013nt inference amber refinement time (all-atom 42000+ atoms, may need precision reduction or block-splitting)
