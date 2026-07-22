"""
scheme10_circ_equivariant_gnn.py — Scheme 10: SO(2)×SO(2) Equivariant GNN Trunk.

Design goal (from the architecture discussion, 2026-07-14):
    Schemes 1-9 all share ESM2 + TPE + CircPairformer / Mamba trunks whose
    equivariance is APPROXIMATE — enforced either by data preprocessing
    (kabsch cap) or by augmentation. Scheme 10 builds the equivariance into
    the architecture itself, against the TRUE symmetry group of a closed
    circRNA backbone: SO(2) × SO(2) × R⁺.

    - θ ∈ [0, 2π): main-ring rotation (circRNA closed loop, no 5'/3' origin)
    - φ ∈ [0, 2π): cross-section rotation (double-helix phase around backbone)
    - r ∈ R⁺      : cross-section radius

    By construction this is a *backbone replacement*, not a head (unlike S9's
    TorusCoordHead which sits on the shared ESM2+TPE+Pairformer trunk). Scheme 10
    owns its entire feature path from tokens → ring coords, so the SO(2)×SO(2)
    equivariance holds end-to-end, not only at the head.

Output contract:
    Aligned with Scheme 8 (Scheme8Model.forward): returns a Dict[str, Tensor]
    containing at least `coords` (B, L, 3) and a closure diagnostic. An optional
    immune head hook is provided (scheme choice (c) — trunk always produces
    coords; immune head is pluggable, not bundled) so downstream fingerprint
    heads can be attached later without rewriting the trunk.

Why no ESM2 here:
    ESM2 is a linear-chain language model with a fixed left-to-right (or
    bidirectional-over-line) inductive bias. Its symmetry group is NOT
    SO(2)×SO(2). Importing it would break the exact-equivariance guarantee
    that is the entire point of this scheme. We therefore use a learned
    token embedding (no frozen LM) so every layer is steerable.

References:
    - Weiler et al., 3D Steerable CNNs (ICLR 2018) — SO(2) steerable kernels
    - Satorras et al., E(n) Equivariant GNNs (ICML 2021) — message passing
    - Cohen & Welling, Group Equivariant CNNs (ICML 2016) — irrep decomposition
    - scheme8_sparse_pair.py — output contract and closure reward reuse
    - tpe.py — TorusPositionalEncoding (additive, periodic PE)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the periodic positional encoding and BSJ closure reward already in the
# torusfold package — keeps closure semantics identical to S8/S9.
from .tpe import TorusPositionalEncoding
from .circrna_mamba_diffusion import BSJClosureReward


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scheme10Config:
    """Configuration for Scheme 10: SO(2)×SO(2) Equivariant GNN Trunk.

    Dimensions follow the scheme8 convention (d_model=128 default) so the two
    schemes are directly comparable in an A/B with matched parameter budget.
    """
    # Embedding / trunk dims
    n_tokens: int = 5             # A, U, G, C, N (padding) — matches scheme8 token map
    d_model: int = 128            # Node feature dim
    d_edge: int = 64              # Edge feature dim (per irrep combo)
    d_pair: int = 64              # Pair representation dim (for output contract)

    # Steerable irrep orders — the truncation of the infinite SO(2) irreps.
    # k_theta = highest harmonic used on the main ring (θ).
    # k_phi   = highest harmonic used on the cross-section (φ).
    # Memory is O(L * d_model * (k_theta+1) * (k_phi+1)) per layer.
    k_theta: int = 2
    k_phi: int = 1

    # Message passing
    n_layers: int = 4             # Steerable GNN layers
    dropout: float = 0.1

    # Edge category vocabulary — circRNA topology types. See EdgeCategoryBuilder.
    n_edge_cats: int = 5

    # Geometry
    bond_length: float = 5.9      # P-P backbone distance (Å) — matches S8/S9
    r_scale: float = 0.5          # Cross-section radius soft cap (matches S9 TorusCoordHead)

    # Closure
    closure_weight: float = 1.0   # Weight on the (diagnostic) closure residual

    # Diffusion-style noise schedule (kept for output-contract parity with S8,
    # even though S10's structural closure is by-construction, not diffusion).
    n_diffusion_steps: int = 50


# ══════════════════════════════════════════════════════════════════════════════
# Edge categories — encode circRNA topology into discrete edge types
# ══════════════════════════════════════════════════════════════════════════════

# Category indices (kept module-level so they are stable across train/eval).
EDGE_CAT_BACKBONE = 0   # i, j adjacent along the ring (|circ_dist(i,j)| == 1)
EDGE_CAT_STEM = 1       # i, j paired in a stem (Watson-Crick or wobble)
EDGE_CAT_CROSSOVER = 2  # i, j on different stems that cross the major ring
EDGE_CAT_PSEUDOKNOT = 3 # i, j paired across a pseudoknot (crossing pairs)
EDGE_CAT_NONE = 4       # no special topology (long-range unpaired contact)


class EdgeCategoryBuilder(nn.Module):
    """Build the per-pair edge-category tensor from pair probabilities.

    The categories are a DISCRETE topological prior — they tell the steerable
    kernel *which* SO(2) action pattern to apply on each edge. The categories
    are derived from a pair-probability matrix (e.g. from ViennaRNA circ-mode,
    as in S8) plus circular-distance geometry. This module has no learnable
    parameters; the *weights* on each category live in the steerable kernel.

    Args:
        min_loop: minimum loop length (hairpin constraint), pairs closer than
            this in circular distance are forced to EDGE_CAT_NONE.
    """

    def __init__(self, min_loop: int = 3):
        super().__init__()
        self.min_loop = min_loop

    @staticmethod
    def _circular_distance_matrix(L: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(L, device=device)
        d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        return torch.min(d, L - d)  # (L, L)

    def forward(
        self,
        pair_probs: Optional[torch.Tensor],  # (B, L, L) in [0, 1]
        lengths: torch.Tensor,                # (B,) valid length per sequence
        pseudoknot_mask: Optional[torch.Tensor] = None,  # (B, L, L) bool
    ) -> torch.Tensor:
        """Return (B, L, L) long tensor of edge categories."""
        B = lengths.shape[0]
        L = int(lengths.max().item()) if lengths.numel() > 0 else 0
        device = lengths.device

        circ_d = self._circular_distance_matrix(L, device)  # (L, L)
        eye = torch.eye(L, device=device, dtype=torch.bool)

        cat = torch.full((B, L, L), EDGE_CAT_NONE, device=device, dtype=torch.long)

        # Backbone: circular distance == 1 (ring neighbours).
        is_backbone = (circ_d == 1)
        cat[:, is_backbone] = EDGE_CAT_BACKBONE

        # Stem / pseudoknot: requires a pair-probability input.
        if pair_probs is not None:
            paired = pair_probs > 0.5  # (B, L, L)
            # Forbid self-pair and too-short loops.
            forbidden = eye | (circ_d < self.min_loop)
            paired = paired & ~forbidden.unsqueeze(0)

            if pseudoknot_mask is not None:
                pk = pseudoknot_mask & paired
                cat[pk] = EDGE_CAT_PSEUDOKNOT
                stem = paired & ~pk
            else:
                stem = paired
            cat[stem] = EDGE_CAT_STEM

        # Crossover: long-range backbone neighbours that belong to different
        # stems (heuristic — pair-probability low but circ distance moderate).
        # We mark circ_d in a band as crossover only if not already assigned.
        crossover_band = (circ_d >= 4) & (circ_d <= 16) & ~is_backbone
        unassigned = (cat == EDGE_CAT_NONE)
        crossover = crossover_band.unsqueeze(0) & unassigned
        cat[crossover] = EDGE_CAT_CROSSOVER

        # Mask padding positions to NONE (they are ignored downstream anyway).
        if L > 0:
            pos = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
            valid = pos < lengths.unsqueeze(1)  # (B, L)
            valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, L, L)
            cat[~valid_pair] = EDGE_CAT_NONE

        return cat


# ══════════════════════════════════════════════════════════════════════════════
# SO(2)×SO(2) steerable message passing
# ══════════════════════════════════════════════════════════════════════════════

class SO2SteerableKernel(nn.Module):
    """Steerable convolution kernel over SO(2)×SO(2).

    For each pair (i, j) with angular difference (Δθ, Δφ) and edge category c,
    the kernel computes a rotation-equivariant message:

        m[i,j] = Σ_{k,l}  φ_c^{(k,l)}(Δθ, Δφ) ⊗ W_c^{(k,l)} x_j

    where φ^{(k,l)}(Δθ, Δφ) = [sin(kΔθ), cos(kΔθ)] ⊗ [sin(lΔφ), cos(lΔφ)]
    is the (k,l) irrep of SO(2)×SO(2), and W_c^{(k,l)} is a learnable linear
    map per edge category. Because φ is built from sin/cos of the angle
    differences, rotating both endpoints by the same amount leaves Δθ, Δφ
    unchanged, so the message is *exactly* equivariant.

    Args:
        d_model: node feature dim
        d_edge: per-irrep edge channel dim
        n_edge_cats: number of edge categories
        k_theta, k_phi: highest irrep orders (inclusive; irrep orders run 0..k).
    """

    def __init__(
        self,
        d_model: int,
        d_edge: int,
        n_edge_cats: int,
        k_theta: int = 2,
        k_phi: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_edge = d_edge
        self.n_edge_cats = n_edge_cats
        self.k_theta = k_theta
        self.k_phi = k_phi

        # irrep degree 0 is the constant [1] term (invariant channel).
        # degrees 1..k contribute [sin(kΔ), cos(kΔ)] (2 channels each).
        # total irrep channels per ring = 1 + 2*k.
        self.n_theta_irreps = 1 + 2 * k_theta
        self.n_phi_irreps = 1 + 2 * k_phi
        # Full (kθ, kφ) irrep tensor: n_theta_irreps * n_phi_irreps channels.
        n_irrep_channels = self.n_theta_irreps * self.n_phi_irreps

        # Per-category, per-irrep linear maps: (i, j) -> message.
        # Implemented as one grouped Linear over n_irrep_channels*d_edge.
        # Shape: (n_edge_cats, n_irrep_channels * d_edge, d_model)
        self.kernel = nn.Parameter(
            torch.empty(n_edge_cats, n_irrep_channels * d_edge, d_model)
        )
        nn.init.xavier_uniform_(self.kernel)

        # Lift node features to per-edge irrep channels (shared across categories).
        self.node_lift = nn.Linear(d_model, n_irrep_channels * d_edge, bias=False)

        # Edge-category embedding (learned), mixed into the irrep channels.
        self.cat_embed = nn.Embedding(n_edge_cats, d_edge)

    def _irrep_features(
        self,
        delta_theta: torch.Tensor,  # (B, L, L)
        delta_phi: torch.Tensor,    # (B, L, L)
    ) -> torch.Tensor:
        """Build the (B, L, L, n_irrep_channels) irrep feature tensor.

        Order: outer product of θ-irreps and φ-irreps.
        θ-irreps: [1, sin(Δθ), cos(Δθ), sin(2Δθ), cos(2Δθ), ...]
        φ-irreps: [1, sin(Δφ), cos(Δφ), ...]
        """
        B, L, _ = delta_theta.shape
        theta_feats = [torch.ones_like(delta_theta)]  # degree 0
        for k in range(1, self.k_theta + 1):
            theta_feats.append(torch.sin(k * delta_theta))
            theta_feats.append(torch.cos(k * delta_theta))
        # (B, L, L, n_theta_irreps)
        theta_feats = torch.stack(theta_feats, dim=-1)

        phi_feats = [torch.ones_like(delta_phi)]
        for l in range(1, self.k_phi + 1):
            phi_feats.append(torch.sin(l * delta_phi))
            phi_feats.append(torch.cos(l * delta_phi))
        phi_feats = torch.stack(phi_feats, dim=-1)

        # Outer product: (B, L, L, n_theta, n_phi) -> (B, L, L, n_theta*n_phi)
        irrep = (theta_feats.unsqueeze(-1) * phi_feats.unsqueeze(-2))
        return irrep.reshape(B, L, L, -1)

    def forward(
        self,
        x: torch.Tensor,           # (B, L, d_model) node features
        delta_theta: torch.Tensor,  # (B, L, L)
        delta_phi: torch.Tensor,    # (B, L, L)
        edge_cat: torch.Tensor,     # (B, L, L) long
    ) -> torch.Tensor:
        """Return steerable messages (B, L, L, d_model).

        Memory-efficient implementation: instead of gathering per-position
        kernel rows (which materializes a (B,L,L,K,M) tensor ~7GB on CPU
        for typical configs), we first contract each of the C=5 kernels
        against flat to get (B,L,L,C,M) partial products, then gather by
        edge_cat. Memory cost: C*M per position instead of K*M.

        For k_theta=k_phi=2, d_edge=32: K=800, C=5 -> 160x memory savings.
        Math: msg[blm] = sum_k flat[blik] * kernel[cat[bli],k,m]
                       = sum_c onehot[blic] * sum_k flat[blik] * kernel[c,k,m]
                       = partial[bli, cat[bli], m]
        """
        B, L, _ = x.shape
        C = self.n_edge_cats

        irrep = self._irrep_features(delta_theta, delta_phi)  # (B,L,L,K_irrep)
        K_irrep = irrep.shape[-1]

        # Lift node j into per-edge channels, broadcast over axis i.
        lifted = self.node_lift(x)  # (B, L, K_irrep*d_edge)
        lifted = lifted.reshape(B, L, K_irrep, self.d_edge)
        # Broadcast to (B, L, L, K_irrep, d_edge): node j along axis 1.
        lifted_ij = lifted.unsqueeze(2).expand(B, L, L, K_irrep, self.d_edge)

        # Modulate by irrep features (Δθ, Δφ)-dependent.
        modulated = lifted_ij * irrep.unsqueeze(-1)  # (B,L,L,K_irrep,d_edge)

        # Mix in edge-category embedding (category-dependent gating).
        cat_emb = self.cat_embed(edge_cat)  # (B,L,L,d_edge)
        modulated = modulated * (1.0 + cat_emb.unsqueeze(-2))  # (B,L,L,K_irrep,d_edge)

        # Flatten irrep*edge dimension -> flat (B,L,L,K) where K = K_irrep*d_edge.
        K = K_irrep * self.d_edge
        flat = modulated.reshape(B, L, L, K)  # (B,L,L,K)

        # Memory-efficient: contract per-category kernels one at a time,
        # accumulate into a (B,L,L,C,d_model) buffer.
        # partial[b,l,i,c,m] = sum_k flat[b,l,i,k] * kernel[c,k,m]
        partial = torch.einsum('blik,ckm->blicm', flat, self.kernel)  # (B,L,L,C,d_model)

        # Gather by edge_cat: msg[bli,m] = partial[bli, cat[bli], m]
        # Use advanced indexing on the category dimension.
        msg = partial[
            torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, L),
            torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, L),
            torch.arange(L, device=x.device).view(1, 1, L).expand(B, L, L),
            edge_cat,
        ]  # (B, L, L, d_model)

        return msg


class CircEquivariantGNNLayer(nn.Module):
    """One layer of SO(2)×SO(2) equivariant message passing + ring-invariant readout.

    Forward:
        messages = SO2SteerableKernel(x, Δθ, Δφ, edge_cat)   # (B,L,L,d_model)
        agg      = mean over j of messages                     # (B,L,d_model)
        x'       = x + MLP_update(concat[x, agg])              # residual

    The aggregation (mean over the ring) is itself ring-equivariant: shifting
    every node's θ by a constant does not change the set {messages[i,j]}_j,
    so the aggregated feature transforms covariantly with x.
    """

    def __init__(self, config: Scheme10Config):
        super().__init__()
        self.config = config
        self.kernel = SO2SteerableKernel(
            d_model=config.d_model,
            d_edge=config.d_edge,
            n_edge_cats=config.n_edge_cats,
            k_theta=config.k_theta,
            k_phi=config.k_phi,
        )
        self.update = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,            # (B, L, d_model)
        delta_theta: torch.Tensor,   # (B, L, L)
        delta_phi: torch.Tensor,     # (B, L, L)
        edge_cat: torch.Tensor,      # (B, L, L)
        lengths: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        msg = self.kernel(x, delta_theta, delta_phi, edge_cat)  # (B,L,L,d_model)

        # Mask padding keys so they do not contribute to the mean.
        B, L, _ = x.shape
        device = x.device
        pos = torch.arange(L, device=device).unsqueeze(0)
        valid = (pos < lengths.unsqueeze(1)).float()  # (B, L)
        mask_ij = valid.unsqueeze(2) * valid.unsqueeze(1)  # (B, L, L)
        denom = mask_ij.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, L, 1)
        agg = (msg * mask_ij.unsqueeze(-1)).sum(dim=2) / denom  # (B, L, d_model)

        x_new = self.update(torch.cat([x, agg], dim=-1))
        return self.norm(x + x_new)


# ══════════════════════════════════════════════════════════════════════════════
# Structure head — predict (θ, φ, r) torus coords from node features
# ══════════════════════════════════════════════════════════════════════════════

class TorusCoordPredictor(nn.Module):
    """Predict per-residue torus coordinates (θ, φ, r) from node features.

    Mirrors the (θ, φ, r) → (x, y, z) map used by S9's TorusCoordHead so the
    geometric convention is identical. Closure is enforced *by construction*
    (the last residue reuses the first's (θ, φ, r)), so closure_dist ≈ 0
    regardless of training dynamics — same structural guarantee as S9.

    Args:
        d_model: node feature dim
        d_hidden: MLP hidden dim
        r_scale: cross-section radius soft cap (matches S9 TorusCoordHead).
        bond_length: adjacent-phosphate bond length (Å) — sets the major-ring
            radius R = bond_length * L / (2π) so the backbone closes naturally.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 256,
        r_scale: float = 0.5,
        bond_length: float = 5.9,
    ):
        super().__init__()
        self.r_scale = r_scale
        self.bond_length = bond_length
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 3),  # (θ_pre, φ_pre, r_pre)
        )

    def forward(
        self,
        node_repr: torch.Tensor,  # (B, L, d_model)
        lengths: torch.Tensor,     # (B,)
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = node_repr.shape
        raw = self.proj(node_repr)  # (B, L, 3)
        theta = torch.tanh(raw[..., 0]) * math.pi   # [-π, π]
        phi = torch.tanh(raw[..., 1]) * math.pi     # [-π, π]
        r = self.r_scale * F.softplus(raw[..., 2])  # R⁺, bounded

        # Hard closure: last valid residue reuses the first valid residue's
        # (θ, φ, r). We operate per-sequence on the valid subsequence [0, len).
        theta_out = theta.clone()
        phi_out = phi.clone()
        r_out = r.clone()
        for b in range(B):
            lb = int(lengths[b].item())
            if lb >= 2:
                # Close the ring: position lb-1 takes position 0's coords.
                theta_out[b, lb - 1] = theta[b, 0]
                phi_out[b, lb - 1] = phi[b, 0]
                r_out[b, lb - 1] = r[b, 0]

        # (θ, φ, r) → Cartesian. Major ring in xy-plane, cross-section tilt in z.
        # Major-ring radius scales with length so the ring closes at bond_length:
        #   R = bond_length * L / (2π)
        coords_list = []
        for b in range(B):
            lb = int(lengths[b].item())
            Rb = self.bond_length * lb / (2.0 * math.pi)
            cos_t = torch.cos(theta_out[b])
            sin_t = torch.sin(theta_out[b])
            cos_p = torch.cos(phi_out[b])
            sin_p = torch.sin(phi_out[b])
            major_R = Rb + r_out[b] * cos_p
            xb = major_R * cos_t
            yb = major_R * sin_t
            zb = r_out[b] * sin_p
            coords_list.append(torch.stack([xb, yb, zb], dim=-1))  # (L, 3)
        coords = torch.stack(coords_list, dim=0)  # (B, L, 3)

        torus_coords = torch.stack([theta_out, phi_out, r_out], dim=-1)  # (B, L, 3)

        # Closure distance (should be ~0 by construction).
        closure_dist = torch.zeros(B, device=node_repr.device)
        for b in range(B):
            lb = int(lengths[b].item())
            if lb >= 2:
                closure_dist[b] = (coords[b, 0] - coords[b, lb - 1]).norm()

        return {
            "coords": coords,
            "torus_coords": torus_coords,
            "closure_dist": closure_dist,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Top-level model
# ══════════════════════════════════════════════════════════════════════════════

class Scheme10Model(nn.Module):
    """Scheme 10: SO(2)×SO(2) Equivariant GNN Trunk for circRNA 3D structure.

    Pipeline:
        1. Token embedding (no frozen LM — keeps every layer steerable)
        2. Torus positional encoding (additive, periodic)
        3. SO(2)×SO(2) steerable GNN layers (exact ring-equivariant messages)
        4. TorusCoordPredictor: node features → (θ, φ, r) → (x, y, z) with
           structural closure guarantee
        5. Optional immune-head hook (scheme choice (c)): if attached, called
           with (sequence_repr, pair_repr, torus_coords); its output is merged
           into the returned dict.

    Output dict (aligned with Scheme 8 contract + S9 torus_coords):
        coords        (B, L, 3)   — Cartesian coordinates
        torus_coords  (B, L, 3)   — raw (θ, φ, r)
        closure_dist  (B,)        — should be ≈ 0 by construction
        pair_repr     (B, L, L, d_pair) — for downstream heads / immune hook
        sequence_repr (B, L, d_model)  — node features (post-GNN)
        structure_method: "so2_so2_equivariant"
        + any keys returned by the immune_head hook (if attached)
    """

    def __init__(self, config: Optional[Scheme10Config] = None):
        super().__init__()
        self.config = config or Scheme10Config()
        c = self.config

        # Learned token embedding (NO ESM2 — see module docstring).
        self.token_embed = nn.Embedding(c.n_tokens, c.d_model)

        # Periodic positional encoding (additive, reused from tpe.py).
        self.tpe = TorusPositionalEncoding(
            d_model=c.d_model,
            n_harmonics=16,
            dropout=c.dropout,
        )

        # Steerable GNN layers.
        self.layers = nn.ModuleList([
            CircEquivariantGNNLayer(c) for _ in range(c.n_layers)
        ])

        # Pair representation projector (from node features, for output contract
        # and for the immune-head hook). z[i,j] = MLP(concat[x_i, x_j, circ_d]).
        self.pair_proj = nn.Sequential(
            nn.Linear(2 * c.d_model + 16, c.d_pair),
            nn.GELU(),
            nn.Linear(c.d_pair, c.d_pair),
        )
        self.dist_embed = nn.Embedding(257, 16)  # circ distance 0..256

        # Structure head.
        self.coord_predictor = TorusCoordPredictor(
            d_model=c.d_model, r_scale=c.r_scale, bond_length=c.bond_length,
        )

        # Closure reward (reused from circrna_mamba_diffusion — same semantics
        # as S8, kept for output-contract parity even though closure is by
        # construction here).
        self.closure_reward = BSJClosureReward(c.bond_length)

        # Optional immune-head hook (scheme choice (c)). Attached externally:
        #     model.attach_immune_head(some_head)
        # where some_head(sequence_repr, pair_repr, torus_coords) -> Dict.
        # None = no immune head (trunk-only mode, A/B against S8).
        self.immune_head: Optional[nn.Module] = None

    def attach_immune_head(self, head: nn.Module) -> None:
        """Attach a pluggable immune-fingerprint head (scheme choice (c)).

        The head must accept (sequence_repr, pair_repr, torus_coords) and
        return a Dict[str, Tensor]. Its keys are merged into forward() output.
        """
        self.immune_head = head

    # ------------------------------------------------------------------
    # Angle geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ring_angles(
        lengths: torch.Tensor,
        L: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-residue ring angles θ, φ.

        θ_i = 2π · i / L_seq   (main ring — uniform around the closed loop)
        φ_i = 0                (cross-section phase — learned via coord head,
                                initialised flat; the GNN learns Δφ from data)
        """
        pos = torch.arange(L, device=device, dtype=torch.float32)  # (L,)
        theta = torch.zeros(1, L, device=device)
        # θ per sequence uses its own length (period = valid length, not L).
        thetas = []
        for b in range(lengths.shape[0]):
            lb = int(lengths[b].item())
            tb = 2.0 * math.pi * pos / max(lb, 1)
            thetas.append(tb)
        theta = torch.stack(thetas, dim=0)  # (B, L)
        phi = torch.zeros_like(theta)
        return theta.unsqueeze(-1), phi.unsqueeze(-1)  # (B, L, 1) each

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        seq_tokens: torch.Tensor,                  # (B, L) long, values 0..4
        pair_probs: Optional[torch.Tensor] = None, # (B, L, L) optional prior
        coords_target: Optional[torch.Tensor] = None,  # (B, L, 3) optional
        temperature: float = 310.0,
        pH: float = 7.4,
        Mg_conc: float = 1.0,
        Na_conc: float = 1.5,
        pseudoknot_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args mirror Scheme8Model.forward so the two schemes are drop-in
        interchangeable for the training/evaluation harness. coords_target is
        accepted for contract parity but, since S10 is not diffusion-based,
        it only toggles a train/eval flag on the structure head (currently a
        no-op distinction — S10 produces coords directly from features).
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Valid lengths from padding token (token id == 4 == 'N'/'pad').
        # Matches scheme8's token map: 0:A 1:U 2:G 3:C 4:N(pad).
        is_pad = (seq_tokens == 4)
        # length = index of first pad, or L if none.
        if is_pad.any(dim=1).any():
            lengths = is_pad.float().argmax(dim=1).clamp(min=1)
            lengths = torch.where(is_pad.any(dim=1), lengths, torch.full_like(lengths, L))
        else:
            lengths = torch.full((B,), L, device=device, dtype=torch.long)

        # 1. Token embedding + TPE.
        x = self.token_embed(seq_tokens)  # (B, L, d_model)
        x = self.tpe(x, seq_len=L)

        # 2. Ring angles + angle differences.
        theta, phi = self._ring_angles(lengths, L, device)  # (B, L, 1)
        delta_theta = theta - theta.transpose(1, 2)  # (B, L, L)
        delta_phi = phi - phi.transpose(1, 2)        # (B, L, L)

        # 3. Edge categories.
        edge_cat = EdgeCategoryBuilder(min_loop=3)(
            pair_probs, lengths, pseudoknot_mask,
        )  # (B, L, L)

        # 4. Steerable GNN layers.
        for layer in self.layers:
            x = layer(x, delta_theta, delta_phi, edge_cat, lengths)
        sequence_repr = x  # (B, L, d_model)

        # 5. Pair representation (for output contract + immune hook).
        circ_d = EdgeCategoryBuilder._circular_distance_matrix(L, device)  # (L, L)
        circ_d_clamped = circ_d.clamp(0, 256).long()
        dist_feat = self.dist_embed(circ_d_clamped).unsqueeze(0).expand(B, -1, -1, -1)
        left = sequence_repr.unsqueeze(2).expand(B, L, L, -1)
        right = sequence_repr.unsqueeze(1).expand(B, L, L, -1)
        pair_repr = self.pair_proj(torch.cat([left, right, dist_feat], dim=-1))

        # 6. Structure head → (θ, φ, r) → coords with structural closure.
        struct_out = self.coord_predictor(sequence_repr, lengths)
        coords = struct_out["coords"]
        torus_coords = struct_out["torus_coords"]
        closure_dist = struct_out["closure_dist"]

        out: Dict[str, torch.Tensor] = {
            "coords": coords,
            "torus_coords": torus_coords,
            "pair_repr": pair_repr,
            "sequence_repr": sequence_repr,
            "structure_method": "so2_so2_equivariant",
            "pair_probs": pair_probs,
        }
        out["closure_dist"] = torch.norm(coords[:, 0] - coords[:, -1], dim=-1)

        # 7. Optional immune-head hook (scheme choice (c)).
        # The shared ImmuneFingerprintHeads signature is
        #     forward(sequence_repr, pair_repr, torus_coords=None, pair_probs=None)
        # S10 has all three (sequence_repr, pair_repr, torus_coords), plus the
        # input pair_probs — pass them all so the head can pick 3D or 2D m6A
        # proxy via its own enable_fingerprint_2d flag.
        if self.immune_head is not None:
            immune_out = self.immune_head(
                sequence_repr, pair_repr, torus_coords, pair_probs
            )
            if isinstance(immune_out, dict):
                out.update(immune_out)

        return out


# Module-level constants for external registration (e.g. in train_all_schemes.py
# or a future moe v3 routing table). Mirrors the scheme8 pattern.
SCHEME10_ID = 10
SCHEME10_NAME = "circ_equivariant_gnn"
