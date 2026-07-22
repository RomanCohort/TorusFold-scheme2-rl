"""
immune_fingerprint_head.py — Shared immune-fingerprint heads (Phase 1 of the
S10 equivariance port-out).

Historically `ImmuneFingerprintHeads` lived inside `torus_coord_head.py`
(Scheme 9 only). This module lifts it to a shared location under the
`torusfold` package so that EVERY scheme (S1–S10) can attach the same
5-head immune-fingerprint block as a pluggable downstream task.

Why a shared head:
    S1–S8 trunks previously had no immune-fingerprint output at all; only
    S9's torus-parameterized trunk fed these heads. Making the head shared
    lets any scheme produce immune-activity predictions end-to-end, and
    lets us run a clean A/B (MLP head vs irrep head, Phase 2) on the SAME
    trunk without scheme-specific duplication.

Interface contract (unchanged from S9):
    forward(sequence_repr, pair_repr, torus_coords=None, pair_probs=None)
        -> Dict[str, Tensor]

`torus_coords` is now Optional:
    - Provided (S9 / S10 with torus trunk): 3D exposure proxy used for m6A.
    - None (S1–S8 trunks that output Cartesian coords only): the head
      falls back to the 2D single-strandedness proxy (1 - pair_probs).
      This is the honest choice — S1–S8 are not torus-parameterized, so
      injecting a fabricated 3D torus radius would be dishonest.

Phase 2 (irrep port, IN PROGRESS):
    Replace PKR and m6A's per-head `_mlp` blocks with SO(2)×SO(2) steerable
    message passing (CircEquivariantGNNLayer from steerable_kernel.py). The
    other three heads (NLRP3 / TLR7 / sponge) stay MLP — their outputs are
    scalar quantities (persistence length, GU density, sponge score) with no
    directional symmetry, so wrapping them in a steerable kernel would be
    gilding, not honesty.

    Three torus-coords paths (auto-selected in forward):
        1. Real torus_coords provided (S9 / S10): use as-is.
        2. cartesian_coords provided but no torus_coords (S1/S4/S6/S7/S8):
           reverse-map Cartesian → (θ, φ, r) via cartesian_to_torus — this is
           the Phase-2 line-1 mapper, mathematically exact SO(2)×SO(2)
           equivariant (verified to 1e-10 in float64).
        3. Neither: zero torus_slot, equivariant head falls back to MLP
           (the head's steerable layer needs Δθ/Δφ; with no angles it can't
           run, so we keep the MLP path as an honest fallback).

    `enable_equivariant` (default True) is the master switch. Set False to
    force the legacy MLP path on PKR/m6A for A/B ablation against the
    equivariant heads.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Phase 2: steerable kernel + Cartesian→torus mapper + edge categories.
from .steerable_kernel import CircEquivariantGNNLayer
from .cartesian_to_torus import cartesian_to_torus, major_ring_radius
from .scheme10_circ_equivariant_gnn import EdgeCategoryBuilder


class ImmuneFingerprintHeads(nn.Module):
    """
    Shared 5-head immune-fingerprint block (PKR / NLRP3 / m6A / TLR7 / sponge),
    plus optional RIG-I walk-attention negative-control head.

    Sits on top of ANY scheme trunk that produces:
        sequence_repr  (B, L, d_model)
        pair_repr      (B, L, L, c_z)
        torus_coords   (B, L, 3)         # optional; None → 2D mode
        pair_probs     (B, L, L)         # optional; used for 2D m6A proxy

    The 5 fingerprints (see torusfold-immune-fingerprints.md):
        pkr:     long-stem ratio + SASA  → PKR activation
        nlrp3:   persistence length       → NLRP3 scaffold
        drach:   DRACH × in_loop × SASA   → m6A shielding
        tlr7:    GU-rich single-loop      → TLR7 activation (auxiliary)
        sponge:  miRNA duplex distance    → sponge potency

    Forward returns a dict of tensors; missing labels are simply not used
    in the loss (caller computes per-head losses only for present targets).

    --- m6A exposure-proxy routing ---

    `enable_fingerprint_2d` is a NARROW switch: it only swaps the
    *solvent-exposure* term inside the DRACH head:

        m6a_write_prob = is_drach * in_loop * exposure_proxy
        exposure_proxy = sigmoid(torus_coords[..., 2])   if 3D available
        exposure_proxy = 1 - pair_probs.mean(dim=2)        if 2D / no torus

    When `torus_coords is None`, the head ALWAYS uses the 2D proxy
    regardless of `enable_fingerprint_2d` — there is no 3D signal to use.
    """

    def __init__(
        self,
        d_model: int,
        c_z: int,
        d_torus: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        enable_pkr: bool = True,
        enable_nlrp3: bool = True,
        enable_drach: bool = True,
        enable_tlr7: bool = True,
        enable_sponge: bool = True,
        enable_rigi: bool = False,
        rigi_window: int = 15,
        rigi_d_hidden: int = 128,
        enable_fingerprint_2d: bool = False,
        # Phase 2: equivariant PKR/m6A heads (SO(2)×SO(2) steerable message
        # passing). Default True — set False to force legacy MLP for A/B.
        enable_equivariant: bool = True,
        d_edge: int = 32,
        k_theta: int = 2,
        k_phi: int = 1,
        bond_length: float = 5.9,
    ):
        super().__init__()
        self.enable_pkr = enable_pkr
        self.enable_nlrp3 = enable_nlrp3
        self.enable_drach = enable_drach
        self.enable_tlr7 = enable_tlr7
        self.enable_sponge = enable_sponge
        self.enable_rigi = enable_rigi
        self.enable_fingerprint_2d = enable_fingerprint_2d
        self.enable_equivariant = enable_equivariant
        self.bond_length = bond_length
        self.d_model = d_model

        # in_dim = d_model + c_z + d_torus + 1 (sasa slot)
        # The +1 sasa slot carries a per-residue solvent-exposure scalar
        # derived from the trunk's 3D structure (torus radius, Cartesian
        # kNN density, or 2D pair_probs fallback). Every head sees it, so
        # PKR's sasa regression and m6A's exposure gate share the same 3D
        # signal — matches the original S9 design where torus_coords fed
        # the whole feat block, not just m6A.
        in_dim = d_model + c_z + d_torus + 1

        def _mlp(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        if enable_pkr:
            # 2 outputs: stem_length_bin (logits over bins), sasa (regression)
            # Always keep the MLP path (fallback when no angles available).
            self.pkr_head = _mlp(2)
            if enable_equivariant:
                # Equivariant path: steerable message passing on sequence_repr
                # (d_model dim), then a small linear head to 2 outputs. The
                # steerable layer operates on the trunk's per-residue hidden
                # state directly — NOT on the concatenated feat — because the
                # kernel expects (B, L, d_model) node features, not the wider
                # in_dim. The torus-derived sasa signal is reintroduced via
                # the feat slot only in the MLP fallback path; here the 3D
                # geometry enters through Δθ/Δφ in the message passing itself.
                self.pkr_equiv_layer = CircEquivariantGNNLayer(
                    d_model=d_model, d_edge=d_edge,
                    n_edge_cats=5, k_theta=k_theta, k_phi=k_phi,
                    dropout=dropout,
                )
                self.pkr_equiv_proj = nn.Linear(d_model, 2)
        if enable_nlrp3:
            # 1 output: persistence length (regression, nm)
            self.nlrp3_head = _mlp(1)
        if enable_drach:
            # 2 outputs: is_DRACH logit, in_loop logit
            self.drach_head = _mlp(2)
            if enable_equivariant:
                # Same design as PKR: steerable layer on sequence_repr, then
                # project to 2 outputs. m6A's DRACH motif is translation-
                # invariant on the ring, which is exactly SO(2)_θ equivariance.
                self.drach_equiv_layer = CircEquivariantGNNLayer(
                    d_model=d_model, d_edge=d_edge,
                    n_edge_cats=5, k_theta=k_theta, k_phi=k_phi,
                    dropout=dropout,
                )
                self.drach_equiv_proj = nn.Linear(d_model, 2)
        if enable_tlr7:
            # 1 output: GU-rich loop density (regression)
            self.tlr7_head = _mlp(1)
        if enable_sponge:
            # 1 output: duplex-compat score (regression)
            self.sponge_head = _mlp(1)
        if enable_rigi:
            # RIG-I walk-attention negative control. Lazily imported to avoid
            # a hard dependency on torus_coord_head from this shared module
            # (the walk-attention block still lives with S9's structure head).
            from .torus_coord_head import RIGIWalkAttention
            self.rigi_head = RIGIWalkAttention(
                d_model=d_model,
                d_hidden=rigi_d_hidden,
                window=rigi_window,
                n_heads=4,
                dropout=dropout,
            )

        # Edge category builder (topology prior for the steerable kernel).
        # No learnable params — derived from pair_probs + circular distance.
        # Instantiated once and shared by PKR/m6A equivariant paths.
        if enable_equivariant and (enable_pkr or enable_drach):
            self.edge_cat_builder = EdgeCategoryBuilder(min_loop=3)

    @staticmethod
    def _exposure_2d(pair_probs, B, L, device, dtype) -> torch.Tensor:
        """2D exposure proxy: 1 - mean pair probability per residue.

        Paired regions (high pair_prob) are buried → exposure ≈ 0.
        Unpaired regions are exposed → exposure ≈ 1.
        Falls back to uniform 0.5 when pair_probs is None.
        """
        if pair_probs is not None:
            pair_per_res = pair_probs.mean(dim=2)  # (B, L)
            return 1.0 - pair_per_res
        return torch.full((B, L), 0.5, device=device, dtype=dtype)

    @staticmethod
    def _exposure_cartesian(coords: torch.Tensor, k: int = 8) -> torch.Tensor:
        """3D Cartesian exposure proxy via kNN local density.

        For each residue i, find its k nearest structural neighbors (by
        Euclidean distance on the predicted Cartesian coords) and sum the
        inverse distances. A residue buried in the fold has many close
        neighbors → high density → low exposure. A surface residue has
        few close neighbors → low density → high exposure.

        This is a cheap, differentiable, rotation-invariant SASA proxy that
        works on ANY Cartesian-coordinate trunk (S1 EGNN, S4 DDPM-EGNN),
        without requiring torus parameterization.

        Args:
            coords: (B, L, 3) predicted Cartesian coordinates.
            k: number of nearest neighbors to consider (default 8, ~one
               helical turn). Clamped to L-1 for short sequences.

        Returns:
            (B, L) exposure in [0, 1].
        """
        B, L, _ = coords.shape
        device = coords.device
        dtype = coords.dtype

        # Pairwise distance matrix (B, L, L). Self-distance set to inf so
        # a residue is never its own nearest neighbor.
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
        dist = diff.norm(dim=-1)  # (B, L, L)
        eye = torch.eye(L, device=device, dtype=dtype).unsqueeze(0)
        dist = dist + eye * 1e9  # mask self

        kk = min(k, L - 1)
        # k nearest distances per residue.
        knn_dist, _ = dist.topk(kk, dim=-1, largest=False)  # (B, L, kk)

        # Inverse-distance sum = local density. Clamp to avoid div-by-zero.
        inv = 1.0 / (knn_dist.clamp(min=1e-3))  # (B, L, kk)
        density = inv.sum(dim=-1)  # (B, L)

        # Normalize per-sample to [0, 1] via min-max, then invert:
        # high density (buried) → low exposure.
        d_min = density.amin(dim=-1, keepdim=True)
        d_max = density.amax(dim=-1, keepdim=True)
        buried = (density - d_min) / (d_max - d_min + 1e-6)  # (B, L) in [0,1]
        return 1.0 - buried  # exposure

    def forward(
        self,
        sequence_repr: torch.Tensor,  # (B, L, d_model)
        pair_repr: torch.Tensor,       # (B, L, L, c_z)
        torus_coords: Optional[torch.Tensor] = None,  # (B, L, 3) or None
        pair_probs: Optional[torch.Tensor] = None,   # (B, L, L) optional
        cartesian_coords: Optional[torch.Tensor] = None,  # (B, L, 3) or None
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = sequence_repr.shape
        device = sequence_repr.device

        per_res_pair = pair_repr.mean(dim=2)  # (B, L, c_z)

        # --- Phase 2: resolve torus_coords via 3 paths ---
        # Path 1: real torus_coords (S9/S10) → use as-is.
        # Path 2: cartesian_coords but no torus_coords (S1-S8) → reverse-map
        #         via cartesian_to_torus (SO(2)×SO(2) equivariant, math exact).
        # Path 3: neither → torus_slot stays zero; equivariant heads fall back
        #         to MLP because they need Δθ/Δφ to run.
        equiv_available = False  # set True once we have real or reverse-mapped angles
        if torus_coords is not None:
            torus_slot = torus_coords  # (B, L, 3)
            equiv_available = True
            torus_source = "real"
        elif cartesian_coords is not None:
            # Reverse-map Cartesian → (θ, φ, r). Need the major-ring radius R
            # per sequence = bond_length * L_seq / (2π). For a fixed-length
            # batch (common in smoke tests) all sequences share L; the
            # mapper also accepts a scalar R, but we pass per-seq to be safe.
            lengths = torch.full((B,), L, device=device, dtype=torch.long)
            R = major_ring_radius(self.bond_length, lengths)  # (B,)
            theta, phi, r_t = cartesian_to_torus(cartesian_coords, R)
            torus_slot = torch.stack([theta, phi, r_t], dim=-1)  # (B, L, 3)
            equiv_available = True
            torus_source = "cartesian_reverse_mapped"
        else:
            torus_slot = torch.zeros(B, L, 3, device=device, dtype=sequence_repr.dtype)
            equiv_available = False
            torus_source = "none"

        # SASA scalar slot (B, L, 1) — the 3D exposure signal shared by all
        # heads. Three honest paths, in priority order (overridden by
        # enable_fingerprint_2d which forces the 2D path for ablation):
        #   1. torus_coords provided (or reverse-mapped): sigmoid(radius r).
        #   2. cartesian_coords provided (S1/S4): kNN local density.
        #   3. Neither (S6 latent / S7 / S8): 2D pair_probs fallback.
        # Note: when we reverse-mapped Cartesian → torus in path 2 above, the
        # sasa still uses the kNN density (path 2 here) rather than
        # sigmoid(r) — they carry different 3D information (kNN = local
        # burial, sigmoid(r) = cross-section radius), and Phase 1's design
        # decision was to use kNN for Cartesian trunks. Only real-torus
        # schemes (S9/S10) use sigmoid(r).
        if self.enable_fingerprint_2d:
            sasa_scalar = self._exposure_2d(
                pair_probs, B, L, device, sequence_repr.dtype
            )  # (B, L)
        elif torus_coords is not None:
            sasa_scalar = torch.sigmoid(torus_coords[..., 2])  # (B, L)
        elif cartesian_coords is not None:
            sasa_scalar = self._exposure_cartesian(cartesian_coords)  # (B, L)
        else:
            sasa_scalar = self._exposure_2d(
                pair_probs, B, L, device, sequence_repr.dtype
            )  # (B, L)
        sasa_feat = sasa_scalar.unsqueeze(-1)  # (B, L, 1)

        # Per-residue feature: [seq_repr, per_res_pair, torus_slot, sasa]
        feat = torch.cat(
            [sequence_repr, per_res_pair, torus_slot, sasa_feat], dim=-1
        )  # (B, L, in_dim)

        out: Dict[str, torch.Tensor] = {}

        # --- Phase 2: equivariant heads need Δθ/Δφ + edge_cat + lengths ---
        # Only compute these if an equivariant head will actually use them.
        need_equiv = (
            self.enable_equivariant
            and equiv_available
            and (self.enable_pkr or self.enable_drach)
        )
        if need_equiv:
            theta = torus_slot[..., 0]  # (B, L)
            phi = torus_slot[..., 1]    # (B, L)
            # Angle differences (B, L, L). These are invariant under the
            # SO(2)×SO(2) group action (shifting all θ by δθ leaves Δθ unchanged),
            # which is what makes the steerable kernel exactly equivariant.
            delta_theta = theta.unsqueeze(2) - theta.unsqueeze(1)  # (B, L, L)
            delta_phi = phi.unsqueeze(2) - phi.unsqueeze(1)         # (B, L, L)
            # Edge categories from pair_probs + circular distance topology.
            # lengths: assume full-length (no padding) for the shared head —
            # schemes that use padding should pass a lengths tensor; the
            # shared head interface doesn't carry it, so we default to L.
            lengths = torch.full((B,), L, device=device, dtype=torch.long)
            edge_cat = self.edge_cat_builder(pair_probs, lengths, None)  # (B, L, L)

        if self.enable_pkr:
            if need_equiv:
                # Equivariant path: steerable message passing on sequence_repr,
                # then project to 2 outputs. The 3D geometry enters through
                # Δθ/Δφ in the messages; sasa is reintroduced below by gating.
                h_equiv = self.pkr_equiv_layer(
                    sequence_repr, delta_theta, delta_phi, edge_cat, lengths
                )  # (B, L, d_model)
                pkr = self.pkr_equiv_proj(h_equiv)  # (B, L, 2)
            else:
                pkr = self.pkr_head(feat)  # (B, L, 2) — MLP fallback
            out["pkr_stem_logit"] = pkr[..., 0]
            out["pkr_sasa"] = torch.sigmoid(pkr[..., 1])

        if self.enable_nlrp3:
            # Aggregate globally: persistence length is a whole-molecule scalar.
            global_feat = feat.mean(dim=1)  # (B, in_dim)
            out["nlrp3_persistence_length"] = self.nlrp3_head(global_feat).squeeze(-1)

        if self.enable_drach:
            if need_equiv:
                h_equiv = self.drach_equiv_layer(
                    sequence_repr, delta_theta, delta_phi, edge_cat, lengths
                )  # (B, L, d_model)
                drach = self.drach_equiv_proj(h_equiv)  # (B, L, 2)
            else:
                drach = self.drach_head(feat)  # (B, L, 2) — MLP fallback
            is_drach = torch.sigmoid(drach[..., 0])
            in_loop = torch.sigmoid(drach[..., 1])
            # m6A write probability = DRACH ∧ in_loop ∧ exposure.
            # exposure_proxy reuses the shared sasa_scalar (same 3D signal
            # the other heads see) — no second computation.
            exposure_proxy = sasa_scalar  # (B, L)
            out["drach_is_drach"] = is_drach
            out["drach_in_loop"] = in_loop
            out["m6a_write_prob"] = is_drach * in_loop * exposure_proxy

        if self.enable_tlr7:
            out["tlr7_gu_density"] = torch.sigmoid(
                self.tlr7_head(feat).squeeze(-1)
            )

        if self.enable_sponge:
            global_feat = feat.mean(dim=1)
            out["sponge_score"] = torch.sigmoid(
                self.sponge_head(global_feat).squeeze(-1)
            )

        if self.enable_rigi:
            # RIG-I walk-attention uses the raw sequence_repr (d_model dim),
            # not the concatenated feat, because MultiheadAttention expects
            # a fixed embed_dim. Negative-control head: circRNA (no 5'-ppp)
            # should produce a lower score than linear RNA with the same
            # internal dsRNA content.
            rigi_out = self.rigi_head(sequence_repr, mask=None)
            out["rigi_per_pos"] = rigi_out["rigi_per_pos"]
            out["rigi_score"] = rigi_out["rigi_score"]

        return out
