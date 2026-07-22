"""trunk_output.py — Shared TrunkOutput protocol + pair-pair synthesis module.

This module defines the UNIFIED output contract that every TorusFold scheme
trunk (S1–S10) must satisfy, plus a light-weight PairPriorSynthesizer used
by schemes that do not internally produce a dense (B, L, L, c_z) pair
representation.

Why a protocol:
    S1–S8 trunks historically returned scheme-specific dicts without the
    `sequence_repr` / `pair_repr` / `pair_probs` fields that downstream
    tasks (immune-fingerprint head, structure analysis) need. The shared
    ImmuneFingerprintHeads module was previously only attachable to S9
    because only S9 natively produced these tensors. This module makes
    every scheme compliant, closing the design debt.

The contract (keys every trunk's forward() MUST return):
    coords:             (B, L, 3)   — predicted Cartesian coordinates
    sequence_repr:      (B, L, d)   — per-residue hidden representation
    pair_repr:          (B, L, L, c)— dense pair representation
    pair_probs:         (B, L, L)   — base-pair probability matrix
    closure_dist:       (B,)        — BSJ closure distance (coords[0]-coords[-1])
    structure_method:   str         — scheme identifier ("scheme8", "torus", ...)
    closure_loss:       (B,)        — same as closure_dist, detached for diagnostics

Optional keys (scheme-specific):
    torus_coords:       (B, L, 3)   — (θ, φ, r) for S9 / S10 with torus trunk
    pair_repr_source:   str         — "synthetic" or "trunk_refined"

The synthesizer is always used for schemes that lack a trunk-refined pair
representation. It produces a *learnable-but-light* dense pair tensor from
token identity + pair_probs (when available). It is honest about its nature:
the output is labeled `pair_repr_source: "synthetic"` so downstream modules
(e.g., papers, ablations) can distinguish it from a trunk-refined pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Required keys for TrunkOutput protocol.
TRUNK_REQUIRED_KEYS = frozenset([
    "coords",
    "sequence_repr",
    "pair_repr",
    "pair_probs",
    "closure_dist",
    "structure_method",
])


def validate_trunk_output(out: Dict[str, torch.Tensor]) -> None:
    """Assert that a trunk's forward() output satisfies the TrunkOutput protocol.

    Raises AssertionError with the missing keys (or None if a required tensor).
    """
    missing = TRUNK_REQUIRED_KEYS - set(out.keys())
    if missing:
        raise AssertionError(f"TrunkOutput missing keys: {sorted(missing)}")
    B, L = out["coords"].shape[:2]
    assert out["sequence_repr"].shape[:2] == (B, L), \
        f"sequence_repr shape {out['sequence_repr'].shape} vs coords {(B, L)}"
    assert out["pair_repr"].shape[:3] == (B, L, L), \
        f"pair_repr shape {out['pair_repr'].shape} vs coords {(B, L, L)}"
    assert out["pair_probs"].shape == (B, L, L), \
        f"pair_probs shape {out['pair_probs'].shape} vs coords {(B, L, L)}"
    assert out["closure_dist"].shape == (B,), \
        f"closure_dist shape {out['closure_dist'].shape} vs (B,)"


@dataclass
class TrunkOutputSpec:
    """Documentation-only spec of the trunk output contract.

    Use validate_trunk_output() at runtime; this class is for readers and
    type checkers that want a concrete reference.
    """
    coords: torch.Tensor                 # (B, L, 3)
    sequence_repr: torch.Tensor          # (B, L, d_model)
    pair_repr: torch.Tensor              # (B, L, L, c_z)
    pair_probs: torch.Tensor             # (B, L, L) in [0, 1]
    closure_dist: torch.Tensor           # (B,)
    structure_method: str                # scheme identifier
    closure_loss: Optional[torch.Tensor] = None  # detached diagnostic
    torus_coords: Optional[torch.Tensor] = None  # (B, L, 3) for S9/S10
    pair_repr_source: str = "synthetic"  # or "trunk_refined"


class PairPriorSynthesizer(nn.Module):
    """Lightweight (B, L, L, c_z) pair representation from tokens + pair_probs.

    Used by schemes that do not internally produce a dense pair representation
    (S1–S8, excluding S10 which has a trunk-refined pair). It combines three
    honest, cheap, learnable signals into a dense pair tensor:

        1. Token outer product: tok_emb[i] @ tok_emb[j]^T → (B,L,L,d_feat)
           Encodes "which bases are at (i,j)".
        2. pair_probs: the input (B,L,L) pair-probability matrix, projected
           through a small MLP → (B,L,L,d_pair). If pair_probs is None, uses
           a learned uniform prior (0.5) as fallback.
        3. Circular distance embedding: ring-aware distance |i-j|_circ ∈ [0,L/2]
           embedded through nn.Embedding. Encodes the backbone proximity prior
           that is universal to circRNA.

    These are concatenated and projected to (B, L, L, c_z) by a final MLP.
    Total parameters: ~100k for c_z=64, d_model=128. Small enough to add to
    any scheme without affecting memory budget meaningfully.

    The output is labeled as `pair_repr_source: "synthetic"` in the trunk
    output — downstream consumers should know this is a learned prior, not
    a trunk-refined pair representation (like S9/S10 produce).
    """

    def __init__(
        self,
        d_token: int,       # per-residue token dim (e.g. d_model of trunk)
        d_feat: int = 32,   # token outer product feature dim
        d_pair: int = 32,   # pair_probs feature dim (from MLP)
        d_circ: int = 16,   # circular distance embedding dim
        c_z: int = 64,      # output pair repr dim
        max_circ_dist: int = 256,
    ):
        super().__init__()
        self.d_token = d_token
        self.d_feat = d_feat
        self.c_z = c_z
        self.max_circ_dist = max_circ_dist

        # Token projection for outer product.
        self.tok_proj = nn.Linear(d_token, d_feat, bias=False)

        # pair_probs MLP: scalar → d_pair.
        self.pp_mlp = nn.Sequential(
            nn.Linear(1, d_pair),
            nn.GELU(),
            nn.Linear(d_pair, d_pair),
        )

        # Circular distance embedding.
        self.circ_embed = nn.Embedding(max_circ_dist + 1, d_circ)

        # Fusion: token_outer + pair_feat + circ_dist → c_z.
        self.fuse = nn.Sequential(
            nn.Linear(d_feat + d_pair + d_circ, c_z),
            nn.GELU(),
            nn.Linear(c_z, c_z),
        )

    @staticmethod
    def circular_distance_matrix(L: int, device: torch.device) -> torch.Tensor:
        """Min distance along the ring between positions i, j. (L, L) long."""
        idx = torch.arange(L, device=device)
        d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        return torch.min(d, L - d)

    def forward(
        self,
        seq_emb: torch.Tensor,                # (B, L, d_token)
        pair_probs: Optional[torch.Tensor] = None,  # (B, L, L) or None
    ) -> torch.Tensor:
        """Return (B, L, L, c_z) dense pair representation."""
        B, L, _ = seq_emb.shape
        device = seq_emb.device

        # 1. Token outer product: per-pair (i,j) → d_feat.
        tok_feat = self.tok_proj(seq_emb)  # (B, L, d_feat)
        tok_outer = tok_feat.unsqueeze(2) * tok_feat.unsqueeze(1)  # (B, L, L, d_feat)

        # 2. pair_probs MLP.
        if pair_probs is not None:
            pp_in = pair_probs.unsqueeze(-1)  # (B, L, L, 1)
            pair_feat = self.pp_mlp(pp_in)    # (B, L, L, d_pair)
        else:
            # No pair_probs: use a learned uniform prior (initialized at 0.5).
            uniform = torch.full((B, L, L, 1), 0.5, device=device)
            pair_feat = self.pp_mlp(uniform)  # (B, L, L, d_pair)

        # 3. Circular distance embedding.
        circ_d = self.circular_distance_matrix(L, device)
        circ_d_clamped = circ_d.clamp(0, self.max_circ_dist)
        circ_emb = self.circ_embed(circ_d_clamped)  # (L, L, d_circ)
        circ_emb = circ_emb.unsqueeze(0).expand(B, -1, -1, -1)

        # 4. Fuse.
        fused = torch.cat([tok_outer, pair_feat, circ_emb], dim=-1)
        return self.fuse(fused)  # (B, L, L, c_z)


class TrunkOutputAdapter(nn.Module):
    """Helper module that wraps any trunk to produce a TrunkOutput-compliant dict.

    Usage (inside a scheme's forward):
        adapter = TrunkOutputAdapter(d_model=..., c_z=...)
        ...
        trunk_output = {
            "coords": coords,
            "sequence_repr": h,
            "pair_probs": pair_probs,  # or fallback
            "structure_method": "scheme8",
            "closure_dist": closure_dist,
        }
        return adapter(trunk_output)

    The adapter:
        1. Fills missing fields with sensible fallbacks.
        2. Synthesizes pair_repr via PairPriorSynthesizer (if not provided).
        3. Calls validate_trunk_output() to ensure contract is met.
        4. Adds `pair_repr_source: "synthetic"` unless caller supplied one.
    """

    def __init__(
        self,
        d_model: int,
        c_z: int,
        structure_method: str,
    ):
        super().__init__()
        self.structure_method = structure_method
        self.synthesizer = PairPriorSynthesizer(
            d_token=d_model, c_z=c_z,
        )

    def forward(self, trunk_output: Dict) -> Dict[str, torch.Tensor]:
        out = dict(trunk_output)

        # Ensure structure_method is set (use adapter's default if missing).
        if "structure_method" not in out:
            out["structure_method"] = self.structure_method

        # closure_dist: compute from coords if missing.
        if "closure_dist" not in out:
            coords = out["coords"]
            out["closure_dist"] = (coords[:, 0] - coords[:, -1]).norm(dim=-1)

        # closure_loss: detached diagnostic (alias of closure_dist).
        if "closure_loss" not in out:
            out["closure_loss"] = out["closure_dist"].detach()

        # pair_probs: fallback to zero if trunk doesn't have it.
        B, L = out["coords"].shape[:2]
        if "pair_probs" not in out or out["pair_probs"] is None:
            device = out["coords"].device
            out["pair_probs"] = torch.zeros(B, L, L, device=device)

        # pair_repr: synthesize if trunk doesn't provide a trunk-refined one.
        if "pair_repr" in out and out["pair_repr"] is not None:
            # Trunk supplied a refined pair repr. Honor it; mark as such.
            out.setdefault("pair_repr_source", "trunk_refined")
        else:
            # Synthesize from sequence_repr + pair_probs.
            seq_repr = out["sequence_repr"]
            pair_probs = out["pair_probs"]
            out["pair_repr"] = self.synthesizer(seq_repr, pair_probs)
            out["pair_repr_source"] = "synthetic"

        # sequence_repr: fallback to zero if trunk doesn't have it (rare).
        if "sequence_repr" not in out or out["sequence_repr"] is None:
            device = out["coords"].device
            d_model = self.synthesizer.d_token
            out["sequence_repr"] = torch.zeros(B, L, d_model, device=device)

        # Validate the contract before returning.
        validate_trunk_output(out)

        return out
