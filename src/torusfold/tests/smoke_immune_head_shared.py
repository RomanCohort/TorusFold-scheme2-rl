"""Smoke test for shared ImmuneFingerprintHeads (Phase 1 of S10 equivariance port).

Verifies:
    1. The shared head module imports from its new location.
    2. It can be attached to Scheme10Model via attach_immune_head().
    3. forward() runs end-to-end and the 5 immune-fingerprint keys appear.
    4. Both 3D mode (torus_coords provided) and 2D mode (torus_coords=None)
       work — the latter is the path S1-S8 will use.

Run from repo root:
    python -m torusfold.tests.smoke_immune_head_shared
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.scheme10_circ_equivariant_gnn import (
    Scheme10Config,
    Scheme10Model,
)
from torusfold.immune_fingerprint_head import (
    ImmuneFingerprintHeads,
)
from torusfold.tests.gradcheck import (
    check_immune_head_gradients,
)


def main() -> int:
    print("[smoke-ih] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[smoke-ih] device:", device)

    # --- 1. trunk ---
    config = Scheme10Config(d_model=64, n_layers=2)
    model = Scheme10Model(config).to(device)

    # --- 2. shared immune head ---
    head = ImmuneFingerprintHeads(
        d_model=config.d_model,
        c_z=config.d_pair,
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=False,  # 3D mode (S10 has torus_coords)
    ).to(device)
    model.attach_immune_head(head)
    print("[smoke-ih] attached shared ImmuneFingerprintHeads to S10")

    # --- 3. dummy input ---
    B, L = 2, 32
    seq_tokens = torch.randint(0, 4, (B, L), device=device)
    pair_probs = torch.full((B, L, L), 0.3, device=device)
    pair_probs = pair_probs * (~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0))
    pair_probs = 0.5 * (pair_probs + pair_probs.transpose(1, 2))

    # --- 4. forward ---
    model.eval()
    with torch.no_grad():
        out = model(seq_tokens, pair_probs=pair_probs)

    expected_immune_keys = {
        "pkr_stem_logit", "pkr_sasa",
        "nlrp3_persistence_length",
        "drach_is_drach", "drach_in_loop", "m6a_write_prob",
        "tlr7_gu_density",
        "sponge_score",
    }
    present = expected_immune_keys & set(out.keys())
    missing = expected_immune_keys - set(out.keys())
    print("[smoke-ih] immune keys present:", sorted(present))
    assert not missing, f"missing immune keys: {missing}"

    # Shape checks (per-residue vs scalar).
    assert out["pkr_sasa"].shape == (B, L), out["pkr_sasa"].shape
    assert out["m6a_write_prob"].shape == (B, L), out["m6a_write_prob"].shape
    assert out["tlr7_gu_density"].shape == (B, L), out["tlr7_gu_density"].shape
    assert out["nlrp3_persistence_length"].shape == (B,), out["nlrp3_persistence_length"].shape
    assert out["sponge_score"].shape == (B,), out["sponge_score"].shape
    print("[smoke-ih] shapes OK (3D mode)")

    # Ranges: sigmoid outputs in [0,1].
    assert (out["pkr_sasa"] >= 0).all() and (out["pkr_sasa"] <= 1).all()
    assert (out["m6a_write_prob"] >= 0).all() and (out["m6a_write_prob"] <= 1).all()
    assert (out["sponge_score"] >= 0).all() and (out["sponge_score"] <= 1).all()
    print("[smoke-ih] ranges OK (sigmoid heads in [0,1])")

    # --- 5. 2D mode (torus_coords=None path) ---
    head_2d = ImmuneFingerprintHeads(
        d_model=config.d_model,
        c_z=config.d_pair,
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=True,
    ).to(device)
    # Standalone call without torus_coords (simulates S1-S8 path).
    with torch.no_grad():
        seq_repr = out["sequence_repr"]
        pair_repr = out["pair_repr"]
        out_2d = head_2d(seq_repr, pair_repr, torus_coords=None, pair_probs=pair_probs)
    assert "m6a_write_prob" in out_2d
    assert out_2d["m6a_write_prob"].shape == (B, L)
    # In 2D mode, m6a should not be all-zero (exposure proxy = 1 - pair_probs).
    assert out_2d["m6a_write_prob"].abs().sum() > 0, "m6a all zero in 2D mode"
    print("[smoke-ih] 2D mode OK (torus_coords=None, m6a not all-zero)")

    # --- 6. backward (trainability of attached head) ---
    model.train()
    out3 = model(seq_tokens, pair_probs=pair_probs)
    # Loss touches every head so all parameters receive grad.
    loss = (
        out3["pkr_stem_logit"].pow(2).mean()
        + out3["pkr_sasa"].pow(2).mean()
        + out3["m6a_write_prob"].pow(2).mean()
        + out3["drach_is_drach"].pow(2).mean()
        + out3["drach_in_loop"].pow(2).mean()
        + out3["sponge_score"].pow(2).mean()
        + out3["nlrp3_persistence_length"].pow(2).mean()
        + out3["tlr7_gu_density"].pow(2).mean()
    )
    loss.backward()
    head_params = [p for p in head.parameters() if p.requires_grad]
    grad_ok, stats = check_immune_head_gradients(head, loss, tag="smoke-ih")
    print(f"[smoke-ih] backward loss={loss.item():.4f}, head grads finite: {grad_ok}")
    assert grad_ok, "non-finite/missing grads in shared immune head"

    print("[smoke-ih] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
