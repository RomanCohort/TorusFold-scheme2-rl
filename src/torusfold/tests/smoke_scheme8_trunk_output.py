"""Smoke test: Scheme8 (sparse-pair diffusion) TrunkOutput protocol compliance.

Verifies:
    1. S8 forward (sample path) returns ALL TRUNK_REQUIRED_KEYS.
    2. validate_trunk_output() passes on the returned dict.
    3. pair_repr_source == "synthetic" (S8 uses PairPriorSynthesizer).
    4. Attaching the shared ImmuneFingerprintHeads works and the 8 immune
       output keys appear in the returned dict.
    5. Backward through immune head is finite (shared head trainable on S8).

Run from repo root:
    python -m torusfold.tests.smoke_scheme8_trunk_output
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.scheme8_sparse_pair import (
    Scheme8Config,
    Scheme8Model,
)
from torusfold.trunk_output import (
    TRUNK_REQUIRED_KEYS,
    validate_trunk_output,
)
from torusfold.immune_fingerprint_head import (
    ImmuneFingerprintHeads,
)
from torusfold.tests.gradcheck import (
    check_immune_head_gradients,
)


def main() -> int:
    print("[smoke-s8] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[smoke-s8] device:", device)

    # Tiny config — n_diffusion_steps small so CPU runs in seconds.
    config = Scheme8Config(
        d_model=64, d_pair=64,
        n_diffusion_steps=4,  # minimal for smoke
        n_recycle=0,          # use single-round _sample path
    )
    model = Scheme8Model(config).to(device)

    # --- 1. without immune head: protocol compliance ---
    B, L = 2, 16
    seq_tokens = torch.randint(0, 4, (B, L), device=device)
    pair_probs = torch.full((B, L, L), 0.3, device=device)
    pair_probs = pair_probs * (~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0))
    pair_probs = 0.5 * (pair_probs + pair_probs.transpose(1, 2))

    model.eval()
    with torch.no_grad():
        out = model(seq_tokens, pair_probs=pair_probs)

    missing = TRUNK_REQUIRED_KEYS - set(out.keys())
    assert not missing, f"S8 missing TrunkOutput keys: {missing}"
    print("[smoke-s8] TrunkOutput keys present:", sorted(TRUNK_REQUIRED_KEYS & set(out.keys())))

    validate_trunk_output(out)
    print("[smoke-s8] validate_trunk_output() PASSED")

    assert out.get("pair_repr_source") == "synthetic", out.get("pair_repr_source")
    assert out["structure_method"] == "scheme8_sparse_pair", out["structure_method"]
    print(f"[smoke-s8] pair_repr_source = {out['pair_repr_source']}  (honest synthetic label)")
    print(f"[smoke-s8] structure_method = {out['structure_method']}")

    # Shape sanity.
    assert out["coords"].shape == (B, L, 3), out["coords"].shape
    assert out["sequence_repr"].shape[:2] == (B, L)
    assert out["pair_repr"].shape[:3] == (B, L, L)
    assert out["pair_probs"].shape == (B, L, L)
    assert out["closure_dist"].shape == (B,)
    print("[smoke-s8] shapes OK")

    # --- 2. with shared immune head attached ---
    # S8's diffusion loop outputs Cartesian coords → 3D Cartesian kNN-density
    # exposure proxy (path 2).
    head = ImmuneFingerprintHeads(
        d_model=config.d_model,
        c_z=config.d_pair,
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=False,  # let Cartesian path activate
    ).to(device)
    model.attach_immune_head(head)
    print("[smoke-s8] attached shared ImmuneFingerprintHeads")

    with torch.no_grad():
        out2 = model(seq_tokens, pair_probs=pair_probs)
    assert "immune" in out2, "immune dict missing after attach"
    immune = out2["immune"]

    expected_immune = {
        "pkr_stem_logit", "pkr_sasa",
        "nlrp3_persistence_length",
        "drach_is_drach", "drach_in_loop", "m6a_write_prob",
        "tlr7_gu_density",
        "sponge_score",
    }
    present = expected_immune & set(immune.keys())
    missing_imm = expected_immune - set(immune.keys())
    assert not missing_imm, f"missing immune keys: {missing_imm}"
    print("[smoke-s8] immune keys present:", sorted(present))

    # 3D Cartesian m6A proxy should be non-zero (kNN density varies)
    assert immune["m6a_write_prob"].abs().sum() > 0, "m6a all-zero (Cartesian path broken)"
    print(f"[smoke-s8] m6a_write_prob stats: min={immune['m6a_write_prob'].min():.3f} max={immune['m6a_write_prob'].max():.3f}")
    print("[smoke-s8] 3D Cartesian m6A proxy non-zero OK")

    # --- 3. backward through immune head on S8 trunk ---
    model.train()
    out3 = model(seq_tokens, pair_probs=pair_probs)
    immune3 = out3["immune"]
    loss = sum(v.pow(2).mean() for v in immune3.values())
    loss.backward()

    head_params = [p for p in head.parameters() if p.requires_grad]
    grad_ok, stats = check_immune_head_gradients(head, loss, tag="smoke-s8")
    print(f"[smoke-s8] backward loss={loss.item():.4f}, head grads finite: {grad_ok}")
    assert grad_ok, "non-finite/missing grads in immune head on S8"

    # --- 4. recycle path (iterative_refinement) ---
    config_rec = Scheme8Config(
        d_model=64, d_pair=64,
        n_diffusion_steps=4,
        n_recycle=1,  # force recycle path
    )
    model_rec = Scheme8Model(config_rec).to(device)
    head_rec = ImmuneFingerprintHeads(
        d_model=config_rec.d_model, c_z=config_rec.d_pair,
        d_torus=3, hidden_dim=128, enable_fingerprint_2d=False,
    ).to(device)
    model_rec.attach_immune_head(head_rec)
    model_rec.eval()
    with torch.no_grad():
        out_rec = model_rec(seq_tokens, pair_probs=pair_probs)
    validate_trunk_output(out_rec)
    assert out_rec["structure_method"] == "scheme8_sparse_pair_recycle", out_rec["structure_method"]
    assert "immune" in out_rec, "recycle path dropped immune dict"
    assert "candidate_mask" in out_rec, "recycle path dropped candidate_mask"
    print(f"[smoke-s8] recycle path OK, structure_method={out_rec['structure_method']}")

    print("[smoke-s8] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
