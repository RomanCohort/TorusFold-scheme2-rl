"""Smoke test: Scheme 4 (DDPM+EGNN guided diffusion) TrunkOutput compliance.

Verifies:
    1. S4 _sample returns ALL TRUNK_REQUIRED_KEYS, passes validate_trunk_output.
    2. pair_repr_source == "synthetic".
    3. Shared ImmuneFingerprintHeads attaches; 8 immune keys present.
    4. Backward through immune head is finite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.circrna_diffusion import (
    CircRNADiffusionModel,
    CircDiffusionConfig,
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
    print("[smoke-s4] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[smoke-s4] device:", device)

    # Small config for quick CPU.
    config = CircDiffusionConfig(
        d_node=64, d_edge=32, d_cond=32,
        n_egnn_layers=2,
        n_diffusion_steps=4,
    )
    model = CircRNADiffusionModel(config).to(device)

    B, L = 2, 16
    seq_tokens = torch.randint(0, 4, (B, L), device=device)
    pair_probs = torch.full((B, L, L), 0.3, device=device)
    pair_probs = pair_probs * (~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0))
    pair_probs = 0.5 * (pair_probs + pair_probs.transpose(1, 2))

    # --- 1. protocol compliance without immune head ---
    model.eval()
    with torch.no_grad():
        out = model(seq_tokens, pair_probs=pair_probs)

    missing = TRUNK_REQUIRED_KEYS - set(out.keys())
    assert not missing, f"S4 missing TrunkOutput keys: {missing}"
    print("[smoke-s4] TrunkOutput keys present:", sorted(TRUNK_REQUIRED_KEYS & set(out.keys())))

    validate_trunk_output(out)
    print("[smoke-s4] validate_trunk_output() PASSED")

    assert out.get("pair_repr_source") == "synthetic", out.get("pair_repr_source")
    assert out["structure_method"] == "circrna_ddpm_egnn_guided", out["structure_method"]
    print(f"[smoke-s4] pair_repr_source = {out['pair_repr_source']}")
    print(f"[smoke-s4] structure_method = {out['structure_method']}")

    assert out["coords"].shape == (B, L, 3)
    assert out["sequence_repr"].shape[:2] == (B, L)
    assert out["pair_repr"].shape[:3] == (B, L, L)
    print("[smoke-s4] shapes OK")

    # --- 2. attach shared immune head ---
    # S4 outputs Cartesian coords → 3D Cartesian kNN-density exposure proxy.
    head = ImmuneFingerprintHeads(
        d_model=config.d_node,
        c_z=config.d_node,  # S4 trunk_adapter uses d_node as c_z
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=False,  # let Cartesian path activate
    ).to(device)
    model.attach_immune_head(head)
    print("[smoke-s4] attached shared ImmuneFingerprintHeads (Cartesian path)")

    with torch.no_grad():
        out2 = model(seq_tokens, pair_probs=pair_probs)
    assert "immune" in out2, "immune dict missing"
    immune = out2["immune"]

    expected = {
        "pkr_stem_logit", "pkr_sasa",
        "nlrp3_persistence_length",
        "drach_is_drach", "drach_in_loop", "m6a_write_prob",
        "tlr7_gu_density",
        "sponge_score",
    }
    missing_imm = expected - set(immune.keys())
    assert not missing_imm, f"missing immune keys: {missing_imm}"
    print("[smoke-s4] immune keys present:", sorted(expected & set(immune.keys())))
    assert immune["m6a_write_prob"].abs().sum() > 0, "m6a all-zero (Cartesian path broken)"
    print(f"[smoke-s4] m6a_write_prob stats: min={immune['m6a_write_prob'].min():.3f} max={immune['m6a_write_prob'].max():.3f}")
    print("[smoke-s4] 3D Cartesian m6A proxy non-zero OK")

    # --- 3. backward ---
    model.train()
    out3 = model(seq_tokens, pair_probs=pair_probs)
    immune3 = out3["immune"]
    loss = sum(v.pow(2).mean() for v in immune3.values())
    loss.backward()

    head_params = [p for p in head.parameters() if p.requires_grad]
    grad_ok, stats = check_immune_head_gradients(head, loss, tag="smoke-s4")
    print(f"[smoke-s4] backward loss={loss.item():.4f}, head grads finite: {grad_ok}")
    assert grad_ok, "non-finite/missing grads in immune head on S4"

    print("[smoke-s4] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
