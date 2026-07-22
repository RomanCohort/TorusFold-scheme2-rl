"""Smoke test: Scheme 6 (GNN latent diffusion) TrunkOutput compliance.

Verifies:
    1. S6 forward returns ALL TRUNK_REQUIRED_KEYS, passes validate_trunk_output.
    2. pair_repr_source == "synthetic".
    3. Shared ImmuneFingerprintHeads attaches; 8 immune keys present.
    4. Backward through immune head is finite.

Note: S6 operates in latent space with no explicit pair_probs; the
synthesizer uses a uniform fallback (pair_probs=None). The immune head's
m6A path uses the uniform exposure proxy (0.5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.gnn_latent_diffusion import (
    GNNLatentDiffusionModel,
    GNNLatentConfig,
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
    print("[smoke-s6] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[smoke-s6] device:", device)

    config = GNNLatentConfig()
    # small dims for CPU smoke
    config.d_latent = 64
    model = GNNLatentDiffusionModel(config).to(device)

    B, L = 2, 16
    seq_tokens = torch.randint(0, 4, (B, L), device=device)

    # --- 1. protocol compliance without immune head ---
    model.eval()
    with torch.no_grad():
        out = model(seq_tokens, mode='sample')

    missing = TRUNK_REQUIRED_KEYS - set(out.keys())
    assert not missing, f"S6 missing TrunkOutput keys: {missing}"
    print("[smoke-s6] TrunkOutput keys present:", sorted(TRUNK_REQUIRED_KEYS & set(out.keys())))

    validate_trunk_output(out)
    print("[smoke-s6] validate_trunk_output() PASSED")

    assert out.get("pair_repr_source") == "synthetic", out.get("pair_repr_source")
    assert out["structure_method"] == "gnn_latent_diffusion", out["structure_method"]
    print(f"[smoke-s6] pair_repr_source = {out['pair_repr_source']}")
    print(f"[smoke-s6] structure_method = {out['structure_method']}")

    assert out["coords"].shape == (B, L, 3)
    assert out["sequence_repr"].shape[:2] == (B, L)
    assert out["pair_repr"].shape[:3] == (B, L, L)
    print("[smoke-s6] shapes OK")

    # --- 2. attach shared immune head ---
    # S6's decoder outputs Cartesian coords → 3D Cartesian kNN-density
    # exposure proxy (path 2). enable_fingerprint_2d=False to let it activate.
    head = ImmuneFingerprintHeads(
        d_model=config.d_latent,
        c_z=config.d_latent,
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=False,
    ).to(device)
    model.attach_immune_head(head)
    print("[smoke-s6] attached shared ImmuneFingerprintHeads (Cartesian path)")

    with torch.no_grad():
        out2 = model(seq_tokens, mode='sample')
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
    print("[smoke-s6] immune keys present:", sorted(expected & set(immune.keys())))
    # 3D Cartesian m6A proxy should be non-zero (kNN density varies)
    assert immune["m6a_write_prob"].abs().sum() > 0, "m6a all-zero (Cartesian path broken)"
    print(f"[smoke-s6] m6a_write_prob stats: min={immune['m6a_write_prob'].min():.3f} max={immune['m6a_write_prob'].max():.3f}")
    print("[smoke-s6] 3D Cartesian m6A proxy non-zero OK")

    # --- 3. backward ---
    model.train()
    out3 = model(seq_tokens, mode='sample')
    immune3 = out3["immune"]
    loss = sum(v.pow(2).mean() for v in immune3.values())
    loss.backward()

    head_params = [p for p in head.parameters() if p.requires_grad]
    grad_ok, stats = check_immune_head_gradients(head, loss, tag="smoke-s6")
    print(f"[smoke-s6] backward loss={loss.item():.4f}, head grads finite: {grad_ok}")
    assert grad_ok, "non-finite/missing grads in immune head on S6"

    print("[smoke-s6] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
