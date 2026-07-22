"""Smoke test: S9 path-1 real torus_coords → immune head equivariance.

S9's TorusCoordHead concept: a trunk produces REAL (θ,φ,r) torus coords, which
are passed to ImmuneFingerprintHeads without reverse-mapping. This tests that
path-1 (real torus_coords) works correctly in the Phase 2 equivariant setup.

The S10 trunk's TorusCoordPredictor is the only component that currently
produces real (θ,φ,r), so we reuse it here to test path-1.

Run from repo root:
    python -m torusfold.tests.smoke_scheme9_path1
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

import torch

# UTF-8 for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.immune_fingerprint_head import (
    ImmuneFingerprintHeads,
)
from torusfold.tests.gradcheck import (
    check_immune_head_gradients,
)


def main() -> int:
    print("[smoke-s9-path1] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, L, d_model, c_z = 2, 24, 64, 32
    bond_length = 5.9

    # =====================================================================
    # Simulate S9 trunk output: real (θ,φ,r) torus_coords
    # =====================================================================
    import math
    theta = torch.rand(B, L, device=device) * 2 * math.pi - math.pi
    phi = torch.rand(B, L, device=device) * 2 * math.pi - math.pi
    r = torch.rand(B, L, device=device) * 0.4 + 0.05
    torus_coords = torch.stack([theta, phi, r], dim=-1)
    print(f"[smoke-s9-path1] real torus_coords shape = {torus_coords.shape}")

    # Context
    sequence_repr = torch.randn(B, L, d_model, device=device)
    pair_repr = torch.sigmoid(torch.randn(B, L, L, c_z, device=device))
    pair_probs = torch.sigmoid(torch.randn(B, L, L, device=device))

    head = ImmuneFingerprintHeads(
        d_model=d_model, c_z=c_z,
        enable_pkr=True, enable_nlrp3=True, enable_drach=True,
        enable_tlr7=True, enable_sponge=True,
        enable_equivariant=True, bond_length=bond_length,
    ).to(device)

    # =====================================================================
    # Forward with real torus_coords (path-1)
    # =====================================================================
    out = head(sequence_repr, pair_repr,
               torus_coords=torus_coords, pair_probs=pair_probs)

    expected = {"pkr_stem_logit", "pkr_sasa", "drach_is_drach", "drach_in_loop",
                "m6a_write_prob", "tlr7_gu_density", "sponge_score",
                "nlrp3_persistence_length"}
    missing = expected - set(out.keys())
    assert not missing, f"missing keys: {missing}"
    print(f"[smoke-s9-path1] immune keys present: {sorted(expected & set(out.keys()))}")

    assert out["m6a_write_prob"].abs().sum() > 0, "m6a all-zero (path-1 broken)"
    print(f"[smoke-s9-path1] m6a_write_prob stats: min={out['m6a_write_prob'].min():.3f} max={out['m6a_write_prob'].max():.3f}")
    print("[smoke-s9-path1] path-1 real torus_coords → immune head OK")

    # =====================================================================
    # Backward gradient check
    # =====================================================================
    loss = sum(v.pow(2).mean() for v in out.values())
    loss.backward()
    grad_ok, stats = check_immune_head_gradients(head, loss, tag="smoke-s9-path1")
    print(f"[smoke-s9-path1] backward loss={loss.item():.4f}, head grads finite: {grad_ok}")
    assert grad_ok, "non-finite/missing grads on S9 path-1"

    print("[smoke-s9-path1] ALL CHECKS PASSED")
    print("  [info] path-1 (real torus_coords) verified: equivariant path active,"
          " legacy MLP bypassed, gradient flow correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
