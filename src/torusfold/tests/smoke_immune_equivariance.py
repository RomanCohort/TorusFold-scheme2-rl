"""Smoke test: immune_fingerprint_head.py SO(2)×SO(2) equivariance (math).

Verifies the REAL equivariance of Phase 2's steerable heads.

Core idea:
    The steerable kernel's irrep basis φ_c^{(k,l)}(Δθ, Δφ) is built from
    angle DIFFERENCES. Shifting every node's (θ, φ) by the same (δθ, δφ)
    leaves all Δθ, Δφ unchanged, so messages are unchanged, so the
    equivariant heads' (PKR, m6A) outputs are unchanged.

    We test this in TORUS coordinate space directly: build (θ, φ, r),
    add (δθ, δφ) to every node, pass both versions to the head as
    `torus_coords` (path 1 = real torus_coords), and compare outputs.

    Heads that go through `feat` (which contains the raw torus_slot θ/φ/r)
    via MLP — NLRP3, TLR7, sponge — are NOT expected to be equivariant
    (they are scalar topology quantities, no directional symmetry). We
    only assert invariance on PKR and m6A (the equivariant heads), and
    merely REPORT the diff on the others.

Tolerance: 1e-5 for float32 (math exact in float64 to 1e-10; float32
catastrophic cancellation on ρ-R pushes the engineering bar to 1e-4 in
the Cartesian path, but path-1 here bypasses Cartesian entirely so we
hold the tighter 1e-5).

Run from repo root:
    python -m torusfold.tests.smoke_immune_equivariance
"""
from __future__ import annotations

import sys
import io
# UTF-8 encoding for Windows terminal output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.immune_fingerprint_head import (
    ImmuneFingerprintHeads,
)
from torusfold.cartesian_to_torus import wrap_to_pi


def main() -> int:
    print("[smoke-immune-equiv] torch", torch.__version__,
          "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, L, d_model, c_z = 4, 36, 64, 32
    bond_length = 5.9

    # =====================================================================
    # Build a torus_coords tensor (θ, φ, r) directly.
    # NOTE: this is what S9/S10 trunks would pass — the head's path-1
    # uses torus_coords as-is, so Δθ/Δφ derived inside are real angles.
    # =====================================================================
    theta = torch.rand(B, L, device=device) * 2 * math.pi - math.pi   # (-π, π]
    phi = torch.rand(B, L, device=device) * 2 * math.pi - math.pi
    r = torch.rand(B, L, device=device) * 0.4 + 0.05                   # small cross-section
    torus_coords = torch.stack([theta, phi, r], dim=-1)  # (B, L, 3)
    print(f"[setup] torus_coords shape = {torus_coords.shape}")

    # Rotation-invariant context (same for both forward passes).
    sequence_repr = torch.randn(B, L, d_model, device=device)
    pair_repr = torch.sigmoid(torch.randn(B, L, L, c_z, device=device))
    pair_probs = torch.sigmoid(torch.randn(B, L, L, device=device))

    head = ImmuneFingerprintHeads(
        d_model=d_model, c_z=c_z,
        enable_pkr=True, enable_nlrp3=True, enable_drach=True,
        enable_tlr7=True, enable_sponge=True,
        enable_equivariant=True, bond_length=bond_length,
    ).to(device)
    head.eval()  # disable dropout for deterministic comparison

    def fwd(tc):
        return head(sequence_repr, pair_repr,
                    torus_coords=tc, pair_probs=pair_probs)

    with torch.no_grad():
        out_orig = fwd(torus_coords)

    def report(name, out_rot, out_ref):
        d_pkr_stem = (out_rot["pkr_stem_logit"] - out_ref["pkr_stem_logit"]).abs().max().item()
        d_pkr_sasa = (out_rot["pkr_sasa"] - out_ref["pkr_sasa"]).abs().max().item()
        d_drach = (out_rot["drach_is_drach"] - out_ref["drach_is_drach"]).abs().max().item()
        d_m6a = (out_rot["m6a_write_prob"] - out_ref["m6a_write_prob"]).abs().max().item()
        d_tlr7 = (out_rot["tlr7_gu_density"] - out_ref["tlr7_gu_density"]).abs().max().item()
        d_sponge = (out_rot["sponge_score"] - out_ref["sponge_score"]).abs().max().item()
        d_nlrp3 = (out_rot["nlrp3_persistence_length"] -
                   out_ref["nlrp3_persistence_length"]).abs().max().item()
        print(f"  [{name}] equivariant heads (should be ~0):")
        print(f"      pkr_stem_logit = {d_pkr_stem:.2e}")
        print(f"      pkr_sasa       = {d_pkr_sasa:.2e}")
        print(f"      drach_is_drach = {d_drach:.2e}")
        print(f"      m6a_write_prob = {d_m6a:.2e}")
        print(f"  [{name}] MLP heads (expected to change — scalar topology):")
        print(f"      tlr7_gu_density    = {d_tlr7:.2e}")
        print(f"      sponge_score       = {d_sponge:.2e}")
        print(f"      nlrp3_persistence  = {d_nlrp3:.2e}")
        return dict(pkr_stem=d_pkr_stem, pkr_sasa=d_pkr_sasa,
                    drach=d_drach, m6a=d_m6a)

    TOL = 1e-5

    # =====================================================================
    # Test 1: SO(2)_θ — shift every node's θ by δθ, φ/r unchanged.
    # Δθ = θ_i - θ_j is invariant under θ → θ + δθ.
    # =====================================================================
    print("\n[Test1] SO(2)_θ equivariance (shift all θ by δθ=0.7)")
    dt = 0.7
    theta_rot = wrap_to_pi(theta + dt)   # keep in (-π, π]
    torus_coords_theta = torch.stack([theta_rot, phi, r], dim=-1)
    with torch.no_grad():
        out_t = fwd(torus_coords_theta)
    d = report("θ-shift", out_t, out_orig)
    assert d["pkr_stem"] < TOL, f"SO(2)_θ broken: pkr_stem {d['pkr_stem']}"
    assert d["pkr_sasa"] < TOL, f"SO(2)_θ broken: pkr_sasa {d['pkr_sasa']}"
    assert d["drach"] < TOL, f"SO(2)_θ broken: drach {d['drach']}"
    assert d["m6a"] < TOL, f"SO(2)_θ broken: m6a {d['m6a']}"
    print("  PASS - SO(2)_θ-equivariance (≤ 1e-5)")

    # =====================================================================
    # Test 2: SO(2)_φ — shift every node's φ by δφ, θ/r unchanged.
    # Δφ = φ_i - φ_j is invariant under φ → φ + δφ.
    # =====================================================================
    print("\n[Test2] SO(2)_φ equivariance (shift all φ by δφ=0.9)")
    dp = 0.9
    phi_rot = wrap_to_pi(phi + dp)
    torus_coords_phi = torch.stack([theta, phi_rot, r], dim=-1)
    with torch.no_grad():
        out_p = fwd(torus_coords_phi)
    d = report("φ-shift", out_p, out_orig)
    assert d["pkr_stem"] < TOL, f"SO(2)_φ broken: pkr_stem {d['pkr_stem']}"
    assert d["pkr_sasa"] < TOL, f"SO(2)_φ broken: pkr_sasa {d['pkr_sasa']}"
    assert d["drach"] < TOL, f"SO(2)_φ broken: drach {d['drach']}"
    assert d["m6a"] < TOL, f"SO(2)_φ broken: m6a {d['m6a']}"
    print("  PASS - SO(2)_φ-equivariance (≤ 1e-5)")

    # =====================================================================
    # Test 3: Combined SO(2)_θ × SO(2)_φ.
    # =====================================================================
    print("\n[Test3] Combined SO(2)_θ × SO(2)_φ (δθ=0.6, δφ=-0.8)")
    theta_rot2 = wrap_to_pi(theta + 0.6)
    phi_rot2 = wrap_to_pi(phi - 0.8)
    torus_coords_both = torch.stack([theta_rot2, phi_rot2, r], dim=-1)
    with torch.no_grad():
        out_b = fwd(torus_coords_both)
    d = report("θ⊕φ-shift", out_b, out_orig)
    assert d["pkr_stem"] < TOL, f"Combined broken: pkr_stem {d['pkr_stem']}"
    assert d["pkr_sasa"] < TOL, f"Combined broken: pkr_sasa {d['pkr_sasa']}"
    assert d["drach"] < TOL, f"Combined broken: drach {d['drach']}"
    assert d["m6a"] < TOL, f"Combined broken: m6a {d['m6a']}"
    print("  PASS - Combined SO(2)_θ × SO(2)_φ-equivariance (≤ 1e-5)")

    print("\n[smoke-immune-equiv] ALL EQUIVARIANCE TESTS PASSED")
    print("  [info] PKR & m6A heads are exactly SO(2)×SO(2)-equivariant via")
    print("    Δθ/Δφ invariance of the steerable irrep basis (≤ 1e-5, float32).")
    print("  [info] NLRP3/TLR7/sponge MLP heads are NOT equivariant by design")
    print("    (scalar topology quantities — no directional symmetry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
