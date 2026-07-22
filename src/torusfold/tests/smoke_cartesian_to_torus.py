"""Smoke test: cartesian_to_torus.py equivariance + inverse correctness.

Verifies (the core scientific honesty requirement of Phase 2):
    1. Inverse correctness: forward (θ,φ,r)→coords then inverse→(θ,φ,r) recovers
       the original (θ,φ,r) to ≤ 1e-5.
    2. SO(2)_θ equivariance: apply θ-action (z-axis rotation) to coords, then
       cartesian_to_torus(action(x)) == θ+δθ, φ unchanged, r unchanged.
    3. SO(2)_φ equivariance: apply φ-action (fiberwise cross-section rotation)
       to coords, then cartesian_to_torus(action(x)) == φ+δφ, θ unchanged, r unchanged.

atol = 1e-5, rtol = 1e-4 (matches the closure_dist precision convention).

Run from repo root:
    python -m torusfold.tests.smoke_cartesian_to_torus
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.cartesian_to_torus import (
    cartesian_to_torus,
    apply_theta_action,
    apply_phi_action,
    wrap_to_pi,
    major_ring_radius,
)


def torus_forward(theta, phi, r, R):
    """S10's (θ,φ,r)→(x,y,z) forward map, replicated for the test."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    major_R = R + r * cos_p  # (B, L)
    x = major_R * cos_t
    y = major_R * sin_t
    z = r * sin_p
    return torch.stack([x, y, z], dim=-1)


def main() -> int:
    print("[smoke-c2t] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, L = 3, 24
    bond_length = 5.9
    lengths = torch.full((B,), L, device=device, dtype=torch.long)
    R = major_ring_radius(bond_length, lengths)  # (B,) float32
    Rf = R  # float32 path
    print(f"[smoke-c2t] R (major-ring radius) = {R[0].item():.4f}")

    # === float64 STRICT path — proves the math is exact (≤ 1e-10) ===
    Rd = bond_length * lengths.double().to(device) / (2.0 * math.pi)
    theta0 = (torch.rand(B, L, device=device).double() * 2 * math.pi - math.pi)
    phi0 = (torch.rand(B, L, device=device).double() * 2 * math.pi - math.pi)
    r0 = (torch.rand(B, L, device=device).double() * 0.4 + 0.05)
    coords = torus_forward(theta0, phi0, r0, Rd.unsqueeze(-1))

    # f64 inverse
    ti, pi_, ri = cartesian_to_torus(coords, Rd)
    assert wrap_to_pi(ti - theta0).abs().max().item() < 1e-10, "f64 θ inverse"
    assert wrap_to_pi(pi_ - phi0).abs().max().item() < 1e-10, "f64 φ inverse"
    assert (ri - r0).abs().max().item() < 1e-10, "f64 r inverse"

    # f64 θ-equivariance
    dt = torch.tensor(0.7, device=device).double()
    cr_t = apply_theta_action(coords, dt)
    tr, pr, rr = cartesian_to_torus(cr_t, Rd)
    assert wrap_to_pi(tr - theta0 - dt).abs().max().item() < 1e-10, "f64 θ-equiv θ"
    assert (pr - phi0).abs().max().item() < 1e-10, "f64 θ-equiv φ leak"
    assert (rr - r0).abs().max().item() < 1e-10, "f64 θ-equiv r leak"

    # f64 φ-equivariance
    dp = torch.tensor(0.4, device=device).double()
    cr_p = apply_phi_action(coords, Rd, dp)
    tp, pp, rp = cartesian_to_torus(cr_p, Rd)
    assert wrap_to_pi(tp - theta0).abs().max().item() < 1e-10, "f64 φ-equiv θ leak"
    assert wrap_to_pi(pp - phi0 - dp).abs().max().item() < 1e-10, "f64 φ-equiv φ"
    assert (rp - r0).abs().max().item() < 1e-10, "f64 φ-equiv r leak"
    print("[smoke-c2t] float64 STRICT path PASSED (math exact to ≤1e-10)")

    # === float32 ENGINEERING path — proves it's usable in real (fp32) training ===
    # float32 catastrophically cancels ρ-R when r·cos φ is small relative to R,
    # so the engineering bar is 1e-4 (not 1e-5). Math itself is exact (shown above).
    theta0f = theta0.float()
    phi0f = phi0.float()
    r0f = r0.float()
    coordsf = torus_forward(theta0f, phi0f, r0f, Rf.unsqueeze(-1))

    ti_f, pi_f, ri_f = cartesian_to_torus(coordsf, Rf)
    d_theta = wrap_to_pi(ti_f - theta0f).abs().max().item()
    d_phi = wrap_to_pi(pi_f - phi0f).abs().max().item()
    d_r = (ri_f - r0f).abs().max().item()
    print(f"[smoke-c2t] f32 inverse: max|dθ|={d_theta:.2e} max|dφ|={d_phi:.2e} max|dr|={d_r:.2e}")
    assert d_theta < 1e-4 and d_phi < 1e-4 and d_r < 1e-4, "f32 inverse exceeds 1e-4"

    dt_f = torch.tensor(0.7, device=device)
    cr_tf = apply_theta_action(coordsf, dt_f)
    tr_f, pr_f, rr_f = cartesian_to_torus(cr_tf, Rf)
    assert wrap_to_pi(tr_f - theta0f - dt_f).abs().max().item() < 1e-4, "f32 θ-equiv"
    assert (pr_f - phi0f).abs().max().item() < 1e-4, "f32 θ-equiv φ leak"
    assert (rr_f - r0f).abs().max().item() < 1e-4, "f32 θ-equiv r leak"
    print("[smoke-c2t] float32 θ-equivariance PASSED (≤1e-4)")

    dp_f = torch.tensor(0.4, device=device)
    cr_pf = apply_phi_action(coordsf, Rf, dp_f)
    tp_f, pp_f, rp_f = cartesian_to_torus(cr_pf, Rf)
    assert wrap_to_pi(tp_f - theta0f).abs().max().item() < 1e-4, "f32 φ-equiv θ leak"
    assert wrap_to_pi(pp_f - phi0f - dp_f).abs().max().item() < 1e-4, "f32 φ-equiv"
    assert (rp_f - r0f).abs().max().item() < 1e-4, "f32 φ-equiv r leak"
    print("[smoke-c2t] float32 φ-equivariance PASSED (≤1e-4)")

    # === batched (per-sequence different δθ) — float32 ===
    dt_b = torch.tensor([0.3, -1.1, 2.0], device=device)
    cr_b = apply_theta_action(coordsf, dt_b)
    theta_b, _, _ = cartesian_to_torus(cr_b, Rf)
    expected = dt_b.unsqueeze(-1)
    assert wrap_to_pi(theta_b - theta0f - expected).abs().max().item() < 1e-4, "batched θ-equiv"
    print("[smoke-c2t] batched (per-sequence) θ-equivariance PASSED")

    # === gradient flow ===
    coords_g = coordsf.clone().requires_grad_(True)
    theta_g, phi_g, r_g = cartesian_to_torus(coords_g, Rf)
    loss = theta_g.sum() + phi_g.sum() + r_g.sum()
    loss.backward()
    assert coords_g.grad is not None and torch.isfinite(coords_g.grad).all(), "grad broken"
    print("[smoke-c2t] gradient flow PASSED")

    print("[smoke-c2t] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
