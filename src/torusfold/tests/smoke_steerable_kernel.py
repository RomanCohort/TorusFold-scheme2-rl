"""Smoke test: steerable_kernel.py standalone import + forward/backward.

Verifies the extracted SO2SteerableKernel / CircEquivariantGNNLayer work
independently of scheme10_circ_equivariant_gnn.py (no Scheme10Config needed).

Run from repo root:
    python -m torusfold.tests.smoke_steerable_kernel
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.steerable_kernel import (
    SO2SteerableKernel,
    CircEquivariantGNNLayer,
)


def main() -> int:
    print("[smoke-steerable] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, L, d_model, d_edge = 2, 16, 64, 32
    n_edge_cats, k_theta, k_phi = 5, 2, 1

    # --- 1. SO2SteerableKernel standalone forward ---
    kernel = SO2SteerableKernel(
        d_model=d_model, d_edge=d_edge,
        n_edge_cats=n_edge_cats, k_theta=k_theta, k_phi=k_phi,
    ).to(device)
    print(f"[smoke-steerable] SO2SteerableKernel params: {sum(p.numel() for p in kernel.parameters()):,}")

    x = torch.randn(B, L, d_model, device=device, requires_grad=True)
    # Angle differences: Δθ, Δφ in [-2π, 2π]
    delta_theta = torch.randn(B, L, L, device=device) * 2.0
    delta_phi = torch.randn(B, L, L, device=device) * 1.0
    edge_cat = torch.randint(0, n_edge_cats, (B, L, L), device=device)

    msg = kernel(x, delta_theta, delta_phi, edge_cat)
    assert msg.shape == (B, L, L, d_model), f"msg shape wrong: {msg.shape}"
    assert torch.isfinite(msg).all(), "msg has NaN/inf"
    print(f"[smoke-steerable] kernel forward OK, msg shape={msg.shape}")

    # --- 2. backward through kernel ---
    loss = msg.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "kernel backward broken"
    kernel_grads = [p.grad for p in kernel.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in kernel_grads), "kernel param grads broken"
    print(f"[smoke-steerable] kernel backward OK, loss={loss.item():.4f}")

    # --- 3. CircEquivariantGNNLayer forward (config-free!) ---
    layer = CircEquivariantGNNLayer(
        d_model=d_model, d_edge=d_edge,
        n_edge_cats=n_edge_cats, k_theta=k_theta, k_phi=k_phi,
        dropout=0.1,
    ).to(device)
    print(f"[smoke-steerable] CircEquivariantGNNLayer params: {sum(p.numel() for p in layer.parameters()):,}")

    x2 = torch.randn(B, L, d_model, device=device, requires_grad=True)
    lengths = torch.full((B,), L, device=device, dtype=torch.long)
    out = layer(x2, delta_theta, delta_phi, edge_cat, lengths)
    assert out.shape == (B, L, d_model), f"layer out shape wrong: {out.shape}"
    assert torch.isfinite(out).all(), "layer out has NaN/inf"
    print(f"[smoke-steerable] layer forward OK, out shape={out.shape}")

    # --- 4. backward through layer ---
    loss2 = out.pow(2).mean()
    loss2.backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all(), "layer backward broken"
    layer_grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in layer_grads), "layer param grads broken"
    print(f"[smoke-steerable] layer backward OK, loss={loss2.item():.4f}")

    # --- 5. irrep channel count sanity ---
    expected_n_irrep = (1 + 2 * k_theta) * (1 + 2 * k_phi)
    assert kernel.n_theta_irreps == 1 + 2 * k_theta
    assert kernel.n_phi_irreps == 1 + 2 * k_phi
    assert kernel.n_theta_irreps * kernel.n_phi_irreps == expected_n_irrep
    print(f"[smoke-steerable] irrep channels: n_theta={kernel.n_theta_irreps} n_phi={kernel.n_phi_irreps} total={expected_n_irrep}")

    print("[smoke-steerable] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
