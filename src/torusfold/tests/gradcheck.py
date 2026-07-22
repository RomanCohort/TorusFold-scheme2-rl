"""Shared gradient-check helper for trunk-output smoke tests.

Phase 2 adds equivariant paths for PKR/m6A heads. When the equivariant path
is active (S1/S4 Cartesian reverse-mapped, or S9/S10 real torus_coords), the
legacy MLP weights (pkr_head, drach_head) are bypassed and legitimately have
NO gradient. This is correct design, not a bug.

Usage (after loss.backward()):
    from .gradcheck import check_immune_head_gradients
    ok, stats = check_immune_head_gradients(head, loss)
    assert ok, f"immune head grads bad: {stats}"
"""
from __future__ import annotations

import torch

EQUIV_PREFIXES = (
    "pkr_equiv_layer.", "pkr_equiv_proj.",
    "drach_equiv_layer.", "drach_equiv_proj.",
)
LEGACY_PREFIXES = ("pkr_head.", "drach_head.")


def check_immune_head_gradients(head, loss, tag=""):
    """Return (ok, stats_dict). ok=True if every param that SHOULD have a
    gradient has a finite one. Legacy MLP weights with no grad (when the
    equivariant path is active) are tolerated and reported.
    """
    named = list(head.named_parameters())
    bad, equiv_no_grad, legacy_no_grad, other_no_grad = [], [], [], []
    for n, p in named:
        if not p.requires_grad:
            continue
        if p.grad is None:
            if n.startswith(EQUIV_PREFIXES):
                equiv_no_grad.append(n)
            elif n.startswith(LEGACY_PREFIXES):
                legacy_no_grad.append(n)
            else:
                other_no_grad.append(n)
        elif not torch.isfinite(p.grad).all():
            bad.append((n, p.shape, p.grad.abs().max().item()))
    ok = (not equiv_no_grad) and (not other_no_grad) and (not bad)
    prefix = f"[{tag}] " if tag else ""
    if legacy_no_grad:
        print(f"{prefix}{len(legacy_no_grad)} legacy MLP params NO grad "
              f"(expected - equivariant path active, MLP bypassed)")
    if equiv_no_grad:
        print(f"{prefix}FAILED: {len(equiv_no_grad)} equivariant params NO grad:")
        for n in equiv_no_grad:
            print(f"    NO GRAD: {n}")
    if other_no_grad:
        print(f"{prefix}FAILED: {len(other_no_grad)} other params NO grad:")
        for n in other_no_grad:
            print(f"    NO GRAD: {n}")
    if bad:
        print(f"{prefix}FAILED: {len(bad)} params non-finite grads:")
        for n, sh, mx in bad[:5]:
            print(f"    {n} shape={sh} max={mx:.3e}")
    return ok, dict(equiv_no_grad=equiv_no_grad, legacy_no_grad=legacy_no_grad,
                    other_no_grad=other_no_grad, bad=bad)
