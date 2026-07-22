"""Smoke test: immune_fingerprint_head.py Phase 2 equivariant path.

Verifies the honest forward/backward behavior across all 4 config paths:

    Path 1: real torus_coords (S9/S10)         → equivariant path active
    Path 2: cartesian_coords (S1-S8)           → reverse-map → equivariant path active
    Path 3: neither (S6/S7/S8 latent)          → MLP fallback (equiv heads inactive)
    Path 4: enable_equivariant=False           → legacy MLP forced (A/B ablation)

Key honesty check: when the equivariant path is active (Path 1/2), the legacy
`pkr_head` / `drach_head` MLP weights are NOT in the computation graph, so they
naturally have no gradient — this is CORRECT behavior, not a bug. We verify the
EQUIVARIANT params (pkr_equiv_layer, pkr_equiv_proj, drach_equiv_layer,
drach_equiv_proj) DO have finite gradients, and the legacy params do NOT (when
equivariant path is taken). When the MLP fallback is taken (Path 3/4), legacy
params DO get gradients.

Run from repo root:
    python -m torusfold.tests.smoke_immune_equivariant
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.immune_fingerprint_head import (
    ImmuneFingerprintHeads,
)


def _make_torus_coords(B, L, R, device):
    """Build coords on a torus surface so reverse-mapping is well-conditioned."""
    theta = (torch.rand(B, L, device=device) * 2 * 3.14159265 - 3.14159265)
    phi = (torch.rand(B, L, device=device) * 2 * 3.14159265 - 3.14159265)
    r = torch.rand(B, L, device=device) * 0.4 + 0.05  # small cross-section radius
    major_R = R + r * torch.cos(phi)  # (B, L)
    x = major_R * torch.cos(theta)
    y = major_R * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)  # (B, L, 3)


def _check_grad_finite(module, prefix, require_grad=True):
    """Check every param under `module` named `prefix.*` has finite grad.

    Returns (ok, n_checked, n_with_grad, n_nan_inf, names_no_grad, names_bad).
    `require_grad=True` means we EXPECT every requires_grad param to have a
    finite grad (the param IS in the graph). `require_grad=False` means we
    EXPECT params to have NO grad (they're skipped, not in the graph) — used
    for legacy MLP weights when the equivariant path is taken.
    """
    n_checked = n_with_grad = n_nan_inf = 0
    names_no_grad, names_bad = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith(prefix):
            continue
        n_checked += 1
        g = p.grad
        if g is None:
            names_no_grad.append(name)
            continue
        n_with_grad += 1
        if not torch.isfinite(g).all():
            n_nan_inf += 1
            names_bad.append(name)
    ok = require_grad and (n_checked == n_with_grad) and (n_nan_inf == 0)
    if not require_grad:
        # Inverse expectation: every checked param should have NO grad.
        ok = (n_with_grad == 0) and (n_checked > 0)
    return ok, n_checked, n_with_grad, n_nan_inf, names_no_grad, names_bad


def main() -> int:
    print("[smoke-immune-equiv] torch", torch.__version__,
          "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, L, d_model, c_z = 2, 16, 64, 32
    bond_length = 5.9
    R = bond_length * L / (2 * 3.14159265)  # scalar major-ring radius

    sequence_repr = torch.randn(B, L, d_model, device=device, requires_grad=False)
    pair_repr = torch.randn(B, L, L, c_z, device=device)
    pair_probs = torch.sigmoid(torch.randn(B, L, L, device=device))
    torus_coords = _make_torus_coords(B, L, R, device)         # on torus surface
    cartesian_coords = _make_torus_coords(B, L, R, device)     # same shape, treated as Cartesian

    def _build(enable_equivariant):
        return ImmuneFingerprintHeads(
            d_model=d_model, c_z=c_z,
            enable_pkr=True, enable_nlrp3=True, enable_drach=True,
            enable_tlr7=True, enable_sponge=True,
            enable_equivariant=enable_equivariant,
            bond_length=bond_length,
        ).to(device)

    def _loss(out):
        # Sum a scalar from each head so every active path gets a gradient.
        return (out["pkr_stem_logit"].sum()
                + out["pkr_sasa"].sum()
                + out["nlrp3_persistence_length"].sum()
                + out["drach_is_drach"].sum()
                + out["drach_in_loop"].sum()
                + out["m6a_write_prob"].sum()
                + out["tlr7_gu_density"].sum()
                + out["sponge_score"].sum())

    # =====================================================================
    # Path 1: real torus_coords → equivariant path active
    # Expect: equiv params HAVE finite grads; legacy pkr_head/drach_head NO grad
    # =====================================================================
    head1 = _build(enable_equivariant=True)
    out1 = head1(sequence_repr, pair_repr,
                 torus_coords=torus_coords, pair_probs=pair_probs)
    _loss(out1).backward()
    assert torch.isfinite(out1["pkr_stem_logit"]).all(), "Path1 forward NaN"
    ok_eq1, nc1, ng1, nbad1, _, bad1 = _check_grad_finite(
        head1, "pkr_equiv_layer", require_grad=True)
    ok_eq1b, _, _, _, _, bad1b = _check_grad_finite(
        head1, "pkr_equiv_proj", require_grad=True)
    ok_eq1c, _, _, _, _, bad1c = _check_grad_finite(
        head1, "drach_equiv_layer", require_grad=True)
    ok_eq1d, _, _, _, _, bad1d = _check_grad_finite(
        head1, "drach_equiv_proj", require_grad=True)
    # Legacy MLP weights should have NO grad (equivariant path taken).
    ok_legacy1, _, nlg1, _, _, _ = _check_grad_finite(
        head1, "pkr_head", require_grad=False)
    ok_legacy1b, _, nlg1b, _, _, _ = _check_grad_finite(
        head1, "drach_head", require_grad=False)
    print(f"[Path1 real-torus] equiv grad finite: "
          f"pkr_layer={ok_eq1} pkr_proj={ok_eq1b} "
          f"drach_layer={ok_eq1c} drach_proj={ok_eq1d} "
          f"(checked {nc1}+ params, bad: {bad1+bad1b+bad1c+bad1d})")
    print(f"[Path1 real-torus] legacy no-grad (expected): "
          f"pkr_head grad={nlg1} drach_head grad={nlg1b} (should be 0)")
    assert ok_eq1 and ok_eq1b and ok_eq1c and ok_eq1d, \
        f"Path1 equivariant grads broken: {bad1+bad1b+bad1c+bad1d}"
    assert ok_legacy1 and ok_legacy1b, \
        "Path1 legacy MLP got unexpected gradient (equiv path should bypass it)"

    # =====================================================================
    # Path 2: cartesian_coords only → reverse-map → equivariant path active
    # =====================================================================
    head2 = _build(enable_equivariant=True)
    out2 = head2(sequence_repr, pair_repr,
                 torus_coords=None, pair_probs=pair_probs,
                 cartesian_coords=cartesian_coords)
    _loss(out2).backward()
    assert torch.isfinite(out2["pkr_stem_logit"]).all(), "Path2 forward NaN"
    ok_eq2, nc2, _, _, _, bad2 = _check_grad_finite(
        head2, "pkr_equiv_layer", require_grad=True)
    ok_eq2b, _, _, _, _, bad2b = _check_grad_finite(
        head2, "pkr_equiv_proj", require_grad=True)
    ok_eq2c, _, _, _, _, bad2c = _check_grad_finite(
        head2, "drach_equiv_layer", require_grad=True)
    ok_eq2d, _, _, _, _, bad2d = _check_grad_finite(
        head2, "drach_equiv_proj", require_grad=True)
    ok_legacy2, _, nlg2, _, _, _ = _check_grad_finite(
        head2, "pkr_head", require_grad=False)
    print(f"[Path2 cartesian-reverse] equiv grad finite: "
          f"pkr_layer={ok_eq2} pkr_proj={ok_eq2b} "
          f"drach_layer={ok_eq2c} drach_proj={ok_eq2d} "
          f"(bad: {bad2+bad2b+bad2c+bad2d})")
    print(f"[Path2 cartesian-reverse] legacy no-grad (expected): pkr_head grad={nlg2}")
    assert ok_eq2 and ok_eq2b and ok_eq2c and ok_eq2d, \
        f"Path2 equivariant grads broken: {bad2+bad2b+bad2c+bad2d}"
    assert ok_legacy2, "Path2 legacy MLP got unexpected gradient"

    # =====================================================================
    # Path 3: neither torus nor cartesian → MLP fallback
    # Expect: legacy pkr_head/drach_head HAVE finite grads; equiv params NO grad
    # (equiv params exist but aren't used in forward → no grad, correct)
    # =====================================================================
    head3 = _build(enable_equivariant=True)
    out3 = head3(sequence_repr, pair_repr,
                 torus_coords=None, pair_probs=pair_probs,
                 cartesian_coords=None)
    _loss(out3).backward()
    assert torch.isfinite(out3["pkr_stem_logit"]).all(), "Path3 forward NaN"
    # Legacy MLP should now HAVE grads.
    ok_legacy3, nc3, nlg3, nbad3, nog3, bad3 = _check_grad_finite(
        head3, "pkr_head", require_grad=True)
    ok_legacy3b, _, nlg3b, _, _, bad3b = _check_grad_finite(
        head3, "drach_head", require_grad=True)
    # Equiv params should have NO grad (forward took MLP fallback).
    ok_eq3_nograd, _, neg3, _, _, _ = _check_grad_finite(
        head3, "pkr_equiv_layer", require_grad=False)
    print(f"[Path3 neither] legacy MLP grad finite: "
          f"pkr_head={ok_legacy3} drach_head={ok_legacy3b} "
          f"(pkr_head with-grad={nlg3}, bad: {bad3+bad3b})")
    print(f"[Path3 neither] equiv no-grad (expected): "
          f"pkr_equiv_layer with-grad={neg3} (should be 0)")
    assert ok_legacy3 and ok_legacy3b, \
        f"Path3 legacy MLP grads broken: {bad3+bad3b}"
    assert ok_eq3_nograd, \
        "Path3 equivariant params got unexpected gradient (MLP fallback taken)"

    # =====================================================================
    # Path 4: enable_equivariant=False → legacy MLP forced (A/B ablation)
    # Expect: legacy pkr_head/drach_head HAVE finite grads; NO equiv params exist
    # =====================================================================
    head4 = _build(enable_equivariant=False)
    out4 = head4(sequence_repr, pair_repr,
                 torus_coords=torus_coords, pair_probs=pair_probs)
    _loss(out4).backward()
    assert torch.isfinite(out4["pkr_stem_logit"]).all(), "Path4 forward NaN"
    ok_legacy4, nc4, nlg4, nbad4, nog4, bad4 = _check_grad_finite(
        head4, "pkr_head", require_grad=True)
    ok_legacy4b, _, _, _, _, bad4b = _check_grad_finite(
        head4, "drach_head", require_grad=True)
    # Confirm no equivariant params exist when disabled.
    has_equiv = any(n.startswith(("pkr_equiv", "drach_equiv"))
                    for n, _ in head4.named_parameters())
    print(f"[Path4 equiv-disabled] legacy MLP grad finite: "
          f"pkr_head={ok_legacy4} drach_head={ok_legacy4b} "
          f"(bad: {bad4+bad4b}); equiv params exist? {has_equiv}")
    assert ok_legacy4 and ok_legacy4b, f"Path4 legacy grads broken: {bad4+bad4b}"
    assert not has_equiv, "Path4 should not instantiate equivariant params"

    print("[smoke-immune-equiv] ALL 4 PATHS PASSED (forward + backward honest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
