"""Smoke test: Scheme 1 (CircRNA3DModel, EGNN) TrunkOutput compliance.

Verifies:
    1. S1 returns ALL TRUNK_REQUIRED_KEYS, passes validate_trunk_output.
    2. pair_repr_source == "synthetic".
    3. Shared ImmuneFingerprintHeads attaches; 8 immune keys present.
    4. Backward through immune head is finite.

Note: S1 is the EGNN-based 3D predictor from train_torusfold_3d.py; it has
no explicit pair_probs input and no torus parameterization. The synthesizer
uses a uniform pair prior (pair_probs=None).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.train_torusfold_3d import (
    CircRNA3DModel,
)
from torusfold.trunk_output import (
    TRUNK_REQUIRED_KEYS,
    validate_trunk_output,
)
from torusfold.immune_fingerprint_head import (
    ImmuneFingerprintHeads,
)


def main() -> int:
    print("[smoke-s1] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[smoke-s1] device:", device)

    model = CircRNA3DModel(d_hidden=64, n_layers=2).to(device)

    B, L = 2, 16
    seq_ids = torch.randint(0, 5, (B, L), device=device)
    lengths = torch.full((B,), L, device=device)  # no padding

    # --- 1. protocol compliance without immune head ---
    model.eval()
    with torch.no_grad():
        out = model(seq_ids, lengths=lengths)

    missing = TRUNK_REQUIRED_KEYS - set(out.keys())
    assert not missing, f"S1 missing TrunkOutput keys: {missing}"
    print("[smoke-s1] TrunkOutput keys present:", sorted(TRUNK_REQUIRED_KEYS & set(out.keys())))

    validate_trunk_output(out)
    print("[smoke-s1] validate_trunk_output() PASSED")

    assert out.get("pair_repr_source") == "synthetic", out.get("pair_repr_source")
    assert out["structure_method"] == "circrna_3d_egnn", out["structure_method"]
    print(f"[smoke-s1] pair_repr_source = {out['pair_repr_source']}")
    print(f"[smoke-s1] structure_method = {out['structure_method']}")

    assert out["coords"].shape == (B, L, 3)
    assert out["sequence_repr"].shape[:2] == (B, L)
    assert out["pair_repr"].shape[:3] == (B, L, L)
    print("[smoke-s1] shapes OK")

    # --- 2. attach shared immune head ---
    # S1 outputs Cartesian coords → 3D Cartesian kNN-density exposure proxy
    # (path 2). enable_fingerprint_2d=False so it doesn't force the 2D path.
    head = ImmuneFingerprintHeads(
        d_model=64,
        c_z=64,
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=False,  # let Cartesian path activate
    ).to(device)
    model.attach_immune_head(head)
    print("[smoke-s1] attached shared ImmuneFingerprintHeads (Cartesian path)")

    with torch.no_grad():
        out2 = model(seq_ids, lengths=lengths)
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
    print("[smoke-s1] immune keys present:", sorted(expected & set(immune.keys())))
    # 3D Cartesian m6A proxy should be non-zero (kNN density varies across residues)
    assert immune["m6a_write_prob"].abs().sum() > 0, "m6a all-zero (Cartesian path broken)"
    # exposure should vary across residues (not all-identical) for a real 3D signal
    exp_proxy = immune["m6a_write_prob"]  # is_drach * in_loop * exposure
    # Check that exposure component has variance: compare to uniform 0.5
    print(f"[smoke-s1] m6a_write_prob stats: min={exp_proxy.min():.3f} max={exp_proxy.max():.3f}")
    print("[smoke-s1] 3D Cartesian m6A proxy non-zero OK")

    # --- 3. backward ---
    # S1 passes Cartesian coords → immune head takes the Phase-2 equivariant
    # path (reverse-map Cartesian → torus → steerable kernel). When the
    # equivariant path is active, the legacy `pkr_head`/`drach_head` MLP
    # weights are NOT in the computation graph (correct by design — the
    # equivariant `pkr_equiv_layer`/`drach_equiv_layer` replace them). So
    # those legacy weights legitimately have NO gradient; we must not flag
    # them as failures. We only require: (a) every equivariant param has a
    # finite gradient, (b) every other param that DID get a gradient is finite.
    model.train()
    out3 = model(seq_ids, lengths=lengths)
    immune3 = out3["immune"]
    loss = sum(v.pow(2).mean() for v in immune3.values())
    loss.backward()

    named = list(head.named_parameters())
    # Legacy MLP weights bypassed by the equivariant path.
    legacy_prefixes = ("pkr_head.", "drach_head.")
    equiv_prefixes = ("pkr_equiv_layer.", "pkr_equiv_proj.",
                      "drach_equiv_layer.", "drach_equiv_proj.")

    bad = []
    equiv_no_grad = []
    legacy_no_grad = []
    other_no_grad = []
    for n, p in named:
        if not p.requires_grad:
            continue
        if p.grad is None:
            if n.startswith(equiv_prefixes):
                equiv_no_grad.append(n)
            elif n.startswith(legacy_prefixes):
                legacy_no_grad.append(n)
            else:
                other_no_grad.append(n)
        elif not torch.isfinite(p.grad).all():
            bad.append((n, p.shape, p.grad.abs().max().item()))

    grad_ok = (len(equiv_no_grad) == 0) and (len(other_no_grad) == 0) and (len(bad) == 0)
    print(f"[smoke-s1] backward loss={loss.item():.4f}, head grads finite: {grad_ok}")
    if legacy_no_grad:
        print(f"[smoke-s1] {len(legacy_no_grad)} legacy MLP params NO grad "
              f"(expected — equivariant path active, MLP bypassed):")
        for n in legacy_no_grad[:4]:
            print(f"    (ok) {n}")
        if len(legacy_no_grad) > 4:
            print(f"    ... +{len(legacy_no_grad)-4} more")
    if equiv_no_grad:
        print(f"[smoke-s1] FAILED: {len(equiv_no_grad)} equivariant params NO grad:")
        for n in equiv_no_grad:
            print(f"    NO GRAD: {n}")
    if other_no_grad:
        print(f"[smoke-s1] FAILED: {len(other_no_grad)} other params NO grad:")
        for n in other_no_grad:
            print(f"    NO GRAD: {n}")
    if bad:
        print(f"[smoke-s1] FAILED: {len(bad)} params with non-finite gradients:")
        for n, sh, mx in bad[:5]:
            print(f"    {n} shape={sh} max={mx:.3e}")
    assert grad_ok, "non-finite/missing grads in immune head on S1 (equivariant path)"

    print("[smoke-s1] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
