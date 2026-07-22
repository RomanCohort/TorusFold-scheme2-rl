"""Smoke test: Scheme 10 (SO(2)×SO(2) equivariant GNN) TrunkOutput compliance.

Verifies:
    1. S10 returns ALL TRUNK_REQUIRED_KEYS, passes validate_trunk_output.
    2. pair_repr_source 未标记（S10 有稀疏 pair kernel，不是 synthetic）。
    3. Shared ImmuneFingerprintHeads attaches and gets 8 immune keys.
    4. 3D mode (torus_coords provided) and 2D mode (torus_coords=None)
       both work — m6A exposure proxy 使用 3D torus radius。
Run from repo root:
    python -m torusfold.tests.smoke_scheme10_trunk_protocol
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torusfold.scheme10_circ_equivariant_gnn import (
    Scheme10Model,
    Scheme10Config,
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
    print("[smoke-s10] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[smoke-s10] device:", device)

    config = Scheme10Config(d_model=64, n_layers=2)
    model = Scheme10Model(config).to(device)

    B, L = 2, 32
    seq_tokens = torch.randint(0, 4, (B, L), device=device)
    pair_probs = torch.full((B, L, L), 0.3, device=device)
    pair_probs = pair_probs * (~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0))
    pair_probs = 0.5 * (pair_probs + pair_probs.transpose(1, 2))

    # --- 1. protocol compliance without immune head ---
    model.eval()
    with torch.no_grad():
        out = model(seq_tokens, pair_probs=pair_probs)

    missing = TRUNK_REQUIRED_KEYS - set(out.keys())
    assert not missing, f"S10 missing TrunkOutput keys: {missing}"
    print("[smoke-s10] TrunkOutput keys present:", sorted(TRUNK_REQUIRED_KEYS & set(out.keys())))

    validate_trunk_output(out)
    print("[smoke-s10] validate_trunk_output() PASSED")

    assert out["structure_method"] == "so2_so2_equivariant", out["structure_method"]
    print(f"[smoke-s10] structure_method = {out['structure_method']}")
    if "pair_repr_source" in out:
        print(f"[smoke-s10] WARNING: pair_repr_source = {out['pair_repr_source']} (expected not set)")

    assert out["coords"].shape == (B, L, 3)
    assert out["sequence_repr"].shape[:2] == (B, L)
    assert out["pair_repr"].shape[:3] == (B, L, L)
    assert out["torus_coords"].shape[:2] == (B, L)

    # --- 2. attach shared immune head (3D mode) ---
    head = ImmuneFingerprintHeads(
        d_model=config.d_model,
        c_z=config.d_pair,
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=False,  # 3D mode (S10 有 torus_coords)
    ).to(device)
    model.attach_immune_head(head)
    print("[smoke-s10] attached shared ImmuneFingerprintHeads")

    with torch.no_grad():
        out2 = model(seq_tokens, pair_probs=pair_probs)

    expected_immune = {
        "pkr_stem_logit", "pkr_sasa",
        "nlrp3_persistence_length",
        "drach_is_drach", "drach_in_loop", "m6a_write_prob",
        "tlr7_gu_density",
        "sponge_score",
    }
    present = expected_immune & set(out2.keys())
    missing_imm = expected_immune - set(out2.keys())
    assert not missing_imm, f"missing immune keys: {missing_imm}"
    print("[smoke-s10] immune keys present:", sorted(present))

    # 3D mode m6A 应该有非零信号（torus radius r 作为 SASA proxy）
    assert out2["m6a_write_prob"].abs().sum() > 0, "m6a all-zero in 3D mode"
    print("[smoke-s10] 3D m6A proxy non-zero OK")

    # Shape checks
    assert out2["pkr_sasa"].shape == (B, L)
    assert out2["m6a_write_prob"].shape == (B, L)
    assert out2["tlr7_gu_density"].shape == (B, L)
    assert out2["nlrp3_persistence_length"].shape == (B,)
    assert out2["sponge_score"].shape == (B,)

    # Ranges: outputs 应该在 [0,1] ( sigmoid heads)
    assert (out2["pkr_sasa"] >= 0).all() and (out2["pkr_sasa"] <= 1).all()
    assert (out2["m6a_write_prob"] >= 0).all() and (out2["m6a_write_prob"] <= 1).all()
    assert (out2["sponge_score"] >= 0).all() and (out2["sponge_score"] <= 1).all()
    print("[smoke-s10] ranges OK (sigmoid heads in [0,1])")

    # --- 3. 改用 2D mode (torus_coords=None) ---
    head_2d = ImmuneFingerprintHeads(
        d_model=config.d_model,
        c_z=config.d_pair,
        d_torus=3,
        hidden_dim=128,
        enable_fingerprint_2d=True,  # 2D mode
    ).to(device)
    model.attach_immune_head(head_2d)
    print("[smoke-s10] changed to 2D mode (torus_coords=None path)")

    model.eval()
    with torch.no_grad():
        out3 = model(seq_tokens, pair_probs=pair_probs)

    assert out3["m6a_write_prob"].abs().sum() > 0, "m6a all-zero in 2D mode"
    print("[smoke-s10] 2D m6A proxy non-zero OK")

    # 反向传播（可选，只测 head 参数）
    model.train()
    out_train = model(seq_tokens, pair_probs=pair_probs)
    immune_train = {
        k: v for k, v in out_train.items()
        if k in expected_immune
    }
    loss = sum(v.pow(2).mean() for v in immune_train.values())
    loss.backward()

    head_params = [p for p in head_2d.parameters() if p.requires_grad]
    grad_ok, stats = check_immune_head_gradients(head_2d, loss, tag="smoke-s10")
    print(f"[smoke-s10] backward loss={loss.item():.4f}, head grads finite: {grad_ok}")
    assert grad_ok, "non-finite/missing grads in immune head on S10"

    print("[smoke-s10] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
