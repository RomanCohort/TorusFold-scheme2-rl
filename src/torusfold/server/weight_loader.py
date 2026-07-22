"""weight_loader.py — load Scheme10 weights safely despite missing config.

The training script ``train_all_schemes.py`` historically does only
``torch.save(model.state_dict(), path)`` — no config snapshot. A bare
``state_dict`` cannot be loaded without rebuilding the model, and rebuilding
requires a :class:`Scheme10Config`. This module resolves that three ways:

  1. **Sidecar config** (preferred): if ``path.with_suffix('.config.json')``
     exists (written by the training-side patch in ``train_all_schemes.py``),
     parse it into ``Scheme10Config``.
  2. **Inline config** (legacy v2 style, see ``torusfold.py`` ``TorusFold.load``):
     if the checkpoint is a dict carrying a ``"config"`` key, use that.
  3. **Hard-coded default**: ``Scheme10Config()`` — every field has a default,
     so this always produces a valid model. Used when neither sidecar nor
     inline config is present (e.g. weights dumped before the patch).

Safety rails (each catches a real failure mode seen in this repo's history):

  * ``torch.load(..., weights_only=False)`` — torch ≥ 2.6 defaults to
    ``weights_only=True`` and refuses to unpickle a state_dict that contains
    anything non-tensor; our checkpoints are plain state_dicts but the default
    still trips on some torch versions, so we force ``False`` explicitly.
  * ``load_state_dict(strict=False)`` + missing-key ratio check — a v1-vs-v2
    architecture mismatch silently loads ~half the weights and produces
    garbage predictions. If >10% of expected keys are missing, we raise
    rather than ship a corrupted model.
  * dummy forward on L=8 — catches shape mismatches that ``strict=False``
  lets through (e.g. a d_model=64 checkpoint loaded into a d_model=128 model).
"""

from __future__ import annotations

import json
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..scheme10_circ_equivariant_gnn import Scheme10Config, Scheme10Model
from ..immune_fingerprint_head import ImmuneFingerprintHeads

# Reject checkpoints missing more than this fraction of expected keys.
MISSING_KEY_TOLERANCE = 0.10


def _config_from_sidecar(weights_path: Path) -> Optional[Scheme10Config]:
    """Read ``<weights>.config.json`` if it exists."""
    sidecar = weights_path.with_suffix(".config.json")
    if not sidecar.exists():
        return None
    with open(sidecar, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(Scheme10Config)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    dropped = set(raw.keys()) - known
    if dropped:
        # Don't fail on extra fields (forward-compat), just note them.
        print(f"    [weight_loader] sidecar 丢弃未知字段: {sorted(dropped)}")
    return Scheme10Config(**kwargs)


def _config_from_inline(state: Dict[str, Any]) -> Optional[Scheme10Config]:
    """Read a ``"config"`` key embedded in the checkpoint dict."""
    if "config" not in state:
        return None
    raw = state["config"]
    if isinstance(raw, Scheme10Config):
        return raw
    known = {f.name for f in dataclasses.fields(Scheme10Config)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return Scheme10Config(**kwargs)


class Scheme10WeightLoader:
    """Load a Scheme10Model + attached immune heads from a checkpoint.

    Usage::

        loader = Scheme10WeightLoader(device="cpu")
        model = loader.load(Path("models/scheme10_best.pt"))
        # model is eval()'d, immune heads attached, ready for forward().
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def load(
        self,
        weights_path: Path | str,
        *,
        attach_immune: bool = True,
    ) -> Scheme10Model:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        state = torch.load(
            str(weights_path), map_location=self.device, weights_only=False
        )

        # Resolve config: sidecar > inline > default.
        config = _config_from_sidecar(weights_path)
        config_source = "sidecar"
        if config is None:
            config = _config_from_inline(state if isinstance(state, dict) else {})
            config_source = "inline"
        if config is None:
            config = Scheme10Config()
            config_source = "default"
        print(f"    [weight_loader] config 来源={config_source}")

        # The actual state_dict may be nested under a key, or be the top-level
        # object (historical train_all_schemes dumps it bare).
        if isinstance(state, dict) and "model_state_dict" in state:
            weights_state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            weights_state = state["state_dict"]
        else:
            weights_state = state

        model = Scheme10Model(config).to(self.device)
        if attach_immune:
            immune = ImmuneFingerprintHeads(
                d_model=config.d_model,
                c_z=config.d_pair,
                enable_equivariant=True,
            ).to(self.device)
            model.attach_immune_head(immune)

        missing, unexpected = model.load_state_dict(weights_state, strict=False)
        total = len(model.state_dict())
        missing_ratio = len(missing) / total if total else 1.0
        if missing_ratio > MISSING_KEY_TOLERANCE:
            raise RuntimeError(
                f"权重与 Scheme10 架构不匹配：missing {len(missing)}/{total} "
                f"({missing_ratio:.0%} > {MISSING_KEY_TOLERANCE:.0%} 阈值)。"
                f"很可能是 Scheme10 旧版权重或别的 scheme 的 .pt。"
                f"样例 missing: {missing[:5]}"
            )
        if unexpected:
            print(f"    [weight_loader] unexpected keys (忽略): {len(unexpected)} 个")
        if missing:
            print(f"    [weight_loader] missing keys (容忍内): {len(missing)} 个")

        # Dummy forward validates shapes that strict=False let through.
        self._validate_forward(model)

        model.eval()
        return model

    def _validate_forward(self, model: Scheme10Model) -> None:
        """Run a tiny forward pass to confirm the loaded model executes."""
        with torch.no_grad():
            dummy = torch.zeros(1, 8, dtype=torch.long, device=self.device)
            pair_dummy = torch.zeros(1, 8, 8, device=self.device)
            out = model(dummy, pair_probs=pair_dummy)
        if "coords" not in out:
            raise RuntimeError("dummy forward 未产出 coords — 模型结构异常")
        coords = out["coords"]
        if coords.shape[1] != 8 or coords.shape[2] != 3:
            raise RuntimeError(
                f"dummy forward coords 形状异常: {tuple(coords.shape)}，期望 (1,8,3)"
            )
