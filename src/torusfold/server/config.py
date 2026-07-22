"""config.py — ServerConfig for the TorusFold web service.

Loaded from a YAML file (``--config path``) with environment-variable
overrides. Environment vars win over the file, file wins over defaults.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class ServerConfig:
    """Runtime configuration for ``torusfold-server``.

    Backend resolution (see :class:`TorusFoldPredictor`): if
    ``TORUSFOLD_BACKEND`` is set it wins; otherwise the predictor auto-detects
    by checking ``weights_path`` exists. ``af3`` and ``polygon`` are always
    available as fallbacks behind the chosen primary backend.
    """

    # Scheme10 weights (cloud-trained .pt). Empty string → AF3 fallback.
    weights_path: str = ""

    # AF3 server URL. Official EBI endpoint; rate-limited but free.
    af3_server_url: str = "https://alphafold.ebi.ac.uk/alphafold3-api/predict"

    # CPU or cuda. CPU is the safe default — GPU needs the right torch build.
    device: str = "cpu"

    # uvicorn bind.
    host: str = "127.0.0.1"
    port: int = 8000

    # Sequence validation bounds (inclusive).
    min_seq_len: int = 10
    max_seq_len: int = 500

    # Force a backend regardless of weights presence.
    # "" = auto; "scheme10" / "scheme2" / "af3" / "polygon" = force.
    backend: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ServerConfig":
        """Load config from a YAML file, then apply env-var overrides."""
        import yaml  # lazy: pyyaml is a core dep but server users have it

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config 文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        dropped = set(raw.keys()) - known
        if dropped:
            print(f"    [config] YAML 未知键已忽略: {sorted(dropped)}")
        cfg = cls(**kwargs)
        return _apply_env(cfg)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Build config from environment variables only (no file)."""
        return _apply_env(cls())

    def resolve_weights_path(self) -> Optional[Path]:
        """Return the weights Path if set and existing, else None."""
        if not self.weights_path:
            return None
        p = Path(self.weights_path)
        return p if p.exists() else None


# Env-var override map. Keys are config field names; values are env var names.
# TORUSFOLD_BACKEND overrides the auto-detect (scheme10 if weights exist).
_ENV_MAP = {
    "weights_path": "TORUSFOLD_WEIGHTS",
    "af3_server_url": "TORUSFOLD_AF3_URL",
    "device": "TORUSFOLD_DEVICE",
    "host": "TORUSFOLD_HOST",
    "port": "TORUSFOLD_PORT",
    "min_seq_len": "TORUSFOLD_MIN_SEQ_LEN",
    "max_seq_len": "TORUSFOLD_MAX_SEQ_LEN",
    "backend": "TORUSFOLD_BACKEND",
}


def _apply_env(cfg: ServerConfig) -> ServerConfig:
    """Override dataclass fields from environment variables, in-place copy."""
    for field_name, env_name in _ENV_MAP.items():
        val = os.environ.get(env_name)
        if val is None:
            continue
        # Coerce to the field's type.
        current = getattr(cfg, field_name)
        if isinstance(current, bool):
            setattr(cfg, field_name, val.lower() in ("1", "true", "yes"))
        elif isinstance(current, int):
            setattr(cfg, field_name, int(val))
        else:
            setattr(cfg, field_name, val)
    return cfg
