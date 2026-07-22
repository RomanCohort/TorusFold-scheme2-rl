"""cli.py — torusfold-server / torusfold-predict entry points.

Two console scripts (declared in ``pyproject.toml [project.scripts]``):

    torusfold-server [--host H --port P --config FILE --device D]
        Start the uvicorn web service (workers pinned to 1; the in-memory job
        store is not shared across processes).

    torusfold-predict SEQ [--output FILE] [--format pdb|json|both]
        Run one prediction headlessly and write the result to a file. Useful
        for scripting / batch / CI.

Both functions degrade gracefully if FastAPI/uvicorn are missing — they print
a clear install hint rather than an opaque ImportError.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ServerConfig
from .predictor import TorusFoldPredictor


def _load_config(args) -> ServerConfig:
    """Config from --config file, with env overrides applied on top."""
    if getattr(args, "config", None):
        return ServerConfig.from_yaml(args.config)
    return ServerConfig.from_env()


def server_main(argv: list[str] | None = None) -> int:
    """Entry point for ``torusfold-server``."""
    parser = argparse.ArgumentParser(
        prog="torusfold-server",
        description="Start the TorusFold web service (AlphaFold3-server style).",
    )
    parser.add_argument("--host", default=None, help="监听地址")
    parser.add_argument("--port", type=int, default=None, help="端口")
    parser.add_argument("--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("--device", default=None, help="cpu | cuda")
    args = parser.parse_args(argv)

    config = _load_config(args)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.device:
        config.device = args.device

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn 未安装。请运行: pip install -e \".[server]\"",
            file=sys.stderr,
        )
        return 1

    from .api import create_app
    app = create_app(config)

    print(f"TorusFold Server 启动中 → http://{config.host}:{config.port}")
    # Show the expected backend without touching app.state (startup hasn't
    # run yet — the predictor is built in the lifespan hook). health() is
    # cheap and lazy, so a throwaway instance is fine for display.
    display = TorusFoldPredictor(config).health
    print(f"  backend: {display.get('backend', 'af3')}")
    uvicorn.run(app, host=config.host, port=config.port, workers=1)
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    """Entry point for ``torusfold-predict SEQ``."""
    parser = argparse.ArgumentParser(
        prog="torusfold-predict",
        description="Predict one circRNA structure headlessly.",
    )
    parser.add_argument("sequence", help="circRNA 序列 (ACGU/T)")
    parser.add_argument(
        "--output", "-o", default="circrna_out",
        help="输出文件前缀（自动加 .pdb / .json 后缀）",
    )
    parser.add_argument(
        "--format", choices=["pdb", "json", "both"], default="both",
        help="输出格式",
    )
    parser.add_argument("--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("--device", default=None, help="cpu | cuda")
    args = parser.parse_args(argv)

    config = _load_config(args)
    if args.device:
        config.device = args.device

    predictor = TorusFoldPredictor(config)
    try:
        result = predictor.predict(args.sequence)
    except Exception as exc:
        print(f"预测失败: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output)
    if args.format in ("pdb", "both"):
        pdb_path = out.with_suffix(".pdb")
        pdb_path.write_text(result.pdb, encoding="utf-8")
        print(f"PDB  → {pdb_path}")
    if args.format in ("json", "both"):
        json_path = out.with_suffix(".json")
        payload = {
            "sequence": result.sequence,
            "method": result.method,
            "metadata": result.metadata,
            "closure_error": result.closure_error,
            "fingerprint": json.loads(result.fp_json),
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON → {json_path}")

    print(f"backend={result.method}  length={len(result.sequence)}  "
          f"closure_error={result.closure_error:.3f} A")
    if result.metadata.get("fallback_reason"):
        print(f"fallback: {result.metadata['fallback_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(server_main())
