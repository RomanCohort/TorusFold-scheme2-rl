"""TorusFold server — AlphaFold3-style local web service.

Exposes :class:`TorusFoldPredictor` for sequence → structure + immune
fingerprints, plus the FastAPI app and CLI entry points (``torusfold-server``,
``torusfold-predict``). Importing this package does NOT pull in FastAPI/uvicorn
— those are optional ``[server]`` deps loaded lazily by ``api``/``cli``.
"""

from __future__ import annotations

from .predictor import PredictionResult, TorusFoldPredictor

__all__ = ["PredictionResult", "TorusFoldPredictor"]
