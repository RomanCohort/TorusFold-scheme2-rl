"""api.py — FastAPI app for the TorusFold web service.

Endpoints:

    GET  /                          → index.html (the SPA)
    GET  /api/health                → {backend, weights_loaded, device, ...}
    POST /api/predict               → {job_id}   (async, runs in a thread)
    GET  /api/jobs/{job_id}         → {status, method?, error?}
    GET  /api/result/{job_id}       → {pdb, fingerprint, method, metadata}
    GET  /api/result/{job_id}/download?format=pdb|json  → file download

Design notes (see plan):

  * Predictions run via ``asyncio.to_thread`` because torch forward blocks the
    event loop, and AF3 fallback hits the network synchronously.
  * Jobs live in an in-memory dict — v1 only, single-worker uvicorn. The CLI
    pins ``workers=1`` so the dict is process-local.
  * Sequence validation is pydantic-level: only ACGU/T/N accepted, T→U,
    lowercase folded up, length bounded by ServerConfig.
  * FastAPI/uvicorn are optional ``[server]`` deps — importing this module
    without them raises a clear message telling the user to ``pip install -e
    ".[server]"``.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from .config import ServerConfig
from .predictor import TorusFoldPredictor
from .tokenizer import clean_sequence

# Web assets live next to this package under ../web
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    sequence: str

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, v: str) -> str:
        cleaned = clean_sequence(v)
        if not cleaned:
            raise ValueError("序列为空")
        bad = set(cleaned) - set("ACGUN")
        if bad:
            raise ValueError(f"序列含非法字符 {bad}，只允许 A/C/G/U/T/N")
        return cleaned


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str          # pending | running | done | error
    method: Optional[str] = None
    error: Optional[str] = None


class JobResult(BaseModel):
    pdb: str
    fingerprint: str     # serialized fingerprint JSON (the FP object)
    method: str
    metadata: dict


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

class _JobStore:
    """Single-process job tracker. Not shared across uvicorn workers."""

    def __init__(self) -> None:
        self._jobs: Dict[str, dict] = {}

    def create(self, sequence: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        self._jobs[job_id] = {
            "status": "pending",
            "sequence": sequence,
            "result": None,
            "error": None,
        }
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job["status"] = "running"

    def set_done(self, job_id: str, result) -> None:
        job = self._jobs.get(job_id)
        if job:
            job["status"] = "done"
            job["result"] = result
            job["method"] = result.method

    def set_error(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job["status"] = "error"
            job["error"] = error


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[ServerConfig] = None) -> "FastAPI":
    """Build the FastAPI app. Loads the predictor lazily on startup."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, Response
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI/uvicorn 未安装。请运行: pip install -e \".[server]\""
        ) from exc

    config = config or ServerConfig.from_env()
    app = FastAPI(title="TorusFold Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = _JobStore()

    @app.on_event("startup")
    def _startup() -> None:
        # Predictor is lazy internally; constructing it is cheap. The actual
        # weight load happens on first predict() (or never, if af3-only).
        app.state.predictor = TorusFoldPredictor(config)
        app.state.config = config

    # ---- pages ----
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = _WEB_DIR / "index.html"
        if not html_path.exists():
            return HTMLResponse(
                "<h1>TorusFold Server</h1><p>web/index.html 未找到</p>",
                status_code=500,
            )
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    # ---- static assets (js/css) ----
    @app.get("/web/{name}")
    def web_asset(name: str):
        # Whitelist filenames — no path traversal.
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="bad asset name")
        asset = _WEB_DIR / name
        if not asset.exists():
            raise HTTPException(status_code=404, detail="asset not found")
        media = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".html": "text/html",
        }
        ct = media.get(asset.suffix, "application/octet-stream")
        return Response(asset.read_text(encoding="utf-8"), media_type=ct)

    # ---- health ----
    @app.get("/api/health")
    def health() -> dict:
        predictor: TorusFoldPredictor = app.state.predictor
        return predictor.health

    # ---- predict (async) ----
    @app.post("/api/predict", response_model=JobCreated)
    async def predict(req: PredictRequest) -> JobCreated:
        # Length bounds checked here (cleaned sequence is already normalised).
        cfg: ServerConfig = app.state.config
        if not (cfg.min_seq_len <= len(req.sequence) <= cfg.max_seq_len):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"序列长度 {len(req.sequence)} 不在 "
                    f"[{cfg.min_seq_len}, {cfg.max_seq_len}] 内"
                ),
            )

        job_id = store.create(req.sequence)
        asyncio.create_task(_run_job(app, store, job_id, req.sequence))
        return JobCreated(job_id=job_id)

    async def _run_job(app, store: _JobStore, job_id: str, sequence: str) -> None:
        predictor: TorusFoldPredictor = app.state.predictor
        store.set_running(job_id)
        try:
            # to_thread keeps the blocking torch forward off the event loop.
            result = await asyncio.to_thread(predictor.predict, sequence)
            store.set_done(job_id, result)
        except Exception as exc:
            store.set_error(job_id, f"{type(exc).__name__}: {exc}")

    # ---- job status ----
    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str) -> JobStatus:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobStatus(
            status=job["status"],
            method=job.get("method"),
            error=job.get("error"),
        )

    # ---- result ----
    @app.get("/api/result/{job_id}", response_model=JobResult)
    def job_result(job_id: str) -> JobResult:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] != "done":
            raise HTTPException(
                status_code=409,
                detail=f"job not done (status={job['status']})",
            )
        result = job["result"]
        return JobResult(
            pdb=result.pdb,
            fingerprint=result.fp_json,
            method=result.method,
            metadata=result.metadata,
        )

    @app.get("/api/result/{job_id}/download")
    def job_download(job_id: str, format: str = "pdb"):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail="job not done")
        result = job["result"]
        if format == "pdb":
            return Response(
                result.pdb,
                media_type="chemical/x-pdb",
                headers={
                    "Content-Disposition": f'attachment; filename="circrna_{job_id}.pdb"'
                },
            )
        if format == "json":
            return Response(
                result.fp_json,
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="circrna_{job_id}.json"'
                },
            )
        raise HTTPException(status_code=400, detail="format must be pdb or json")

    return app
