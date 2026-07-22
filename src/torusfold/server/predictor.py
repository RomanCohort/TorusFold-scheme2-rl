"""predictor.py — TorusFoldPredictor, the unified sequence→structure entry.

This is the AlphaFold3-server-style inference core. Given a circRNA sequence,
produce a :class:`PredictionResult` carrying 3D coords, PDB string, immune
fingerprints, and provenance (which backend ran).

Backend precedence (see plan, four-tier graceful degradation):

    scheme10  →  scheme2  →  af3  →  polygon

  * ``scheme10``: the trained SO(2)×SO(2) equivariant GNN. Used only when
    weights are available. If forward raises or returns NaN, fall through.
  * ``scheme2``: zero-training geometric constraint solver + OpenMM coarse-
    grain refine (``torusfold.scheme2.predict_3d``). No weights, no network —
    real geometric structure. The *intended placeholder* during the weight-
    training window: better than a regular polygon, fully local.
  * ``af3``: AlphaFold3 server linear RNA prediction + circularization.
    Real structure, but network-dependent and rate-limited. Reached only
    if scheme2 can't run (e.g. OpenMM/ViennaRNA missing).
  * ``polygon``: regular polygon closure. No network, no weights. Last resort.

The chosen backend and any fallback reason are recorded in
``PredictionResult.method`` / ``.metadata`` — the front-end displays this so
the user is never misled about whether they are seeing a real prediction.

Key correctness notes (see plan "风险与坑"):

  * ``torch.no_grad()`` + ``model.eval()`` wraps every forward — dropout=0.1
    would otherwise inject randomness into a prediction.
  * Scheme10's ``forward`` internally calls the attached immune head
    (scheme10 L686-689), so we do NOT call ``ImmuneFingerprintHeads`` again —
    attaching once at load time is sufficient.
  * ``pair_probs`` is fed as zeros when no pairing prior is available; the
    m6A head's 2D proxy path needs the tensor present or it crashes.
  * Post-forward ``_circularize`` is a safety net: scheme10's torus head
    produces near-zero closure by construction, but if it doesn't (NaN, bad
    weights) we close it here so the PDB is always a valid ring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .config import ServerConfig
from .tokenizer import clean_sequence, tokenize
from .weight_loader import Scheme10WeightLoader
from .export import coords_to_pdb, fingerprints_to_json


@dataclass
class PredictionResult:
    """Everything a prediction yields — PDB for Mol*, fingerprints for the
    panel, provenance for honesty."""

    sequence: str
    coords: np.ndarray              # (L, 3) Cartesian Å
    pdb: str                        # PDB string (P-only, CONECT-closed)
    fingerprints: Dict[str, Any]   # raw immune-fingerprint tensors
    fp_json: str                    # serialized fingerprint JSON for the FE
    method: str                     # "scheme10" | "af3" | "polygon"
    confidence: np.ndarray          # (L,) in [0,1]
    closure_error: float            # ‖coords[0] - coords[-1] - bond_length‖
    metadata: Dict[str, Any] = field(default_factory=dict)


class TorusFoldPredictor:
    """Unified predictor with three-tier backend fallback.

    Instantiate once (e.g. in the FastAPI startup hook) and reuse — model
    loading is expensive. Thread-safe for inference calls because
    ``model.eval()`` + ``no_grad`` makes forward stateless.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = config.device

        self._scheme10: Optional[Any] = None      # lazy-loaded Scheme10Model
        self._scheme10_tried: bool = False
        self._af3: Optional[Any] = None           # lazy AlphaFold3Initializer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, sequence: str) -> PredictionResult:
        """Predict structure + fingerprints for one circRNA sequence."""
        seq = clean_sequence(sequence)
        if len(seq) < self.config.min_seq_len:
            raise ValueError(
                f"序列过短: {len(seq)} < 最小 {self.config.min_seq_len}"
            )
        if len(seq) > self.config.max_seq_len:
            raise ValueError(
                f"序列过长: {len(seq)} > 最大 {self.config.max_seq_len}"
            )

        t0 = time.time()
        forced = self.config.backend or os_backend_override()

        # Resolve backend: forced > scheme10-if-weights > af3 > polygon.
        backend = self._resolve_backend(forced)

        if backend == "scheme10":
            try:
                return self._predict_scheme10(seq, elapsed=time.time() - t0)
            except Exception as exc:
                # Scheme10 failed → degrade to scheme2 (local geometric placeholder).
                # Record the reason.
                print(f"[predictor] scheme10 失败 → 降级 scheme2: {exc!r}")
                return self._predict_scheme2(
                    seq, fallback_reason=f"scheme10: {exc!r}", t0=t0
                )

        if backend == "scheme2":
            return self._predict_scheme2(seq, t0=t0)

        if backend == "af3":
            return self._predict_af3(seq, t0=t0)

        return self._predict_polygon(seq, t0=t0)

    @property
    def health(self) -> Dict[str, Any]:
        """Status snapshot for /api/health."""
        weights = self.config.resolve_weights_path()
        backend = self._resolve_backend(self.config.backend or os_backend_override())
        return {
            "backend": backend,
            "weights_loaded": self._scheme10 is not None,
            "weights_path": str(weights) if weights else None,
            "device": self.device,
        }

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self, forced: str) -> str:
        if forced in ("scheme10", "scheme2", "af3", "polygon"):
            return forced
        # Auto: scheme10 if weights resolvable, else scheme2 (local placeholder).
        if self._can_load_scheme10():
            return "scheme10"
        return "scheme2"

    def _can_load_scheme10(self) -> bool:
        """True if weights exist and load successfully (cached)."""
        if self._scheme10_tried:
            return self._scheme10 is not None
        self._scheme10_tried = True
        weights = self.config.resolve_weights_path()
        if weights is None:
            return False
        try:
            loader = Scheme10WeightLoader(device=self.device)
            self._scheme10 = loader.load(weights)
            print(f"[predictor] Scheme10 权重加载成功: {weights}")
            return True
        except Exception as exc:
            print(f"[predictor] Scheme10 权重加载失败，将走 AF3: {exc!r}")
            self._scheme10 = None
            return False

    # ------------------------------------------------------------------
    # Backend: Scheme10
    # ------------------------------------------------------------------

    def _predict_scheme10(
        self,
        seq: str,
        *,
        elapsed: float = 0.0,
    ) -> PredictionResult:
        if self._scheme10 is None:
            # Should have been lazy-loaded by _can_load_scheme10.
            if not self._can_load_scheme10():
                raise RuntimeError("Scheme10 weights unavailable")
        model = self._scheme10
        if model is None:  # defensive — mypy can't see the lazy load
            raise RuntimeError("Scheme10 model None after load attempt")

        tokens = tokenize(seq).to(self.device)
        B, L = tokens.shape
        pair_probs = torch.zeros(B, L, L, device=self.device)

        model.eval()
        with torch.no_grad():
            out = model(tokens, pair_probs=pair_probs)

        coords = out["coords"].detach().cpu().numpy()
        if np.isnan(coords).any() or np.isinf(coords).any():
            raise RuntimeError("Scheme10 forward 产出 NaN/Inf coords")

        immune: Dict[str, Any] = {}
        for k, v in out.items():
            if k in (
                "coords", "torus_coords", "pair_repr", "sequence_repr",
                "structure_method", "pair_probs", "closure_dist",
            ):
                continue
            if torch.is_tensor(v):
                immune[k] = v.detach().cpu().numpy()

        # Confidence: scheme10 has no explicit pLDDT head; synthesise a
        # soft score from closure residual (smaller closure → higher conf).
        closure_raw = float(
            np.linalg.norm(coords[0, 0] - coords[0, -1])
        )
        confidence = np.full(
            L, max(0.0, 1.0 - closure_raw / 10.0), dtype=np.float32
        )

        return self._assemble(
            seq=seq,
            coords=coords,            # (1, L, 3)
            immune=immune,
            confidence=confidence,
            method="scheme10",
            fallback_reason=None,
            t0_offset=elapsed,
        )

    # ------------------------------------------------------------------
    # Backend: Scheme2 (zero-training geometric constraints + OpenMM refine)
    # ------------------------------------------------------------------

    def _predict_scheme2(
        self,
        seq: str,
        *,
        t0: float,
        fallback_reason: Optional[str] = None,
    ) -> PredictionResult:
        """scheme2 后端: 优先走全原子 (CG 重建 + amber14 OL3 精修),
        任何一步失败 fallback 到 CG P-only 粗粒度。"""
        try:
            from ..scheme2 import predict_3d_allatom, predict_3d as cg_predict_3d
            from ..scheme2.allatom_reconstruct import get_atom_xyzs
            from ..server.export import coords_to_pdb_allatom, fingerprints_to_json
        except ImportError as exc:
            print(f"[predictor] scheme2 模块缺失 → 降级 af3: {exc!r}")
            return self._predict_af3(
                seq, t0=t0,
                fallback_reason=(fallback_reason or "") + f" | scheme2 import: {exc!r}",
            )

        # --- 优先: 全原子路径 ---
        try:
            out = predict_3d_allatom(seq, platform_name="CPU")
            return self._assemble_allatom(seq, out, t0=t0, fallback_reason=fallback_reason)
        except Exception as exc:
            # 全原子失败 → fallback 到 CG P-only
            print(f"[predictor] scheme2 全原子失败 → fallback CG: {exc!r}")
            cg_fallback_reason = (fallback_reason or "") + f" | allatom: {exc!r}"
            try:
                cg_out = cg_predict_3d(seq, platform_name="CPU")
                return self._assemble_cg_fallback(seq, cg_out, t0=t0, fallback_reason=cg_fallback_reason)
            except Exception as cg_exc:
                print(f"[predictor] scheme2 CG fallback 也失败 → 降级 af3: {cg_exc!r}")
                return self._predict_af3(
                    seq, t0=t0,
                    fallback_reason=(cg_fallback_reason) + f" | cg: {cg_exc!r}",
                )

    def _assemble_allatom(
        self, seq: str, out: dict, *, t0: float, fallback_reason: Optional[str],
    ) -> PredictionResult:
        """全原子路径的 PredictionResult 构造。"""
        from ..server.export import coords_to_pdb_allatom, fingerprints_to_json

        structure = out["atoms"]
        coords_aa = out["coords_aa"]  # (N, 3) Å 含 H
        coords_cg = out["coords_cg"]  # (L, 3) Å
        pairs = out["pairs"]
        L = len(seq)

        # 构造 atom_records (per-atom dict list)
        atom_records = []
        for i, atom in enumerate(structure.atoms):
            atom_records.append({
                "serial": atom.serial + 1,  # PDB 1-based
                "res_seq": atom.res_seq,
                "res_name": atom.res_name,
                "atom_name": atom.atom_name,
                "element": atom.element,
                "xyz": coords_aa[i],
            })

        # confidence: BSJ-based (CG)
        bsj_after = float(np.linalg.norm(coords_cg[0] - coords_cg[-1]))
        confidence = np.full(
            L, max(0.0, 1.0 - abs(bsj_after - 5.9) / 2.0), dtype=np.float32
        )

        pdb = coords_to_pdb_allatom(atom_records, seq, confidence=confidence, circular=True)

        # 几何启发式免疫指纹 (scheme2 全原子路径)。
        # scheme10 训练头云端权重没下来前, 用全原子坐标+序列+ViennaRNA 配对
        # 算 10 个指标, 让前端 Coloring Scheme 下拉恢复 pkr/m6a/tlr7/rigi/nlrp3/sponge。
        # 启发式挂了不能让整个预测挂 → 退回 {} 走原无指标路径, warning 记进 metadata。
        immune_warn: Optional[str] = None
        try:
            from ..scheme2.immune_heuristic import compute_immune_fingerprints
            immune_fp = compute_immune_fingerprints(
                coords_aa, structure, pairs, seq,
            )
        except Exception as exc:  # noqa: BLE001 - 启发式兜底, 任何错都退回空
            immune_fp = {}
            immune_warn = f"immune_heuristic failed: {type(exc).__name__}: {exc}"

        fp_dict = fingerprints_to_json(
            coords_cg, seq,
            immune_fingerprints=immune_fp,
            confidence=confidence,
            pairs=pairs,
        )
        fp_json = json.dumps(fp_dict, ensure_ascii=False)

        closure_error = float(abs(bsj_after - 5.9))
        metadata = {
            "backend": "scheme2",
            "allatom": True,
            "n_atoms": out["amber_info"].get("n_atoms"),
            "n_h": out["amber_info"].get("n_h"),
            "max_p_drift": out["amber_info"].get("max_p_drift"),
            "e0_aa": out["e0_aa"],
            "e1_aa": out["e1_aa"],
            "elapsed_s": round(time.time() - t0, 3),
            "fallback_to_cg": False,
        }
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason
        if immune_warn:
            metadata["immune_warning"] = immune_warn

        return PredictionResult(
            sequence=seq,
            coords=coords_cg,  # CG 坐标给前端 (mol* 画全原子, 但 confidence 用 CG)
            pdb=pdb,
            fingerprints=immune_fp,
            fp_json=fp_json,
            method="scheme2",
            confidence=confidence,
            closure_error=closure_error,
            metadata=metadata,
        )

    def _assemble_cg_fallback(
        self, seq: str, out: dict, *, t0: float, fallback_reason: Optional[str],
    ) -> PredictionResult:
        """CG fallback 路径 (P-only)。保留原 scheme2 行为, 但标记 fallback_to_cg=True。"""
        coords = np.asarray(out["coords"], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise RuntimeError(f"CG fallback coords 形状异常: {coords.shape}")
        coords = coords[np.newaxis]
        L = coords.shape[1]
        bsj_after = float(out.get("bsj_after", 0.0))
        confidence = np.full(
            L, max(0.0, 1.0 - abs(bsj_after - 5.9) / 2.0), dtype=np.float32
        )
        # CG fallback 没有 pairs 数据 → 不传 pairs (fingerprints 里没有 stem/loop)
        result = self._assemble(
            seq=seq, coords=coords, immune={}, confidence=confidence,
            method="scheme2", fallback_reason=fallback_reason,
            t0_offset=time.time() - t0,
        )
        result.metadata["fallback_to_cg"] = True
        return result

    # ------------------------------------------------------------------
    # Backend: AlphaFold3 server (real structure, network-dependent)
    # ------------------------------------------------------------------

    def _predict_af3(
        self,
        seq: str,
        *,
        fallback_reason: Optional[str] = None,
        t0: float,
    ) -> PredictionResult:
        initializer = self._get_af3()
        try:
            coords = initializer.predict(seq)   # (L, 3) already circularized
        except Exception as exc:
            print(f"[predictor] AF3 失败 → 降级 polygon: {exc!r}")
            return self._predict_polygon(
                seq, t0=t0,
                fallback_reason=(fallback_reason or "") + f" | af3: {exc!r}",
            )

        coords = coords.astype(np.float32)
        if coords.ndim == 2:
            coords = coords[np.newaxis]          # (1, L, 3)
        L = coords.shape[1]

        # AF3 has no per-residue confidence in this path; uniform mid-high.
        confidence = np.full(L, 0.7, dtype=np.float32)

        # No immune fingerprints from AF3 — leave empty dict; the FE panel
        # will show "no data" rather than fabricated values.
        method = "af3" if fallback_reason is None else "af3"
        return self._assemble(
            seq=seq,
            coords=coords,
            immune={},
            confidence=confidence,
            method=method,
            fallback_reason=fallback_reason,
            t0_offset=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Backend: regular polygon (no weights, no network)
    # ------------------------------------------------------------------

    def _predict_polygon(
        self,
        seq: str,
        *,
        t0: float,
        fallback_reason: Optional[str] = None,
    ) -> PredictionResult:
        initializer = self._get_af3()
        coords = initializer._fallback_polygon(len(seq)).astype(np.float32)
        coords = coords[np.newaxis]            # (1, L, 3)
        L = coords.shape[1]
        confidence = np.full(L, 0.2, dtype=np.float32)
        reason = fallback_reason or "no weights and af3 unavailable"
        return self._assemble(
            seq=seq,
            coords=coords,
            immune={},
            confidence=confidence,
            method="polygon",
            fallback_reason=reason,
            t0_offset=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Shared assembly
    # ------------------------------------------------------------------

    def _assemble(
        self,
        *,
        seq: str,
        coords: np.ndarray,        # (1, L, 3) or (L, 3)
        immune: Dict[str, Any],
        confidence: np.ndarray,    # (L,)
        method: str,
        fallback_reason: Optional[str],
        t0_offset: float,
    ) -> PredictionResult:
        # Safety: ensure BSJ closure regardless of backend.
        # _circularize wants (L, 3); we keep batch dim through to PDB export.
        initializer = self._get_af3()
        had_batch = coords.ndim == 3
        coords_2d = coords[0] if had_batch else coords
        coords_2d = initializer._circularize(coords_2d)
        coords = coords_2d[np.newaxis]          # (1, L, 3)

        bond_length = 5.9
        closure_error = float(
            abs(np.linalg.norm(coords[0, 0] - coords[0, -1]) - bond_length)
        )

        pdb = coords_to_pdb(
            coords, seq, confidence=confidence * 100.0, circular=True
        )
        fp_dict = fingerprints_to_json(
            coords, seq, immune_fingerprints=immune, confidence=confidence * 100.0
        )
        fp_json = json.dumps(fp_dict, ensure_ascii=False)

        metadata: Dict[str, Any] = {
            "backend": method,
            "elapsed_s": round(t0_offset, 3),
        }
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason

        return PredictionResult(
            sequence=seq,
            coords=coords[0],
            pdb=pdb,
            fingerprints=immune,
            fp_json=fp_json,
            method=method,
            confidence=confidence,
            closure_error=closure_error,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Lazy AF3 initializer (shared by af3 + polygon + circularize)
    # ------------------------------------------------------------------

    def _get_af3(self):
        if self._af3 is None:
            from ..alphafold3_init import AlphaFold3Initializer
            self._af3 = AlphaFold3Initializer()
        return self._af3


def os_backend_override() -> str:
    """Read TORUSFOLD_BACKEND env if set to a known value, else ''."""
    import os
    val = os.environ.get("TORUSFOLD_BACKEND", "").strip().lower()
    return val if val in ("scheme10", "scheme2", "af3", "polygon") else ""
