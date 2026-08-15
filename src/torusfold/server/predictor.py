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

    def predict(self, sequence: str, **kwargs) -> PredictionResult:
        """Predict structure + fingerprints for one circRNA sequence."""
        self._params = kwargs  # 存储前端传来的参数
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
        # 优先用前端传来的 backend, 其次 config, 其次 env
        forced = getattr(self, '_params', {}).get('backend', '') or self.config.backend or os_backend_override()

        # Resolve backend: forced > rhofoldcirclong > scheme10 > ...
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

        if backend == "rhofoldcirclong":
            return self._predict_rhofoldcirclong(seq, t0=t0)

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
        if forced in ("rhofoldcirclong", "scheme10", "scheme2", "af3", "polygon"):
            return forced
        # 默认走 RhoFoldCircLong (我们的完整管线)
        return "rhofoldcirclong"

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

    def _predict_rhofoldcirclong(
        self,
        seq: str,
        *,
        t0: float,
    ) -> PredictionResult:
        """RhoFoldCircLong: 完整管线 (ViennaRNA + 分段 RhoFold+ + close_ends + REST2)."""
        import tempfile
        from pathlib import Path

        # 1. ViennaRNA 二级结构预测
        try:
            from ..scheme2.refine import vienna_pair_probs
            probs = vienna_pair_probs(seq)
            L = len(seq)
            ss_lines = []
            for i in range(L):
                ss_lines.append(f"{i+1} {seq[i]} {probs[i]}")
            ss_str = "\n".join(ss_lines)
        except Exception as exc:
            print(f"[rhofoldcirclong] ViennaRNA 失败: {exc}")
            ss_str = "\n".join(f"{i+1} {c} 0" for i, c in enumerate(seq))

        # 2. 运行完整管线 (使用前端传来的参数)
        params = getattr(self, '_params', {})
        with tempfile.TemporaryDirectory(prefix="torusfold_") as tmpdir:
            try:
                from ..scheme2.isrnaclong import isrnaclong_pipeline
                result = isrnaclong_pipeline(
                    sequence=seq,
                    secondary_structure=ss_str,
                    output_dir=tmpdir,
                    max_seg_len=params.get("max_seg_len", 200),
                    overlap=params.get("overlap", 20),
                    n_relax_rounds=params.get("n_relax_rounds", 1),
                    use_rl_mcts=params.get("use_rl_mcts", False),
                    use_rhofold=params.get("use_rhofold", True),
                    n_rest2_replicas=params.get("n_rest2_replicas", 4),
                    rest2_nsteps=params.get("rest2_nsteps", 50000),
                    verbose=True,
                )
            except Exception as exc:
                print(f"[rhofoldcirclong] 管线失败: {exc}")
                # fallback: 用 scheme2 CG
                return self._predict_scheme2(
                    seq, t0=t0,
                    fallback_reason=f"rhofoldcirclong: {exc!r}",
                )

            # 3. 读取最终 PDB
            pdb_path = Path(tmpdir) / "isrnaclong_final.pdb"
            if not pdb_path.exists():
                # 尝试找 merged_aa.pdb
                pdb_path = Path(tmpdir) / "cg2aa" / "merged_aa.pdb"
            if not pdb_path.exists():
                return self._predict_scheme2(
                    seq, t0=t0,
                    fallback_reason="rhofoldcirclong: PDB not found",
                )
            pdb_text = pdb_path.read_text(encoding="utf-8")

        # 4. 构造结果
        coords = result.coords_cg  # (L, 3)
        L = len(seq)
        confidence = np.full(L, max(0.0, 1.0 - result.energy_cg / 1e6), dtype=np.float32)
        closure_error = float(np.linalg.norm(coords[0] - coords[-1]))

        # Compute physical/immune signals from 3D coords
        # Try to get pair_probs from ViennaRNA for dsRNA signals
        pair_probs_for_signals = None
        try:
            from ..scheme2.refine import vienna_pair_probs
            vp = vienna_pair_probs(seq)
            pair_probs_for_signals = np.array(vp, dtype=np.float32)
        except Exception:
            pass
        signals = compute_signals(seq, coords, pair_probs=pair_probs_for_signals)
        # circDesign scores (MFE, CAI, IRES deviation)
        circdesign = compute_circdesign_signals(seq)
        signals.update(circdesign)
        # rsRNASP1 score (if binary available)
        try:
            from .rsrasp_wrapper import score_pdb_file
            rsp1_score = score_pdb_file(pdb_text)
            if rsp1_score is not None:
                signals["rsrasp1_energy"] = float(rsp1_score)
                signals["rsrasp1_energy_per_nt"] = float(rsp1_score / max(L, 1))
        except Exception:
            pass

        fp_dict = {
            "sequence": seq,
            "length": L,
            "per_residue": {},
            "scalar": {
                "energy_cg": result.energy_cg,
                "pair_rate": result.pair_rate,
                "cross_segment_ok_rate": result.cross_segment_ok_rate,
                "n_segments": result.n_segments,
                "runtime_seconds": result.runtime_seconds,
            },
            "coloring_schemes": [
                {"key": "confidence", "label": "Confidence", "type": "continuous"},
            ],
            "signals": signals,
        }

        return PredictionResult(
            sequence=seq,
            coords=coords,
            pdb=pdb_text,
            fingerprints={},
            fp_json=json.dumps(fp_dict, ensure_ascii=False),
            method="rhofoldcirclong",
            confidence=confidence,
            closure_error=closure_error,
            metadata={
                "backend": "rhofoldcirclong",
                "energy_cg": result.energy_cg,
                "pair_rate": result.pair_rate,
                "n_segments": result.n_segments,
                "elapsed_s": round(time.time() - t0, 3),
            },
        )

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

        # Compute physical/immune signals from 3D coords
        pair_probs_arr = None
        if pairs is not None:
            # pairs is list of (i, j) tuples → build (L, L) matrix
            pp = np.zeros((L, L), dtype=np.float32)
            for i, j in pairs:
                if 0 <= i < L and 0 <= j < L:
                    pp[i, j] = 1.0
                    pp[j, i] = 1.0
            pair_probs_arr = pp
        signals = compute_signals(seq, coords_cg, pair_probs=pair_probs_arr)
        circdesign = compute_circdesign_signals(seq)
        signals.update(circdesign)
        # rsRNASP1 score — only meaningful with full-atom PDB
        try:
            from .rsrasp_wrapper import score_pdb_file
            rsp1_score = score_pdb_file(pdb)
            if rsp1_score is not None:
                signals["rsrasp1_energy"] = float(rsp1_score)
                signals["rsrasp1_energy_per_nt"] = float(rsp1_score / max(L, 1))
        except Exception:
            pass

        fp_dict = fingerprints_to_json(
            coords_cg, seq,
            immune_fingerprints=immune_fp,
            confidence=confidence,
            pairs=pairs,
            signals=signals,
        )
        fp_dict["signals"] = signals
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
        # Compute physical/immune signals from 3D coords
        signals = compute_signals(seq, coords[0])
        circdesign = compute_circdesign_signals(seq)
        signals.update(circdesign)
        # rsRNASP1 — only when PDB has full-atom records (ATOM count > L)
        try:
            n_atom_lines = pdb.count("ATOM")
            if n_atom_lines > len(seq):
                from .rsrasp_wrapper import score_pdb_file
                rsp1_score = score_pdb_file(pdb)
                if rsp1_score is not None:
                    signals["rsrasp1_energy"] = float(rsp1_score)
                    signals["rsrasp1_energy_per_nt"] = float(rsp1_score / max(L, 1))
        except Exception:
            pass
        fp_dict = fingerprints_to_json(
            coords, seq, immune_fingerprints=immune, confidence=confidence * 100.0,
            signals=signals,
        )
        fp_dict["signals"] = signals
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


# ---------------------------------------------------------------------------
# Physical / immune signal computation
# ---------------------------------------------------------------------------

def compute_signals(
    seq: str,
    coords: np.ndarray,
    pair_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute derived physical and immune signals from 3D coords + pair_probs.

    Called after coordinate generation; result is embedded in the fingerprint
    JSON under ``signals`` for the frontend to render as gauges and bars.

    Args:
        seq: circRNA sequence (ACGU, length L)
        coords: (L, 3) Cartesian coordinates in Angstroms
        pair_probs: (L, L) symmetric pairing probability matrix, or None

    Returns:
        dict with scalar physical/immune signals
    """
    coords = np.asarray(coords, dtype=np.float32)
    L = len(seq)
    signals: Dict[str, Any] = {}

    # --- Physical signals ---

    # 1. Closure distance (Å)
    signals["closure_distance"] = float(np.linalg.norm(coords[0] - coords[-1]))

    # 2. Bond RMSD (Å) — reference bond length = 5.9 Å
    if L > 1:
        bond_lengths = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        signals["bond_rmsd"] = float(np.sqrt(np.mean((bond_lengths - 5.9) ** 2)))
    else:
        signals["bond_rmsd"] = 0.0

    # 3. SASA — Lee & Richards spherical-cap approximation
    R = 5.0        # water probe radius (Å)
    nuc_r = 1.7    # nucleotide radius ≈ bond_length / 2
    diff = coords[:, None, :] - coords[None, :, :]  # (L, L, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))       # (L, L)
    eff = np.maximum(dist - nuc_r, 0.0)
    cap_mask = (eff > 0.0) & (eff < 2.0 * R)
    h = np.clip(R - eff / 2.0, 0.0, R)
    occl = np.where(cap_mask, h / (2.0 * R), 0.0)
    sasa = np.clip(1.0 - np.sum(occl, axis=1), 0.0, 1.0)
    signals["sasa_mean"] = float(np.mean(sasa))

    # 4. SASA at BSJ — adaptive window w = max(4, min(L//6, 12))
    w = max(4, min(L // 6, 12))
    bsj_idx = list(range(w)) + list(range(L - w, L))
    signals["sasa_bsj"] = float(np.mean(sasa[bsj_idx]))

    # 5. BSJ 3D closure tightness — exponential decay, 1/e at bond_length
    #    (reference bond length = 5.9 Å for CG P-P backbone)
    signals["bsj_3d_closure_tightness"] = float(
        np.exp(-signals["closure_distance"] / 5.9)
    )

    # 6. dsRNA fraction (requires pair_probs)
    # bpp 概率是热力学集合平均，单个配对很少 > 0.5，旧阈值(>0.5)会让
    # dsRNA_fraction 几乎恒为 0。降到 > 0.1 才反映真实配对密度。
    PAIR_THRESHOLD = 0.1
    if pair_probs is not None:
        pp = pair_probs[0] if pair_probs.ndim == 3 else pair_probs
        try:
            upper = np.triu(pp, diagonal=1)
        except TypeError:  # numpy >= 2.0 renamed diagonal -> k
            upper = np.triu(pp, k=1)
        signals["dsRNA_fraction"] = float(np.mean(upper > PAIR_THRESHOLD))
        signals["mean_pair_prob"] = float(np.mean(upper))
        # Long-range pairs: circular distance > L/4
        idx = np.arange(L)
        cdist = np.minimum(
            np.abs(idx[:, None] - idx[None, :]),
            L - np.abs(idx[:, None] - idx[None, :]).astype(float),
        )
        lr_mask = cdist > L / 4
        lr_upper = upper[lr_mask]
        signals["long_range_pair_fraction"] = (
            float(np.mean(lr_upper > PAIR_THRESHOLD)) if len(lr_upper) > 0 else 0.0
        )
    else:
        signals["dsRNA_fraction"] = 0.0
        signals["mean_pair_prob"] = 0.0
        signals["long_range_pair_fraction"] = 0.0

    # --- Immune signals ---

    # 7. Immune motif accessibility (mean SASA at motif positions)
    immune_motifs = ["CCUCC", "UCUCC", "GUGU", "GUUG", "AUUA", "AUUU"]
    motif_acc: Dict[str, float] = {}
    for motif in immune_motifs:
        start = 0
        while True:
            pos = seq.find(motif, start)
            if pos == -1:
                break
            end = pos + len(motif)
            if end <= len(sasa):
                motif_acc[f"{motif}_{pos}"] = float(np.mean(sasa[pos:end]))
            start = pos + 1
    signals["motif_accessibility"] = motif_acc
    signals["buried_motif_count"] = sum(1 for v in motif_acc.values() if v < 0.2)

    # 8. IRES 3D accessibility
    ires_motifs = ["GCGCC", "GGGG", "UUGU", "AUGG", "CCUG", "GGAAGG"]
    ires_pos: list = []
    for m in ires_motifs:
        start = 0
        while True:
            pos = seq.find(m, start)
            if pos == -1:
                break
            ires_pos.extend(range(pos, pos + len(m)))
            start = pos + 1
    valid = [p for p in ires_pos if p < len(sasa)]
    signals["ires_3d_accessibility"] = (
        float(np.mean(sasa[valid])) if valid else float(np.mean(sasa))
    )

    return signals


# ---------------------------------------------------------------------------
# circDesign scoring (Xu et al., 2023)
# ---------------------------------------------------------------------------

# Standard E. coli codon usage table for CAI computation
_ECOLI_CODON_COUNTS: Dict[str, int] = {
    'UUU': 22046, 'UUC': 16230, 'UUA': 1370, 'UUG': 1328,
    'CUU': 15351, 'CUC': 10965, 'CUA': 2287, 'CUG': 52796,
    'AUU': 30285, 'AUC': 24590, 'AUA': 4645, 'AUG': 27197,
    'GUU': 25136, 'GUC': 15087, 'GUA': 10982, 'GUG': 25812,
    'UAU': 9232, 'UAC': 7133, 'UAA': 290, 'UAG': 126,
    'CAU': 12860, 'CAC': 9462, 'CAA': 15247, 'CAG': 28860,
    'AAU': 17988, 'AAC': 21591, 'AAA': 33529, 'AAG': 1145,
    'GAU': 32135, 'GAC': 19153, 'GAA': 39881, 'GAG': 15927,
    'UGU': 5127, 'UGC': 6326, 'UGA': 108, 'UGG': 1519,
    'CGU': 20716, 'CGC': 21586, 'CGA': 5650, 'CGG': 5588,
    'AGU': 8774, 'AGC': 15900, 'AGA': 2370, 'AGG': 1214,
    'GGU': 24758, 'GGC': 28599, 'GGA': 12549, 'GGG': 11949,
}

# Codons per amino acid (from E. coli counts)
_CODON_TABLE = {
    'F': ['UUU', 'UUC'], 'L': ['CUU', 'CUC', 'CUA', 'CUG', 'UUA', 'UUG'],
    'I': ['AUU', 'AUC', 'AUA'], 'M': ['AUG'], 'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'], 'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'], 'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'Y': ['UAU', 'UAC'], 'H': ['CAU', 'CAC'], 'Q': ['CAA', 'CAG'],
    'N': ['AAU', 'AAC'], 'K': ['AAA', 'AAG'], 'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'], 'C': ['UGU', 'UGC'], 'W': ['UGG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], 'G': ['GGU', 'GGC', 'GGA', 'GGG'],
}

# Homo sapiens codon usage counts (frequencies ×10000, high-expression genes,
# Kazusa codon usage database). Used for CAI of human-expressed vaccines —
# E. coli counts are the wrong reference for a human vaccine.
_HUMAN_CODON_COUNTS: Dict[str, int] = {
    'UUU': 1756, 'UUC': 2051, 'UUA': 76, 'UUG': 129,
    'CUU': 133, 'CUC': 195, 'CUA': 70, 'CUG': 397,
    'AUU': 157, 'AUC': 208, 'AUA': 75, 'AUG': 220,
    'GUU': 109, 'GUC': 145, 'GUA': 71, 'GUG': 285,
    'UAU': 122, 'UAC': 155, 'UAA': 7, 'UAG': 5,
    'CAU': 107, 'CAC': 151, 'CAA': 121, 'CAG': 342,
    'AAU': 170, 'AAC': 197, 'AAA': 241, 'AAG': 318,
    'GAU': 219, 'GAC': 250, 'GAA': 290, 'GAG': 396,
    'UGU': 101, 'UGC': 122, 'UGA': 16, 'UGG': 131,
    'CGU': 45, 'CGC': 105, 'CGA': 61, 'CGG': 114,
    'AGU': 120, 'AGC': 198, 'AGA': 121, 'AGG': 121,
    'GGU': 108, 'GGC': 222, 'GGA': 165, 'GGG': 157,
}

_CODON_USAGE_TABLES: Dict[str, Dict[str, int]] = {
    "ecoli": _ECOLI_CODON_COUNTS,
    "human": _HUMAN_CODON_COUNTS,
}


def _compute_cai(
    seq: str,
    codon_table: Optional[Dict] = None,
    usage_table: Optional[str] = None,
) -> float:
    """Compute Codon Adaptation Index (Sharp & Li, 1987).

    CAI = exp( (1/L) * Σ_i log w(c_i) )
    where w(c_i) = count(c_i) / max_count(synonymous), L = number of codons.

    Args:
        seq: coding sequence (ACGU)
        codon_table: amino-acid -> codon list mapping (defaults to _CODON_TABLE)
        usage_table: 'ecoli' | 'human' | None. 'human' uses the Homo sapiens
            codon usage counts (Kazusa), appropriate for human-expressed
            vaccines. None/'' falls back to E. coli (historical default).
    """
    table = codon_table or _CODON_TABLE
    counts = (
        _CODON_USAGE_TABLES.get(usage_table, _ECOLI_CODON_COUNTS)
        if usage_table
        else _ECOLI_CODON_COUNTS
    )
    L = len(seq) // 3
    if L == 0:
        return 0.0

    log_weights = []
    for i in range(L):
        codon = seq[i * 3: i * 3 + 3].upper()
        if len(codon) < 3:
            continue
        # Find which amino acid this codon encodes
        aa = None
        for amino_acid, codons in table.items():
            if codon in codons:
                aa = amino_acid
                break
        if aa is None:
            continue
        synonymous = table[aa]
        counts_syn = [counts.get(c, 0) for c in synonymous]
        max_count = max(counts_syn) if counts_syn else 1
        if max_count == 0:
            continue
        c_count = counts.get(codon, 0)
        if c_count > 0:
            log_weights.append(np.log(c_count / max_count))

    if not log_weights:
        return 0.0
    return float(np.exp(np.mean(log_weights)))


def _detect_ires_region(seq: str, ires_len: int = 0) -> Tuple[int, int]:
    """Heuristic IRES detection: find the longest GC-rich stretch.

    Returns (start, end) indices. If ires_len > 0, uses that fixed length
    from the beginning of the sequence.
    """
    if ires_len > 0:
        return (0, min(ires_len, len(seq)))

    # Heuristic: find 200-nt window with highest GC content
    L = len(seq)
    best_start, best_gc = 0, 0.0
    window = min(200, L)
    for i in range(L - window + 1):
        region = seq[i:i + window]
        gc = sum(1 for c in region if c in 'GC') / window
        if gc > best_gc:
            best_gc = gc
            best_start = i
    return (best_start, best_start + window)


def _compute_stem_loop_stability(seq: str, struct: str) -> Dict[str, Any]:
    """Compute stem-loop (hairpin) stability metrics from RNA secondary structure.

    A hairpin is defined as a closing pair (i, j) where all positions
    between i and j are unpaired dots in the dot-bracket notation.
    Multiple nested closing pairs enclosing the same dot region belong
    to the same hairpin — only the innermost pair is kept.

    Returns dict with:
      stem_loop_count: int — number of independent hairpins
      stem_loop_stability: float — mean ΔG per hairpin (kcal/mol, lower = more stable)
      stem_loop_min_stability: float — min ΔG (most stable)
      stem_loop_max_stability: float — max ΔG (least stable)
      stem_loop_stem_lengths: list — stem length per hairpin
      stem_loop_loop_lengths: list — loop size per hairpin
    """
    result: Dict[str, Any] = {}

    # Parse dot-bracket into pairs  ( ViennaRNA: 1-indexed pairs, we use 0-indexed )
    stack = []
    pairs = {}
    for i, ch in enumerate(struct):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs[j] = i   # j opens, i closes
                pairs[i] = j

    # ── Find hairpin closing pairs ──
    # A pair (i, j) with i < j closes a hairpin iff every position
    # k in (i+1 .. j-1) is a dot (unpaired).
    hairpin_closing = []  # list of (i, j) with i < j, innermost only

    for i in sorted(pairs.keys()):
        j = pairs[i]
        if i >= j:
            continue
        inner = struct[i + 1:j]
        if inner and '.' * len(inner) == inner:
            # All dots — candidate hairpin closing pair
            hairpin_closing.append((i, j))

    # Remove redundant: if (i1, j1) ⊂ (i2, j2), keep only the inner one
    hairpin_closing.sort(key=lambda p: p[0])
    filtered = []
    for ic in hairpin_closing:
        if filtered and ic[0] > filtered[-1][0] and ic[1] < filtered[-1][1]:
            # ic is inside the previous — replace
            filtered[-1] = ic
        elif filtered and ic[0] >= filtered[-1][0] and ic[1] <= filtered[-1][1]:
            # ic is same or subset — skip
            continue
        else:
            filtered.append(ic)

    # ── For each hairpin, compute stem length and loop size ──
    loops = []
    for (ci, cj) in filtered:
        loop_len = cj - ci - 1  # number of dots in loop

        # Stem length: walk outward from closing pair
        stem_len = 0
        si, sj = ci, cj
        while si in pairs and pairs[si] == sj:
            stem_len += 1
            si -= 1
            sj += 1

        loops.append({
            "stem_len": stem_len,
            "loop_len": loop_len,
            "ci": ci,
            "cj": cj,
        })

    result["stem_loop_count"] = len(loops)

    if not loops:
        result["stem_loop_stability"] = 0.0
        result["stem_loop_min_stability"] = 0.0
        result["stem_loop_max_stability"] = 0.0
        result["stem_loop_stem_lengths"] = []
        result["stem_loop_loop_lengths"] = []
        return result

    # ── Compute per-hairpin ΔG ──
    energies = []
    stem_lens = []
    loop_lens = []
    for loop in loops:
        # Extract subsequence/structure for this hairpin (stem + loop only)
        si = loop["ci"] - loop["stem_len"] + 1
        sj = loop["cj"] + loop["stem_len"] - 1
        sub_seq = seq[si:sj + 1]
        sub_struct = struct[si:sj + 1]
        try:
            import RNA
            mfe_fold = RNA.fold(sub_seq)
            # ViennaRNA >= 2.6: fold() returns just the structure string;
            # older versions returned (structure, mfe). Handle both.
            mfe_struct = mfe_fold[0] if isinstance(mfe_fold, (list, tuple)) else mfe_fold
            energy = RNA.energy_of_structure(sub_seq, mfe_struct, 0)
            energies.append(float(energy))
        except Exception:
            # Nearest-neighbor approximation:
            # ΔG ≈ stem × (-1.5 kcal/mol per bp) + loop penalty
            # Loop penalty ≈ 4.0 + 0.4 × loop_size (entropy term)
            est = -1.5 * loop["stem_len"] + 4.0 + 0.4 * loop["loop_len"]
            energies.append(est)

        stem_lens.append(loop["stem_len"])
        loop_lens.append(loop["loop_len"])

    result["stem_loop_stability"] = float(np.mean(energies))
    result["stem_loop_min_stability"] = float(min(energies))
    result["stem_loop_max_stability"] = float(max(energies))
    result["stem_loop_stem_lengths"] = stem_lens
    result["stem_loop_loop_lengths"] = loop_lens

    return result


def compute_circdesign_signals(
    seq: str,
    ires_start: int = 0,
    ires_end: int = 0,
    cds_start: int = 0,
    cds_end: int = 0,
    cai_usage_table: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute circDesign-derived scores (Xu et al., 2023).

    Three key metrics:
    1. MFE — minimum free energy from ViennaRNA folding
    2. CAI — codon adaptation index of CDS region
    3. IRES structural deviation — L2 norm of base-pairing probability
       difference between IRES-in-full-circRNA and IRES-standalone

    Args:
        seq: full circRNA sequence (ACGU)
        ires_start, ires_end: IRES region bounds (0 = auto-detect)
        cds_start, cds_end: CDS region bounds (0 = auto-detect after IRES)
        cai_usage_table: 'human' | 'ecoli' | None. For human-expressed
            vaccines pass 'human' so CAI is computed against Homo sapiens
            codon usage; None/'' keeps the historical E. coli default.

    Returns:
        dict with circDesign signals
    """
    L = len(seq)
    signals: Dict[str, Any] = {}

    # Auto-detect regions if not specified
    if ires_end <= ires_start:
        ires_start, ires_end = _detect_ires_region(seq)
    if cds_end <= cds_start:
        # CDS typically follows IRES; assume coding region is the rest
        cds_start = ires_end
        cds_end = L

    ires_seq = seq[ires_start:ires_end]
    cds_seq = seq[cds_start:cds_end]

    # 1. MFE — fold the full sequence with ViennaRNA
    mfe = 0.0
    try:
        from RNA import fold, energy_of_structure
        mfe_fold = fold(seq)
        # ViennaRNA >= 2.6: fold() returns just the structure string;
        # older versions returned (structure, mfe). Handle both.
        mfe_struct = mfe_fold[0] if isinstance(mfe_fold, (list, tuple)) else mfe_fold
        # Compute actual MFE from structure
        mfe = energy_of_structure(seq, mfe_struct, 0)
    except ImportError:
        # ViennaRNA not available — estimate from GC content
        gc = sum(1 for c in seq if c in 'GC') / max(L, 1)
        mfe = -L * gc * 0.5  # rough kcal/mol estimate
    except Exception:
        mfe = 0.0
    signals["circdesign_mfe"] = float(mfe)
    signals["circdesign_mfe_per_nt"] = float(mfe / max(L, 1))

    # 2. CAI — Codon Adaptation Index of CDS
    signals["circdesign_cai"] = _compute_cai(cds_seq, usage_table=cai_usage_table)

    # 3. IRES structural deviation — partition function based
    #
    # circDesign paper Eq.5:
    #   L_IRES = sqrt( Σ_{(i,j)∈S, i∈IRES∨j∈IRES} (P_cand(i,j) - P_ref(i,j))² )
    #
    # P_cand = base-pairing probs from partition function fold of full circRNA
    # P_ref  = base-pairing probs from partition function fold constrained to
    #           IRES-internal pairs only
    #
    # We use ViennaRNA's cofold / inverse_fold for constrained folding,
    # or fall back to comparing standalone IRES fold vs full-context fold.
    ires_dev = 0.0
    cross_talk_frac = 0.0
    try:
        import RNA

        def _bpp_to_dict(fc_obj, L_full):
            """Convert fold_compound.bpp() to a 0-indexed {(i,j): prob} dict.

            ViennaRNA < 2.6 returns a dict {(i,j): prob} (1-indexed).
            ViennaRNA >= 2.6 returns a tuple of per-position probability
            rows, row i holding probs against every other position.

            Only pairs with probability > 0.01 are kept — sub-threshold
            noise pairs otherwise dominate the L2 norm (circDesign Eq.5)
            and blow up the deviation to the clamp ceiling.
            """
            b = fc_obj.bpp()
            out = {}
            if hasattr(b, "items"):  # old API
                for (i, j), prob in b.items():
                    if i > 0 and j > i and prob > 0.01:
                        out[(i - 1, j - 1)] = prob
            else:  # new API — tuple of rows
                # row i holds P(position i pairs position j), both 1-indexed.
                # 0-indexed mapping: (i-1, j-1).
                for i in range(len(b)):
                    row = b[i]
                    for j in range(i + 1, len(row)):
                        if j > 0 and j < L_full + 1:
                            p = row[j]
                            if p > 0.01:
                                out[(i - 1, j - 1)] = p
            return out

        # --- P_cand: partition function of full sequence ---
        fc = RNA.fold_compound(seq)
        fc.pf()
        # Get base-pairing probabilities as a sparse dict
        # bp[i][j] = probability that i pairs with j (i < j)
        bp_cand = _bpp_to_dict(fc, len(seq))

        # --- P_ref: IRES standalone fold ---
        # Fold IRES alone to get reference base-pairing probs
        fc_ires = RNA.fold_compound(ires_seq)
        fc_ires.pf()
        bp_ref_local = _bpp_to_dict(fc_ires, len(ires_seq))

        # Re-index P_ref to full-sequence coordinates
        bp_ref = {(i + ires_start, j + ires_start): p
                  for (i, j), p in bp_ref_local.items()}

        # --- L2 norm deviation (circDesign Eq.5) ---
        # Sum over all (i,j) where i∈IRES or j∈IRES
        ires_indices = set(range(ires_start, ires_end))
        all_pairs = set(bp_cand.keys()) | set(bp_ref.keys())
        relevant_pairs = {(i, j) for i, j in all_pairs
                          if i in ires_indices or j in ires_indices}

        if relevant_pairs:
            l2_sq = sum(
                (bp_cand.get(pair, 0.0) - bp_ref.get(pair, 0.0)) ** 2
                for pair in relevant_pairs
            )
            ires_dev = float(np.sqrt(l2_sq))

        # --- Normalized pair-retention (complementary to raw L2) ---
        # L2 deviation grows with IRES length and is not comparable across
        # sequences. Pair-retention rate is length-invariant and directly
        # interpretable: what fraction of the IRES's own strong pairs
        # (bpp > 0.01 when folded standalone) survive intact when the IRES
        # is embedded in the full circRNA.
        #
        #   retention = |{ (i,j) ∈ P_ref : P_cand(i,j) > 0.01 }| / |P_ref|
        #   loss_rate = 1 - retention
        #
        # IRES-internal pairs only (both ends inside the IRES region) — this
        # is what "does the IRES keep its own structure" means.
        ires_internal_ref = {
            (i, j) for (i, j) in bp_ref
            if i in ires_indices and j in ires_indices
        }
        if ires_internal_ref:
            retained = sum(
                1 for (i, j) in ires_internal_ref if bp_cand.get((i, j), 0.0) > 0.01
            )
            ires_pair_retention = retained / len(ires_internal_ref)
        else:
            ires_pair_retention = 1.0
        ires_pair_loss_rate = 1.0 - ires_pair_retention

        # --- Cross-talk: IRES-CDS inter-region pairs ---
        cds_indices = set(range(cds_start, cds_end))
        cross_talk_pairs = {
            (i, j) for i, j in bp_cand
            if bp_cand[(i, j)] > 0.01  # threshold significant pairs
            and ((i in ires_indices and j in cds_indices)
                 or (i in cds_indices and j in ires_indices))
        }
        cross_talk_prob = sum(bp_cand[p] for p in cross_talk_pairs)
        # Normalize by total IRES pairing probability
        total_ires_prob = sum(
            p for (i, j), p in bp_cand.items()
            if i in ires_indices or j in ires_indices
        )
        cross_talk_frac = cross_talk_prob / max(total_ires_prob, 1e-10)

    except ImportError:
        # No ViennaRNA — fallback to heuristic
        ires_dev = 0.5
        cross_talk_frac = 0.0
        ires_pair_retention = 1.0
        ires_pair_loss_rate = 0.0
    except Exception:
        ires_dev = 0.5
        cross_talk_frac = 0.0
        ires_pair_retention = 1.0
        ires_pair_loss_rate = 0.0

    signals["circdesign_ires_deviation"] = float(max(0.0, min(2.0, ires_dev)))
    signals["ies_structural_dev"] = signals["circdesign_ires_deviation"]  # alias
    signals["ires_crosstalk_fraction"] = float(max(0.0, min(1.0, cross_talk_frac)))
    # Normalized pair-retention / loss rate (complement to raw L2):
    # fraction of the IRES's own strong pairs surviving in full-circRNA
    # context. 0 = all lost, 1 = fully retained. Loss rate = 1 - retention.
    signals["ires_pair_retention"] = float(max(0.0, min(1.0, ires_pair_retention)))
    signals["ires_pair_loss_rate"] = float(max(0.0, min(1.0, ires_pair_loss_rate)))

    # Region info
    signals["ires_length"] = ires_end - ires_start
    signals["cds_length"] = cds_end - cds_start

    # 4. Stem-loop stability
    try:
        import RNA
        mfe_fold = RNA.fold(seq)
        # ViennaRNA >= 2.6: fold() returns just the structure string;
        # older versions returned (structure, mfe). Handle both.
        mfe_struct = mfe_fold[0] if isinstance(mfe_fold, (list, tuple)) else mfe_fold
        sl_result = _compute_stem_loop_stability(seq, mfe_struct)
    except Exception:
        # Fallback: estimate from sequence only
        gc = sum(1 for c in seq if c in 'GC') / max(L, 1)
        sl_result = {
            "stem_loop_count": 0,
            "stem_loop_stability": -L * gc * 0.3,
            "stem_loop_min_stability": 0.0,
            "stem_loop_max_stability": 0.0,
            "stem_loop_stem_lengths": [],
            "stem_loop_loop_lengths": [],
        }
    signals["stem_loop_count"] = sl_result["stem_loop_count"]
    signals["stem_loop_stability"] = sl_result["stem_loop_stability"]
    signals["stem_loop_min_stability"] = sl_result["stem_loop_min_stability"]
    signals["stem_loop_max_stability"] = sl_result["stem_loop_max_stability"]
    signals["stem_loop_stem_lengths"] = sl_result["stem_loop_stem_lengths"]
    signals["stem_loop_loop_lengths"] = sl_result["stem_loop_loop_lengths"]

    return signals
