"""生成 60nt circRNA 全原子 + amber 精修 结构演示 HTML。

完整管线: ViennaRNA 配对 → scheme2 CG 几何 → CG openmm 精修
        → 1EHZ 晶体模板全原子重建 → amber14 OL3 + OBC1 约束最小化
        → 全原子 PDB (amber14 命名, CONECT 闭环)
        → 稳定性指标 (能量/BSJ/键长/碰撞/RMSD/amber drift)
        → 独立 HTML (CDN Mol* 5.10.1, 无后端)

输出: D:/TorusFold/scripts/demo_60nt_allatom.html
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torusfold.scheme2 import predict_3d_allatom
from torusfold.scheme2.refine import BOND_LEN
from torusfold.server.export import coords_to_pdb_allatom

L = 60
# 精确 60nt, AUG 开头, GC 含量适中 (~55%)
SEQ = "AUGCGUAACGCGAUGCUAGCAGUACGAUCGUAUCGUAACGCGAUGCUAGCAGUACGAUCG"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>TorusFold — 60nt circRNA 全原子结构演示</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.css">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #0e1117; color: #c9d1d9; min-height: 100vh; display: flex; flex-direction: column; }
  header { padding: 14px 24px; background: #161b22; border-bottom: 1px solid #30363d;
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 20px; color: #58a6ff; }
  header .tag { font-size: 13px; color: #8b949e; }
  header .stat { font-size: 12px; color: #7d8590; }
  header .stat code { color: #f0883e; }
  header .verdict { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 10px; }
  #main { flex: 1; display: flex; min-height: 0; }
  #viewer-wrap { flex: 1; position: relative; min-width: 0; }
  #viewer { position: absolute; inset: 0; }
  aside { width: 360px; flex-shrink: 0; background: #161b22; border-left: 1px solid #30363d;
    padding: 16px; overflow-y: auto; }
  aside h2 { font-size: 13px; color: #58a6ff; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
  aside .panel { margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid #21262d; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px; }
  .stat-card .k { font-size: 11px; color: #7d8590; }
  .stat-card .v { font-size: 15px; color: #f0883e; font-weight: 600; margin-top: 2px; }
  .seq-box { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px;
    font-family: "Consolas", "Courier New", monospace; font-size: 12px; line-height: 1.6;
    word-break: break-all; color: #7ee787; max-height: 90px; overflow-y: auto; }
  #status { font-size: 11px; color: #7d8590; padding: 6px 24px; background: #161b22; border-top: 1px solid #30363d; }
  .pair-row { display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; color: #c9d1d9; }
  .pair-row .w { color: #d2a8ff; }
  .loader { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #8b949e; font-size: 14px; }
  .stab-row { margin-bottom: 14px; }
  .stab-row .label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
  .stab-row .label .name { color: #c9d1d9; }
  .stab-row .label .val { color: #f0883e; font-weight: 600; font-family: "Consolas", monospace; }
  .bar { height: 8px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; overflow: hidden; position: relative; }
  .bar .fill { height: 100%; border-radius: 3px; transition: width .4s; }
  .bar .target { position: absolute; top: -2px; bottom: -2px; width: 2px; background: #58a6ff; opacity: 0.6; }
  .stab-row .note { font-size: 10px; color: #7d8590; margin-top: 2px; }
  .stab-row.good .fill { background: linear-gradient(90deg, #2ea043, #3fb950); }
  .stab-row.warn .fill { background: linear-gradient(90deg, #d29922, #e3b341); }
  .stab-row.bad  .fill { background: linear-gradient(90deg, #da3633, #f85149); }
  .gauge { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
  .gauge .ring { width: 70px; height: 70px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; border: 4px solid; }
  .gauge .ring-text .t { font-size: 12px; color: #c9d1d9; }
  .gauge .ring-text .d { font-size: 11px; color: #7d8590; margin-top: 2px; }
  .energy-row { display: flex; justify-content: space-between; font-size: 11px; color: #8b949e; padding: 2px 0; }
  .energy-row .v { color: #c9d1d9; font-family: "Consolas", monospace; }
</style>
</head>
<body>
<header>
  <h1>TorusFold</h1>
  <span class="tag">60nt circRNA · 全原子 · amber14 OL3 精修</span>
  <span class="stat">atoms: <code id="m-atoms">—</code></span>
  <span class="stat">pairs: <code id="m-pairs">—</code></span>
  <span class="stat">BSJ: <code id="m-bsj">—</code></span>
  <span class="verdict" id="verdict">—</span>
</header>

<div id="main">
  <div id="viewer-wrap">
    <div id="viewer"></div>
    <div class="loader" id="loader">载入 Mol*…</div>
  </div>
  <aside>
    <div class="panel">
      <h2>结构稳定性</h2>
      <div class="gauge">
        <div class="ring" id="gauge-ring">—</div>
        <div class="ring-text">
          <div class="t" id="gauge-label">综合稳定度</div>
          <div class="d" id="gauge-desc">—</div>
        </div>
      </div>
      <div id="stab-list"></div>
    </div>
    <div class="panel">
      <h2>能量收敛</h2>
      <div id="energy-list"></div>
    </div>
    <div class="panel">
      <h2>几何统计</h2>
      <div class="stat-grid">
        <div class="stat-card"><div class="k">序列长度</div><div class="v" id="s-len">—</div></div>
        <div class="stat-card"><div class="k">重原子数</div><div class="v" id="s-heavy">—</div></div>
        <div class="stat-card"><div class="k">总原子数</div><div class="v" id="s-atoms">—</div></div>
        <div class="stat-card"><div class="k">氢原子数</div><div class="v" id="s-h">—</div></div>
        <div class="stat-card"><div class="k">BSJ 距离</div><div class="v" id="s-bsj">—</div></div>
        <div class="stat-card"><div class="k">amber P 漂移</div><div class="v" id="s-drift">—</div></div>
      </div>
    </div>
    <div class="panel">
      <h2>说明</h2>
      <div style="font-size:12px; color:#8b949e; line-height:1.6;">
        完整全原子管线: ViennaRNA → scheme2 CG → CG openmm → 1EHZ 晶体模板全原子重建 → amber14 OL3 + OBC1 约束最小化。<br>
        琥珀色糖环 + 碱基, BSJ O3'-P 化学闭环, B-factor = per-residue 置信度。<br>
        右侧 Mol* 面板: cartoon/spacefill 表示, 按残基类型着色。
      </div>
    </div>
    <div class="panel">
      <h2>序列</h2>
      <div class="seq-box" id="seq-box"></div>
    </div>
    <div class="panel">
      <h2>碱基配对 (高置信, threshold=0.5)</h2>
      <div id="pair-list"></div>
    </div>
  </aside>
</div>

<div id="status">init…</div>

<script type="application/json" id="pdb-data">__PDB_JSON__</script>
<script type="application/json" id="meta-data">__META_JSON__</script>
<script type="application/json" id="stab-data">__STAB_JSON__</script>

<script src="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.js"></script>
<script>
const PDB = JSON.parse(document.getElementById('pdb-data').textContent);
const META = JSON.parse(document.getElementById('meta-data').textContent);
const STAB = JSON.parse(document.getElementById('stab-data').textContent);
const setStatus = (s) => { document.getElementById('status').textContent = s; };

document.getElementById('m-atoms').textContent = META.n_atoms;
document.getElementById('m-pairs').textContent = META.pairs.length;
document.getElementById('m-bsj').textContent = META.bsj.toFixed(2) + ' A';
document.getElementById('s-len').textContent = META.length + ' nt';
document.getElementById('s-heavy').textContent = META.n_heavy;
document.getElementById('s-atoms').textContent = META.n_atoms;
document.getElementById('s-h').textContent = META.n_h;
document.getElementById('s-bsj').textContent = META.bsj.toFixed(2) + ' A';
document.getElementById('s-drift').textContent = META.max_p_drift.toFixed(2) + ' A';
document.getElementById('seq-box').textContent = META.sequence;

const pairList = document.getElementById('pair-list');
META.pairs.forEach(p => {
  const row = document.createElement('div');
  row.className = 'pair-row';
  const ring = Math.min(Math.abs(p.i - p.j), META.length - Math.abs(p.i - p.j));
  row.innerHTML = '<span>' + (p.i + 1) + ' <-> ' + (p.j + 1)
    + ' <span style="color:#7d8590">(环距 ' + ring + ')</span></span>'
    + '<span class="w">w=' + p.w.toFixed(2) + '</span>';
  pairList.appendChild(row);
});

function stabClass(s) { return s >= 0.75 ? 'good' : (s >= 0.45 ? 'warn' : 'bad'); }
function stabColor(s) { const c = stabClass(s); return c === 'good' ? '#3fb950' : (c === 'warn' ? '#e3b341' : '#f85149'); }
function verdictText(s) { return s >= 0.75 ? '稳定' : (s >= 0.45 ? '基本稳定' : '不稳定'); }

const overall = STAB.overall_score;
const vc = stabColor(overall);
const verdictEl = document.getElementById('verdict');
verdictEl.textContent = verdictText(overall) + ' (' + (overall * 100).toFixed(0) + ')';
verdictEl.style.color = vc; verdictEl.style.background = vc + '22'; verdictEl.style.border = '1px solid ' + vc + '55';

const ring = document.getElementById('gauge-ring');
ring.textContent = (overall * 100).toFixed(0);
ring.style.color = vc; ring.style.borderColor = vc;
document.getElementById('gauge-label').textContent = '综合结构稳定度';
document.getElementById('gauge-desc').textContent = '加权: 闭合/键长/碰撞/drift/能量';

const stabList = document.getElementById('stab-list');
STAB.metrics.forEach(m => {
  const row = document.createElement('div');
  row.className = 'stab-row ' + stabClass(m.score);
  const fillW = (m.score * 100).toFixed(0);
  let targetHtml = '';
  if (m.target_pos != null) targetHtml = '<div class="target" style="left:' + (m.target_pos * 100).toFixed(0) + '%"></div>';
  row.innerHTML =
    '<div class="label"><span class="name">' + m.name + '</span>'
    + '<span class="val">' + m.display + '</span></div>'
    + '<div class="bar"><div class="fill" style="width:' + fillW + '%"></div>' + targetHtml + '</div>'
    + '<div class="note">' + m.note + '</div>';
  stabList.appendChild(row);
});

const eList = document.getElementById('energy-list');
STAB.energy.forEach(e => {
  const row = document.createElement('div');
  row.className = 'energy-row';
  row.innerHTML = '<span>' + e.label + '</span><span class="v">' + e.display + '</span>';
  eList.appendChild(row);
});

async function build() {
  setStatus('init Mol* viewer...');
  const viewer = await molstar.Viewer.create('viewer', {
    layoutIsExpanded: false, viewportShowExpand: true, viewportShowControls: false,
    viewportShowAnimation: false, viewportShowSettings: false,
    backgroundColor: { r: 0.055, g: 0.067, b: 0.091 },
  });
  setStatus('loading all-atom PDB...');
  await viewer.loadStructureFromData(PDB, 'pdb', { dataLabel: 'circRNA-60nt-allatom' });
  document.getElementById('loader').style.display = 'none';
  setStatus('ready | 60nt circRNA (全原子, amber 精修后) loaded');
}
build().catch(e => { setStatus('ERROR: ' + e); console.error(e); });
</script>
</body>
</html>
"""


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    try:
        from scipy.spatial.transform import Rotation
        R, _ = Rotation.align_vectors(a_c, b_c, return_sensitivity=False)
        a_rot = a_c @ R.as_matrix().T
        return float(np.sqrt(((a_rot - b_c) ** 2).sum(axis=1).mean()))
    except Exception:
        return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def compute_stability_aa(coords_aa, coords_cg, pairs, e0_aa, e1_aa, amber_info) -> dict:
    """全原子版稳定性。coords_cg 用作 BSJ/闭合判据 (P 原子)。"""
    L = len(coords_cg)
    bsj = float(np.linalg.norm(coords_cg[0] - coords_cg[-1]))
    clo_err = abs(bsj - BOND_LEN)
    clo_score = max(0.0, 1.0 - clo_err / 2.0)

    # 重原子非键碰撞率 (重原子 = 非 H, amber14 重原子排在前段)
    n_heavy = int(amber_info.get("n_heavy", len(coords_aa)))
    heavy = coords_aa[:n_heavy] if n_heavy <= len(coords_aa) else coords_aa
    clash = 0
    n_nonbond = 0
    # 采样降复杂度: 全原子 O(N^2) 对 1368^2=187万, 可接受但慢; 降采样到步长 2
    step = 2 if len(heavy) > 800 else 1
    idx = list(range(0, len(heavy), step))
    sub = heavy[idx]
    # 同残基原子不算碰撞 (近似: 序号接近的跳过)
    for a in range(len(sub)):
        for b in range(a + 1, len(sub)):
            # 跳过同残基 (索引接近, 因 atom_records 按残基连续排列)
            if abs(idx[a] - idx[b]) < 20:
                continue
            d = float(np.linalg.norm(sub[a] - sub[b]))
            n_nonbond += 1
            if d < 1.6:  # 重原子碰撞阈值 (Å)
                clash += 1
    clash_rate = clash / max(1, n_nonbond)
    clash_score = max(0.0, 1.0 - clash_rate * 10.0)

    # amber P 漂移 (max_p_drift): 全原子精修中 P 原子最大位移, 越小越稳
    drift = float(amber_info.get("max_p_drift", 0.0))
    drift_score = max(0.0, 1.0 - drift / 8.0)  # 0-2Å 满分, 8Å+ 零分

    # 能量收敛: e0_aa 通常巨大 (初始穿模), e1_aa 应降到负值
    if e0_aa > 0:
        drop = (e0_aa - e1_aa) / e0_aa
    else:
        drop = 0.0
    energy_score = max(0.0, min(1.0, drop / 0.99)) if e0_aa > 1e6 else max(0.0, min(1.0, drop / 0.9))

    # 键长 (CG P-P, 全原子也用 P 残基间距)
    bb = np.linalg.norm(np.diff(coords_cg, axis=0), axis=1)
    bb_dev = np.abs(bb - BOND_LEN)
    bb_ok = float((bb_dev < 0.3).mean())

    def fmt_e(x):
        if abs(x) >= 1e6: return f"{x:.2e}"
        return f"{x:.1f}"

    metrics = [
        {"name": "BSJ 闭合度", "score": clo_score,
         "display": f"{bsj:.2f} A / 目标 {BOND_LEN}",
         "note": f"BSJ 误差 {clo_err:.3f} A", "target_pos": 1.0},
        {"name": "骨架 P-P 达标", "score": bb_ok,
         "display": f"{bb_ok*100:.0f}%",
         "note": f"相邻 P-P 在 {BOND_LEN}±0.3 的比例"},
        {"name": "重原子碰撞率", "score": clash_score,
         "display": f"{clash_rate*100:.2f}% 碰撞",
         "note": f"重原子 <1.6 A 的非键比例 (越低越稳)", "target_pos": 0.0},
        {"name": "amber P 漂移", "score": drift_score,
         "display": f"{drift:.2f} A",
         "note": "amber 精修中 P 原子最大位移, 越小构型越守恒"},
        {"name": "amber 能量收敛", "score": energy_score,
         "display": f"drop {drop*100:.1f}%",
         "note": f"{fmt_e(e0_aa)} -> {fmt_e(e1_aa)} kJ/mol"},
    ]
    weights = [0.25, 0.10, 0.25, 0.20, 0.20]
    overall = sum(m["score"] * w for m, w in zip(metrics, weights))

    return {"overall_score": round(overall, 3), "metrics": metrics}


def main() -> None:
    assert len(SEQ) == L, f"seq len {len(SEQ)} != {L}"
    print(f"[gen] L={L} (全原子 + amber 精修)", flush=True)

    t0 = time.time()
    print("[gen] running full all-atom pipeline (CPU)...", flush=True)
    out = predict_3d_allatom(SEQ, platform_name="CPU")
    dt = time.time() - t0
    print(f"[gen] pipeline done in {dt:.1f}s", flush=True)

    coords_cg = out["coords_cg"]
    coords_aa = out["coords_aa"]
    pairs = out["pairs"]
    structure = out["atoms"]
    e0_cg, e1_cg = out["e0_cg"], out["e1_cg"]
    e0_aa, e1_aa = out["e0_aa"], out["e1_aa"]
    amber_info = out["amber_info"]
    n_heavy = amber_info.get("n_heavy", len(coords_aa))
    n_atoms = amber_info.get("n_atoms", len(coords_aa))
    n_h = amber_info.get("n_h", 0)
    drift = float(amber_info.get("max_p_drift", 0.0))
    bsj = float(np.linalg.norm(coords_cg[0] - coords_cg[-1]))

    print(f"[gen] atoms: heavy={n_heavy} total={n_atoms} H={n_h}", flush=True)
    print(f"[gen] CG BSJ={bsj:.2f} A, drift={drift:.2f} A", flush=True)
    print(f"[gen] CG energy {e0_cg:.1f} -> {e1_cg:.1f}", flush=True)
    print(f"[gen] amber energy {e0_aa:.2e} -> {e1_aa:.1f}", flush=True)

    # 全原子 PDB (amber14 命名, CONECT O3'-P 闭环)
    atom_records = []
    for i, atom in enumerate(structure.atoms):
        atom_records.append({
            "serial": atom.serial + 1,
            "res_seq": atom.res_seq,
            "res_name": atom.res_name,
            "atom_name": atom.atom_name,
            "element": atom.element,
            "xyz": coords_aa[i],
        })
    closure_err = abs(bsj - BOND_LEN)
    confidence = np.full(L, max(0.0, 1.0 - closure_err / 2.0), dtype=np.float32)
    for i, j, w in pairs:
        confidence[i] = min(1.0, confidence[i] + 0.15 * w)
        confidence[j] = min(1.0, confidence[j] + 0.15 * w)
    pdb_str = coords_to_pdb_allatom(atom_records, SEQ, confidence=confidence, circular=True)

    # 稳定性
    stab = compute_stability_aa(coords_aa, coords_cg, pairs, e0_aa, e1_aa, amber_info)
    # 补能量历史
    stab["energy"] = [
        {"label": "CG openmm 精修前", "display": f"{e0_cg:.1f}" + " kJ/mol"},
        {"label": "CG openmm 精修后", "display": f"{e1_cg:.1f}" + " kJ/mol"},
        {"label": "amber 精修前 (全原子)", "display": (f"{e0_aa:.2e}" if abs(e0_aa) >= 1e6 else f"{e0_aa:.1f}") + " kJ/mol"},
        {"label": "amber 精修后 (全原子)", "display": f"{e1_aa:.1f}" + " kJ/mol"},
    ]
    print(f"[gen] 综合稳定度: {stab['overall_score']:.3f}", flush=True)

    meta = {
        "length": L, "sequence": SEQ, "bsj": round(bsj, 3),
        "n_heavy": int(n_heavy), "n_atoms": int(n_atoms), "n_h": int(n_h),
        "max_p_drift": round(drift, 3),
        "pairs": [{"i": int(i), "j": int(j), "w": float(w)} for i, j, w in pairs],
        "e0_cg": round(float(e0_cg), 2), "e1_cg": round(float(e1_cg), 2),
        "e0_aa": float(e0_aa), "e1_aa": round(float(e1_aa), 2),
    }
    print(f"[gen] PDB 字节数: {len(pdb_str)}", flush=True)

    pdb_json = json.dumps(pdb_str)
    meta_json = json.dumps(meta, ensure_ascii=False)
    stab_json = json.dumps(stab, ensure_ascii=False)
    html = (HTML_TEMPLATE.replace("__PDB_JSON__", pdb_json)
            .replace("__META_JSON__", meta_json)
            .replace("__STAB_JSON__", stab_json))

    out_path = Path(__file__).resolve().parent / "demo_60nt_allatom.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[gen] HTML saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
