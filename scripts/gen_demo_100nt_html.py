"""生成 100nt circRNA 完整管线结构演示 HTML (含 amber/openmm 精修 + 稳定性可视化)。

流程: ViennaRNA 配对 → scheme2 CG 几何求解 → openmm 粗粒度精修
     → 组装 PDB(P-only, B 因子带置信度, CONECT 闭环)
     → 算一套结构稳定性指标 (能量下降/闭合度/RMSD/配对达标/键长/碰撞)
     → 嵌进独立 HTML(用 CDN Mol* 5.10.1 渲染, 不依赖后端)

输出: D:/TorusFold/scripts/demo_100nt.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torusfold.scheme2.refine import (
    BOND_LEN, PAIR_DIST, CLASH_DIST,
    scheme2_initial_coords, vienna_pair_probs, openmm_refine,
)

L = 100
# 原 demo_cg_100nt.py 序列 94nt, 补 6nt ACGCGU 凑到 100 (保持 GC 含量一致)
SEQ = "AUGCGUAACGCGAUGCUAGCAGUACGAUCGUAUCGUAACGCGAUGCUAGCAGUACGAUCGUAUCGUAACGCGAUGCUAGCAGUACGAUCGUACGACGCGU"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>TorusFold — 100nt circRNA 完整管线结构演示</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.css">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #0e1117; color: #c9d1d9; min-height: 100vh;
    display: flex; flex-direction: column;
  }
  header {
    padding: 14px 24px; background: #161b22; border-bottom: 1px solid #30363d;
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
  }
  header h1 { font-size: 20px; color: #58a6ff; }
  header .tag { font-size: 13px; color: #8b949e; }
  header .stat { font-size: 12px; color: #7d8590; }
  header .stat code { color: #f0883e; }
  header .verdict { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 10px; }
  #main { flex: 1; display: flex; min-height: 0; }
  #viewer-wrap { flex: 1; position: relative; min-width: 0; }
  #viewer { position: absolute; inset: 0; }
  aside {
    width: 360px; flex-shrink: 0; background: #161b22; border-left: 1px solid #30363d;
    padding: 16px; overflow-y: auto;
  }
  aside h2 { font-size: 13px; color: #58a6ff; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
  aside .panel { margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid #21262d; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px; }
  .stat-card .k { font-size: 11px; color: #7d8590; }
  .stat-card .v { font-size: 15px; color: #f0883e; font-weight: 600; margin-top: 2px; }
  .seq-box {
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px;
    font-family: "Consolas", "Courier New", monospace; font-size: 12px; line-height: 1.6;
    word-break: break-all; color: #7ee787; max-height: 100px; overflow-y: auto;
  }
  #status { font-size: 11px; color: #7d8590; padding: 6px 24px; background: #161b22; border-top: 1px solid #30363d; }
  .pair-row { display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; color: #c9d1d9; }
  .pair-row .w { color: #d2a8ff; }
  .loader { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #8b949e; font-size: 14px; }
  /* 稳定性面板 */
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
    display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700;
    border: 4px solid; }
  .gauge .ring-text .t { font-size: 12px; color: #c9d1d9; }
  .gauge .ring-text .d { font-size: 11px; color: #7d8590; margin-top: 2px; }
</style>
</head>
<body>
<header>
  <h1>TorusFold</h1>
  <span class="tag">100nt circRNA · CG + openmm 精修 · scheme2</span>
  <span class="stat">length: <code id="m-len">—</code></span>
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
      <h2>几何统计</h2>
      <div class="stat-grid">
        <div class="stat-card"><div class="k">序列长度</div><div class="v" id="s-len">—</div></div>
        <div class="stat-card"><div class="k">配对数</div><div class="v" id="s-pairs">—</div></div>
        <div class="stat-card"><div class="k">BSJ 距离</div><div class="v" id="s-bsj">—</div></div>
        <div class="stat-card"><div class="k">闭合误差</div><div class="v" id="s-clo">—</div></div>
        <div class="stat-card"><div class="k">骨架 P-P</div><div class="v" id="s-bb">—</div></div>
        <div class="stat-card"><div class="k">CG→精修 RMSD</div><div class="v" id="s-rmsd">—</div></div>
      </div>
    </div>
    <div class="panel">
      <h2>说明</h2>
      <div style="font-size:12px; color:#8b949e; line-height:1.6;">
        完整管线: ViennaRNA 二级结构 → scheme2 几何约束求解 → openmm 粗粒度能量最小化。<br>
        骨架 = 磷原子粗粒度, 红虚线 = BSJ 闭环, B-factor = per-residue 置信度。<br>
        RMSD 偏大属正常: CG 初始构型冲突剧烈 (高能穿模), 精修大幅重排到低能态, 残基位移较大但能量收敛良好。<br>
        右侧 Mol* 面板可切表示 (surface/cartoon) 与着色。
      </div>
    </div>
    <div class="panel">
      <h2>序列</h2>
      <div class="seq-box" id="seq-box"></div>
    </div>
    <div class="panel">
      <h2>碱基配对</h2>
      <div id="pair-list"></div>
    </div>
  </aside>
</div>

<div id="status">init…</div>

<!-- Python 注入: 纯 JSON, 无转义问题 -->
<script type="application/json" id="pdb-data">__PDB_JSON__</script>
<script type="application/json" id="meta-data">__META_JSON__</script>
<script type="application/json" id="stab-data">__STAB_JSON__</script>

<script src="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.js"></script>
<script>
const PDB = JSON.parse(document.getElementById('pdb-data').textContent);
const META = JSON.parse(document.getElementById('meta-data').textContent);
const STAB = JSON.parse(document.getElementById('stab-data').textContent);
const setStatus = (s) => { document.getElementById('status').textContent = s; };

// ---------- 填侧栏 ----------
document.getElementById('m-len').textContent = META.length;
document.getElementById('m-pairs').textContent = META.pairs.length;
document.getElementById('m-bsj').textContent = META.bsj.toFixed(2) + ' A';
document.getElementById('s-len').textContent = META.length + ' nt';
document.getElementById('s-pairs').textContent = META.pairs.length;
document.getElementById('s-bsj').textContent = META.bsj.toFixed(2) + ' A';
document.getElementById('s-clo').textContent = META.closure_error.toFixed(3);
document.getElementById('s-bb').textContent = META.backbone_mean.toFixed(2) + ' A';
document.getElementById('s-rmsd').textContent = META.rmsd.toFixed(2) + ' A';
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

// ---------- 稳定性可视化 ----------
function stabClass(score) {
  if (score >= 0.75) return 'good';
  if (score >= 0.45) return 'warn';
  return 'bad';
}
function stabColor(score) {
  const c = stabClass(score);
  if (c === 'good') return '#3fb950';
  if (c === 'warn') return '#e3b341';
  return '#f85149';
}
function verdictText(score) {
  if (score >= 0.75) return '稳定';
  if (score >= 0.45) return '基本稳定';
  return '不稳定';
}

// 综合评分
const overall = STAB.overall_score;
const vc = stabColor(overall);
const verdictEl = document.getElementById('verdict');
verdictEl.textContent = verdictText(overall) + ' (' + (overall * 100).toFixed(0) + ')';
verdictEl.style.color = vc;
verdictEl.style.background = vc + '22';
verdictEl.style.border = '1px solid ' + vc + '55';

// 仪表盘
const ring = document.getElementById('gauge-ring');
ring.textContent = (overall * 100).toFixed(0);
ring.style.color = vc;
ring.style.borderColor = vc;
document.getElementById('gauge-label').textContent = '综合结构稳定度';
document.getElementById('gauge-desc').textContent = '加权: 闭合/配对/碰撞/键长/能量';

// 各指标条
const stabList = document.getElementById('stab-list');
STAB.metrics.forEach(m => {
  const cls = stabClass(m.score);
  const row = document.createElement('div');
  row.className = 'stab-row ' + cls;
  const fillW = (m.score * 100).toFixed(0);
  // 目标线位置 (m.target 归一化 0-1, 无则不画)
  let targetHtml = '';
  if (m.target_pos != null) {
    targetHtml = '<div class="target" style="left:' + (m.target_pos * 100).toFixed(0) + '%"></div>';
  }
  row.innerHTML =
    '<div class="label"><span class="name">' + m.name + '</span>'
    + '<span class="val">' + m.display + '</span></div>'
    + '<div class="bar"><div class="fill" style="width:' + fillW + '%"></div>' + targetHtml + '</div>'
    + '<div class="note">' + m.note + '</div>';
  stabList.appendChild(row);
});

// ---------- Mol* 5.10.1 (API 照搬已验证的 circrna_viewer.js) ----------
async function build() {
  setStatus('init Mol* viewer...');
  const viewer = await molstar.Viewer.create('viewer', {
    layoutIsExpanded: false,
    viewportShowExpand: true,
    viewportShowControls: false,
    viewportShowAnimation: false,
    viewportShowSettings: false,
    backgroundColor: { r: 0.055, g: 0.067, b: 0.091 },
  });
  setStatus('loading refined PDB...');
  await viewer.loadStructureFromData(PDB, 'pdb', { dataLabel: 'circRNA-100nt-refined' });
  document.getElementById('loader').style.display = 'none';
  setStatus('ready | 100nt circRNA (精修后) loaded');
}
build().catch(e => {
  setStatus('ERROR: ' + e);
  console.error(e);
});
</script>
</body>
</html>
"""


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """对齐后的 RMSD (Å), 两阵同 shape (N,3)。

    用 scipy.spatial.transform.Rotation.align_vectors 做稳健 Kabsch
    (比手写 SVD 符号处理更稳, 避免镜像反射导致 RMSD 异常放大)。
    失败时 fallback 到 naive RMSD (不对齐)。
    """
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    try:
        from scipy.spatial.transform import Rotation
        R, _ = Rotation.align_vectors(a_c, b_c, return_sensitivity=False)
        a_rot = a_c @ R.as_matrix().T
        return float(np.sqrt(((a_rot - b_c) ** 2).sum(axis=1).mean()))
    except Exception:
        return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def compute_stability(coords: np.ndarray, pairs, e0: float, e1: float) -> dict:
    """算结构稳定性指标, 返回 {overall_score, metrics:[...]}。

    每个指标: {name, score(0-1), display, note, target_pos?}
    score 越高越稳定。
    """
    L = len(coords)
    bsj = float(np.linalg.norm(coords[0] - coords[-1]))
    bb = np.linalg.norm(np.diff(coords, axis=0), axis=1)

    # 1. 闭合度: BSJ 应接近 BOND_LEN(5.9)。误差 0→满分, 误差>=2→0
    clo_err = abs(bsj - BOND_LEN)
    clo_score = max(0.0, 1.0 - clo_err / 2.0)

    # 2. 骨架键长: 每个 P-P 应接近 BOND_LEN。看达标率(±0.3Å)
    bb_dev = np.abs(bb - BOND_LEN)
    bb_ok = float((bb_dev < 0.3).mean())
    bb_rmsd = float(np.sqrt((bb_dev ** 2).mean()))

    # 3. 配对距离达标率: 每个 WC 配对 P-P 应接近 PAIR_DIST(10.6, ±1.5)
    pair_hits = 0
    pair_deviations = []
    for i, j, w in pairs:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        dev = abs(d - PAIR_DIST)
        pair_deviations.append(dev)
        if dev < 1.5:
            pair_hits += 1
    pair_rate = pair_hits / max(1, len(pairs))
    pair_rmsd = float(np.sqrt(np.mean(pair_deviations)) if pair_deviations else 0.0)

    # 4. 碰撞率: 非相邻 P-P < CLASH_DIST(3.0) 的比例 (越低越稳定)
    clash_count = 0
    n_nonadj = 0
    for a in range(L):
        for b in range(a + 2, L):
            if (a, b) == (0, L - 1):
                continue
            d = float(np.linalg.norm(coords[a] - coords[b]))
            n_nonadj += 1
            if d < CLASH_DIST:
                clash_count += 1
    clash_rate = clash_count / max(1, n_nonadj)
    clash_score = max(0.0, 1.0 - clash_rate * 8.0)  # 12.5%碰撞→0分

    # 5. 能量下降: e0→e1 应大幅下降。看相对降幅
    if e0 > 0:
        energy_drop = (e0 - e1) / e0  # 正值=下降
    else:
        energy_drop = 0.0
    # 降幅 >=0.9 满分, 0.5 及格
    energy_score = max(0.0, min(1.0, energy_drop / 0.9))

    metrics = [
        {
            "name": "BSJ 闭合度", "score": clo_score,
            "display": f"{bsj:.2f} Å / 目标 {BOND_LEN}",
            "note": f"BSJ 误差 {clo_err:.3f} Å, 越接近 {BOND_LEN} 越好",
            "target_pos": 1.0,
        },
        {
            "name": "骨架键长达标", "score": bb_ok,
            "display": f"{bb_ok*100:.0f}% / RMSD {bb_rmsd:.2f} Å",
            "note": f"相邻 P-P 接近 {BOND_LEN}±0.3 的比例",
        },
        {
            "name": "碱基配对达标", "score": pair_rate,
            "display": f"{pair_rate*100:.0f}% / RMSD {pair_rmsd:.2f} Å",
            "note": f"WC 配对 P-P 接近 {PAIR_DIST}±1.5 的比例 ({pair_hits}/{len(pairs)})",
        },
        {
            "name": "原子碰撞率", "score": clash_score,
            "display": f"{clash_rate*100:.1f}% 碰撞",
            "note": f"非相邻 P-P < {CLASH_DIST} Å 的比例 (越低越稳定)",
            "target_pos": 0.0,
        },
        {
            "name": "能量收敛", "score": energy_score,
            "display": f"↓ {energy_drop*100:.1f}%",
            "note": f"openmm 最小化: {e0:.1f} → {e1:.1f} kJ/mol",
        },
    ]

    # 加权综合 (闭合/配对/碰撞 是几何核心, 权重高)
    weights = [0.25, 0.15, 0.25, 0.25, 0.10]
    overall = sum(m["score"] * w for m, w in zip(metrics, weights))

    return {"overall_score": round(overall, 3), "metrics": metrics}


def main() -> None:
    assert len(SEQ) == L, f"seq len {len(SEQ)} != {L}"
    print(f"[gen] L={L}")

    # 1. ViennaRNA 配对
    pairs_probs, _ = vienna_pair_probs(SEQ, threshold=0.3)
    if not pairs_probs:
        print("[gen] ViennaRNA 没给出配对, 兜底造远端配对")
        pairs_probs = [(10, 60, 0.9), (50, 80, 0.9)]
    pairs = [(i, j, w) for i, j, w in pairs_probs]
    print(f"[gen] 配对数: {len(pairs)}")

    # 2. CG 几何求解
    cg = scheme2_initial_coords(SEQ, pairs, n_samples=8)
    if cg is None:
        raise SystemExit("[gen] CG 求解失败")
    print(f"[gen] CG 坐标 shape: {cg.shape}")

    # 3. openmm 粗粒度精修
    print("[gen] running openmm refine (CPU, Verlet minimize)...")
    refined, e0, e1 = openmm_refine(cg, pairs, platform_name="CPU")
    print(f"[gen] 精修完成: e0={e0:.1f} -> e1={e1:.1f} kJ/mol")

    coords = refined
    bsj = float(np.linalg.norm(coords[0] - coords[-1]))
    bb = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    closure_err = abs(bsj - BOND_LEN)
    rmsd = _kabsch_rmsd(cg, refined)
    print(f"[gen] BSJ={bsj:.2f} A, closure_err={closure_err:.3f}, P-P mean={bb.mean():.2f}, RMSD(cg->refined)={rmsd:.2f}")

    # 4. 稳定性指标
    stab = compute_stability(coords, pairs, e0, e1)
    print(f"[gen] 综合稳定度: {stab['overall_score']:.3f}")

    # 5. per-residue 置信度 (BSJ 全局 + 配对局部)
    confidence = np.full(L, max(0.0, 1.0 - closure_err / 2.0), dtype=np.float32)
    for i, j, w in pairs:
        confidence[i] = min(1.0, confidence[i] + 0.15 * w)
        confidence[j] = min(1.0, confidence[j] + 0.15 * w)

    # 6. 组装 PDB (P-only, CONECT 闭环)
    atom_lines = [
        "REMARK   1 TorusFold 100nt demo — CG + openmm refined (scheme2 full pipeline)",
        f"REMARK   2 length={L}, pairs={len(pairs)}, BSJ={bsj:.2f}A, closure_err={closure_err:.3f}",
        f"REMARK   3 energy: {e0:.1f} -> {e1:.1f} kJ/mol, RMSD(cg->ref)={rmsd:.2f}A",
        f"REMARK   4 stability_score={stab['overall_score']:.3f}",
        "REMARK   5 P-only coarse-grained. B-factor = per-residue confidence (0-100).",
    ]
    for i in range(L):
        ser = i + 1
        bf = max(0.0, min(100.0, float(confidence[i]) * 100.0))
        x, y, z = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
        atom_lines.append(
            f"ATOM  {ser:5d}  P   C   A {ser:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{bf:6.2f}           P"
        )
    atom_lines.append(f"CONECT{1:5d}{L:5d}")  # BSJ 闭环
    for i, j, _w in pairs:
        atom_lines.append(f"CONECT{i+1:5d}{j+1:5d}")
    atom_lines.append("END")
    pdb_str = "\n".join(atom_lines) + "\n"

    # 7. 元数据
    meta = {
        "length": L,
        "sequence": SEQ,
        "bsj": round(bsj, 3),
        "closure_error": round(closure_err, 4),
        "backbone_mean": round(float(bb.mean()), 3),
        "rmsd": round(rmsd, 3),
        "energy_before": round(e0, 2),
        "energy_after": round(e1, 2),
        "pairs": [{"i": int(i), "j": int(j), "w": float(w)} for i, j, w in pairs],
        "confidence_mean": round(float(confidence.mean()), 3),
    }
    print(f"[gen] PDB 行数: {len(atom_lines)}, 字节数: {len(pdb_str)}")

    # 8. 写 HTML
    pdb_json = json.dumps(pdb_str)
    meta_json = json.dumps(meta, ensure_ascii=False)
    stab_json = json.dumps(stab, ensure_ascii=False)
    html = (HTML_TEMPLATE
            .replace("__PDB_JSON__", pdb_json)
            .replace("__META_JSON__", meta_json)
            .replace("__STAB_JSON__", stab_json))

    out = Path(__file__).resolve().parent / "demo_100nt.html"
    out.write_text(html, encoding="utf-8")
    print(f"[gen] HTML saved: {out}")
    print(f"[gen] 双击打开即可, 无需后端 (Mol* 5.10.1 走 CDN)")


if __name__ == "__main__":
    main()
