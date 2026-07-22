"""100nt 合成 circRNA: 序列 -> ViennaRNA 配对 -> CG 几何求解 -> 看构型。

不跑 openmm 精修, 不跑 RL, 只看 CG 粒度原始构型。
输出: 统计 + matplotlib 3D 图 (保存 D:/TorusFold/scripts/demo_cg_100nt.png)。
"""
from __future__ import annotations

import sys
import numpy as np

# 加 src 到 path (本地跑不需要 install)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from torusfold.scheme2.refine import scheme2_initial_coords
from torusfold.scheme2.refine import vienna_pair_probs

L = 100
seq = "AUGCGUAACGCGAUGCUAGCAGUACGAUCGUAUCGUAACGCGAUGCUAGCAGUACGAUCGUAUCGUAACGCGAUGCUAGCAGUACGAUCGUACG"
# 长度校验
if len(seq) != L:
    raise SystemExit(f"seq 长度 {len(seq)} != {L}")

print(f"[demo] L={L}, seq={seq}")

# 1. ViennaRNA 配对
pairs_probs, _ = vienna_pair_probs(seq, pair_threshold=0.3)
print(f"[demo] ViennaRNA 配对数: {len(pairs_probs)}")

if not pairs_probs:
    print("[demo] ViennaRNA 没给出配对, 退出手动造一对远端配对")
    # 手动造一对远端配对让 CG 求解能跑
    pairs_probs = [(10, 60, 0.9), (50, 80, 0.9)]

pairs = [(i, j, w) for i, j, w in pairs_probs]
print(f"[demo] 配对 (i, j, weight):")
for i, j, w in pairs:
    print(f"  ({i}, {j}, {w:.2f})  环距={min(abs(i-j), L-abs(i-j))}")

# 2. CG 几何求解
cg_coords = scheme2_initial_coords(seq, pairs, n_samples=8)
if cg_coords is None:
    raise SystemExit("CG 求解失败")

print(f"[demo] CG 坐标 shape: {cg_coords.shape}, dtype={cg_coords.dtype}")

# 3. 构型统计
bsj = float(np.linalg.norm(cg_coords[0] - cg_coords[-1]))
backbone = np.linalg.norm(np.diff(cg_coords, axis=0), axis=1)
print(f"\n=== 构型统计 ===")
print(f"  BSJ 距离 (P[0]-P[L-1]): {bsj:.2f} Å")
print(f"  骨架相邻 P-P: mean={backbone.mean():.2f}, min={backbone.min():.2f}, max={backbone.max():.2f} Å")

print(f"\n  配对距离:")
for i, j, w in pairs:
    d = float(np.linalg.norm(cg_coords[i] - cg_coords[j]))
    print(f"    ({i:3d}, {j:3d})  d={d:6.2f} Å  (WC 目标 ~10.5 Å, w={w:.2f})")

# 4. 端到端统计: 非相邻 P-P 最近邻
n_nonadj = 0
min_nonadj = np.inf
for a in range(L):
    for b in range(a + 2, L):
        if (a, b) == (0, L - 1):  # BSJ 算相邻
            continue
        d = float(np.linalg.norm(cg_coords[a] - cg_coords[b]))
        min_nonadj = min(min_nonadj, d)
        n_nonadj += 1
print(f"\n  非相邻 P-P 最小距离: {min_nonadj:.2f} Å  (共 {n_nonadj} 对)")

# 5. 画图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 骨架: 按 residue index 渐变着色
    cmap = plt.cm.viridis
    for k in range(L - 1):
        ax.plot(cg_coords[k:k+2, 0], cg_coords[k:k+2, 1], cg_coords[k:k+2, 2],
                color=cmap(k / L), linewidth=1.5)
    # BSJ 闭合: 红色虚线
    ax.plot([cg_coords[-1, 0], cg_coords[0, 0]],
            [cg_coords[-1, 1], cg_coords[0, 1]],
            [cg_coords[-1, 2], cg_coords[0, 2]],
            color="red", linestyle="--", linewidth=1.5, label=f"BSJ {bsj:.1f} Å")

    # 配对: 青色连线, 端点标红
    for i, j, w in pairs:
        ax.plot([cg_coords[i, 0], cg_coords[j, 0]],
                [cg_coords[i, 1], cg_coords[j, 1]],
                [cg_coords[i, 2], cg_coords[j, 2]],
                color="cyan", linewidth=1.2, alpha=0.7)
        ax.scatter(cg_coords[i, 0], cg_coords[i, 1], cg_coords[i, 2],
                   color="red", s=30, alpha=0.8)
        ax.scatter(cg_coords[j, 0], cg_coords[j, 1], cg_coords[j, 2],
                   color="red", s=30, alpha=0.8)

    ax.set_title(f"100nt circRNA CG构型 (P atoms) | {len(pairs)} pairs | BSJ={bsj:.1f} Å")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    out = __import__("pathlib").Path(__file__).resolve().parents[0] / "demo_cg_100nt.png"
    fig.savefig(out, dpi=120)
    print(f"\n[demo] 图已保存: {out}")
except Exception as exc:
    print(f"[demo] 画图失败 (非致命): {exc!r}")
