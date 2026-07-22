"""
CircRNA 3D viewer 着色方案 — 低水平版本（D, 7/17）

不依赖 Mol* 实例，只提供「torus 坐标 → 颜色数组」的纯计算接口。
Mol* viewer 实例化时直接把颜色数组喂给 components.params 颜色参数即可。

两种着色方案：
1. rainbow_torus_theta: 按 torus θ 角度做彩虹着色（环面坐标可视化）
2. motif_accessibility_heat: 按 per-nucleotide 可及性做红绿热力图（暴露度可视化）

低水平说明：
- 仅算颜色值（hex / rgb 数组），不触发 Mol* API
- Mol* 5.10.1 着色 API 见 torusfold-molstar-coloring-api 记忆（b-factor/components/{color:'x'}）
- 等 Mol* viewer 实例落地后，把 colors 数组灌进 theme 即可

依赖：TorusFoldSignals.torus_coords (N,3) 或 motif_accessibility dict
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


def _hsv_to_hex(h: float, s: float = 0.85, v: float = 0.95) -> str:
    """HSV → hex，h ∈ [0,1)（彩虹）。"""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


def rainbow_torus_theta(
    torus_coords: Optional[np.ndarray] = None,
    closure_positions: Optional[List[int]] = None,
    n: Optional[int] = None,
) -> List[str]:
    """按 torus θ 角度做彩虹着色。

    低水平：无 torus_coords 时退化为「按碱基序号线性彩虹」（dummy），
    有 torus_coords 时按 atan2(z, x) 角度映射 hue。

    Args:
        torus_coords: (N,3) 环面坐标，None 时走 dummy 线性
        closure_positions: BSJ 闭合点（高亮），可选
        n: 序列长度（torus_coords 为 None 时必需）

    Returns:
        长度 N 的 hex 颜色数组
    """
    if torus_coords is not None:
        coords = np.asarray(torus_coords, dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 3 and coords.shape[0] > 0:
            theta = np.arctan2(coords[:, 2], coords[:, 0])  # atan2(z, x)
            theta_norm = (theta - theta.min()) / max(theta.max() - theta.min(), 1e-9)
            colors = [_hsv_to_hex(float(t)) for t in theta_norm]
            # BSJ 闭合点高亮成白色
            if closure_positions:
                for p in closure_positions:
                    if 0 <= p < len(colors):
                        colors[p] = "#ffffff"
            return colors
    # dummy: 按序号线性彩虹
    length = n if n is not None else (len(closure_positions) if closure_positions else 100)
    return [_hsv_to_hex(i / max(length, 1)) for i in range(length)]


def motif_accessibility_heat(
    motif_accessibility: Optional[Dict[str, float]] = None,
    sasa: Optional[np.ndarray] = None,
    n: Optional[int] = None,
) -> List[str]:
    """按可及性做红绿热力图（高暴露=红，低暴露=绿）。

    低水平：无 motif_accessibility/sasa 时退化为全中性灰。
    有数据时按值映射：低(0)→绿 #4caf50, 高(1)→红 #f44336, 中(0.5)→黄 #ffeb3b。

    Args:
        motif_accessibility: {motif_id: accessibility[0,1]}, 可选
        sasa: (N,) per-nucleotide SASA, 可选
        n: 序列长度

    Returns:
        长度 N 的 hex 颜色数组
    """
    if sasa is not None:
        s = np.asarray(sasa, dtype=float).flatten()
        n = len(s) if n is None else n
        # 补齐长度
        if len(s) < n:
            s = np.concatenate([s, np.full(n - len(s), 0.5)])
        else:
            s = s[:n]
        colors = []
        for v in s:
            v = float(np.clip(v, 0.0, 1.0))
            # 三段映射: 绿→黄→红
            if v < 0.5:
                # 绿 #4caf50 → 黄 #ffeb3b
                t = v / 0.5
                r, g, b = int(76 + (255-76)*t), int(175 + (235-175)*t), int(80 + (59-80)*t)
            else:
                # 黄 #ffeb3b → 红 #f44336
                t = (v - 0.5) / 0.5
                r, g, b = int(255 + (244-255)*t), int(235 + (67-235)*t), int(59 + (54-59)*t)
            colors.append("#{:02x}{:02x}{:02x}".format(r, g, b))
        return colors
    # 有 motif_accessibility 但无 sasa: 用 motif 平均值粗算
    if motif_accessibility:
        avg = float(np.mean(list(motif_accessibility.values())))
        n = n if n is not None else 100
        return motif_accessibility_heat(sasa=np.full(n, avg), n=n)
    # dummy: 全中性灰
    length = n if n is not None else 100
    return ["#9e9e9e"] * length


def torus_coord_color_summary(
    torus_coords: Optional[np.ndarray],
    motif_accessibility: Optional[Dict[str, float]],
    n: int,
) -> Dict[str, List[str]]:
    """一次性产出两种着色方案，供 viewer 切换。"""
    return {
        "rainbow_torus_theta": rainbow_torus_theta(torus_coords=torus_coords, n=n),
        "motif_accessibility_heat": motif_accessibility_heat(
            motif_accessibility=motif_accessibility, n=n
        ),
    }
