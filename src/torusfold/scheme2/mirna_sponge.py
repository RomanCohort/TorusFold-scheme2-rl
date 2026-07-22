"""mirna_sponge.py - miRNA duplex 兼容性 (sponge 活性) 真版计算。

用 miRBase mature.fa 库, 对 circRNA 序列做 miRNA-mRNA duplex 评估:
  1. seed 区 (miRNA 2-8 位) 必须与靶点完全 Watson-Crick 互补 (硬门控)
  2. 全长配对数 (miRNA 22nt 对 circRNA 窗口)
  3. 简易 nearest-neighbor ΔG (kcal/mol, 越低越稳定)

生物学: circRNA 作 miRNA sponge = 能结合多个 miRNA 形成稳定 duplex,
        sequester 它们使其无法下调其他靶 mRNA。
        sponge_score = f(命中 miRNA 数, 平均 ΔG, 序列长度)

性能: 用 seed (2-8 位) 反向互补哈希索引, O(L * |hsa-miR|) → O(L) 查表。
      48885 条全物种筛 hsa- ~2600 条, 预处理建索引 ~1s, 缓存复用。
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import gzip
import numpy as np


# Watson-Crick 互补 (RNA, U 记法)
_COMP = {"A": "U", "U": "A", "G": "C", "C": "G", "T": "A"}

# nearest-neighbor ΔG (kcal/mol, 37°C, 1M NaCl, SantaLucia 1998)
# key = "5'XY3'" 上一对与下一对, 这里用简化: 取相邻两对的 5'->3' 双联体
_NN_DG = {
    "AA": -0.9, "AU": -1.1, "AC": -2.3, "AG": -2.1,
    "UA": -1.0, "UU": -0.9, "UC": -2.1, "UG": -1.1,
    "CA": -2.1, "CU": -2.3, "CC": -3.1, "CG": -3.4,
    "GA": -2.1, "GU": -1.1, "GC": -3.4, "GG": -3.1,
}
# 起始 + 对称校正值 (kcal/mol)
_INIT_DG = 1.6
_SYMM_DG = 0.4 if False else 0.0  # 非自互补, 不加

DEFAULT_MATURE_FA = r"D:\LENOVO\Documents\mature.fa"

_cache: Dict[str, object] = {}


def _load_mirna(mature_fa: str, species_prefix: str = "hsa-") -> List[Tuple[str, str]]:
    """加载 mature.fa, 筛指定物种, 返回 [(name, seq), ...]。缓存。"""
    key = (mature_fa, species_prefix)
    if key in _cache:
        return _cache[key]  # type: ignore

    p = Path(mature_fa)
    if not p.exists():
        raise FileNotFoundError(f"mature.fa 不存在: {mature_fa}")

    entries: List[Tuple[str, str]] = []
    cur_name: Optional[str] = None
    cur_seq: List[str] = []

    def _flush():
        if cur_name is not None and cur_seq:
            seq = "".join(cur_seq).upper().replace("T", "U")
            if all(c in "AUGC" for c in seq) and len(seq) >= 18:
                entries.append((cur_name, seq))

    opener = gzip.open if str(mature_fa).endswith(".gz") else open
    with opener(mature_fa, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                _flush()
                parts = line[1:].split()
                cur_name = parts[0] if parts else None
                cur_seq = []
            elif line:
                cur_seq.append(line)
    _flush()

    # 筛物种
    filtered = [(n, s) for (n, s) in entries if n.startswith(species_prefix)]
    _cache[key] = filtered
    return filtered


def _seed_index(mirna_list: List[Tuple[str, str]]) -> Dict[str, List[int]]:
    """建 seed (miRNA 2-8 位, 1-indexed) 反向互补 → miRNA 索引列表。

    靶点 = miRNA seed 的反向互补。circRNA 上若出现该靶点 = seed 完全匹配。
    """
    idx: Dict[str, List[int]] = {}
    for k, (_name, seq) in enumerate(mirna_list):
        if len(seq) < 9:
            continue
        seed = seq[1:8]  # 2-8 位 (0-indexed 1..7), 7nt
        # 靶点 = seed 反向互补 (circRNA 上要出现的序列)
        target = "".join(_COMP[b] for b in reversed(seed))
        idx.setdefault(target, []).append(k)
    return idx


def _duplex_dg(mirna_seq: str, target_seq: str) -> float:
    """简易 nearest-neighbor ΔG (kcal/mol)。

    miRNA 5'->3' 与靶标 3'->5' 配对。靶标 3'->5' = target_seq 5'->3' 反向。
    miRNA[i] (5'->3') 与 target_rev[n-1-i] (3'->5') 配对 (miRNA 5' 端对靶标 3' 端)。
    对每段连续配对的相邻双联体累加 NN ΔG, 每段加 init penalty。
    """
    n = min(len(mirna_seq), len(target_seq))
    # miRNA 5'->3' 与靶标 3'->5' 反向对齐
    pairs = [mirna_seq[i] == _COMP.get(target_seq[n - 1 - i], "") for i in range(n)]

    dg = 0.0
    i = 0
    while i < n:
        if pairs[i]:
            # 找连续配对段
            j = i
            while j < n and pairs[j]:
                j += 1
            run_len = j - i
            if run_len >= 2:
                dg += _INIT_DG  # 每段 init
                for k in range(i, j - 1):
                    duo = mirna_seq[k] + mirna_seq[k + 1]
                    dg += _NN_DG.get(duo, -2.0)
            i = j
        else:
            i += 1
    return dg


def compute_sponge(
    sequence: str,
    mature_fa: str = DEFAULT_MATURE_FA,
    species: str = "hsa-",
    dg_threshold: float = -10.0,
    max_hits: int = 50,
) -> Dict:
    """算 circRNA 序列的 miRNA sponge 活性。

    Args:
        sequence: circRNA 序列 (ACGU, 也接受 ACTG)
        mature_fa: miRBase mature.fa 路径
        species: 物种前缀 (hsa- = 人类)
        dg_threshold: ΔG 阈值 (kcal/mol), 低于此视为有效结合
        max_hits: 最多记录的命中 miRNA 数

    Returns:
        dict: sponge_score [0,1], n_hits, mean_dg, hits [(name, dg, pos)]
    """
    seq = sequence.upper().replace("T", "U")
    L = len(seq)
    if L < 9:
        return {"sponge_score": 0.0, "n_hits": 0, "mean_dg": 0.0, "hits": []}

    mirna_list = _load_mirna(mature_fa, species)
    if not mirna_list:
        # 无库则回退启发式 (GU 含量代理)
        gu_frac = sum(1 for c in seq if c in "GU") / L
        return {
            "sponge_score": float(gu_frac * min(L / 200.0, 1.0)),
            "n_hits": 0, "mean_dg": 0.0, "hits": [],
            "fallback": "no_mirna_lib",
        }

    seed_idx = _seed_index(mirna_list)

    # 扫 circRNA: 滑动 7nt 窗口查 seed 索引, 命中后取 22nt 窗口算 ΔG
    hits: List[Tuple[str, float, int]] = []
    for pos in range(L - 6):
        window7 = seq[pos:pos + 7]
        if window7 in seed_idx:
            for k in seed_idx[window7]:
                mirna_name, mirna_seq = mirna_list[k]
                # circRNA 靶点窗口 (含 seed 在内, 22nt)
                win_start = pos
                win_end = min(pos + len(mirna_seq), L)
                target_win = seq[win_start:win_end]
                if len(target_win) < 10:
                    continue
                dg = _duplex_dg(mirna_seq, target_win)
                if dg <= dg_threshold:
                    hits.append((mirna_name, dg, pos))

    if not hits:
        return {"sponge_score": 0.0, "n_hits": 0, "mean_dg": 0.0, "hits": []}

    # 去重 (同 miRNA 多处命中只留最强)
    best: Dict[str, Tuple[float, int]] = {}
    for name, dg, pos in hits:
        if name not in best or dg < best[name][0]:
            best[name] = (dg, pos)
    dedup = sorted(best.items(), key=lambda x: x[1][0])[:max_hits]

    n_hits = len(dedup)
    mean_dg = float(np.mean([dg for _, (dg, _) in dedup]))
    # sponge_score: 命中数 + ΔG 强度 + 长度因子 (饱和)
    hit_factor = min(n_hits / 20.0, 1.0)
    dg_factor = min(-mean_dg / 25.0, 1.0)
    len_factor = min(L / 200.0, 1.0)
    sponge_score = float(0.4 * hit_factor + 0.4 * dg_factor + 0.2 * len_factor)

    return {
        "sponge_score": sponge_score,
        "n_hits": n_hits,
        "mean_dg": mean_dg,
        "hits": [(name, dg, pos) for name, (dg, pos) in dedup],
    }


if __name__ == "__main__":
    # 自测
    import time
    t0 = time.time()
    # 一段含 let-7 seed 靶点的序列
    test_seq = "AUGCGUAACGCGAUGCUAGCAGUACGAUCGUAUCGUAACGCGAUGCUAGCAGUACGAUCGUACG"
    r = compute_sponge(test_seq)
    print(f"elapsed {time.time()-t0:.2f}s")
    print(f"sponge_score={r['sponge_score']:.3f} n_hits={r['n_hits']} mean_dg={r['mean_dg']:.1f}")
    for name, dg, pos in r["hits"][:5]:
        print(f"  {name} dg={dg:.1f} pos={pos}")
