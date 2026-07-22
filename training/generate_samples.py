"""generate_samples.py - 从 CircBase 生成 RL 训练样本。

每条样本: (seq, p_coords, far_pairs, near_pairs, stem_blocks)
  - ViennaRNA 全配对 + 互补性扫描补充
  - 拓扑距离分近端/远端 (环距>50 且 去自身边图距>50 = 远端)
  - CG 只用近端配对求解 (正多边形起点 + 近端约束, 远端区域未约束)
  - 起点 P 坐标 = CG 精修结果, RL 从此出发拉拢远端配对

链路 (见 docs/scheme2_rl_design.md):
  序列 -> ViennaRNA + 互补扫描 -> 配对图 -> 近端/远端分
       -> CG(只用近端) -> 起点P坐标 -> 存样本

用法:
  python generate_samples.py --n 1000            # 全量
  python generate_samples.py --n 10 --smoke      # 小样本验证
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import random
import sys
from pathlib import Path

import numpy as np

# 让 src 可 import (脚本放 training/, 源码在 src/torusfold/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfold.scheme2.refine import (  # noqa: E402
    vienna_pair_probs, scheme2_initial_coords, openmm_refine,
)
from torusfold.scheme2.pair_graph import (  # noqa: E402
    complementarity_scan, build_pair_graph, far_end_pairs,
    extract_stem_blocks, W,
)

CIRCBASE = r"C:\Users\颜子壹\data\circrna\human_hg19_circRNAs_putative_spliced_sequence.fa.gz"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "rl_samples"
MIN_LEN, MAX_LEN = 2000, 3200  # RL 甜区: ViennaRNA pf() 在这个区间能给真配对 (实测 2676nt=681对);
                                # 3354nt 起 ViennaRNA 失效给 0 配对, 上限留余量到 3200 (2026-07-22 改)


def iter_circbase(path: str, min_len: int = MIN_LEN, max_len: int = MAX_LEN):
    """流式读 CircBase fasta, yield 长度合格的序列。

    不存全列表 (CircBase max=1.8M nt, 全存触发 MemoryError)。
    超长序列累积超 max_len 即提前 skip, 单条内存可控。

    注: circBase 存 hg19 DNA 文库, 碱基用 T 不用 U; 读入时 T->U 转 RNA,
    否则 set(s)<=set("AUGC") 过滤会把含 T 的序列全毙 (2026-07-22 修)。
    """
    def _ok(s, L):
        # T->U 转 RNA, 去掉 N 等非 AUGC 字符后判长度与字符集
        s = s.replace("T", "U")
        if not (min_len <= L <= max_len):
            return None
        if set(s) <= set("AUGC"):
            return s
        return None

    with gzip.open(path, "rt", encoding="utf-8") as f:
        cur: list[str] = []
        cur_len = 0
        skip = False
        for line in f:
            if line.startswith(">"):
                if not skip and cur:
                    s = _ok("".join(cur), cur_len)
                    if s is not None:
                        yield s
                cur = []
                cur_len = 0
                skip = False
            else:
                if not skip:
                    ls = line.strip().upper()
                    cur_len += len(ls)
                    if cur_len > max_len:
                        # 超长序列提前 skip, 不再累积 (省内存)
                        skip = True
                        cur = []
                    else:
                        cur.append(ls)
        if not skip and cur:
            s = _ok("".join(cur), cur_len)
            if s is not None:
                yield s


def reservoir_sample(stream, n: int, seed: int = 42):
    """蓄水池抽样: 流式采 n 条, 不需知道总数 (适合无法全加载的流)。"""
    random.seed(seed)
    reservoir: list[str] = []
    for i, seq in enumerate(stream):
        if len(reservoir) < n:
            reservoir.append(seq)
        else:
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = seq
    return reservoir


def split_near_far(seq, vienna_pairs, scan_pairs):
    """建图 + 拓扑距离分近端/远端。

    far_end_pairs 内部已做: all_pairs 合并 + 环距>50 + 去自身边图距>50。
    near = all_pairs - far (带 weight 1.0 给 CG 求解)。
    """
    L = len(seq)
    adj = build_pair_graph(seq, vienna_pairs, scan_pairs)
    far = far_end_pairs(adj, vienna_pairs, scan_pairs)
    far_set = {(min(i, j), max(i, j)) for i, j in far}

    # all_pairs: ViennaRNA + 扫描窗口展开成逐残基配对
    all_pairs: set[tuple[int, int]] = set()
    for (i, j, _w) in vienna_pairs:
        if 0 <= i < L and 0 <= j < L and i != j:
            all_pairs.add((min(i, j), max(i, j)))
    for (i0, j0, _dg) in scan_pairs:
        for k in range(W):
            ik = (i0 + k) % L
            jk = (j0 + W - 1 - k) % L
            if ik != jk:
                all_pairs.add((min(ik, jk), max(ik, jk)))

    near_weighted = [(i, j, 1.0) for (i, j) in all_pairs if (i, j) not in far_set]
    return near_weighted, far


def cg_solve_near(seq, near_pairs, platform="CPU"):
    """CG 只用近端配对求解 -> 起点 P 坐标 (远端区域停在正多边形附近, 未约束)。"""
    init = scheme2_initial_coords(seq, near_pairs)
    if init is None:
        return None
    refined, _e0, _e1 = openmm_refine(init, near_pairs, platform)
    return refined


def generate_one(seq, platform="CPU"):
    """生成一条训练样本。

    返回 (sample_dict, skip_reason):
      sample_dict: {seq, p_coords, far_pairs, near_pairs, stem_blocks} 或 None
      skip_reason: "no_far" / "cg_fail" / None
    """
    vienna, _ = vienna_pair_probs(seq, 0.3)
    scan = complementarity_scan(seq)
    near_pairs, far_pairs = split_near_far(seq, vienna, scan)

    if not far_pairs:
        return None, "no_far"

    p_coords = cg_solve_near(seq, near_pairs, platform)
    if p_coords is None:
        return None, "cg_fail"

    stem_blocks = extract_stem_blocks(vienna, scan)
    sample = {
        "seq": seq,
        "p_coords": p_coords.astype(np.float32),
        "far_pairs": far_pairs,
        "near_pairs": near_pairs,
        "stem_blocks": stem_blocks,
    }
    return sample, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="采样条数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--platform", default="CPU")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--smoke", action="store_true", help="小样本验证 (覆盖 n=10)")
    args = p.parse_args()

    if args.smoke:
        args.n = 10

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"读 CircBase (流式): {CIRCBASE}")
    print(f"筛选: {MIN_LEN}-{MAX_LEN}nt, AUGC only, 蓄水池采样 {args.n} 条")
    sampled = reservoir_sample(iter_circbase(CIRCBASE), args.n, args.seed)
    print(f"采样完成: {len(sampled)} 条")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skip = {"no_far": 0, "cg_fail": 0}
    for idx, seq in enumerate(sampled):
        sample, reason = generate_one(seq, args.platform)
        if sample is None:
            skip[reason] += 1
            continue
        with open(out_dir / f"sample_{ok:05d}.pkl", "wb") as f:
            pickle.dump(sample, f)
        ok += 1
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[{idx+1}/{n_sample}] 成功 {ok}, 无远端 {skip['no_far']}, "
                  f"CG失败 {skip['cg_fail']}, L={len(seq)}")

    print(f"\n完成: {ok} 样本 -> {out_dir}")
    print(f"跳过: 无远端配对 {skip['no_far']}, CG失败 {skip['cg_fail']}")


if __name__ == "__main__":
    main()
