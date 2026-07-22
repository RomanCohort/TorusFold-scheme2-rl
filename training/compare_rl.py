"""compare_rl.py - RL 远端配对优化的端到端对比实验。

对比 (同一条 CircBase 序列):
  baseline: predict_3d_allatom(seq, use_rl=False)         ViennaRNA->CG->amber
  +RL     : predict_3d_allatom(seq, use_rl=True, ...)    ViennaRNA->CG->RL->amber

关键诚实性 (见 docs/scheme2_rl_design.md / 记忆 torusfold-scheme2-rl-positioning):
  - RL 的 compute_reward 是搜索信号, 不是评估结论。reward 高 != 结构好。
  - 最终结构评估走独立物理指标: amber 能量 e1_aa (负越多越合理)。
  - improvement (RL reward 提升) 只反映 RL 跑没跑动, 不当结构好坏结论。
  - +RL 后必回 amber 收尾, 最终结构由 amber 定。

指标 (每条序列):
  L, n_far_pairs (pair_graph 扫出的远端配对数), rl_improvement (reward 提升),
  baseline_e1_aa, rl_e1_aa, delta_e1 (rl - baseline),
  baseline_fallback / rl_fallback (amber 是否失败回退),
  n_coding_pinned (RL 路径 amber 钉死数)。

用法:
  python compare_rl.py --fa D:/IGEM集成方案/data/circrna/circbase_seqs.fa.gz \
      --min-len 1200 --max-len 2200 --limit 5 --policy models/rl/ppo_gnn_final.pth
  python compare_rl.py --smoke   # 短序列少量快测, 验证管线
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfold.scheme2 import predict_3d_allatom  # noqa: E402
from torusfold.scheme2.pair_graph import (  # noqa: E402
    build_full_pair_graph, parse_case_annotation,
)
from torusfold.scheme2.refine import vienna_pair_probs  # noqa: E402


def iter_fasta(fa_path: Path) -> Iterator[Tuple[str, str]]:
    """读 fasta (支持 .gz)。yield (header, seq)。序列转大写 ACGU (T->U)。"""
    opener = gzip.open if fa_path.suffix == ".gz" else open
    with opener(fa_path, "rt", encoding="utf-8", errors="replace") as f:
        header, seq_lines = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header, seq_lines = line[1:], []
            elif line:
                seq_lines.append(line)
        if header is not None:
            yield header, "".join(seq_lines)


def clean_seq(raw: str) -> str:
    """fasta 序列 -> ACGU。非 ACGUT 字符丢弃, T->U。"""
    s = raw.upper().replace("T", "U")
    return "".join(c for c in s if c in "ACGU")


def count_far_pairs(sequence: str, pairs) -> int:
    """pair_graph 扫出该序列的远端配对数 (baseline/RL 共用判定)。"""
    _, _, far = build_full_pair_graph(sequence, pairs, do_scan=True)
    return len(far)


def run_one(sequence: str, use_rl: bool, policy_path: str,
            n_sim: int, coding_mask, max_iter: int) -> dict:
    """跑一次 predict_3d_allatom, 返回指标 dict (含 fallback 标记)。"""
    t0 = time.time()
    try:
        res = predict_3d_allatom(
            sequence,
            max_iterations=max_iter,
            use_rl=use_rl,
            rl_policy_path=policy_path,
            rl_n_simulations=n_sim,
            coding_mask=coding_mask,
        )
        rl_info = res.get("rl_info") or {}
        amber_info = res.get("amber_info") or {}
        return {
            "e1_aa": float(res["e1_aa"]),
            "fallback": bool(amber_info.get("fallback", False)),
            "rl_skipped": bool(rl_info.get("skipped", False)) if rl_info else (not use_rl),
            "rl_improvement": float(rl_info.get("improvement", 0.0)) if rl_info else 0.0,
            "n_coding_pinned": int(amber_info.get("n_coding_pinned", 0)),
            "max_p_drift": float(amber_info.get("max_p_drift", 0.0)),
            "elapsed": time.time() - t0,
        }
    except Exception as exc:
        return {
            "e1_aa": float("nan"), "fallback": True,
            "rl_skipped": False, "rl_improvement": 0.0,
            "n_coding_pinned": 0, "max_p_drift": float("nan"),
            "elapsed": time.time() - t0, "error": str(exc),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fa", default=r"D:\IGEM集成方案\data\circrna\circbase_seqs.fa.gz")
    p.add_argument("--min-len", type=int, default=1200)
    p.add_argument("--max-len", type=int, default=2200)
    p.add_argument("--limit", type=int, default=5, help="最多跑几条序列")
    p.add_argument("--policy", default="models/rl/ppo_gnn_final.pth")
    p.add_argument("--n-sim", type=int, default=50, help="MCTS 模拟次数")
    p.add_argument("--max-iter", type=int, default=1500, help="amber 最小化迭代 (长序列降一点)")
    p.add_argument("--out", default="models/rl/compare_rl_result.csv")
    p.add_argument("--smoke", action="store_true", help="短序列少量快测")
    args = p.parse_args()

    if args.smoke:
        args.min_len, args.max_len = 60, 120
        args.limit = 2
        args.n_sim = 5
        args.max_iter = 400

    fa_path = Path(args.fa)
    if not fa_path.exists():
        print(f"[compare_rl] CircBase 数据不存在: {fa_path}")
        print("            改用合成序列 smoke (64nt)")
        seqs = [("synthetic_64", clean_seq("AUGCAUGC" * 8))]
    else:
        seqs = []
        for hdr, raw in iter_fasta(fa_path):
            s = clean_seq(raw)
            if args.min_len <= len(s) <= args.max_len:
                seqs.append((hdr.split()[0], s))
            if len(seqs) >= args.limit:
                break
        print(f"[compare_rl] 筛出 {len(seqs)} 条 (len {args.min_len}-{args.max_len})")

    policy_path = args.policy if Path(args.policy).exists() else None
    if policy_path is None:
        print(f"[compare_rl] 策略权重不存在 ({args.policy}), RL 用随机策略")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (name, seq) in enumerate(seqs):
        L = len(seq)
        print(f"\n[{idx+1}/{len(seqs)}] {name}  L={L}")
        # ViennaRNA 配对 (baseline 和 RL 共用, 只算一次 far_pairs 供报告)
        try:
            pairs, _ = vienna_pair_probs(seq, 0.5)
        except Exception as e:
            print(f"  ViennaRNA 失败: {e!r}, 跳过")
            continue
        n_far = count_far_pairs(seq, pairs)
        print(f"  far_pairs={n_far}")
        # CircBase 全小写 -> parse_case_annotation 默认全非 coding (RL 全序列可动)
        mask = parse_case_annotation(seq, default_coding=False)

        # baseline
        print(f"  [baseline] running...")
        b = run_one(seq, use_rl=False, policy_path=None,
                    n_sim=args.n_sim, coding_mask=None, max_iter=args.max_iter)
        print(f"  [baseline] e1_aa={b['e1_aa']:.0f} fallback={b['fallback']} "
              f"({b['elapsed']:.0f}s)")

        # +RL
        print(f"  [+RL] running... (policy={policy_path or 'random'}, sim={args.n_sim})")
        r = run_one(seq, use_rl=True, policy_path=policy_path,
                    n_sim=args.n_sim, coding_mask=mask, max_iter=args.max_iter)
        print(f"  [+RL] e1_aa={r['e1_aa']:.0f} fallback={r['fallback']} "
              f"rl_skipped={r['rl_skipped']} improvement={r['rl_improvement']:+.4f} "
              f"pinned={r['n_coding_pinned']} ({r['elapsed']:.0f}s)")

        delta = r["e1_aa"] - b["e1_aa"] if not (np.isnan(b["e1_aa"]) or np.isnan(r["e1_aa"])) else float("nan")
        rows.append({
            "name": name, "L": L, "n_far_pairs": n_far,
            "rl_improvement": r["rl_improvement"], "rl_skipped": r["rl_skipped"],
            "baseline_e1_aa": b["e1_aa"], "rl_e1_aa": r["e1_aa"],
            "delta_e1_aa": delta,
            "baseline_fallback": b["fallback"], "rl_fallback": r["fallback"],
            "n_coding_pinned": r["n_coding_pinned"],
        })

    # 写 CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n[compare_rl] CSV -> {out_path}")

    # 汇总
    if not rows:
        print("[compare_rl] 无有效结果")
        return
    valid = [r for r in rows if not np.isnan(r["delta_e1_aa"])]
    n_rl_ran = sum(1 for r in rows if not r["rl_skipped"])
    print(f"\n=== 汇总 ({len(rows)} 条, RL 真跑 {n_rl_ran} 条) ===")
    print(f"{'name':<20} {'L':>5} {'far':>4} {'rl_imp':>8} {'base_e1':>9} {'rl_e1':>9} {'d_e1':>9} {'fb':>4}")
    for r in rows:
        print(f"{r['name'][:20]:<20} {r['L']:>5} {r['n_far_pairs']:>4} "
              f"{r['rl_improvement']:>+8.4f} {r['baseline_e1_aa']:>9.0f} "
              f"{r['rl_e1_aa']:>9.0f} {r['delta_e1_aa']:>+9.0f} "
              f"{'B!' if r['baseline_fallback'] else ('R!' if r['rl_fallback'] else 'ok'):>4}")
    if valid:
        deltas = [r["delta_e1_aa"] for r in valid]
        print(f"\ndelta_e1_aa (rl-baseline): mean={np.mean(deltas):+.0f} "
              f"min={np.min(deltas):+.0f} max={np.max(deltas):+.0f}")
        print("(delta<0 = RL 后 amber 能量更低更合理; delta>0 = RL 让结构稍紧)")
    print(f"\n诚实声明: rl_improvement 是 RL reward 提升信号, 不是结构好坏结论;")
    print(f"          最终结构评估看 e1_aa (amber 物理能量), RL 只是搜索步骤。")


if __name__ == "__main__":
    main()
