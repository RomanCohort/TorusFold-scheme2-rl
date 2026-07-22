"""train_ppo.py - PPO + GAE 训练 RL 策略网络 (scheme2 远端配对优化)。

链路:
  samples (generate_samples.py 产出 .pkl) -> build_rl_state ->
  policy.roll_episode 采轨迹 -> GAE 估 advantage -> PPO clip 更新

消融: --variant gnn|mlp, gnn 用消息传递 (n_mp_layers=3), mlp 退化
  (n_mp_layers=0, 消息传递层空, 等价旧版 MLP)。同 seed 同数据对比 reward 曲线。

用法:
  python train_ppo.py --samples data/rl_samples --epochs 50 --batch 8 --variant gnn
  python train_ppo.py --smoke   # 5 条合成样本, 2 epoch 验证管线通

产出: models/rl/ppo_<variant>_epochXXX.pth + models/rl/ppo_<variant>_final.pth
      + models/rl/ppo_<variant>_log.json (每 epoch 的 mean reward / loss)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# src 可 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfold.scheme2.rl_optimizer import (  # noqa: E402
    RLOptimizerState, PolicyNetwork, MCTS, apply_action, compute_reward,
    build_rl_state, _rebuild_blocks, _get_torch,
    N_DIRECTIONS, N_STEPS, DIRECTIONS, STEP_SIZES, WC_TARGET_DIST,
)

# ---------- PPO 超参 ----------
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
LR = 3e-4
WEIGHT_DECAY = 0.01
VALUE_COEF = 0.5        # V 头 MSE 系数 (reward 标准化后 ~N(0,1), MSE 不大, 恢复 0.5)
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4          # 每批数据重用次数
EPISODE_LEN = 20        # 单条样本采多少步动作
ROLLOUT_DEPTH = 3       # 采轨迹时评估 reward 用短 rollout (省时间)
MAX_GRAD_NORM = 0.5     # 梯度裁剪 (防 KL 爆炸)


class RunningMeanStd:
    """Welford 在线均值/方差, 标准化 reward 到 ~N(0,1)。

    bug② 修: compute_reward 绝对值 ~-7, V 头拟合它 MSE 天然大;
    标准化后 V 头在 ~N(0,1) 空间学, 数值稳, MSE 自然降。
    """
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps  # 防除零

    def update(self, x: float):
        """单样本更新 (Welford)。"""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    def normalize(self, x):
        """标准化 (向量/标量通用)。"""
        std = max(self.var ** 0.5, 1e-6)
        return (np.asarray(x, dtype=np.float32) - self.mean) / std

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd):
        self.mean = sd["mean"]; self.var = sd["var"]; self.count = sd["count"]


def load_samples(sample_dir: Path) -> List[dict]:
    """加载 generate_samples.py 产出的 .pkl 样本。"""
    samples = []
    for pkl in sorted(sample_dir.glob("sample_*.pkl")):
        with open(pkl, "rb") as f:
            samples.append(pickle.load(f))
    return samples


def make_synthetic_samples(n: int = 5, seed: int = 0) -> List[dict]:
    """无真实样本时造合成样本 (smoke 用)。

    环形 CG P 坐标 + 远端配对故意拉远, stem_blocks 覆盖远端配对。
    """
    rng = np.random.RandomState(seed)
    samples = []
    for k in range(n):
        L = rng.randint(80, 120)
        R = L * 5.9 / (2 * np.pi)
        ang = np.linspace(0, 2 * np.pi, L, endpoint=False)
        p = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(L)], axis=1)
        # 远端配对: 两个块, 隔开
        i0 = rng.randint(5, L // 4)
        j0 = i0 + L // 2 + rng.randint(0, 10)
        far_pairs = [
            (i0, j0 % L), (i0 + 1, (j0 + 1) % L),
            (i0 + 2, (j0 + 2) % L), (i0 + 3, (j0 + 3) % L),
        ]
        stem_blocks = [list(zip([i0, i0 + 1, i0 + 2, i0 + 3],
                                [j0 % L, (j0 + 1) % L, (j0 + 2) % L, (j0 + 3) % L]))]
        samples.append({
            "seq": "A" * L,
            "p_coords": p.astype(np.float32),
            "far_pairs": far_pairs,
            "near_pairs": [],
            "stem_blocks": stem_blocks,
        })
    return samples


def roll_episode(
    policy: PolicyNetwork,
    sample: dict,
    ep_len: int = EPISODE_LEN,
    rms: "RunningMeanStd" = None,
) -> List[dict]:
    """用 policy 采一条轨迹。

    每步: policy.forward 采样动作 -> apply_action -> compute_reward。
    记录 (state_snapshot, action, log_prob, reward, value)。
    reward 经 rms 标准化 (bug② 修), V 头拟合标准化后值。
    """
    torch = _get_torch()
    state = build_rl_state(
        sample["p_coords"], sample["seq"],
        sample["far_pairs"], sample["stem_blocks"],
    )
    if not state.far_blocks:
        return []

    trajectory = []
    cur_p = state.p_coords.copy()
    for step in range(ep_len):
        cur_state = RLOptimizerState(
            p_coords=cur_p, sequence=state.sequence,
            far_blocks=_rebuild_blocks(state, cur_p),
            far_pairs=state.far_pairs, block_edges=state.block_edges,
        )
        pi_b, pi_d, pi_s, v = policy.forward(cur_state, return_value=True)
        n_blocks = len(cur_state.far_blocks)
        if pi_b is None:
            break

        # 采样动作 (从 policy 分布)
        bidx = int(torch.multinomial(pi_b, 1).item())
        didx = int(torch.multinomial(pi_d, 1).item())
        sidx = int(torch.multinomial(pi_s, 1).item())
        # log_prob = log π_block[b] + log π_dir[d] + log π_step[s]
        logp = (torch.log(pi_b[bidx] + 1e-8) +
                torch.log(pi_d[didx] + 1e-8) +
                torch.log(pi_s[sidx] + 1e-8))

        # 执行 + reward (绝对值, 不用差分 shaping)
        new_p = apply_action(cur_state, bidx, didx, sidx)
        r_raw = float(compute_reward(new_p, state.far_pairs))
        # 标准化 reward: 更新 rms 统计 + 归一 (bug② 修)
        if rms is not None:
            rms.update(r_raw)
            r_step = float(rms.normalize(r_raw))
        else:
            r_step = r_raw

        trajectory.append({
            "p_coords": cur_p.copy(),
            "far_blocks": cur_state.far_blocks,   # 重建状态用
            "block_edges": state.block_edges,
            "sequence": state.sequence,
            "far_pairs": state.far_pairs,
            "action": (bidx, didx, sidx),
            "log_prob_old": float(logp.detach()),
            "reward": r_step,
            "reward_raw": r_raw,
            "value_old": float(v.detach()),
        })
        cur_p = new_p

    return trajectory


def compute_gae(trajectory: List[dict], policy: PolicyNetwork,
                gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> Tuple[List[float], List[float]]:
    """GAE 估 advantage + return。

    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    A_t = δ_t + (γλ)·A_{t+1}
    R_t (return) = A_t + V(s_t)
    """
    torch = _get_torch()
    n = len(trajectory)
    advantages = [0.0] * n
    returns = [0.0] * n
    last_gae = 0.0

    # 预算每个状态的 V (bootstrap), 末态 V=0 (episode 结束)
    values = []
    for t, trans in enumerate(trajectory):
        if t == n - 1:
            values.append(0.0)
        else:
            # 下一个状态的 V (用当前 policy 估, off-policy 近似)
            nxt = trajectory[t + 1]
            nxt_state = _rebuild_state(nxt)
            v = policy.value(nxt_state)
            values.append(float(v.detach()) if v is not None else 0.0)

    for t in reversed(range(n)):
        delta = trajectory[t]["reward"] + gamma * values[t] - trajectory[t]["value_old"]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + trajectory[t]["value_old"]

    # advantage 标准化 (减均值除标准差, 稳定训练)
    adv_arr = np.array(advantages, dtype=np.float32)
    if adv_arr.std() > 1e-6:
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
    return adv_arr.tolist(), returns


def _rebuild_state(trans: dict) -> RLOptimizerState:
    """从 trajectory 条目重建 RLOptimizerState (PPO 更新时前向用)。"""
    return RLOptimizerState(
        p_coords=trans["p_coords"], sequence=trans["sequence"],
        far_blocks=trans["far_blocks"], far_pairs=trans["far_pairs"],
        block_edges=trans["block_edges"],
    )


def ppo_update(
    policy: PolicyNetwork,
    batch_episodes: List[List[dict]],
    optimizer,
) -> dict:
    """对一批 episode 轨迹做 PPO 更新 (bug① 修: 真 batch 聚合梯度)。

    batch_episodes: [[trans_t for t in episode], ...] 按 episode 分组
    每轮 PPO_EPOCHS: 前向所有 trans -> 求和 loss -> 一次 backward+step
    (不是单样本更新, ratio clip 才有意义)。
    返回 {policy_loss, value_loss, entropy, kl} 均值。
    """
    torch = _get_torch()
    # 算每条 episode 的 GAE, flatten
    all_adv, all_ret, flat_traj = [], [], []
    for traj in batch_episodes:
        adv, ret = compute_gae(traj, policy)
        all_adv.extend(adv)
        all_ret.extend(ret)
        flat_traj.extend(traj)
    n = len(flat_traj)
    if n == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    all_adv_t = torch.tensor(all_adv, dtype=torch.float32)
    all_ret_t = torch.tensor(all_ret, dtype=torch.float32)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    n_updates = 0

    for _ in range(PPO_EPOCHS):
        total_loss = torch.tensor(0.0)
        total_p = torch.tensor(0.0)
        total_v = torch.tensor(0.0)
        total_ent = torch.tensor(0.0)
        total_kl = torch.tensor(0.0)
        cnt = 0
        for t, trans in enumerate(flat_traj):
            state = _rebuild_state(trans)
            pi_b, pi_d, pi_s, v = policy.forward(state, return_value=True)
            if pi_b is None:
                continue
            bidx, didx, sidx = trans["action"]
            logp_new = (torch.log(pi_b[bidx] + 1e-8) +
                        torch.log(pi_d[didx] + 1e-8) +
                        torch.log(pi_s[sidx] + 1e-8))
            logp_old = trans["log_prob_old"]
            ratio = torch.exp(logp_new - logp_old)
            adv = all_adv_t[t]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv
            policy_loss = -torch.min(surr1, surr2)
            value_loss = (v - all_ret_t[t]) ** 2
            entropy = -(pi_b * torch.log(pi_b + 1e-8)).sum() - \
                      (pi_d * torch.log(pi_d + 1e-8)).sum() - \
                      (pi_s * torch.log(pi_s + 1e-8)).sum()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            total_loss = total_loss + loss
            total_p = total_p + policy_loss.detach()
            total_v = total_v + value_loss.detach()
            total_ent = total_ent + entropy.detach()
            with torch.no_grad():
                total_kl = total_kl + (ratio - 1) - torch.log(ratio + 1e-8)
            cnt += 1

        if cnt == 0:
            continue
        # bug① 修: 整 batch 聚合后一次 backward (非单样本)
        optimizer.zero_grad()
        total_loss.backward()
        # bug③ 修: 梯度裁剪防 KL 爆炸
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        metrics["policy_loss"] += float(total_p) / cnt
        metrics["value_loss"] += float(total_v) / cnt
        metrics["entropy"] += float(total_ent) / cnt
        metrics["kl"] += float(total_kl) / cnt
        n_updates += 1

    for k in metrics:
        metrics[k] /= max(n_updates, 1)
    return metrics


def _group_traj_by_episode(batch_traj: List[dict]) -> List[List[dict]]:
    """[已弃用] 旧版按 EPISODE_LEN 切组, 现收集时已按 episode 分组。保留空壳防 import 报错。"""
    return [batch_traj]


def evaluate(policy: PolicyNetwork, samples: List[dict],
             n_simulations: int = 20) -> float:
    """用 MCTS 跑 evaluate, 返回平均 reward 提升。"""
    mcts = MCTS(policy=policy, n_simulations=n_simulations, use_rollout=True)
    improvements = []
    for s in samples[:8]:  # 评估子集
        state = build_rl_state(s["p_coords"], s["seq"], s["far_pairs"], s["stem_blocks"])
        if not state.far_blocks:
            continue
        r_before = compute_reward(s["p_coords"], s["far_pairs"])
        opt = mcts.search(state, s["far_pairs"])
        r_after = compute_reward(opt, s["far_pairs"])
        improvements.append(r_after - r_before)
    return float(np.mean(improvements)) if improvements else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/rl_samples")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--variant", choices=["gnn", "mlp"], default="gnn",
                   help="gnn=消息传递 n_mp_layers=3; mlp=退化 n_mp_layers=0 (消融对照)")
    p.add_argument("--ep_len", type=int, default=EPISODE_LEN)
    p.add_argument("--out", default="models/rl")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch = 4
        args.ep_len = 8

    torch = _get_torch()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载样本
    sample_dir = Path(args.samples)
    if args.smoke or not sample_dir.exists():
        print(f"[train_ppo] 用合成样本 (smoke 或样本目录不存在: {sample_dir})")
        samples = make_synthetic_samples(8 if args.smoke else 50, args.seed)
    else:
        samples = load_samples(sample_dir)
    print(f"[train_ppo] 样本数: {len(samples)}, variant={args.variant}")

    # 消融: gnn vs mlp
    n_mp = 3 if args.variant == "gnn" else 0
    policy = PolicyNetwork(hidden_dim=128, n_mp_layers=n_mp)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rms = RunningMeanStd()  # bug②: reward 标准化统计

    log = {"variant": args.variant, "epochs": [], "config": {
        "lr": LR, "gamma": GAMMA, "lam": GAE_LAMBDA, "clip": PPO_CLIP,
        "n_mp_layers": n_mp, "ep_len": args.ep_len, "batch": args.batch,
    }}

    for epoch in range(args.epochs):
        np.random.shuffle(samples)
        # 收集一批轨迹 (按 episode 分组, 不 flatten)
        batch_episodes = []
        for s in samples[:args.batch]:
            traj = roll_episode(policy, s, ep_len=args.ep_len, rms=rms)
            if traj:
                batch_episodes.append(traj)

        if not batch_episodes:
            print(f"[epoch {epoch}] 无有效轨迹, 跳过")
            continue

        metrics = ppo_update(policy, batch_episodes, optimizer)

        # 每 5 epoch 评估一次 (省时间)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            mean_improve = evaluate(policy, samples)
        else:
            mean_improve = -1.0  # 占位, 未评估

        # epoch 平均 reward (用 raw reward, 标准化后的看 kl/v_loss)
        flat = [t for ep in batch_episodes for t in ep]
        ep_rewards_raw = [t["reward_raw"] for t in flat]
        log["epochs"].append({
            "epoch": epoch,
            "mean_step_reward_raw": float(np.mean(ep_rewards_raw)),
            "improvement": mean_improve,
            **metrics,
        })
        print(f"[epoch {epoch}] r_raw={np.mean(ep_rewards_raw):+.4f} "
              f"improve={mean_improve:+.4f} "
              f"p_loss={metrics['policy_loss']:.4f} "
              f"v_loss={metrics['value_loss']:.4f} "
              f"ent={metrics['entropy']:.4f} kl={metrics['kl']:.4f} "
              f"rms={rms.mean:+.2f}±{rms.var**0.5:.2f}")

        # 快照
        if (epoch + 1) % 10 == 0:
            policy.save(str(out_dir / f"ppo_{args.variant}_epoch{epoch+1:03d}.pth"))

    # 最终权重 + log (含 rms, 推理时反标准化 V 用)
    final_path = out_dir / f"ppo_{args.variant}_final.pth"
    policy.save(str(final_path))
    rms_path = out_dir / f"ppo_{args.variant}_rms.json"
    with open(rms_path, "w", encoding="utf-8") as f:
        json.dump(rms.state_dict(), f)
    log_path = out_dir / f"ppo_{args.variant}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[train_ppo] 完成: {final_path}")
    print(f"[train_ppo] log: {log_path}")
    print(f"[train_ppo] variant={args.variant} 最末 improvement={log['epochs'][-1]['improvement']:+.4f}")


if __name__ == "__main__":
    main()
