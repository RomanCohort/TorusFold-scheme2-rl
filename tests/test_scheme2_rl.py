"""scheme2 RL 优化器 smoke test。

不依赖真实 RNA 数据 / torch 权重, 全程用合成环形 CG P 坐标验证
rl_optimizer 的状态构建、reward、动作、MCTS、端到端入口的正确性。

合成构造: L=100 环形 (半径 = L*5.9/(2π)), 远端配对块 [(10,60),(11,59),(12,58),(13,57)]。
环距 = 50 > FAR_DIST=50 边界, 这里直接喂 far_pairs 绕过 pair_graph 判定,
只测 rl_optimizer 本身 (pair_graph 的远端判定另有覆盖)。
"""
from __future__ import annotations

import numpy as np
import pytest

from torusfold.scheme2.rl_optimizer import (
    BlockState,
    N_DIRECTIONS,
    N_ROT_AXES,
    N_STEPS,
    MCTS,
    PolicyNetwork,
    RLOptimizerState,
    WC_TARGET_DIST,
    apply_action,
    build_rl_state,
    compute_reward,
    optimize_far_pairs,
)


# ---------- fixtures ----------
def make_ring_p(L: int = 100, seed: int = 42) -> np.ndarray:
    """合成环形 CG P 坐标 (L, 3), z=0 平面圆。"""
    R = L * 5.9 / (2.0 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    return np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(L)], axis=1)


def make_far_block() -> list[tuple[int, int]]:
    """4 连续反向平行远端配对 (茎块)。"""
    return [(10, 60), (11, 59), (12, 58), (13, 57)]


@pytest.fixture
def ring_state():
    p = make_ring_p()
    far = make_far_block()
    state = build_rl_state(p, "A" * len(p), far, [far])
    return p, far, state


# ---------- build_rl_state ----------
class TestBuildRLState:
    def test_far_blocks_built(self, ring_state):
        _, _, state = ring_state
        assert len(state.far_blocks) == 1, "整块远端配对应聚成 1 个 block"
        b = state.far_blocks[0]
        assert isinstance(b, BlockState)
        assert len(b.residues_i) == 4
        assert len(b.residues_j) == 4
        assert b.centroid_i.shape == (3,)
        assert b.centroid_j.shape == (3,)
        assert b.current_deviation > 0.0

    def test_block_deviation_matches_direct(self, ring_state):
        p, far, state = ring_state
        b = state.far_blocks[0]
        direct = float(np.mean([np.linalg.norm(p[i] - p[j]) for i, j in far]))
        assert abs(b.current_deviation - direct) < 1e-9

    def test_block_edges_self_only(self, ring_state):
        """单个 block 没有 block 间边。"""
        _, _, state = ring_state
        assert state.block_edges == []

    def test_far_pairs_passthrough(self, ring_state):
        _, far, state = ring_state
        assert state.far_pairs == far

    def test_non_far_block_filtered(self):
        """块内含非远端配对时应被过滤掉。"""
        p = make_ring_p()
        # far_pairs 只含 (10,60), 但 stem_blocks 给整块 -> block 内 (11,59) 不在 far_set
        far_pairs = [(10, 60)]
        stem = make_far_block()
        state = build_rl_state(p, "A" * len(p), far_pairs, [stem])
        assert state.far_blocks == [], "块内配对不全在 far_pairs 的应被过滤"


# ---------- compute_reward ----------
class TestComputeReward:
    def test_empty_far_pairs_zero(self):
        p = make_ring_p()
        assert compute_reward(p, []) == 0.0

    def test_target_distance_high_reward(self):
        """把配对距离正好拉到 WC_TARGET_DIST 时 R_pair 接近最大。"""
        L = 100
        p = np.zeros((L, 3))
        # (0,1) 距离 = WC_TARGET_DIST -> dev=0 -> exp(0)=1
        p[0] = np.array([0.0, 0.0, 0.0])
        p[1] = np.array([WC_TARGET_DIST, 0.0, 0.0])
        r = compute_reward(p, [(0, 1)], use_regularization=False)
        # 单配对 dev=0: R_pair = exp(0) - 0.01*0 = 1.0
        assert abs(r - 1.0) < 1e-9

    def test_far_distance_lower_reward(self):
        """拉远后 reward 低于目标距离。"""
        L = 100
        p_target = np.zeros((L, 3))
        p_target[0] = [0.0, 0.0, 0.0]
        p_target[1] = [WC_TARGET_DIST, 0.0, 0.0]
        p_far = np.zeros((L, 3))
        p_far[0] = [0.0, 0.0, 0.0]
        p_far[1] = [WC_TARGET_DIST + 20.0, 0.0, 0.0]
        r_target = compute_reward(p_target, [(0, 1)], use_regularization=False)
        r_far = compute_reward(p_far, [(0, 1)], use_regularization=False)
        assert r_target > r_far

    def test_regularization_reduces_on_distortion(self):
        """骨架扭曲时正则版 reward 低于无正则版。"""
        L = 100
        # 让相邻 P-P 严重偏离 5.9 (扭曲)
        p = np.zeros((L, 3))
        for k in range(L - 1):
            p[k + 1] = p[k] + np.array([20.0, 0.0, 0.0])  # 相邻 20Å, 远超 5.9±1
        # 加一个目标距离配对让 R_pair 非零
        p[0] = [0.0, 0.0, 0.0]
        p[50] = [WC_TARGET_DIST, 0.0, 0.0]
        far = [(0, 50)]
        r_no_reg = compute_reward(p, far, use_regularization=False)
        r_reg = compute_reward(p, far, use_regularization=True)
        assert r_reg < r_no_reg, "骨架扭曲时正则应压低总 reward"

    def test_returns_float(self, ring_state):
        p, far, _ = ring_state
        r = compute_reward(p, far)
        assert isinstance(r, float)

    def test_closure_penalizes_opened_bsj(self):
        """BSJ 拉开时正则版 reward 低于闭合时 (R_closure 起作用)。"""
        L = 100
        p = make_ring_p()
        far = [(10, 60)]
        # 闭合状态: p[0] 和 p[-1] 很近
        r_closed = compute_reward(p, far, use_regularization=True)
        # 把末端拉开
        p_opened = p.copy()
        p_opened[-1] = p_opened[-1] + np.array([50.0, 0.0, 0.0])
        r_opened = compute_reward(p_opened, far, use_regularization=True)
        assert r_opened < r_closed, \
            f"BSJ 拉开应降低 reward: closed={r_closed:.2f} vs opened={r_opened:.2f}"

    def test_target_dists_per_pair(self):
        """逐对目标距离: 自定义 target 影响 reward。"""
        L = 10
        p = np.zeros((L, 3))
        p[0] = np.array([0.0, 0.0, 0.0])
        p[1] = np.array([30.0, 0.0, 0.0])
        far = [(0, 1)]
        # 用固定目标 20Å: dev=10 -> reward 低
        r_20 = compute_reward(p, far, use_regularization=False,
                               target_dists=[WC_TARGET_DIST])
        # 用 30Å 目标: dev=0 -> reward=1.0
        r_30 = compute_reward(p, far, use_regularization=False,
                               target_dists=[30.0])
        assert r_30 > r_20, f"30Å 目标应高于 20Å 目标: {r_20:.4f} vs {r_30:.4f}"
        assert abs(r_30 - 1.0) < 1e-9, "dev=0 时 reward 应 = 1.0"


# ---------- apply_action ----------
class TestApplyAction:
    def test_only_i_side_moved(self, ring_state):
        p, far, state = ring_state
        b = state.far_blocks[0]
        j_before = p[b.residues_j].copy()
        new_p = apply_action(state, block_idx=0, dir_idx=0, step_idx=2)  # +x, 5Å
        # j 侧不动
        assert np.allclose(new_p[b.residues_j], j_before), "j 侧不应被移动"
        # i 侧 x 增加 5
        for r in b.residues_i:
            assert abs(new_p[r, 0] - p[r, 0] - 5.0) < 1e-9

    def test_action_changes_pair_distance(self, ring_state):
        """移动 i 侧应改变 i-j 配对距离 (旧版 i/j 同向平移 bug 的回归测试)。"""
        p, far, state = ring_state
        d_before = np.linalg.norm(p[10] - p[60])
        # 选一个朝向 j 的方向 (启发式: +x 不一定对, 多试几个找到一个改变距离的)
        changed = False
        for didx in range(6):
            new_p = apply_action(state, 0, didx, 2)
            d_after = np.linalg.norm(new_p[10] - new_p[60])
            if not np.isclose(d_after, d_before):
                changed = True
                break
        assert changed, "至少有一个方向应改变配对距离 (否则 i/j 同向平移 bug 回归)"

    def test_rotation_changes_pair_distance(self, ring_state):
        """旋转动作 (dir_idx>=6) 也应改变 i-j 配对距离。"""
        p, far, state = ring_state
        d_before = np.linalg.norm(p[10] - p[60])
        changed = False
        for didx in range(N_DIRECTIONS, N_DIRECTIONS + N_ROT_AXES):
            for sidx in range(N_STEPS):
                new_p = apply_action(state, 0, didx, sidx)
                d_after = np.linalg.norm(new_p[10] - new_p[60])
                if not np.isclose(d_after, d_before):
                    changed = True
                    break
            if changed:
                break
        assert changed, "旋转动作应能改变 i-j 配对距离"
        # j 侧在旋转中不应移动 (只动 i 侧)
        j_before = p[state.far_blocks[0].residues_j].copy()
        new_p = apply_action(state, 0, N_DIRECTIONS, 0)
        assert np.allclose(new_p[state.far_blocks[0].residues_j], j_before), \
            "旋转应只动 i 侧"

    def test_does_not_mutate_input(self, ring_state):
        p, far, state = ring_state
        p_copy = p.copy()
        _ = apply_action(state, 0, 0, 0)
        assert np.allclose(p, p_copy), "apply_action 不应修改输入 p_coords"


# ---------- MCTS ----------
class TestMCTS:
    def test_search_returns_same_shape(self, ring_state):
        p, far, state = ring_state
        mcts = MCTS(policy=None, n_simulations=10)
        out = mcts.search(state, far)
        assert out.shape == p.shape

    def test_search_reward_not_degraded(self, ring_state):
        """MCTS 输出 reward 应 >= 初始 (best 初始化为 root)。"""
        p, far, state = ring_state
        mcts = MCTS(policy=None, n_simulations=30)
        out = mcts.search(state, far)
        r_before = compute_reward(p, far)
        r_after = compute_reward(out, far)
        assert r_after >= r_before - 1e-9, \
            f"MCTS 不应降低 reward: {r_before:.4f} -> {r_after:.4f}"

    def test_empty_blocks_returns_input(self):
        """无远端 block 时返回输入。"""
        p = make_ring_p()
        state = RLOptimizerState(
            p_coords=p, sequence="A" * len(p),
            far_blocks=[], far_pairs=[], block_edges=[],
        )
        mcts = MCTS(policy=None, n_simulations=5)
        out = mcts.search(state, [])
        assert np.allclose(out, p)

    def test_with_random_policy(self, ring_state):
        """带 (未训练) 策略网络的 MCTS 也能跑完不崩。"""
        p, far, state = ring_state
        try:
            policy = PolicyNetwork()
        except Exception as exc:  # torch 缺失等
            pytest.skip(f"PolicyNetwork 不可用: {exc!r}")
        mcts = MCTS(policy=policy, n_simulations=10)
        out = mcts.search(state, far)
        assert out.shape == p.shape


# ---------- 端到端入口 ----------
class TestOptimizeFarPairs:
    def test_info_fields(self, ring_state):
        p, far, _ = ring_state
        _, _, info = optimize_far_pairs(p, "A" * len(p), far, [far], n_simulations=10)
        for key in ("reward_before", "reward_after", "improvement",
                    "n_blocks", "n_far_pairs", "n_simulations", "policy_loaded",
                    "coding_mask"):
            assert key in info, f"info 缺字段 {key}"
        assert info["n_far_pairs"] == len(far)
        assert info["n_simulations"] == 10
        assert info["policy_loaded"] is False
        # 不传 coding_mask 时 info["coding_mask"] = None (透传占位)
        assert info["coding_mask"] is None

    def test_no_degradation(self, ring_state):
        p, far, _ = ring_state
        out, _, info = optimize_far_pairs(p, "A" * len(p), far, [far], n_simulations=30)
        assert info["reward_after"] >= info["reward_before"] - 1e-9
        assert out.shape == p.shape

    def test_missing_policy_file_fallback(self, ring_state, capsys):
        """policy_path 指向不存在文件时回退随机策略不抛。"""
        p, far, _ = ring_state
        out, _, info = optimize_far_pairs(
            p, "A" * len(p), far, [far],
            policy_path="/nonexistent/rl_policy.pt", n_simulations=5,
        )
        assert info["policy_loaded"] is False
        assert out.shape == p.shape
