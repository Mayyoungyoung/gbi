# GbI: 共享任务条件 state-based RSSM 世界模型（GbI.md §3.1，v5）。
#
# 结构与 TD-MPC2 (ICLR 2024) 的 state-based 模型同构（简化版）：
#   obs --encoder--> z (512)                     共享
#   [z; a] --GRU--> h (512 确定性隐状态)            共享
#   h --state_head--> Δz（残差预测下一 z）             共享（动力学）
#   h --obs_head--> Δŝ（残差解码下一 obs，仅诊断/供策略观测） 共享
#   [h; onehot(z_i)] --reward_head_k--> two-hot logits  5 成员仅在此分叉
#
# 关键设计（均为 GbI.md 明确约束）：
#   1. 任务条件 one-hot 仅注入奖励头，动力学不注入（转移与任务无关）。
#   2. ensemble×5 共享动力学、仅奖励头分叉 → 分歧信号只来自奖励头（U_score 归因干净）。
#   3. 奖励头 two-hot(101 bins) + symlog(±400 原始奖励上下界) + 交叉熵。
#   4. 双流采样：均匀流训练动力学（保住物理覆盖）；残差加权流只进奖励头梯度
#      （high-residual 样本高权重，quantile-sigmoid 权重，上限截断）。残差缓存每 T 步刷新。
#   5. TTA（§3.4）：adapt_reward_head() 冻结 GRU/encoder/state/obs head，
#      只对奖励头（可选仅最后一层）做 1-3 步梯度更新，lr = adapt_lr（默认 0.1×wm_lr）。
#   6. 编码器 EMA 副本提供 next-z 目标（防表征坍缩）。

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mtrl.agent.components import base as base_component
from mtrl.agent.components.twohot import (
    TwoHotSymlog,
    bins_to_reward,
    symlog,
    twohot_loss,
)
from mtrl.utils.types import TensorType


class SymLogTwoHotHead(nn.Module):
    """单个奖励头成员：[h; onehot(task)] -> MLP -> bins 个 logits。"""

    def __init__(self, hidden_dim: int, num_envs: int, reward_bins: int, mlp_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + num_envs, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, reward_bins),
        )

    def forward(self, h: TensorType, task_onehot: TensorType) -> TensorType:
        return self.fc(torch.cat([h, task_onehot], dim=-1))  # (B, bins)


class WorldModel(base_component.Component):
    """共享任务条件 RSSM（state-based）。"""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        num_envs: int,
        hidden_dim: int = 512,
        num_heads: int = 5,
        reward_bins: int = 101,
        reward_min: float = -400.0,
        reward_max: float = 400.0,
        obs_eps: float = 1e-5,
        bootstrap_low: float = 0.7,
        bootstrap_high: float = 0.9,
        encoder_ema_tau: float = 0.995,
        lr: float = 1e-4,
        adapt_lr: float = 1e-5,  # 0.1 × wm_lr（GbI §3.4 上限）
        adapt_scope: str = "last_layer",  # 'last_layer' | 'all_heads'
        grad_clip: float = 10.0,
        reward_weight: float = 1.0,
        obs_weight: float = 1.0,
        state_weight: float = 1.0,
        cache_size: int = 4096,
        cache_refresh_every: int = 512,
        residual_quantile: float = 0.8,  # quantile-sigmoid 拐点
        residual_temperature: float = 5.0,
        residual_max_weight: float = 5.0,  # 上限截断防异常重尾样本绑架
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.obs_dim = int(np.prod(env_obs_shape))
        self.act_dim = int(np.prod(action_shape))
        self.num_envs = num_envs
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device = device
        self.lr = lr
        self.adapt_lr = adapt_lr
        self.adapt_scope = adapt_scope
        self.bootstrap_low = bootstrap_low
        self.bootstrap_high = bootstrap_high
        self.grad_clip = grad_clip
        self.reward_weight = reward_weight
        self.obs_weight = obs_weight
        self.state_weight = state_weight
        self.cache_size = cache_size
        self.cache_refresh_every = cache_refresh_every
        self.residual_quantile = residual_quantile
        self.residual_temperature = residual_temperature
        self.residual_max_weight = residual_max_weight
        self._step = 0
        self._last_cache_refresh = -1

        # 奖励空间 two-hot 变换
        self.twohot = TwoHotSymlog(reward_bins, reward_min, reward_max)

        # ---- 共享动力学 ----
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.encoder_ema = deepcopy(self.encoder)
        for p in self.encoder_ema.parameters():
            p.requires_grad_(False)
        self.encoder_ema_tau = encoder_ema_tau

        self.rnn = nn.GRU(
            input_size=hidden_dim + self.act_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        # 残差预测（zeros init → 初始时 z' ≈ z、s' ≈ s，训练稳定）
        self.state_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                        nn.Linear(hidden_dim, hidden_dim))
        self.obs_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, self.obs_dim))
        for module in (self.state_head, self.obs_head):
            for m in module:
                if isinstance(m, nn.Linear):
                    m.weight.data.fill_(0.0)
                    m.bias.data.fill_(0.0)

        # ---- 任务条件奖励头（5 成员） ----
        self.reward_heads = nn.ModuleList(
            [SymLogTwoHotHead(hidden_dim, num_envs, reward_bins) for _ in range(num_heads)]
        )

        # ---- obs 归一化（running mean/std；dynamics 预测与目标都在归一化空间） ----
        self.register_buffer("obs_mean", torch.zeros(1, self.obs_dim))
        self.register_buffer("obs_std", torch.ones(1, self.obs_dim))
        self.obs_eps = obs_eps
        self._obs_sum = torch.zeros(self.obs_dim)
        self._obs_sqsum = torch.zeros(self.obs_dim)
        self._obs_count = 0

        # ---- 残差加权缓存（双流采样的高残差流数据源） ----
        self._rw_cache: Optional[Dict[str, TensorType]] = None
        self._rw_cache_weights: Optional[np.ndarray] = None
        self._rw_cache_tasks: Optional[np.ndarray] = None

        # ---- 优化器（延迟创建：agent 置 device 后显式 init） ----
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.adapt_optimizer: Optional[torch.optim.Optimizer] = None

        self._task_onehot_eye = torch.eye(num_envs)

        # ---- 确保 buffer / 统计 tensor 在正确 device 上 ----
        self.to(self.device)

    # ------------------------------------------------------------------ #
    # 基础设施
    # ------------------------------------------------------------------ #
    def init_optimizers(self) -> None:
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.adapt_optimizer is None:
            if self.adapt_scope == "last_layer":
                adapt_params = []
                for head in self.reward_heads:
                    adapt_params += list(head.fc[-1].parameters())
            elif self.adapt_scope == "all_heads":
                adapt_params = list(self.reward_heads.parameters())
            else:
                raise ValueError(f"unknown adapt_scope: {self.adapt_scope}")
            # 兼容旧 torch 的每个参数 lr 设置
            self.adapt_optimizer = torch.optim.Adam(adapt_params, lr=self.adapt_lr)

    def to(self, device=None, *args, **kwargs):  # type: ignore[override]
        if device is None:
            return super().to(*args, **kwargs)
        self.device = torch.device(device)
        move = super().to(device, *args, **kwargs)
        # 归一化统计与 one-hot 眼矩阵不参与参数注册的部分一并迁移
        self._obs_sum = self._obs_sum.to(self.device)
        self._obs_sqsum = self._obs_sqsum.to(self.device)
        return move

    def _task_onehot(self, task_ids: TensorType) -> TensorType:
        """(B, 1) 的 task id → (B, num_envs) one-hot。"""
        ids = task_ids.long().squeeze(-1).clamp(0, self.num_envs - 1)
        return self._task_onehot_eye.to(self.device)[ids]

    def _update_obs_stats(self, obs: TensorType) -> None:
        """running mean/std（逐 batch 累积，不参与梯度）。"""
        with torch.no_grad():
            flat = obs.detach().float().reshape(-1, self.obs_dim).to(self.device)
            B = flat.shape[0]
            self._obs_sum += flat.sum(dim=0)
            self._obs_sqsum += (flat ** 2).sum(dim=0)
            self._obs_count += B
            mean = self._obs_sum / self._obs_count
            var = torch.clamp(self._obs_sqsum / self._obs_count - mean ** 2, min=0.0)
            self.obs_mean.copy_(mean.reshape(1, -1))
            self.obs_std.copy_((torch.sqrt(var) + self.obs_eps).reshape(1, -1))

    def normalize_obs(self, obs: TensorType) -> TensorType:
        return (obs.float() - self.obs_mean) / self.obs_std

    def denormalize_obs(self, obs_norm: TensorType) -> TensorType:
        return obs_norm * self.obs_std + self.obs_mean

    # ------------------------------------------------------------------ #
    # 前向 / 想象
    # ------------------------------------------------------------------ #
    def encode(self, obs: TensorType, normalize: bool = True) -> TensorType:
        x = self.normalize_obs(obs) if normalize else obs.float()
        return self.encoder(x)

    def predict_reward(
        self,
        h: TensorType,
        task_onehot: TensorType,
    ) -> TensorType:
        """返回 (num_heads, B, bins) 的 two-hot logits。"""
        return torch.stack([head(h, task_onehot) for head in self.reward_heads], dim=0)

    def close_loop_rollout(
        self,
        obs: TensorType,
        task_ids: TensorType,
        action_fn: Callable[[TensorType], TensorType],
        H: int,
    ) -> Dict[str, TensorType]:
        """闭环想象（供裁决器打分）。

        Args:
            obs: (B, obs_dim) 起始观测（多状态批量）。
            task_ids: (B, 1) 每个样本打分的任务 id（批内可混合任务）。
            action_fn: TensorType(B, obs_dim) -> TensorType(B, act_dim)，
                闭包绑定候选 π_j 的均值动作。
            H: 展开长度。返回 H 步想象奖励（裁决公式用 τ=1..H-1）。

        Returns:
            dict: rewards (H, num_heads, B) 原始空间奖励；obs (H, B, obs_dim)。
        """
        with torch.no_grad():
            B = obs.shape[0]
            if task_ids.ndim == 0:
                task_ids = torch.full((B, 1), int(task_ids), dtype=torch.long, device=self.device)
            task_onehot = self._task_onehot(task_ids)
            z = self.encode(obs)
            lo = self.obs_mean - 10.0 * self.obs_std
            hi = self.obs_mean + 10.0 * self.obs_std
            hidden = None
            s_cur = obs.float()
            rewards_list: List[TensorType] = []
            obs_list: List[TensorType] = []
            for _ in range(H):
                a = action_fn(s_cur).to(self.device)
                z, h, s_next = self.observe_with_latent(z, a, s_cur, hidden)
                logits = self.predict_reward(h, task_onehot)  # (M, B, bins)
                r = bins_to_reward(logits, self.twohot)  # (M, B)
                rewards_list.append(r)
                s_next = s_next.clamp(min=lo, max=hi)
                obs_list.append(s_next)
                hidden = h
                s_cur = s_next
            rewards = torch.stack(rewards_list, dim=0)  # (H, M, B)
            return {"rewards": rewards, "obs": torch.stack(obs_list, dim=0)}

    def observe_with_latent(
        self,
        z: TensorType,
        action: TensorType,
        s_cur: TensorType,
        hidden: Optional[TensorType],
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """由上一隐状态 z 直接做 GRU 转移（rollout 内部用，不重编码）。
        返回 (z_next, h, s_next 的原始 obs 空间重建)。obs 重建与训练损失同构：
        在归一化空间做残差预测 s_n + obs_head(h)。"""
        inp = torch.cat([z, action.float()], dim=-1).unsqueeze(1)
        if hidden is None:
            h = self.rnn(inp)[0].squeeze(1)
        else:
            h = self.rnn(inp, hidden.unsqueeze(0))[0].squeeze(1)
        z_next = z + self.state_head(h)
        s_next = self.denormalize_obs(self.normalize_obs(s_cur) + self.obs_head(h))
        return z_next, h, s_next

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    def _transition_stats(self, h: TensorType, task_onehot: TensorType) -> TensorType:
        logits = self.predict_reward(h, task_onehot)  # (M, B, bins)
        r_pred = bins_to_reward(logits, self.twohot)  # (M, B) 原始空间
        return r_pred.mean(dim=0)  # (B,) 成员均值

    @torch.no_grad()
    def refresh_residual_cache(self, buffer) -> None:
        """每 T 步重算残差 rank 权重（GbI §3.1 的缓存刷新）。

        从 buffer 中抽 cache_size 条（任务均衡），用当前模型打分，
        残差 |r - r̂| 的 quantile-sigmoid 权重 + 上限截断。
        """
        batch = self._sample_from_buffer(buffer, self.cache_size, balanced=True)
        obs, act, rew, next_obs, task = self._unpack_batch(batch)
        self._update_obs_stats(obs)
        z = self.encode(obs)
        inp = torch.cat([z, act], dim=-1).unsqueeze(1)
        h = self.rnn(inp)[0].squeeze(1)
        r_pred = self._transition_stats(h, self._task_onehot(task))  # (B,)
        residual = (rew.squeeze(-1) - r_pred).abs()  # (B,)
        self._fill_residual_cache(obs, act, rew, next_obs, task, residual)

    def _fill_residual_cache(
        self,
        obs: TensorType, act: TensorType, rew: TensorType,
        next_obs: TensorType, task: TensorType, residual: TensorType,
    ) -> None:
        """按残差做分位 sigmoid 权重，存入缓存（采样流只用奖励头）。"""
        B = residual.shape[0]
        # 分位：residual 排名 → [0, 1]
        q = residual.argsort().argsort().float() / max(B - 1, 1)
        w = torch.sigmoid((q - self.residual_quantile) * self.residual_temperature)
        w = w.clamp(max=self.residual_max_weight).clamp(min=0.02)
        self._rw_cache = {
            "obs": obs.detach().clone(),
            "act": act.detach().clone(),
            "rew": rew.detach().clone(),
            "next_obs": next_obs.detach().clone(),
            "task": task.detach().clone(),
        }
        self._rw_cache_weights = w.cpu().numpy()
        self._rw_cache_tasks = task.squeeze(-1).cpu().numpy()

    def _sample_from_buffer(self, buffer, size: int, balanced: bool = True, rng=None):
        rng = rng or np.random
        if balanced:
            per_task = size // buffer.task_num
            idx_list = []
            for task_i in range(buffer.task_num):
                high = buffer.single_capacity if buffer.full else max(buffer.idx, 1)
                idx_list.append(rng.randint(0, high, size=per_task))
            return buffer.sample(index=idx_list)
        raise NotImplementedError("weighted sampling goes through residual cache")

    def _unpack_batch(self, batch) -> Tuple[TensorType, ...]:
        return (
            batch.env_obs.float().to(self.device),
            batch.action.float().to(self.device),
            batch.reward.float().to(self.device),
            batch.next_env_obs.float().to(self.device),
            batch.task_obs.float().to(self.device),
        )

    def _dynamics_and_reward_losses(
        self,
        obs: TensorType, act: TensorType, rew: TensorType,
        next_obs: TensorType, task: TensorType,
        with_obs_stats_update: bool = False,
        bootstrap: bool = True,
    ) -> Dict[str, TensorType]:
        """均匀流核心：动力学（state/obs）+ 全部 5 头奖励损失。"""
        if with_obs_stats_update:
            self._update_obs_stats(obs)
            self._update_obs_stats(next_obs)
        z = self.encode(obs)
        inp = torch.cat([z, act], dim=-1).unsqueeze(1)
        h = self.rnn(inp)[0].squeeze(1)
        z_next_pred = z + self.state_head(h)
        s_next_pred_norm = self.normalize_obs(obs) + self.obs_head(h)
        # 目标：EMA 编码器 + 归一化 next obs（M4 修复：原实现在线 encoder
        # 自为目标，encoder_ema 维护却从未参与损失；改用 EMA 副本提供
        # next-z 目标，防表征坍缩——与文件头设计约束第 6 条一致）
        with torch.no_grad():
            z_next_target = self.encoder_ema(self.normalize_obs(next_obs))
            s_next_target_norm = self.normalize_obs(next_obs)
        state_loss = F.mse_loss(z_next_pred, z_next_target) / self.hidden_dim
        obs_loss = F.mse_loss(s_next_pred_norm, s_next_target_norm)

        task_onehot = self._task_onehot(task)
        logits = self.predict_reward(h, task_onehot)  # (M, B, bins)
        rew_losses = []
        B = rew.shape[0]
        for m in range(self.num_heads):
            mask = None
            if bootstrap:
                frac = np.random.uniform(self.bootstrap_low, self.bootstrap_high)
                keep = torch.rand(B, device=self.device) < frac
                # bootstrap 子抽样按 GbI §3.1: 每成员抽 0.7-0.9 数据子集
                mask_logits = logits[m][keep]
                mask_rew = rew[keep]
            else:
                mask_logits = logits[m]
                mask_rew = rew
            rew_losses.append(twohot_loss(mask_logits, mask_rew, self.twohot, reduction="mean"))
        reward_loss = torch.stack(rew_losses).mean()
        return {
            "state_loss": state_loss,
            "obs_loss": obs_loss,
            "reward_loss": reward_loss,
        }

    def _residual_stream_loss(self) -> TensorType:
        """残差加权流：只作用于奖励头（从缓存按权重采样）。"""
        assert self._rw_cache is not None, "residual cache not initialized"
        B = int(self.cache_size * 0.25)
        idx = np.random.choice(
            len(self._rw_cache_weights), size=B, p=self._rw_cache_weights / self._rw_cache_weights.sum()
        )
        obs = self._rw_cache["obs"][idx]
        act = self._rw_cache["act"][idx]
        rew = self._rw_cache["rew"][idx]
        task = self._rw_cache["task"][idx]
        with torch.no_grad():
            z = self.encode(obs)
            h = self.rnn(torch.cat([z, act], dim=-1).unsqueeze(1))[0].squeeze(1)
            task_onehot = self._task_onehot(task)
        logits = self.predict_reward(h, task_onehot)  # (M, B, bins)
        losses = [
            twohot_loss(logits[m], rew, self.twohot, reduction="mean")
            for m in range(self.num_heads)
        ]
        return torch.stack(losses).mean()

    def update(self, buffer, step: int, logger=None, batch_size: Optional[int] = None) -> Dict[str, float]:
        """一次世界模型训练步（双流）。

        Args:
            buffer: ReplayBuffer（并集 buffer，任务均衡采样）。
            step: 环境步数（控制残差缓存刷新）。
            batch_size: 世界模型专属 batch（缺省复用 buffer.batch_size）。

        Returns:
            metrics: 各损失与诊断指标的 python float。
        """
        self.init_optimizers()
        self._step = step

        # 冷启动保护：缓冲不足时跳过
        available = buffer.idx * buffer.task_num if not buffer.full else buffer.capacity
        if available < 2 * buffer.batch_size:
            return {}

        metrics: Dict[str, float] = {}

        # 残差缓存刷新（每 T 步）
        if (
            self._rw_cache is None
            or step - self._last_cache_refresh >= self.cache_refresh_every
        ):
            self.refresh_residual_cache(buffer)
            self._last_cache_refresh = step

        # 流 1: 均匀流（动力学 + 奖励）
        if batch_size is None:
            batch_size = buffer.batch_size
        batch = self._sample_from_buffer(buffer, batch_size)
        obs, act, rew, next_obs, task = self._unpack_batch(batch)
        losses = self._dynamics_and_reward_losses(
            obs, act, rew, next_obs, task, with_obs_stats_update=True
        )
        total = (
            self.state_weight * losses["state_loss"]
            + self.obs_weight * losses["obs_loss"]
            + self.reward_weight * losses["reward_loss"]
        )

        # 流 2: 残差加权流（只奖励头）
        if self._rw_cache is not None:
            res_loss = self._residual_stream_loss()
            total = total + self.reward_weight * res_loss
            metrics["residual_stream_reward_loss"] = float(res_loss.detach().cpu())

        # EMA 编码器更新
        with torch.no_grad():
            for p_ema, p in zip(self.encoder_ema.parameters(), self.encoder.parameters()):
                p_ema.data.mul_(self.encoder_ema_tau).add_(p.data, alpha=1 - self.encoder_ema_tau)

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        for k, v in losses.items():
            metrics[k] = float(v.detach().cpu())
        metrics["wm_loss"] = float(total.detach().cpu())
        metrics["wm_grad_norm"] = float(grad_norm.detach().cpu())

        # 动力学 MAE 诊断（obs 空间，一步）
        with torch.no_grad():
            z = self.encode(obs)
            h = self.rnn(torch.cat([z, act], dim=-1).unsqueeze(1))[0].squeeze(1)
            s_next_pred = self.denormalize_obs(self.normalize_obs(obs) + self.obs_head(h))
            metrics["dyn_obs_mae"] = float(
                (s_next_pred - next_obs).abs().mean().cpu()
            )
        return metrics

    # ------------------------------------------------------------------ #
    # TTA：测试时自适应对准奖励头（GbI §3.4）
    # ------------------------------------------------------------------ #
    def adapt_reward_head(
        self,
        transitions: List[Tuple[TensorType, TensorType, TensorType, TensorType, int]],
        num_steps: int = 1,
        mode: str = "forward+backward",
    ) -> Dict[str, float]:
        """对 recent-N 真实转移微调奖励头（冻结 GRU/encoder/state/obs head）。

        Args:
            transitions: [(s, a, r, s', task_id)]（recent-N=5，真实执行段数据）。
            num_steps: 梯度步数 ∈ {1..3}（由 U_score 触发逻辑控制）。

        Returns:
            metrics: adapt loss 下降等诊断。
        """
        self.init_optimizers()
        old_lr = self.adapt_optimizer.param_groups[0]["lr"]
        # adapt_lr = 0.1 × wm_lr 上限（GbI §3.4）
        effective_lr = min(old_lr, 0.1 * self.lr)
        for pg in self.adapt_optimizer.param_groups:
            pg["lr"] = effective_lr

        losses_before: List[float] = []
        losses_after: List[float] = []

        for (s, a, r, s_next, task_id) in transitions:
            s = s.float().to(self.device).unsqueeze(0)
            a = a.float().to(self.device).unsqueeze(0)
            r = r.float().to(self.device).unsqueeze(0).unsqueeze(0)
            task = torch.tensor([[task_id]], device=self.device).float()
            with torch.no_grad():
                z = self.encode(s)
                h = self.rnn(torch.cat([z, a], dim=-1).unsqueeze(1))[0].squeeze(1)
                onehot = self._task_onehot(task)
                logits0 = self.predict_reward(h, onehot)
                losses_before.append(
                    float(twohot_loss(logits0[0], r, self.twohot, reduction="mean").cpu())
                )
            for _ in range(num_steps):
                z = self.encode(s).detach()  # 冻结编码器输出
                with torch.no_grad():
                    h_det = self.rnn(torch.cat([z, a], dim=-1).unsqueeze(1))[0].squeeze(1).detach()
                    onehot = self._task_onehot(task)
                self.adapt_optimizer.zero_grad(set_to_none=True)
                if self.adapt_scope == "last_layer":
                    h_in = h_det
                    logits = torch.stack([head(h_in, onehot) for head in self.reward_heads], dim=0)
                else:
                    logits = self.predict_reward(h_det, onehot)
                loss = twohot_loss(logits.mean(dim=0), r, self.twohot, reduction="mean")
                loss.backward()
                self.adapt_optimizer.step()
            with torch.no_grad():
                logits1 = self.predict_reward(h_det, onehot)
                losses_after.append(
                    float(twohot_loss(logits1[0], r, self.twohot, reduction="mean").cpu())
                )

        for pg in self.adapt_optimizer.param_groups:
            pg["lr"] = old_lr
        return {
            "adapt_loss_before": float(np.mean(losses_before)),
            "adapt_loss_after": float(np.mean(losses_after)),
        }

    # ------------------------------------------------------------------ #
    # 诊断 / 打分
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reward_bias_report(self, buffer, num_tasks: int, samples: int = 2048) -> Dict[str, np.ndarray]:
        """逐任务奖励偏置表（bias / MAE），Phase 0 验收口径。"""
        bias = np.zeros(num_tasks)
        mae = np.zeros(num_tasks)
        count = np.zeros(num_tasks)
        batch = self._sample_from_buffer(buffer, samples, balanced=True)
        obs, act, rew, next_obs, task = self._unpack_batch(batch)
        z = self.encode(obs)
        h = self.rnn(torch.cat([z, act], dim=-1).unsqueeze(1))[0].squeeze(1)
        r_pred = self._transition_stats(h, self._task_onehot(task))  # (B,)
        r = rew.squeeze(-1)
        tid = task.squeeze(-1).long().cpu().numpy()
        r = r.cpu().numpy()
        r_pred = r_pred.cpu().numpy()
        for i in range(len(r)):
            t = tid[i]
            bias[t] += r_pred[i] - r[i]
            mae[t] += abs(r_pred[i] - r[i])
            count[t] += 1
        count = np.maximum(count, 1)
        return {"bias": bias / count, "mae": mae / count, "count": count}

    def get_last_shared_layers(self) -> List[nn.Module]:
        return [self.state_head, self.obs_head] + list(self.reward_heads)