# GbI: 行为裁决器（GbI.md §3.2/§3.3/§3.5，v5）。
#
# 一元统一公式（§3.2）：
#   S_j(s_t) = Q_φi(s_t, a_j) + λ_t·(Î_j - Î_i)
#   a_j = π_j(s_t)（SAC 均值动作）；Î_j = Σ_{τ=1}^{H-1} γ^{τ-1} r̂(s^j_τ, a^j_τ, z_i)
#   （5 奖励头对同一共享动力学轨迹打分的均值）；λ_t = max(0, Spearman κ_t)，
#   由 L2 标注缓冲（预测相对增益 vs 真实执行增益）在线校准。
#
# U_score 自适应触发（§3.3）：
#   U_score(s) = max_j Var_k(Î^k_j)（评分层分歧，5 奖励头对同一轨迹）；
#   τ_on = 近 W=1e4 步 U 滑窗的 0.9 分位；τ_reject = 0.99 分位；
#   K_min=3 / K_max=15 执行段；连续 m=3 步 U < τ_off=0.7τ_on 提前终止。
#
# 护栏（§3.5）：U > τ_reject → 保持自身策略；换手要求相对增益 > 0。
#   （护栏仅 gbi/imag 生效；qmp 基线无护栏、执行段退化为单步——见 score()/step_segment()）
#
# 模式（agent.gbi.arbiter_mode）：
#   gbi  —— 完整公式（Q 锚 + λ_t 相对想象增益）
#   qmp  —— λ_t ≡ 0（QMP 基线：每步 argmax Q_i）
#   imag —— 纯想象（无 Q 锚：S_j = Î_j）（消融用）
# 触发（agent.gbi.trigger_mode）：
#   adaptive —— U_score 对 τ_on 自适应触发（缺省）
#   fixed_K  —— 每 K 步固定触发（消融，K = agent.gbi.fixed_K）
#   always   —— 每步触发（QMP 缺省行为）

from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from mtrl.agent.components.world_model import WorldModel
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.utils.types import TensorType


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（自实现，避免引入 scipy 依赖）。"""
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    def _rank(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v))
        ranks[order] = np.arange(len(v))
        return ranks
    rx, ry = _rank(np.asarray(x, dtype=np.float64)), _rank(np.asarray(y, dtype=np.float64))
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


class Candidate:
    """候选策略：名称 + actor + 均值/采样动作接口。

    freeze=True 用于磁盘加载的独立快照（冻结防漂移）；
    own 候选传入现役 actor 本体引用，必须 freeze=False ——
    否则 requires_grad_(False) 会关闭现役 actor 的梯度，
    导致训练静默失效（高危 #7，2026-09-02 实锤）。
    """

    def __init__(self, name: str, actor: torch.nn.Module, action_range: Tuple[float, float],
                 device: torch.device, freeze: bool = True):
        self.name = name
        self.actor = actor.to(device).eval()
        if freeze:
            for p in self.actor.parameters():
                p.requires_grad_(False)
        self.action_range = action_range
        self.device = device

    @torch.no_grad()
    def mean_action(self, obs: TensorType, task_ids: TensorType) -> TensorType:
        """(B, obs_dim) x (B, 1) 任务 id → (B, act_dim) 均值动作。

        注意：task actor 的 forward 约定与 sac.Agent.act 一致：
        actor(mtobs) -> (mu, pi, _, _)。
        """
        B = obs.shape[0]
        obs_t = obs.float().to(self.device)
        task_t = task_ids.long().to(self.device).reshape(B, 1)
        mtobs = MTObs(
            env_obs=obs_t,
            task_obs=task_t,
            task_info=TaskInfo(encoding=None, compute_grad=False, env_index=task_t),
        )
        mu, _pi, _, _ = self.actor(mtobs=mtobs)
        lo, hi = self.action_range
        return mu.clamp(lo, hi)

    @torch.no_grad()
    def sample_action(self, obs: TensorType, task_ids: TensorType) -> TensorType:
        """(B, obs_dim) x (B, 1) 任务 id → (B, act_dim) 采样动作（保持探索）。

        执行段内用候选策略的采样动作：裁决用均值 μ，执行用 π_j 采样——
        产生的数据带回来源标记入 buffer（SAC off-policy 可无偏吸收）。
        """
        B = obs.shape[0]
        obs_t = obs.float().to(self.device)
        task_t = task_ids.long().to(self.device).reshape(B, 1)
        mtobs = MTObs(
            env_obs=obs_t,
            task_obs=task_t,
            task_info=TaskInfo(encoding=None, compute_grad=False, env_index=task_t),
        )
        _mu, pi, _, _ = self.actor(mtobs=mtobs)
        lo, hi = self.action_range
        return pi.clamp(lo, hi)


class Arbiter:
    """裁决器：Q 锚 + λ_t·相对想象增益 + U_score 触发/护栏。"""

    def __init__(
        self,
        world_model: WorldModel,
        critic,
        candidates: List[Candidate],
        num_tasks: int,
        device: torch.device,
        H: int = 5,
        gamma: float = 0.99,
        mode: str = "gbi",           # gbi | qmp | imag
        trigger_mode: str = "adaptive",  # adaptive | fixed_K | always
        W: int = 10000,
        rho_on: float = 0.90,
        rho_reject: float = 0.99,
        tau_off_ratio: float = 0.7,
        K_min: int = 3,
        K_max: int = 15,
        m_low: int = 3,
        fixed_K: int = 10,
        calibrate_window: int = 500,
        lambda_ramp: bool = False,
        quantile_refresh_steps: int = 32,
        guardrail_reject: bool = True,
        switch_requires_positive_gain: bool = True,
    ):
        self.wm = world_model
        self.critic = critic
        self.candidates = candidates
        self.num_tasks = num_tasks
        self.device = device
        self.H = H
        self.gamma = gamma
        self.mode = mode
        self.trigger_mode = trigger_mode
        self.W = W
        self.rho_on = rho_on
        self.rho_reject = rho_reject
        self.tau_off_ratio = tau_off_ratio
        self.K_min = K_min
        self.K_max = K_max
        self.m_low = m_low
        self.fixed_K = fixed_K
        self.calibrate_window = calibrate_window
        self.lambda_ramp = lambda_ramp
        self.quantile_refresh_steps = quantile_refresh_steps
        self.guardrail_reject = guardrail_reject
        self.switch_requires_positive_gain = switch_requires_positive_gain
        assert mode in ("gbi", "qmp", "imag")

        # γ^{τ-1} 权重：τ=1..H-1
        disc = torch.tensor(
            [self.gamma ** max(0, tau - 1) for tau in range(1, H)], device=device
        )  # (H-1,)
        self.register_discount(disc)

        # ---- 触发统计：U 滑窗 + 分位阈值（惰性刷新） ----
        self.u_window: Deque[float] = deque(maxlen=W)
        self.tau_on: float = float("inf")     # 启动时 inf → 不触发，直到滑窗 ≥ 512
        self.tau_reject: float = float("inf")
        self._steps_since_quantile = 0

        # ---- λ_t 校准（L2 标注缓冲） ----
        self.calib_deque: Deque[Tuple[float, float]] = deque(maxlen=calibrate_window)
        self.lambda_t: float = 0.0
        self._kappa_t: float = 0.0
        self._steps_since_calib = 0

        # ---- 执行段状态机（每 env） ----
        self.exec: Dict[int, Dict[str, int]] = {}
        self._pending_decision: Dict[int, Dict[str, float]] = {}
        self._decision_counter = 0

        # ---- 逐 env 自身步数（fixed_K 触发用） ----
        self.env_step_counter = np.zeros(num_tasks, dtype=np.int64)

        self._quantile_warmup = 512

    def register_discount(self, disc: TensorType) -> None:
        self.discount = disc  # (H-1,)

    # ------------------------------------------------------------------ #
    # 打分核心
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def q_anchor(self, obs: TensorType, task_ids: TensorType,
                 candidates: Optional[List[Candidate]] = None) -> TensorType:
        """Q 锚：Q_φi(s, π_j(s))，现役 critic（均值动作）。返回 (B, N)。"""
        cands = candidates or self.candidates
        B = obs.shape[0]
        cols = []
        for c in cands:
            a = c.mean_action(obs, task_ids)
            task_t = task_ids.long().to(self.device).reshape(B, 1)
            mtobs = MTObs(
                env_obs=obs.float().to(self.device),
                task_obs=task_t,
                task_info=TaskInfo(encoding=None, compute_grad=False, env_index=task_t),
            )
            q1, q2 = self.critic(mtobs=mtobs, action=a, detach_encoder=False)
            # min 口径（2026-09-02）：与 SAC 值学习（target 用 min）一致，
            # 防止打分高估；此前 (q1+q2)/2 会系统性偏高且偏向过乐观的候选
            cols.append(torch.min(q1, q2).squeeze(-1))
        return torch.stack(cols, dim=1)  # (B, N)

    @torch.no_grad()
    def imagine(
        self,
        obs: TensorType,
        task_ids: TensorType,
        candidates: Optional[List[Candidate]] = None,
    ) -> Tuple[TensorType, TensorType]:
        """想象打分。返回：
        - I_mean: (B, N) 每候选 5 头均值打分 Î_j（Σ_{τ=1}^{H-1} γ^{τ-1} r̂_τ）
        - U: (B, N) 每候选 Var_k(Î^k_j) 评分层分歧
        """
        cands = candidates or self.candidates
        B = obs.shape[0]
        I_mean_list, U_list = [], []
        for c in cands:
            def _act(cur_obs):
                return c.mean_action(cur_obs, task_ids)
            out = self.wm.close_loop_rollout(obs, task_ids, _act, self.H)
            # rewards: (H, M, B)；取 τ=1..H-1（公式 Σ_{τ=1}^{H-1}）
            rewards = out["rewards"][1:]  # (H-1, M, B)
            I_head = (rewards * self.discount.view(-1, 1, 1)).sum(0)  # (M, B)
            I_mean_list.append(I_head.mean(0))   # (B,)
            U_list.append(I_head.var(0))         # (B,)
        I_mean = torch.stack(I_mean_list, dim=1)  # (B, N)
        U = torch.stack(U_list, dim=1)            # (B, N)
        return I_mean, U

    def score(
        self,
        obs: TensorType,
        task_ids: TensorType,
    ) -> Dict[str, TensorType]:
        """完整裁决打分：
        返回 {S(B,N), Q(B,N), I(B,N), U(B,N), chosen(B,), chosen_idx(B,), u_trigger(B,)}。
        task_ids 每 env 是自己的任务 i；batch 内必须同任务（rollout 的 task 条件按
        首个元素设定，Phase 0 评估按任务分批调用）。
        """
        B = obs.shape[0]
        Q = self.q_anchor(obs, task_ids)          # (B, N)
        if self.mode == "qmp":
            # QMP 基线：S=Q 每步 argmax，想象通道（gain/U）不参与裁决。
            # 跳过 close-loop rollout 省约 4× 算力（u_trigger 仅归档用途，填 0；
            # qmp 的 τ 统计无语义，2026-09-02 性能修复）
            I = torch.zeros_like(Q)
            U = torch.zeros_like(Q)
        else:
            I, U = self.imagine(obs, task_ids)    # (B, N), (B, N)
        # 自身基线 = 候选集第一项（index 0 = 现役自身策略）
        I_own = I[:, 0:1]                          # (B, 1)
        Q_own = Q[:, 0:1]
        gain = I - I_own                           # (B, N) 相对想象增益
        if self.mode == "qmp":
            S = Q
        elif self.mode == "imag":
            S = gain
        else:  # gbi
            S = Q + self.lambda_t * gain
        # 护栏：U > τ_reject → 保持自身（S 第 0 列拉满）。
        # 仅 gbi/imag 生效：τ_reject 拒绝属于 GbI 触发机制，QMP 基线（每步
        # argmax Q）不得引入该行为，否则破坏与主实验的“唯一变量”对照。
        u_trigger = U.max(dim=1).values          # (B,)
        chosen = S.argmax(dim=1)                  # (B,)
        if self.guardrail_reject and self.mode != "qmp":
            reject_mask = u_trigger > self.tau_reject
            chosen = torch.where(reject_mask, torch.zeros_like(chosen), chosen)
        # 换手要求相对增益 > 0
        if self.switch_requires_positive_gain and self.mode == "gbi":
            gain_of_chosen = gain.gather(1, chosen.unsqueeze(1)).squeeze(-1)
            chosen = torch.where(gain_of_chosen > 0, chosen, torch.zeros_like(chosen))
        return {
            "S": S, "Q": Q, "I": I, "U": U,
            "chosen": chosen, "u_trigger": u_trigger, "gain": gain,
        }

    # ------------------------------------------------------------------ #
    # 触发 / 执行段状态机
    # ------------------------------------------------------------------ #
    def update_u_statistics(self, u_trigger_values: np.ndarray, step: int) -> None:
        """把本步的 U_trigger 值推入滑窗并惰性刷新分位阈值。"""
        self.u_window.extend([float(v) for v in u_trigger_values])
        self._steps_since_quantile += 1
        if (
            len(self.u_window) >= self._quantile_warmup
            and self._steps_since_quantile >= self.quantile_refresh_steps
        ):
            self._steps_since_quantile = 0
            arr = np.asarray(self.u_window)
            self.tau_on = float(np.quantile(arr, self.rho_on))
            self.tau_reject = float(np.quantile(arr, self.rho_reject))

    @property
    def tau_off(self) -> float:
        return max(0.0, self.tau_off_ratio * self.tau_on)

    def should_trigger(self, env_i: int, u_trigger: float, step: int) -> bool:
        """未在执行段时判断是否发起重规划。"""
        self.env_step_counter[env_i] += 1
        if self.trigger_mode == "always":
            return True
        if self.trigger_mode == "fixed_K":
            return bool(self.env_step_counter[env_i] % self.fixed_K == 0)
        # adaptive
        if len(self.u_window) < self._quantile_warmup:
            return False
        return bool(u_trigger > self.tau_on)

    def begin_segment(self, env_i: int, j: int, delta_pred: float, u_trigger: float) -> int:
        """执行段开始：返回 decision_id。"""
        self._decision_counter += 1
        self.exec[env_i] = {"j": int(j), "steps": 0, "low_count": 0}
        self._pending_decision[env_i] = {
            "decision_id": self._decision_counter,
            "delta_pred": float(delta_pred),
            "u_trigger": float(u_trigger),
        }
        return self._decision_counter

    def step_segment(self, env_i: int, u_exec: float) -> Optional[Dict[str, float]]:
        """执行段推进一步。返回：
        - "continue": 继续执行；
        - "stop": 段结束（并弹出 pending 标注）；
        - None: env 不在执行段。
        """
        if env_i not in self.exec:
            return None
        seg = self.exec[env_i]
        seg["steps"] += 1
        if self.trigger_mode == "always":
            # always 触发（QMP）：每步重新裁决，执行段退化为单步——
            # K_min/K_max/m_low 惯性属于 GbI 触发机制，对基线不适用。
            self.exec.pop(env_i, None)
            return "stop"
        if seg["steps"] < self.K_min:
            return "continue"
        if u_exec < self.tau_off and len(self.u_window) >= self._quantile_warmup:
            seg["low_count"] += 1
        else:
            seg["low_count"] = 0
        # 段边界：清执行段状态、返回 "stop"。pending 标注保留在
        # _pending_decision，由 agent 在归集完最后一步真实 reward 后
        # 调用 record_segment_outcome() 弹出（见 gbi_sac.py 时序说明）。
        if seg["low_count"] >= self.m_low:
            self.exec.pop(env_i, None)
            return "stop"
        if seg["steps"] >= self.K_max:
            self.exec.pop(env_i, None)
            return "stop"
        return "continue"

    def end_segment(self, env_i: int) -> Dict[str, float]:
        self.exec.pop(env_i, None)
        return self._pending_decision.pop(env_i, {})

    def cancel_segment(self, env_i: int) -> None:
        """外力终止执行段（env done 等）：清段状态；
        pending 保留，待 agent 归集完最后一步 reward 后标注。"""
        self.exec.pop(env_i, None)

    # ------------------------------------------------------------------ #
    # λ_t 校准（L2 标注回流）
    # ------------------------------------------------------------------ #
    def record_segment_outcome(
        self,
        env_i: int,
        delta_real: float,
        *,
        decision_id: Optional[int] = None,
    ) -> None:
        """执行段结束后的真实增益标注（agent 在段边界调用）。"""
        pending = self._pending_decision.pop(env_i, None)
        if pending is None:
            return
        self.calib_deque.append((pending["delta_pred"], float(delta_real)))
        self._steps_since_calib += 1
        if self._steps_since_calib >= 32 and len(self.calib_deque) >= 64:
            self._steps_since_calib = 0
            xs = np.asarray([p[0] for p in self.calib_deque])
            ys = np.asarray([p[1] for p in self.calib_deque])
            kappa = spearman_rank(xs, ys)
            self._kappa_t = float(kappa)
            if self.mode != "qmp":
                # QMP 基线 λ_t ≡ 0：κ_t 照常统计（诊断用），但不驱动 λ_t，
                # 保证 arbiter/lambda_t metric 口径与公式一致。
                self.lambda_t = max(0.0, float(kappa))

    def set_lambda(self, value: float) -> None:
        self.lambda_t = float(value)

    # ------------------------------------------------------------------ #
    # 诊断
    # ------------------------------------------------------------------ #
    def metrics(self) -> Dict[str, float]:
        return {
            "arbiter/lambda_t": self.lambda_t,
            "arbiter/kappa_t": self._kappa_t,
            "arbiter/tau_on": self.tau_on,
            "arbiter/tau_reject": self.tau_reject,
            "arbiter/u_window_len": len(self.u_window),
            "arbiter/calib_len": len(self.calib_deque),
        }