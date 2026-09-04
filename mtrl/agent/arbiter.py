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

    task_id（P0.1，2026-09-03）：跨任务候选固定用「候选自己的任务 head」
    出动作（CTPG 同池语义：任务 j 的策略可在任务 i 上被裁决接管）；
    None = legacy 行为，沿用调用方传入的 env 任务 id（快照/own/online 候选）。
    """

    def __init__(self, name: str, actor: torch.nn.Module, action_range: Tuple[float, float],
                 device: torch.device, freeze: bool = True,
                 task_id: Optional[int] = None):
        self.name = name
        self.actor = actor.to(device).eval()
        if freeze:
            for p in self.actor.parameters():
                p.requires_grad_(False)
        self.action_range = action_range
        self.device = device
        self.task_id = task_id

    def _resolve_task_ids(self, task_ids: TensorType, B: int) -> TensorType:
        """task_ids: (B, 1) env 任务 id → (B, 1) 实际用于选 head 的 id。
        跨任务候选覆盖为自身 task_id（多头 mask 按 env_index 选 head）。"""
        task_t = task_ids.long().to(self.device).reshape(B, 1)
        if self.task_id is not None:
            task_t = torch.full_like(task_t, int(self.task_id))
        return task_t

    @torch.no_grad()
    def mean_action(self, obs: TensorType, task_ids: TensorType) -> TensorType:
        """(B, obs_dim) x (B, 1) 任务 id → (B, act_dim) 均值动作。

        注意：task actor 的 forward 约定与 sac.Agent.act 一致：
        actor(mtobs) -> (mu, pi, _, _)。
        """
        B = obs.shape[0]
        obs_t = obs.float().to(self.device)
        task_t = self._resolve_task_ids(task_ids, B)
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
        task_t = self._resolve_task_ids(task_ids, B)
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
        gain_gate_requires_trust: bool = True,
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
        # G3：正增益换手门是否只在想象通道被信任（λ_t>0）时生效（详见 score()）
        self.gain_gate_requires_trust = bool(gain_gate_requires_trust)
        assert mode in ("gbi", "qmp", "imag")
        # 决策场密度诊断（每次 score 刷新，由 metrics() 上报）
        self._diag: Dict[str, float] = {
            "arbiter/gain_abs_median": 0.0,
            "arbiter/gain_max_median": 0.0,
            "arbiter/gain_pos_frac": 0.0,
            "arbiter/q_spread_median": 0.0,
            "arbiter/i_own_median": 0.0,
        }

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

    def own_index(self, task_ids: TensorType) -> TensorType:
        """每行的「自身候选」下标 (B,)。

        跨任务候选（task_id != None）：行任务 id == i → 自身候选 =
        task_id == i 的第一个候选（cross 池下 own = 现役 actor 的任务 i head）。
        legacy 候选（own/快照/online，task_id=None）：自身候选恒为 index 0，
        与 v5 行为完全一致（全 0 向量）。
        """
        B = task_ids.reshape(-1).shape[0]
        ids = task_ids.long().to(self.device).reshape(-1)  # (B,)
        own = torch.zeros(B, dtype=torch.long, device=self.device)
        for idx, c in enumerate(self.candidates):
            if c.task_id is None:
                continue
            own = torch.where(ids == c.task_id, torch.full_like(own, idx), own)
        return own

    # ------------------------------------------------------------------ #
    # 打分核心
    # ------------------------------------------------------------------ #
    def _shared_actor_pool(self, cands: List[Candidate]) -> bool:
        """所有候选是否共享同一个 actor 对象（cross 池：N 个任务 head 同一网络）。
        成立时可把「候选维」折进「batch 维」，一次前向算完所有候选。"""
        return len(cands) > 1 and len({id(c.actor) for c in cands}) == 1

    def _candidate_task_matrix(self, cands: List[Candidate], task_env: TensorType,
                              B: int) -> TensorType:
        """(B, N) 矩阵：第 (i, j) 元 = 候选 j 在行 i 上应用的任务 head id。
        跨任务候选用自身 task_id；legacy 候选（task_id=None）沿用 env 任务 id。"""
        cand_tid = torch.tensor(
            [-1 if c.task_id is None else int(c.task_id) for c in cands],
            dtype=torch.long, device=self.device,
        ).view(1, len(cands))
        env_tid = task_env.view(B, 1).expand(B, len(cands))
        return torch.where(cand_tid >= 0, cand_tid.expand(B, len(cands)), env_tid)

    @torch.no_grad()
    def q_anchor(self, obs: TensorType, task_ids: TensorType,
                 candidates: Optional[List[Candidate]] = None) -> TensorType:
        """Q 锚：Q_φi(s, π_j(s))，现役 critic（均值动作）。返回 (B, N)。

        性能（G4）：旧实现对每个候选各做一次 actor 前向 + 一次 critic 前向
        （N=10 → 20 次 batch=10 的小算子）。现把候选维折进 batch 维：
        actor 共享时 1 次前向算完 (B*N) 行动作，critic 恒为 1 次 (B*N) 前向。
        行布局为 row-major（行 i*N+j 对应 env i / 候选 j），数值与逐候选版一致。
        """
        cands = candidates or self.candidates
        B = obs.shape[0]
        N = len(cands)
        obs_d = obs.float().to(self.device)
        task_env = task_ids.long().to(self.device).reshape(B, 1)
        lo, hi = cands[0].action_range

        # 1) 所有候选的均值动作 a_j = π_j(s)
        if self._shared_actor_pool(cands):
            tid_flat = self._candidate_task_matrix(cands, task_env, B).reshape(-1, 1)
            obs_rep = obs_d.repeat_interleave(N, dim=0)          # (B*N, obs_dim)
            mtobs = MTObs(
                env_obs=obs_rep,
                task_obs=tid_flat,
                task_info=TaskInfo(encoding=None, compute_grad=False, env_index=tid_flat),
            )
            mu, _pi, _, _ = cands[0].actor(mtobs=mtobs)
            acts = mu.clamp(lo, hi).reshape(B, N, -1)            # (B, N, act_dim)
        else:
            acts = torch.stack(
                [c.mean_action(obs, task_ids) for c in cands], dim=1
            )                                                    # (B, N, act_dim)

        # 2) critic 条件始终是任务 i 自己的尺子（与候选无关）→ 可整体批量
        obs_rep = obs_d.repeat_interleave(N, dim=0)              # (B*N, obs_dim)
        task_rep = task_env.repeat_interleave(N, dim=0)          # (B*N, 1) = env 任务 i
        mtobs = MTObs(
            env_obs=obs_rep,
            task_obs=task_rep,
            task_info=TaskInfo(encoding=None, compute_grad=False, env_index=task_rep),
        )
        q1, q2 = self.critic(
            mtobs=mtobs, action=acts.reshape(B * N, -1), detach_encoder=False
        )
        # min 口径（2026-09-02）：与 SAC 值学习（target 用 min）一致，
        # 防止打分高估；此前 (q1+q2)/2 会系统性偏高且偏向过乐观的候选
        return torch.min(q1, q2).reshape(B, N)

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

        性能（G4）：cross 池下 N 个候选共享动力学与 actor，旧实现做 N 次
        独立 close_loop_rollout（N×H×M ≈ 250 次小前向/env step，实测 173ms/步）。
        现合并为一次 (B*N) 行的 rollout：动力学/奖励头条件逐行用 env 的任务 i，
        动作闭包逐行用候选自己的 head j——与 GbI.md §4.5「想象 = 任务 j 的策略在
        任务 i 的奖励场里展开」语义一致，世界模型侧零改动。
        """
        cands = candidates or self.candidates
        B = obs.shape[0]
        N = len(cands)
        obs_d = obs.float().to(self.device)
        task_env = task_ids.long().to(self.device).reshape(B, 1)

        if self._shared_actor_pool(cands):
            tid_flat = self._candidate_task_matrix(cands, task_env, B).reshape(-1, 1)
            obs_rep = obs_d.repeat_interleave(N, dim=0)          # (B*N, obs_dim)
            env_tid_rep = task_env.repeat_interleave(N, dim=0)   # (B*N, 1) 奖励头条件=任务 i
            actor = cands[0].actor
            lo, hi = cands[0].action_range

            def _act_batched(cur_obs):
                mtobs = MTObs(
                    env_obs=cur_obs,
                    task_obs=tid_flat,
                    task_info=TaskInfo(encoding=None, compute_grad=False, env_index=tid_flat),
                )
                mu, _pi, _, _ = actor(mtobs=mtobs)
                return mu.clamp(lo, hi)

            out = self.wm.close_loop_rollout(obs_rep, env_tid_rep, _act_batched, self.H)
            rewards = out["rewards"][1:]                         # (H-1, M, B*N)
            I_head = (rewards * self.discount.view(-1, 1, 1)).sum(0)   # (M, B*N)
            I_mean = I_head.mean(0).reshape(B, N)
            U = I_head.var(0).reshape(B, N)
            return I_mean, U

        I_mean_list, U_list = [], []
        for c in cands:
            def _act(cur_obs, _c=c):
                return _c.mean_action(cur_obs, task_ids)
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
        返回 {S(B,N), Q(B,N), I(B,N), U(B,N), chosen(B,), chosen_idx(B,),
        u_trigger(B,), own_idx(B,), gain(B,N)}。
        task_ids 每 env 是自己的任务 i；own_idx 为每行的自身候选下标
        （cross 池 = 对角；legacy 池 = 全 0，v5 行为不变）。
        """
        B = obs.shape[0]
        own_idx = self.own_index(task_ids)        # (B,)
        Q = self.q_anchor(obs, task_ids)          # (B, N)
        if self.mode == "qmp":
            # QMP 基线：S=Q 每步 argmax，想象通道（gain/U）不参与裁决。
            # 跳过 close-loop rollout 省约 4× 算力（u_trigger 仅归档用途，填 0；
            # qmp 的 τ 统计无语义，2026-09-02 性能修复）
            I = torch.zeros_like(Q)
            U = torch.zeros_like(Q)
        else:
            I, U = self.imagine(obs, task_ids)    # (B, N), (B, N)
        # 自身基线：每行取 own_idx 指向的候选（legacy 池 = index 0）
        I_own = I.gather(1, own_idx.view(-1, 1))  # (B, 1)
        gain = I - I_own                           # (B, N) 相对想象增益
        if self.mode == "qmp":
            S = Q
        elif self.mode == "imag":
            S = gain
        else:  # gbi
            S = Q + self.lambda_t * gain
        # 护栏：U > τ_reject → 保持自身（S 的 own 列拉满）。
        # 仅 gbi/imag 生效：τ_reject 拒绝属于 GbI 触发机制，QMP 基线（每步
        # argmax Q）不得引入该行为，否则破坏与主实验的“唯一变量”对照。
        u_trigger = U.max(dim=1).values          # (B,)
        chosen = S.argmax(dim=1)                  # (B,)
        if self.guardrail_reject and self.mode != "qmp":
            reject_mask = u_trigger > self.tau_reject
            chosen = torch.where(reject_mask, own_idx, chosen)
        # 换手要求相对增益 > 0（§3.5 护栏②）。
        # G3 修复：该门必须受信任度约束。设计 §3.2 声称「κ_t≤0 时系统平滑退化
        # 为纯 QMP」，但原实现里该门与 λ_t 无关、始终生效：λ_t=0 时 S=Q，argmax Q
        # 选中的候选只要想象增益≤0 就被打回 own，系统实际退化成「纯 own 策略」
        # （比 QMP 更保守）——这正是 gbi 臂 SWR 仅 0.4% 而 qmp 臂 31.2% 的机制性原因，
        # 也使「gbi(λ_t=0) ≡ qmp」的消融声明不成立。现改为：仅当 λ_t>0（想象通道
        # 被校准门认可）时才用增益门否决换手；λ_t=0 时严格等价于 QMP。
        if (
            self.switch_requires_positive_gain
            and self.mode == "gbi"
            and (not self.gain_gate_requires_trust or self.lambda_t > 0.0)
        ):
            gain_of_chosen = gain.gather(1, chosen.unsqueeze(1)).squeeze(-1)
            chosen = torch.where(gain_of_chosen > 0, chosen, own_idx)
        self._update_decision_field_diag(Q, I_own, gain, own_idx)
        return {
            "S": S, "Q": Q, "I": I, "U": U,
            "chosen": chosen, "u_trigger": u_trigger, "gain": gain,
            "own_idx": own_idx,
        }

    @torch.no_grad()
    def _update_decision_field_diag(self, Q: TensorType, I_own: TensorType,
                                    gain: TensorType, own_idx: TensorType) -> None:
        """决策场密度诊断。

        Phase A 的 go/no-go 口径明确要求「跨任务决策场 gap 中位数 > 0.1」与
        「gain 分布非退化」，但原实现无任何在线指标可观测（只能事后跑
        eval_decision_field.py 离线复测），跑完 300k 步才知道决策场是不是空的。
        这里把四个关键量随 score() 顺带算出（纯 reduction，开销可忽略）。
        """
        N = gain.shape[1]
        not_own = torch.ones_like(gain, dtype=torch.bool)
        not_own.scatter_(1, own_idx.view(-1, 1), False)
        g_off = gain[not_own] if N > 1 else gain.new_zeros(0)
        if g_off.numel() > 0:
            self._diag["arbiter/gain_abs_median"] = float(g_off.abs().median())
            self._diag["arbiter/gain_pos_frac"] = float((g_off > 0).float().mean())
            row_max = gain.masked_fill(~not_own, float("-inf")).max(dim=1).values
            finite = row_max[torch.isfinite(row_max)]
            self._diag["arbiter/gain_max_median"] = (
                float(finite.median()) if finite.numel() > 0 else 0.0
            )
        else:
            self._diag["arbiter/gain_abs_median"] = 0.0
            self._diag["arbiter/gain_pos_frac"] = 0.0
            self._diag["arbiter/gain_max_median"] = 0.0
        q_spread = Q.max(dim=1).values - Q.min(dim=1).values
        self._diag["arbiter/q_spread_median"] = (
            float(q_spread.median()) if q_spread.numel() > 0 else 0.0
        )
        self._diag["arbiter/i_own_median"] = (
            float(I_own.median()) if I_own.numel() > 0 else 0.0
        )

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
        m = {
            "arbiter/lambda_t": self.lambda_t,
            "arbiter/kappa_t": self._kappa_t,
            "arbiter/tau_on": self.tau_on,
            "arbiter/tau_reject": self.tau_reject,
            "arbiter/u_window_len": len(self.u_window),
            "arbiter/calib_len": len(self.calib_deque),
        }
        m.update(self._diag)
        # G5：τ 的 warmup 哨兵是 +inf，而 Logger 的 AverageMeter 直接累加求均，
        # 一个 inf 会把整个日志窗口的 tau_on/tau_reject 污染成 inf（实测冷启动
        # 窗口 TON/TREJ 均为 inf，丢失全部信息）。未就绪时干脆不上报该键。
        for key in ("arbiter/tau_on", "arbiter/tau_reject"):
            if not np.isfinite(m[key]):
                del m[key]
        return m