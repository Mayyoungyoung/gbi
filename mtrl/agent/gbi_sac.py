# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# (骨架基于 mtrl/agent/sac.py；GbI 裁决通道为新增实现)

# GbI (Guide-by-Imagination, v5) agent —— 在标准 SAC 上加"想象前推裁决"通道。
#
# 设计（GbI.md §3.2/§3.3/§3.4/§3.5，与 arbiter.py 分工：
#   打分/触发/护栏/λ_t 校准在 arbiter.py；本文件负责把它们挂进实验循环）：
#   - act(sample=True) 每步跑裁决打分
#       S_j = Q_φi(s, π_j(s)) + λ_t·(Î_j − Î_i)，
#     换手（chosen j>0）后进入执行段：K_min 护栏、K_max 上限、
#     连续 m 步 U < 0.7τ_on 提前终止；段内动作用 π_j 的采样动作（保持探索，
#     数据带来源标记入 buffer，SAC off-policy 可无偏吸收）。
#   - update() 尾挂世界模型训练（双流 two-hot/symlog）+ 段收尾 L2 标注
#     （delta_pred vs 段真实增益 → Spearman κ_t → λ_t 校准门）
#     + TTA 奖励头适配（recent-N=5 执行段真实转移，步数按 U/τ_on 动态 1..3）。
#   - 来源标记 {step, env, task, source_j, S_chosen, U, triggered} 旁路归档。
#
# 与 experiment 循环的时序协议（实验循环不可改，见 multitask.Experiment.run；
# L3 修复：原注释将 update/act 顺序写反）：
#   [act_t 决策+出动作] → [update_t（读 buffer 最近行 = row_{t-1}，即上一步转移）]
#   → [env.step_t] → [buffer.add(写 row_t)] → [act_{t+1}] → ...
#   因此 update_t 时 buffer 最近写入行 = row_{t-1}（上一步的转移）。
#   agent 在 update 中归集"上一步"的 reward/转移（含执行段最后一步），
#   段收尾、L2 标注、TTA 全部在 update 内完成；act 只负责决策与出动作——
#   这样执行段第 K 步的真实 reward 也能完整计入段增益。
#
# 模式开关（config agent.gbi）：
#   arbiter_mode: gbi | qmp | imag     —— 完整公式 / λ≡0 消融 / 纯想象消融
#   trigger_mode: adaptive | fixed_K | always

from collections import deque
import copy
import os
import time
from typing import Any, Deque, Dict, List, Optional

import hydra
import numpy as np
import torch

from mtrl.agent.arbiter import Arbiter, Candidate
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.agent.sac import Agent as SACAgent
from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer
from mtrl.utils.types import ConfigType, TensorType


class Agent(SACAgent):
    """GbI：SAC + 想象前推裁决。"""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: tuple,
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        discount: float,
        init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float,
        loss_reduction: str = "mean",
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
        logger: Logger = None,
        # ---- GbI 新增 ----
        gbi_cfg: Optional[ConfigType] = None,
        world_model_cfg: Optional[ConfigType] = None,
    ):
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            device=device,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
            alpha_optimizer_cfg=alpha_optimizer_cfg,
            actor_optimizer_cfg=actor_optimizer_cfg,
            critic_optimizer_cfg=critic_optimizer_cfg,
            multitask_cfg=multitask_cfg,
            discount=discount,
            init_temperature=init_temperature,
            actor_update_freq=actor_update_freq,
            critic_tau=critic_tau,
            critic_target_update_freq=critic_target_update_freq,
            encoder_tau=encoder_tau,
            loss_reduction=loss_reduction,
            cfg_to_load_model=None,
            should_complete_init=False,
            logger=logger,
        )
        self.logger = logger
        self.gbi_cfg = dict(gbi_cfg) if gbi_cfg is not None else {}
        self.train_step = 0

        # ---- 世界模型（加入 _components → 自动 checkpoint 保存/恢复） ----
        self.world_model = hydra.utils.instantiate(
            world_model_cfg,
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            num_envs=multitask_cfg.num_envs,
            device=device,
        )
        self._components["world_model"] = self.world_model

        # ---- 候选集：index 0 = 现役自身策略（同一 actor 引用，天然同步）；
        #      其余为 state_sac 训练快照（early/mid/late）
        # freeze=False：own 传入 self.actor 本体引用，若被 Candidate 冻结
        # 则现役 actor 梯度永久关闭（高危 #7，2026-09-02 修复）
        self.candidates: List[Candidate] = [
            Candidate("own", self.actor, self.action_range, device, freeze=False)
        ]
        self._load_snapshot_candidates(
            actor_cfg=actor_cfg,
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            device=device,
        )

        # ---- 在线自举候选（免预训练直接开训，2026-09-02 新增）：
        #      每 interval 步将现役 actor 快照（deepcopy+冻结）入池，
        #      FIFO 保留最近 max_count 个；与磁盘快照池可叠加（先淘汰更老的磁盘快照）
        oc = (self.gbi_cfg.get("online_candidates", None) or {})
        self._oc_enabled = bool(oc.get("enabled", False))
        self._oc_interval = max(1, int(oc.get("interval", 25000)))
        self._oc_warmup = int(oc.get("warmup", 20000))
        self._oc_max_count = max(0, int(oc.get("max_count", 4)))
        self._oc_last_snap = -1
        self._oc_pending_evict = False

        # ---- 裁决器 ----
        self.arbiter = Arbiter(
            world_model=self.world_model,
            critic=self.critic,
            candidates=self.candidates,
            num_tasks=self.num_envs,
            device=device,
            H=int(self.gbi_cfg.get("H", 5)),
            gamma=float(self.gbi_cfg.get("gamma", 0.99)),
            mode=str(self.gbi_cfg.get("arbiter_mode", "gbi")),
            trigger_mode=str(self.gbi_cfg.get("trigger_mode", "adaptive")),
            W=int(self.gbi_cfg.get("W", 10000)),
            rho_on=float(self.gbi_cfg.get("rho_on", 0.9)),
            rho_reject=float(self.gbi_cfg.get("rho_reject", 0.99)),
            tau_off_ratio=float(self.gbi_cfg.get("tau_off_ratio", 0.7)),
            K_min=int(self.gbi_cfg.get("K_min", 3)),
            K_max=int(self.gbi_cfg.get("K_max", 15)),
            m_low=int(self.gbi_cfg.get("m_low", 3)),
            fixed_K=int(self.gbi_cfg.get("fixed_K", 10)),
            calibrate_window=int(self.gbi_cfg.get("calibrate_window", 500)),
        )

        # ---- 执行段 / 标注 / 来源归档 状态 ----
        # _seg[i]: env i 处于执行段中（agent 侧维护 reward 归集与转移列表）
        #          {"j", "rew_cum", "steps_total", "trans": deque(maxlen=N),
        #           "u_start"}
        # 时序对齐（row_{t-1} 由 act_{t-1} 的动作产生，在 update_t 被读到）：
        # _seg_just_created[i]: act_t 刚创建的段——update_t 读到的 row_{t-1}
        #     属于段前策略，不得计入段增益；跳过归集后清除。
        # _seg_finalized[i]: 已关闭待收尾的段——update 还需再归集最后一步的
        #     reward（row 晚一拍写入），归集完毕才做 L2 标注 + TTA 并弹出。
        self._seg: Dict[int, Dict[str, Any]] = {}
        self._seg_just_created: set = set()
        self._seg_finalized: set = set()
        self._closing: List[int] = []      # 段已到边界，等 update 归集最后一步后收尾
        self._env_tasks = np.arange(self.num_envs, dtype=np.int64)
        self._own_reward_ema = np.zeros(self.num_envs, dtype=np.float64)
        self._own_ema_decay = 0.99

        # 来源标记旁路归档
        self._source_records: List[Dict[str, Any]] = []
        self._eval_steps = 0            # 参与 U 打分的 agent 步数（≈每步 1）
        self._trigger_count_total = 0   # 触发重规划的 env 次数
        self._switch_count_total = 0    # 换手（chosen>0）的 env 次数
        self._uscore_ema = 0.0
        self._seg_count_total = 0
        self._seg_len_sum = 0
        self._last_adapt_metrics: Dict[str, float] = {}
        self._tta_recent_n = int(self.gbi_cfg.get("tta_recent_n", 5))

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    # ------------------------------------------------------------------ #
    # 候选快照加载
    # ------------------------------------------------------------------ #
    def _load_snapshot_candidates(
        self,
        actor_cfg: ConfigType,
        env_obs_shape: List[int],
        action_shape: List[int],
        device: torch.device,
    ) -> None:
        candidates_dir = self.gbi_cfg.get("candidates_dir", None)
        candidate_steps = list(self.gbi_cfg.get("candidate_steps", []) or [])
        if not candidates_dir or not candidate_steps:
            print("[gbi] no snapshot candidates configured; candidates=[own]")
            return
        for step in candidate_steps:
            path = os.path.join(candidates_dir, f"actor_{step}.pt")
            if not os.path.exists(path):
                print(f"[gbi] snapshot candidate missing, skip: {path}")
                continue
            actor = hydra.utils.instantiate(
                actor_cfg,
                env_obs_shape=env_obs_shape,
                action_shape=action_shape[0] * 2,
            )
            actor.load_state_dict(torch.load(path, map_location="cpu"))
            self.candidates.append(
                Candidate(f"snap_{step}", actor, self.action_range, device)
            )
            print(f"[gbi] loaded snapshot candidate snap_{step} from {path}")
        print(f"[gbi] candidates: {[c.name for c in self.candidates]}")

    # ------------------------------------------------------------------ #
    # act 挂载（sample 分支接仲裁）
    # ------------------------------------------------------------------ #
    def act(
        self,
        multitask_obs: ObsType,
        modes: List[str],
        sample: bool,
    ) -> np.ndarray:
        """eval（sample=False）照旧用自身均值策略；训练分支走 gbi_sample。"""
        if not sample or not bool(self.gbi_cfg.get("enabled", True)):
            return super().act(multitask_obs=multitask_obs, modes=modes, sample=sample)
        return self.gbi_sample(multitask_obs=multitask_obs, modes=modes)

    def gbi_sample(
        self,
        multitask_obs: ObsType,
        modes: List[str],
    ) -> np.ndarray:
        self.train(True)
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            obs = obs.reshape(obs.shape[0], -1)  # (B, obs_dim)
            B = obs.shape[0]
            task_t = env_index.long().reshape(B, 1)
            self._env_tasks = task_t.squeeze(-1).cpu().numpy().astype(np.int64)

            # 1) 全量裁决打分（Q 锚 + λ_t·想象增益；护栏内嵌在 score 里）
            res = self.arbiter.score(obs, task_t)
            u_trigger_np = res["u_trigger"].detach().cpu().numpy()  # (B,)
            chosen_np = res["chosen"].detach().cpu().numpy()        # (B,)
            gain_np = res["gain"].detach().cpu().numpy()            # (B, N)
            S_np = res["S"].detach().cpu().numpy()

            # 2) U 滑窗统计（惰性刷新 τ_on / τ_reject）
            self.arbiter.update_u_statistics(u_trigger_np, self.train_step)

            # 3) 自身策略的批量采样动作（clip 后转 numpy，供装配）
            mu, pi, _, _ = self.actor(
                mtobs=MTObs(
                    env_obs=obs,
                    task_obs=task_t,
                    task_info=TaskInfo(
                        encoding=None, compute_grad=False, env_index=task_t
                    ),
                )
            )
            lo, hi = self.action_range
            own_act = pi.clamp(lo, hi).cpu().numpy()  # (B, act_dim)

            # 4) 逐 env 决策与动作装配
            actions = np.zeros((B, self.action_shape[0]), dtype=np.float32)
            sources = np.zeros(B, dtype=np.int64)
            triggered = np.zeros(B, dtype=bool)

            for i in range(B):
                u_i = float(u_trigger_np[i])
                if i in self._closing:
                    # 安全网：段已到边界但 update 尚未收尾（正常时序不会走到）
                    actions[i] = own_act[i]
                    continue
                seg = self._seg.get(i)
                if seg is not None:
                    # 段内：用候选 π_j 的采样动作执行（本步计入段），
                    # 再推进段状态机（K 上限 / 低 U 提前终止）。
                    c = self.candidates[seg["j"]]
                    actions[i] = (
                        c.sample_action(obs[i : i + 1], task_t[i : i + 1])
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )
                    seg["steps_total"] += 1
                    sources[i] = int(seg["j"])
                    triggered[i] = True
                    status = self.arbiter.step_segment(i, u_i)
                    if status == "stop":
                        # 段到边界：pending 保留，等 update 归集完最后一步
                        # reward 后走 record_segment_outcome + TTA。
                        self._closing.append(i)
                    continue
                # 空闲 env：U 触发判断
                if self.arbiter.should_trigger(i, u_i, self.train_step):
                    self._trigger_count_total += 1
                    triggered[i] = True
                    j = int(chosen_np[i])
                    if j > 0:
                        c = self.candidates[j]
                        actions[i] = (
                            c.sample_action(obs[i : i + 1], task_t[i : i + 1])
                            .detach()
                            .cpu()
                            .numpy()[0]
                        )
                        delta_pred = float(gain_np[i, j])
                        self.arbiter.begin_segment(i, j, delta_pred, u_i)
                        self._seg[i] = {
                            "j": j,
                            "rew_cum": 0.0,
                            "steps_total": 1,
                            "trans": deque(maxlen=2 * self._tta_recent_n),
                            "u_start": u_i,
                        }
                        # 本步创建：紧接的 update 读到的 row 属于段前策略，跳过归集
                        self._seg_just_created.add(i)
                        self._switch_count_total += 1
                        self._seg_count_total += 1
                        sources[i] = j
                        continue
                actions[i] = own_act[i]

            # 5) 台账与来源归档
            self._eval_steps += 1
            self._uscore_ema = 0.99 * self._uscore_ema + 0.01 * float(
                u_trigger_np.mean()
            )
            for i in range(B):
                j_i = int(sources[i]) if sources[i] < len(self.candidates) else 0
                self._source_records.append(
                    {
                        "step": int(self.train_step),
                        "env": int(i),
                        "task": int(self._env_tasks[i]),
                        "source": int(sources[i]),
                        "source_name": self.candidates[j_i].name,
                        "triggered": bool(triggered[i]),
                        "u": float(u_trigger_np[i]),
                        "s_chosen": float(S_np[i, chosen_np[i]]),
                    }
                )
            return actions  # (B, act_dim) 与 sac.act 协议一致

    # ------------------------------------------------------------------ #
    # update 尾挂：世界模型训练 + 段收尾（L2 标注 / TTA）+ metric 落盘
    # ------------------------------------------------------------------ #
    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """SAC 更新 + GbI 尾挂。"""
        ret = super().update(
            replay_buffer=replay_buffer,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=kwargs_to_compute_gradient,
            buffer_index_to_sample=buffer_index_to_sample,
        )
        self.train_step = step

        # ---- 归集上一个 env step 的真实数据（段推进 / own ema / 收尾） ----
        self._post_update_bookkeeping(replay_buffer)

        # ---- 世界模型训练（双流） ----
        wm_every = int(self.gbi_cfg.get("wm_update_every", 1))
        if step >= wm_every and step % wm_every == 0:
            self.world_model.train()
            wm_batch = int(self.gbi_cfg.get("wm_batch_size", 1024))
            wm_metrics = self.world_model.update(
                replay_buffer, step, logger=None, batch_size=wm_batch
            )
            for key, value in wm_metrics.items():
                log_key = key if key.startswith("wm_") else f"wm_{key}"
                logger.log(f"train/{log_key}", value, step)

        # ---- 裁决器/台账 metric ----
        arb = self.arbiter.metrics()
        for key, value in arb.items():
            logger.log(f"train/{key}", value, step)
        trigger_rate = 0.0
        switch_rate = 0.0
        if self._eval_steps > 0:
            trigger_rate = self._trigger_count_total / (
                self._eval_steps * self.num_envs
            )
            switch_rate = self._switch_count_total / (
                self._eval_steps * self.num_envs
            )
        logger.log("train/gbi_trigger_rate", trigger_rate, step)
        logger.log("train/gbi_switch_rate", switch_rate, step)
        logger.log("train/gbi_uscore_ema", self._uscore_ema, step)
        if self._seg_count_total > 0:
            logger.log(
                "train/gbi_seg_len_mean",
                self._seg_len_sum / self._seg_count_total,
                step,
            )
            logger.log("train/gbi_seg_count", float(self._seg_count_total), step)
        for key, value in self._last_adapt_metrics.items():
            logger.log(f"train/gbi_{key}", value, step)

        # ---- 在线自举候选：到点注册新快照；段活跃期间延迟淘汰（避免段引用的
        #      候选 index 因 FIFO 平移而错位，段最长 K_max+1 步，延迟无碍） ----
        if self._oc_enabled and step >= self._oc_warmup and (
            self._oc_last_snap < 0 or step - self._oc_last_snap >= self._oc_interval
        ):
            self._register_online_candidate(step)
        elif self._oc_pending_evict and not self._seg:
            self._evict_old_candidates()

        # ---- 来源归档落盘 ----
        dump_every = int(self.gbi_cfg.get("source_log_dump_every", 10000))
        if len(self._source_records) >= dump_every:
            self._dump_source_log(final=False)

        return ret

    def _post_update_bookkeeping(self, replay_buffer: ReplayBuffer) -> None:
        """读 buffer 最近写入行（上一步转移），推进段 reward 归集并收尾。

        时序（与文件头协议一致）：update_t 读到 row_{t-1}，由 act_{t-1} 的动作产生。
        因此：① act_t 刚创建的段，本次读到的 row 属于段前策略（自身策略），
        计入 own EMA 而非段；② 已关闭（_closing）的段，其最后一步 reward 晚一拍
        写入，本次归集完毕后才做 L2 标注 + TTA 并弹出。
        """
        if replay_buffer.idx == 0 and not replay_buffer.full:
            return
        last_row = (replay_buffer.idx - 1) % replay_buffer.single_capacity
        rewards = replay_buffer.rewards[last_row]        # (task_num, 1)
        env_obses = replay_buffer.env_obses[last_row]    # (task_num, *obs_shape)
        actions = replay_buffer.actions[last_row]        # (task_num, *act_shape)
        next_env_obses = replay_buffer.next_env_obses[last_row]
        not_dones = replay_buffer.not_dones[last_row]    # (task_num, 1)

        closing_set = set(self._closing)
        done_finalize: List[int] = []   # 本步归集完最后一步、可以收尾的段
        for i, task in enumerate(self._env_tasks):
            r = float(rewards[task, 0])
            seg = self._seg.get(i)
            just_created = i in self._seg_just_created
            if just_created:
                self._seg_just_created.discard(i)
            if seg is None:
                # 自身策略执行步（含触发判“自身”的步）→ own 收益 EMA
                self._own_reward_ema[i] = (
                    self._own_ema_decay * self._own_reward_ema[i]
                    + (1 - self._own_ema_decay) * r
                )
                continue
            if just_created and i not in closing_set:
                # 段创建于本步的 act：读到的 row 由段前（自身）策略产生，
                # 不入段增益，按自身策略步计入 own EMA。
                self._own_reward_ema[i] = (
                    self._own_ema_decay * self._own_reward_ema[i]
                    + (1 - self._own_ema_decay) * r
                )
                continue
            # 段内步（含 closing 段最后一步）reward 归集到段
            seg["rew_cum"] += r
            if i in closing_set:
                # 最后一步归集完毕 → 本 update 内收尾（L2 标注 + TTA）
                done_finalize.append(i)
            elif float(not_dones[task, 0]) > 0:
                # 转移留档（TTA recent-N）：排除 done 步
                # （done 步的 next_env_obs 已是下一 episode 状态，语义不符）
                seg["trans"].append(
                    (
                        torch.from_numpy(
                            np.asarray(env_obses[task], dtype=np.float32)
                            .reshape(-1)
                            .copy()
                        ),
                        torch.from_numpy(
                            np.asarray(actions[task], dtype=np.float32)
                            .reshape(-1)
                            .copy()
                        ),
                        torch.tensor(r, dtype=torch.float32),
                        torch.from_numpy(
                            np.asarray(next_env_obses[task], dtype=np.float32)
                            .reshape(-1)
                            .copy()
                        ),
                        int(task),
                    )
                )
                if len(seg["trans"]) > self._tta_recent_n:
                    seg["trans"] = deque(
                        list(seg["trans"])[-self._tta_recent_n :],
                        maxlen=2 * self._tta_recent_n,
                    )

        # ---- 收尾：最后一步 reward 已归集的段完成 L2 标注 + TTA ----
        for i in done_finalize:
            self._closing.remove(i)
            seg = self._seg.pop(i, None)
            if seg is None:
                continue
            self._seg_len_sum += seg["steps_total"]
            # 真实增益 = 段累计奖励 − K·(自身策略每步收益 EMA)
            # （与 delta_pred = Î_j − Î_i 同为“相对自身”口径，Spearman 只取秩）
            delta_real = seg["rew_cum"] - seg["steps_total"] * self._own_reward_ema[
                i
            ]
            self.arbiter.record_segment_outcome(i, float(delta_real))
            # TTA：U 触发过的高分歧执行段 → 奖励头适配（§3.4）
            if bool(self.gbi_cfg.get("tta_enabled", True)) and len(seg["trans"]) > 0:
                num_steps = self._tta_num_steps(
                    float(seg["u_start"]), self.arbiter.tau_on
                )
                self._last_adapt_metrics = self.world_model.adapt_reward_head(
                    list(seg["trans"]), num_steps=num_steps
                )

    def _tta_num_steps(self, u_start: float, tau_on: float) -> int:
        """TTA 步数 ∈ {1..3}，按段首 U/τ_on 动态（§3.4）。"""
        if tau_on is None or tau_on == float("inf") or tau_on <= 0:
            return 1
        ratio = u_start / tau_on
        if ratio < 1.0:
            return 1
        if ratio < 2.0:
            return 2
        return 3

    # ------------------------------------------------------------------ #
    # episode 边界 / 归档
    # ------------------------------------------------------------------ #
    def reset_at_begin(self, index: int, step: int) -> None:
        """env done：若在执行段中，强制终止段并标记收尾（下一 update
        归集完该 env 的最后一步 reward 后完成 L2 标注 + TTA）。"""
        if index in self._seg and index not in self._closing:
            self.arbiter.cancel_segment(index)
            self._closing.append(index)

    def _register_online_candidate(self, step: int) -> None:
        """deepcopy 现役 actor 入池（冻结）；超出 max_count 时 FIFO 淘汰
        index 1 起的最老快照（candidates[0] 恒为 own 现役引用）。"""
        snap = copy.deepcopy(self.actor)
        cand = Candidate(
            f"online_{step}", snap, self.action_range, self.device, freeze=True
        )
        self.candidates.append(cand)
        self._oc_last_snap = step
        if len(self.candidates) > 1 + self._oc_max_count:
            if self._seg:
                self._oc_pending_evict = True  # 段活跃，延迟到段清空后再淘汰
            else:
                self._evict_old_candidates()
        print(
            f"[gbi] online candidate registered: {cand.name} "
            f"(pool={[c.name for c in self.candidates]})"
        )

    def _evict_old_candidates(self) -> None:
        """FIFO 淘汰最老快照，直到回到 1(own) + max_count 个候选。"""
        self._oc_pending_evict = False
        evicted = []
        while len(self.candidates) > 1 + self._oc_max_count:
            evicted.append(self.candidates.pop(1).name)
        if evicted:
            print(f"[gbi] online candidates evicted (FIFO): {evicted}")

    def _dump_source_log(self, final: bool = False) -> None:
        if not self._source_records:
            return
        log_dir = getattr(self.logger, "_log_dir", None) or "."
        path = os.path.join(
            log_dir, f"gbi_source{'_final' if final else f'_step{self.train_step}'}.npz"
        )
        records = self._source_records
        self._source_records = []
        arrays = {
            "step": np.asarray([r["step"] for r in records], dtype=np.int64),
            "env": np.asarray([r["env"] for r in records], dtype=np.int64),
            "task": np.asarray([r["task"] for r in records], dtype=np.int64),
            "source": np.asarray([r["source"] for r in records], dtype=np.int64),
            # FIFO 淘汰会使 source index 跨时间漂移，分析时按 source_name 聚合
            "source_name": np.asarray(
                [r.get("source_name", "") for r in records], dtype="U32"
            ),
            "triggered": np.asarray([r["triggered"] for r in records], dtype=bool),
            "u": np.asarray([r["u"] for r in records], dtype=np.float32),
            "s_chosen": np.asarray([r["s_chosen"] for r in records], dtype=np.float32),
        }
        np.savez_compressed(path, **arrays)
        print(f"[gbi] dumped {len(records)} source records to {path}")