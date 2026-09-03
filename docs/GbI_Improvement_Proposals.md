# GbI 改进提案：根因诊断 × 文献借鉴 × v6 路线

> 生成时间：2026-09-03 05:30 UTC　|　基于：正在运行的 benchmark 批次（未中断）、GbI.md v5、代码级审查（行号可点）、2023–2026 顶会文献检索
> 关联文档：[GbI.md](GbI.md)（v5 设计）、[GbI_Experiment_Status.md](GbI_Experiment_Status.md)（运行状态）、[GbI_Experiment_Guide.md](GbI_Experiment_Guide.md)（复现指南）

## TL;DR

1. **当前批次最大的发现不是"QMP 更强"，而是实现与设计发生了结构性偏离**：候选池被实现为"同一条训练轨迹的时间快照"（执行时仍用任务 i 自己的 head），而设计（GbI.md §2.1/§3.5）与竞品 CTPG 的候选池都是**跨任务策略**。时间快照池 → 候选之间真实差异极小（own 最优率 0.8–0.9）→ 想象增益与真实增益的 Spearman κ_t 退化为噪声 → λ_t 归零 → GbI 退化成"迟钝版 QMP"。**这不是方法失败，是方法还没被真正测过。**
2. 三个根因：**R1 候选池结构偏离**（致命）、**R2 校准标签非配对**（gbi_sac.py:523 用 own EMA 当反事实基线）、**R3 触发模式混杂**（gbi=adaptive vs qmp=always，λ_t 效应与重判频率效应未分离）。
3. 改进按优先级：**P0 恢复跨任务候选池 + 解耦触发**（前提修复，约 50 行改动）→ **P1 跨任务反事实闭环**（L2 语义恢复）→ **P2 成功率头**（指标对齐）→ **P3 配对校准 + doubly-robust 残差修正** → **P4 状态局部信任门 λ_t(s)** → **P5 软仲裁（蒸馏/动作混合）**。
4. 文献给出两个关键正当性支撑：世界模型**保排序但失绝对值、OOD 区高估**（WPE, 2025）——GbI 的"差分设计 + 不确定性门控"恰好是对的方向，缺的是真候选池和真标签；**DR 估计器**（OPE 文献）可以直接搬进在线仲裁做增益修正，是论文级创新点。
5. 论文叙事三选一：A. 方法论文（Computation vs Memorization，同池对打 CTPG）；B. 机制研究论文（When Does Imagination Help Arbitration?——条件刻画 + 诊断框架）；C. OPE 视角（在线仲裁 = 逐状态 DR 策略评估）。

---

## 1. 当前实验快照（2026-09-03 04:55 UTC，未中断）

| stage | 步数/300k | 状态 | 同步数成功率（最近 eval 点） |
|---|---|---|---|
| qmp:0 | 300k | ✅ COMPLETED @ 02:01 | **0.60**（280k，7/10 任务 ≥50%） |
| gbi_online:0 | ~274k | 91%，ETA ~07:20 | 0.25（260k） |
| gbi:0 | ~164k | 55%，ETA ~22:40 | 0.29（160k） |
| qmp_online:0 | ~62k | 21%，ETA ~16:45 | 0.08（60k） |
| guide:0 (CTPG) | 排队 | gbi_online 结束后自动补位 | — |

关键观测：

- `qmp`（λ≡0、每步重判）末期 reward 91.1，超过 `state_sac_indep` 基线 400k 的 81.4——**裁决框架本身有效**；
- `gbi`/`gbi_online` 的 λ_t 在 100k–137k 区间精确归零、κ_t 为负（−0.003 ~ −0.055），想象通道被校准门关闭——系统在按设计"安全退化"，但退化原因是 R1+R2 的噪声，不是真信号；
- 触发率对比：gbi TGR≈5–6%（几乎不重判，段长 13.3 步）vs qmp TGR≈68%（段长 2.0 步）——**每步重判本身就带来巨大收益**，与 λ_t 无关（R3 混杂）；
- 全程无 NaN/traceback，ES_ok 全 True，监控的 α 塌缩告警（qmp α≈0.017）与高性能并存，属观察项。

注意：`reports/benchmark_report.md` 是 09-02 基于已归档"白跑"数据的过期报告，勿引用；`state_sac_indep` 的 success 记录全废（#6 bug），对基线只能用 episode_reward。

---

## 2. 根因诊断（代码级证据）

### R1（致命）：候选池结构偏离设计——"时间快照"冒充了"跨任务策略"

**设计意图**（GbI.md §2.1、§3.5）：候选 = 任务控制策略 π_j, j∈{1..N}，"候选 π_j 在任务 i 状态上的 (s_i, a_j) 转移可能从未入池"——整个 L1/L2 反事实覆盖、跨任务指导叙事都建立在**跨任务候选**上。

**实现真相**：

- `gbi_sac.py:116-128`：候选 = index 0 的 own + `_load_snapshot_candidates` 加载的 10 个磁盘快照（50k–500k，**state_sac_indep 同一条训练轨迹的整网快照**）；
- `arbiter.py:72-108`：候选执行时以**当前 env 的任务 id i** 构造 MTObs——即取"快照 j 的任务 i head"。**没有任何跨任务成分**；
- 在线池（`gbi_sac.py:559-576`）更窄：现役 actor 每 25k 步快照一次、FIFO 保留 4 个——候选间差异只有"最近 100k 步的训练进度"。

**对照 CTPG 原版**（本地即官方代码库）：`guide_sac.py:96-100` 的 guide actor `action_shape=multitask_cfg.num_envs`——guide 的动作空间就是"选哪个任务策略"，候选池 = N 个任务 head（跨任务）。

**后果链**（每一环都有实测对应）：

```
时间快照池（R1）
  → 候选间真实增益差异 ≈ 0（Phase 0 own 最优率 0.8–0.9，gap 中位数 0.02，"决策场空"）
  → 想象增益 Î_j−Î_i 与真实增益都是噪声（R2 的标签噪声叠加）
  → Spearman κ_t 在 0 附近随机摆动，长时间为负
  → λ_t = max(0, κ_t) 归零（arbiter.py:401，按设计"安全退化"）
  → GbI = Q 锚 + 5–6% 触发率的迟钝 QMP（还付了世界模型的算力：3.9 步/s vs 9 步/s）
```

**为什么这是一个"好消息"**：它统一解释了所有阴性结果（决策场空、κ_t 负、QMP 反超、λ_t 归零），且**修复成本极低**。跨任务候选被换上后，想象通道要回答的问题从"过去的我 vs 现在的我"（几乎没有信息）变成"任务 j 的技能接管任务 i 会怎样"（信息量大、且有 CTPG 证明存在可挖的增益）。

### R2：κ_t 校准标签不是配对反事实

`gbi_sac.py:523-527`：

```python
# 真实增益 = 段累计奖励 − K·(自身策略每步收益 EMA)
delta_real = seg["rew_cum"] - seg["steps_total"] * self._own_reward_ema[i]
```

问题：(a) own EMA 是跨状态的平均基线，**混杂了状态价值差异**——裁决恰好发生在高 U_score 的异常状态，那里的奖励分布天然偏离均值；(b) 单段单标签，方差极大；(c) own EMA 本身漂移。Spearman 在这种标签上算出来就是噪声。**即使候选池修好，这个口径也会污染 κ_t。**

### R3：触发模式混杂，λ_t 效应无法归因

gbi 系配 `trigger_mode=adaptive`（K_min=3/K_max=15，段长实测 13.3），qmp 系配 `always`（每步重判，段长 2.0）。**gbi vs qmp 的差距同时含两个变量**。设计文档 §3.3 自己也承认"实测 QMP 每步重判体系列最强"——但这个结论是在坏候选池上得出的；跨任务候选下假设可能反转（技能迁移需要连续执行，每步硬切换可能摧毁行为连贯性，CTPG 用 K 步引导正因如此）。**必须跑 trigger 消融。**

### R4：次要问题清单（修复顺手、不单独致命）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| a | H=5 且只累 4 步奖励（γ⁰..γ³），想象用 mean 动作 | arbiter.py:164-167, 218-243 | 视距短 + 低方差但悲观；跨任务候选下 H 可能需要 ≥8 |
| b | 世界模型无 success 头，奖励头只学 dense reward | world_model.py（grep 无 success）；two-hot 界 ±400 对应 dense reward | MT10 评测指标是 success；dense reward 是噪声代理。#6 修复后 success 信号已可用但没被模型用 |
| c | λ_t 全局标量、阶跃更新，`lambda_ramp` 参数存在但未实现 | arbiter.py:134,157（仅存不用） | κ_t 抖动 → λ_t 抖动；无平滑 |
| d | 执行段内每步仍全量计算 N 候选打分，仅不用结果 | gbi_sac.py:296-311 注释 | 纯算力浪费（约 40% 打分开销） |
| e | 奖励头 loss 平台在 ~0.44（WRL 曲线全程平） | train.log | two-hot+symlog 修了口径但没修好排序能力；跨任务候选对排序质量要求更高 |
| f | q_anchor 已改 min(q1,q2)（防高估，与 SAC 一致） | arbiter.py:197-216 | 合理，保留 |
| g | 单 seed（RUN_STAGES 只有 :0） | benchmark_runner.log | n=1 无法报显著性 |

---

## 3. 文献检索结果与可借鉴思想

### 3.1 直接竞品与近邻

| 工作 | 机制 | 对 GbI 的启示 |
|---|---|---|
| **CTPG**（He et al., NeurIPS 2024）[arXiv](https://arxiv.org/abs/2507.06615) / [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d5cd70b708f726737e2ebace18c3f71b-Abstract-Conference.html) / [代码](https://github.com/DarkDawn233/CTPG) | guide policy 从**全部任务策略**中选一个接管执行 K 步；policy-filter gate 拒绝坏指导；hindsight correction 处理非平稳 | 候选池应是跨任务的（R1 的修复方向）；"三笔账单"叙事（guide 训练债/门控网络债/K 调参债）只有在**同池对比**下才能兑现——把 CTPG 的 guide 换成 GbI 裁决器是唯一干净归因 |
| **Diffusion Model as Planner & Data Synthesizer for MTRL**（He et al., NeurIPS 2023）[arXiv](https://arxiv.org/abs/2305.18459) | 扩散规划器在多任务间合成轨迹迁移知识 | 同作者前作；GbI.md §8 未决项——它做的是**轨迹级**迁移，GbI 做**策略级裁决**，差异化成立，可在 related work 明确切割 |
| **To Mix or To Merge**（2026）[arXiv](https://arxiv.org/html/2602.12566v1) | 多任务联合训练（mix）有协同效应 vs 权重合并（merge）只继承单任务技能；多任务训练用 33% GPU 时数达到可比性能 | 支撑"在行为层混用多任务策略"的合理性；merge 视角可作为候选生成的备选（见 P6） |
| **Merging Decision Transformers** [OpenReview](https://openreview.net/forum?id=7NcrDeuMM8) | 权重平均形成多任务 policy | 同上，候选多样化工具箱 |

### 3.2 世界模型评估策略：GbI 差分设计的正当性证据

| 工作 | 发现 | 启示 |
|---|---|---|
| **WPE: Evaluating Robot Policies in a World Model**（2025）[arXiv](https://arxiv.org/html/2506.00613v1) | 用动作条件视频世界模型评估策略：**保排序（ranking preserved）**，但 in-distribution 低估、**OOD 高估** | 直接支撑 GbI 的两个核心设计：(1) 只用差分/排序不用绝对值——对；(2) OOD 必须门控——κ_t/U_score 的方向对，但现在的全局标量粒度不够（→ P4 状态局部门） |
| **TD-MPC2**（ICLR 2024）/ **DreamerV3**（Nature） | state-based RSSM 多任务训练成熟方案 | 已是 GbI 基座，无新增动作 |

### 3.3 不确定性、悲观与门控

| 工作 | 机制 | 启示 |
|---|---|---|
| **MOPO 系 / Combining Pessimism with Optimism**（ICML 2021）[link](https://icml.cc/virtual/2021/poster/9877) | ensemble 分歧 → 奖励惩罚 → 悲观规划 | 把"分歧惩罚"从规划目标搬到**信任门**：λ_t(s) = max(0,κ_t)·exp(−β·U(s))（P4） |
| **Uncertainty-Aware Robotic World Model**（2025）[arXiv](https://arxiv.org/html/2504.16680v3) | 不确定性感知世界模型做离线 MBRL | 离线版"用不确定性约束模型信任"的在线对应 |
| **Confidence-Gated Progress Reward Modeling**（2026）[arXiv](https://arxiv.org/html/2606.22027v4) | 进度奖励建模 + 置信度门控 | 与 P2 成功率/进展头同思路：预测"任务完成进度"比回归 dense reward 更稳 |

### 3.4 OPE / DR：把"在线仲裁"形式化为"逐状态策略评估"

| 工作 | 机制 | 启示 |
|---|---|---|
| **Doubly Robust OPE**（Jiang & Li, ICML 2016）[PDF](http://proceedings.mlr.press/v48/jiang16.pdf) | 估计 = 模型项 + 真实观测残差修正，两者只要一个无偏则估计无偏 | **P3 的理论骨架**：候选增益估计 = 想象 Î_j + 该候选历史执行段的（真实回报 − 当时预测）残差修正；把 GbI 从"启发式仲裁"升级为"在线 DR 策略评估"，这是论文级创新点 |
| **More Robust DR (MRDR)**（ICML 2018）[PDF](https://mohammadghavamzadeh.github.io/PUBLICATIONS/icml18-MRDR.pdf） | 学习残差项最小化估计方差 | 残差修正项的学习目标设计 |
| **DR for Ranking Policies**（2022）[ACM](https://dl.acm.org/doi/abs/10.1145/3488560.3498380) | DR 用于策略排序而非单值估计 | 排序场景的 DR 使用范式 |

### 3.5 多任务世界模型与评测协议

| 工作 | 机制 | 启示 |
|---|---|---|
| **Mixture-of-World Models (MoWM)**（ICLR 2026）[arXiv](https://arxiv.org/abs/2602.01270) / [OpenReview](https://openreview.net/forum?id=qUQARlAx5y) | 模块化 VAE + 混合 Transformer 专家解决多任务世界模型的**任务干扰**，Meta-World 74.5%，参数比 ensemble 少 50% | GbI 的 5 成员"共享动力学、奖励头分叉"是简化版 mixture；若跨任务候选下奖励头仍不准，MoWM 式专家路由是升级路径（P2 备选） |
| **Meta-World+**（2025）[PDF](https://openreview.net/pdf/e99efd74499d0685f991408922f91b3462404e77.pdf) | 标准化改进的 Meta-World 协议 | 最终论文的评测协议建议对齐 Meta-World+（避免 reviewer 挑评测毛病）；多 seed + 置信区间强制 |

### 3.6 策略库 / 技能库 / TTA

| 工作 | 机制 | 启示 |
|---|---|---|
| **SkillRL**（ICLR 2026）[OpenReview](https://openreview.net/pdf?id=56D2hjARkn) / **RL with Skill Library** [arXiv](https://arxiv.org/html/2512.17102v2) | 递归技能增剂演化智能体 | "维护策略/技能库并组合复用"的近期正当性；GbI 的候选池 = 隐式技能库，可在 related work 挂靠 |
| **Promoting Exploration of Ensemble Policies**（NeurIPS 2023）[link](https://neurips.cc/virtual/2023/poster/72558) | ensemble 策略显式促探索 | U_score 的第二用途：高分歧候选可作为**探索加分**（bandit UCB 视角）而非只做悲观惩罚（P6） |
| **TTA with Binary Feedback**（ICML 2025）[arXiv](https://arxiv.org/html/2505.18514v1) | 二值反馈驱动的测试时自适应 | P2 的 success 头 + TTA 可以共用二值信号（比回归 dense reward 的 TTA 更稳） |

**未决项核查**（GbI.md §8）：SWIM 检索未见同名的"逐任务想象 + 其他任务策略展开"工作（检索到的 SWIM 相关是人类视频结构世界模型，[RSS23](https://human-world-model.github.io/)）；He et al. 2023 是轨迹级扩散规划（已切割）。**"策略级想象裁决"的 novelty 目前仍然成立**——但必须先把 R1 修了，否则测的是另一个东西。

---

## 4. 改进方案（v6 路线，按优先级）

### P0 —— 对齐修复（前提，非创新，约 1 天）

**P0.1 恢复跨任务候选池**。`Candidate` 增加 `task_id` 字段（`arbiter.py:53-108`，mean/sample action 用候选自己的 task_id 而非 env 的）；候选集改为：

- **cross 池**：现役 actor 的 N 个任务 head（own = 任务 i 的 head，其余 j≠i 为跨任务候选）——与 CTPG 完全同池，零预训练依赖；
- 可选叠加：磁盘快照池/在线池作为**第二维**（(任务 j, 时间 t) 候选矩阵），用于 P6 决策场密度实验。

改动量约 50 行（Candidate 构造 + gbi_sac.py 候选列表 + imagine 的奖励头条件不变——奖励头本来就以任务 i 的 one-hot 条件，想象"任务 j 策略在任务 i"语义天然成立）。

**P0.2 触发解耦消融**。跨任务池下跑 2×2：{adaptive, always/fixed_K=2} × {gbi, qmp}。R3 的混杂即可归因；跨任务下"每步硬切换 vs K 步连续执行"本身就是一个有信号的实验点（CTPG 用 K 步的理由是技能连贯性）。

**P0.3 多 seed**。≥3 seeds（RUN_STAGES 已支持 `gbi:1 qmp:1 ...`；单卡时间允许下 PARALLEL=3 并行）。

### P1 —— 跨任务反事实闭环（R1 修复后自动激活，核心创新恢复）

设计文档 §3.5 的 L2 闭环"指导产生数据 → 数据改善模型 → 模型改善指导"**只有在跨任务候选下才有语义**：π_j 在任务 i 执行产生 (s_i, a_j, r_i)，是世界模型最缺的反事实转移。P0.1 之后此闭环免费激活，需在论文里作为机制图重点呈现（ imagination 通道的增益来源）。

### P2 —— 成功率/进展头（指标对齐，R4-b，约 1 天）

- 世界模型加 success 头：sigmoid 概率，监督信号用 #6 修复后可用的 success；目标设为"未来 H 步内成功"（per-step success 稀疏到学不动，进展型目标更稠密）；
- 裁决打分改为 `S_j = Q锚 + λ_t·(想象成功率增益)`（或 reward/success 双头加权），**与 MT10 评测指标直接对齐**；
- U_score 与 TTA 可切换到 success 头上（二值信号更稳，借鉴 TTA-with-Binary-Feedback）；
- 消融：dense reward 头 vs success 头 vs 双头。

### P3 —— 配对校准 + doubly-robust 残差修正（修 R2，创新点）

两级：

1. **标签口径修复**（必做）：δ_real 改配对口径——对齐 Phase 0 的 `set_state` 双轨机制，在线抽样状态做真配对 rollout 代价高，先用**同段内模型预测残差**修正：
2. **DR 增益估计**（创新）：维护每候选 EMA 残差 `b_j = E[G_real − Î_pred]`（该候选历史执行段的真实回报 − 决策时预测），打分用 `Î_j − b_j`（系统偏差修正后的差分），κ_t 在修正后的预测上重算。这是把 OPE 的 DR 公式搬进在线仲裁——"在线逐状态策略评估"的形式化是论文的理论卖点。

### P4 —— 状态局部信任门 λ_t(s)（R4-c，创新点）

全局 λ_t → `λ_t(s) = max(0, κ_t) · exp(−β · Û(s))`（Û 为归一化 U_score）：分歧低的状态放行想象、分歧高的状态压成纯 QMP（WPE 的 OOD 高估证据直接支撑）；顺手实现 `lambda_ramp` 平滑（参数已在 arbiter.py:134 挂着）。可选 LCB 版：想象增益取 ensemble 均值 − c·std。消融：全局 λ_t vs 局部 λ_t(s) vs LCB。

### P5 —— 软仲裁：从硬切换到蒸馏/混合（可选，加分项）

- **动作混合**：`a = (1−w)·a_i + w·a_j`，w 由 S_j 差值 sigmoid 决定——避免每步硬换手的振荡（如果 P0.2 发现 always 触发在跨任务下伤行为连贯性，此项对症）；
- **裁决即监督**：被选中候选的动作给 π_i 加 BC 正则（DQfD 式），把"指导"内化进本体策略而非只污染 buffer——指导结束后的性能保持是 CTPG 没有的性质，可作为差异化卖点。

### P6 —— 决策场密度作为实验变量（诊断框架升格为贡献）

把 Phase 0 的"决策场"从验收工具升格为**可测量、可调控的核心实验变量**：候选池 ∈ {cross, cross×时间快照, cross×探索温度, online-only}，画出"决策场密度（真实 gap 分布）× 想象增益"关系曲线。这回答"什么时候想象裁决有价值"——无论主结果正负，这张图都是论文的核心 figure，也是叙事 B 的骨架。

---

## 4.5 P0.1 实现记录（2026-09-03 06:30 UTC，已落地）

> 在当前批次继续运行期间实现（编辑 Python/yaml 文件不影响已加载进内存的运行进程；**未改动** `run_full_benchmark.sh`——它正被运行中的 bash 解释执行，就地编辑有破坏编排器的风险）。默认关闭（`cross_task_candidates.enabled=False`），所有现有 arm 行为逐位不变。

### 改动清单

| 文件 | 改动 |
|---|---|
| `CTPG-main/CTPG-main/mtrl/agent/arbiter.py` | ① `Candidate` 增加 `task_id` 参数 + `_resolve_task_ids()`：跨任务候选的动作固定走「候选自己的任务 head」（多头 mask 按 env_index 选 head，actor.py:541-545），legacy 候选（own/快照/online，task_id=None）沿用 env 任务 id；② `Arbiter.own_index(task_ids)`：每行的自身候选下标（cross 池 = 对角，legacy 池 = 全 0）；③ `score()` 基线全面 per-env 化：`I_own = gather(I, own_idx)`（替代 `I[:, 0:1]`）、护栏回退与正增益换手门回退均改为 `own_idx`（替代硬编码 0）；返回 dict 新增 `own_idx`；顺带删除死代码 `Q_own`。qmp 模式跳过想象的 4× 提速保持不变 |
| `CTPG-main/CTPG-main/mtrl/agent/gbi_sac.py` | ① 候选池双模式：`cross_task_candidates.enabled=True` 时候选 = 现役 actor 的 N 个任务 head（共享同一 actor 引用，零显存开销，freeze=False），打印池构成；快照池/在线池在该模式下忽略并禁用（`_oc_enabled and not _ctc_enabled`）；② `gbi_sample`：换手判断 `j > 0` → `j != own_idx[i]`（legacy 池等价于 `j != 0`）；`sources` 台账初始化为 per-env own 下标（自身执行步在 cross 池下正确记 `task_i` 而非 `task0`） |
| `CTPG-main/CTPG-main/config/agent/gbi_sac.yaml` | 新增 `cross_task_candidates: {enabled: False}` |
| `CTPG-main/CTPG-main/scripts/run_cross_benchmark.sh` | **新建**（独立于正在运行的 run_full_benchmark.sh）：Phase A 三臂 `gbi_cross:0`（gbi/adaptive）/ `qmp_cross:0`（qmp/always）/ `gbi_cross_fast:0`（gbi/always，P0.2 触发解耦）；支持 `SMOKE=1`（3000 步）、`PARALLEL=1 MAX_PARALLEL=3`、断点续跑；**内置运行中训练检测**（检测到 `main.py setup.alg` 进程时拒绝启动，FORCE=1 越过；已实测：当前批次运行中启动 → exit 1） |

### 语义要点

- **想象语义**：`close_loop_rollout` 的奖励头条件逐行用 **env 的任务 i** one-hot（world_model.py:258/269，批内可混合任务），动作闭包经 `task_id` 覆盖走 **head j** —— 想象 = "任务 j 的策略在任务 i 的奖励场里展开"，跨任务反事实语义天然成立，世界模型侧零改动。
- **Q 锚语义**：`q_anchor` 的 critic 条件仍是任务 i 的 critic（不变），`a_j` 自动变为 head j 的动作——"用任务 i 的尺子量任务 j 的动作"。
- **own 候选**：cross 池下 own = 现役 actor 的任务 i head（对角定位），拒绝坏指导的涌现性质保留（gain 对角恒 0）。
- **qmp+adaptive 组合无意义**：qmp 跳过想象 → U 恒 0 → adaptive 永不触发；P0.2 的 2×2 因此取 {gbi, qmp} × {adaptive, always} 中有意义的 3 臂。

### 验证（已执行）

1. **单元测试 20/20 通过**（临时脚本，跑完即删）：task_id 动作覆盖（mean/sample 双路径）、own_index 对角/全 0、gain 对角 = 0 且 `gain[i,j] = disc·(j−i)`、λ_t=0+正增益门全行回退 own、护栏回退 per-env own（非 index 0）、λ_t 主导时可换手到跨任务候选、qmp 模式 S=Q 且护栏/正增益门不生效、**legacy 池行为与 v5 逐位一致**、cross 候选共享 actor 引用且不冻结梯度。
2. `py_compile` 两个改动文件通过；yaml 解析通过。
3. **hydra 1.0.5 离线组装通过**：`agent.gbi.cross_task_candidates.enabled=True` override 合法，gbi_cfg 正常流入 builder（setup.alg 仅作 run 目录标签，config.py:154，任意值可用）。
4. `run_cross_benchmark.sh`：`bash -n` 语法通过；运行中训练护栏实测生效（exit 1）。

### Phase A 启动方式（当前批次结束后）

```bash
# 1) 冒烟验收（~10 分钟）：看 stdout 的 "cross-task candidate pool: 10 live task heads"，
#    train.log 的 λ_t/κ_t/换手率是否摆脱全零、entropy 正常下降
SMOKE=1 bash scripts/run_cross_benchmark.sh

# 2) 正式三臂（批次结束后；护栏会自动放行）
nohup bash scripts/run_cross_benchmark.sh \
  > /root/rivermind-data/lost+found/gbi/experiments/runs/cross_runner.log 2>&1 &
# 并行（T4 实测 3 并发安全）：
PARALLEL=1 MAX_PARALLEL=3 nohup bash scripts/run_cross_benchmark.sh \
  > /root/rivermind-data/lost+found/gbi/experiments/runs/cross_runner.log 2>&1 &
```

监控无需改动：`monitor_training.py` 按 `runs/*/metaworld/mt10/seed*/...` glob 自动发现新 run 目录。

### Phase A 验收口径（smoke → go/no-go）

1. stdout 出现 `cross-task candidate pool: N live task heads`（池构建正确）；
2. 决策场 gap 上移：跨任务候选的 `gain` 分布非退化（对照旧池的"空裁决场"）；
3. κ_t 信噪比改善：不再长区间精确为 0/负（λ_t 有非零时段）；
4. 换手率 SWR > 0（裁决确实在跨任务候选间切换；对照旧 gbi 的 0.4%）；
5. smoke 无 NaN/崩溃、ES_i=200 正常。

---

## 5. 论文叙事选项（NeurIPS 定位）

| 选项 | 叙事 | 依赖 | 风险 |
|---|---|---|---|
| **A. 方法论文** | "Computation vs Memorization"：同池对打 CTPG——guide 网络（学出来的选择）vs 想象+Q 锚（算出来的选择），裁决器级干净归因；三笔账单逐一兑现 | P0+P1 后 GbI ≥ CTPG | 若想象通道仍不敌 QMP-cross，退到 B |
| **B. 机制研究论文** | "When Does Imagination Help Behavior Arbitration?"：决策场密度、模型排序保真度、信任门行为、触发频率的系统刻画 + 阴性条件 + 诊断框架 | P0+P6，即使主结果阴性也成立 | 方法新颖性弱于 A，但结论稳健；符合 GbI.md 退路 D |
| **C. OPE 视角** | 在线多任务仲裁 = 逐状态 DR 策略评估：理论（估计误差界）+ P3/P4 实现 + 实证 | P3 完整落地 | 理论工作量最大 |

推荐 **A 为主、B 兜底**（A 的实验全部是 B 的子集，先跑 A，阴性自动降级 B）；C 作为 A/B 里的一节而非独立论文。评测协议对齐 Meta-World+，≥3 seeds 报置信区间。

## 6. 建议实验路线（不打断当前批次）

```
当前批次（跑完为止，预计 09-03 ~22:40 全部结束）
  │  其中 qmp/qmp_online/guide 提供时间快照池与 CTPG 的完整对照数字（保留价值）
  ▼
Phase A（~1 天）：P0.1 跨任务候选池 + P0.2 触发消融 + 3 seeds smoke（3k 步验收：
  │   候选间真实 gap 分布上移、κ_t 信噪比、想象开销）
  ▼
Phase B（~2–3 天）：跨任务池全量 4 arms（gbi-cross / qmp-cross / guide(已有) / indep(已有)）
  │   × 3 seeds —— go/no-go：gbi-cross ≥ qmp-cross 且 ≥ guide
  ▼
Phase C（~3–5 天）：按 B 的短板装配 P2/P3/P4（success 头 / DR / λ_t(s)）+ P6 决策场曲线
  ▼
写作（叙事 A 或 B）
```

关键 go/no-go 判据：

- Phase A smoke：跨任务候选的决策场 gap 中位数 > 0.1（否则回到"空场"问题，转 P6 候选多样化）；
- Phase B：gbi-cross 的 λ_t 不再长区间归零（κ_t 信噪比改善的直接证据）；
- 全程：`analyze_results.py` 重跑（当前 reports/ 是过期数据），对齐相同 eval 步配对比较。

---

## 7. 参考链接清单

**直接竞品/近邻**：[CTPG (NeurIPS 2024)](https://arxiv.org/abs/2507.06615) · [CTPG NeurIPS 页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d5cd70b708f726737e2ebace18c3f71b-Abstract-Conference.html) · [CTPG 代码](https://github.com/DarkDawn233/CTPG) · [Diffusion Planner MTRL (NeurIPS 2023)](https://arxiv.org/abs/2305.18459) · [To Mix or To Merge](https://arxiv.org/html/2602.12566v1) · [Merging Decision Transformers](https://openreview.net/forum?id=7NcrDeuMM8)

**世界模型评估/多任务 WM**：[WPE: Evaluating Robot Policies in a World Model](https://arxiv.org/html/2506.00613v1) · [Mixture-of-World Models (ICLR 2026)](https://arxiv.org/abs/2602.01270) · [MoWM OpenReview](https://openreview.net/forum?id=qUQARlAx5y) · [Meta-World+](https://openreview.net/pdf/e99efd74499d0685f991408922f91b3462404e77.pdf) · [Meta-World 官网](https://meta-world.github.io/)

**不确定性/OPE**：[Doubly Robust OPE (ICML 2016)](http://proceedings.mlr.press/v48/jiang16.pdf) · [MRDR (ICML 2018)](https://mohammadghavamzadeh.github.io/PUBLICATIONS/icml18-MRDR.pdf) · [DR for Ranking Policies](https://dl.acm.org/doi/abs/10.1145/3488560.3498380) · [Pessimism+Optimism (ICML 2021)](https://icml.cc/virtual/2021/poster/9877) · [Uncertainty-Aware Robotic World Model](https://arxiv.org/html/2504.16680v3)

**技能库/TTA/探索**：[SkillRL (ICLR 2026)](https://openreview.net/pdf?id=56D2hjARkn) · [RL with Skill Library](https://arxiv.org/html/2512.17102v2) · [TTA with Binary Feedback (ICML 2025)](https://arxiv.org/html/2505.18514v1) · [Confidence-Gated Progress Reward Modeling](https://arxiv.org/html/2606.22027v4) · [Ensemble Policies Exploration (NeurIPS 2023)](https://neurips.cc/virtual/2023/poster/72558)
