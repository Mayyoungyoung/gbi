# GbI 实验当前情况总结

> 更新时间：2026-09-03 15:40 UTC（benchmark 批次近尾声：qmp/gbi_online ~COMPLETED，gbi 81%，guide 89%，qmp_online 86%；**R1 根因确诊 + P0.1 跨任务候选池已落地**；Phase A 就绪）
> 关联文档：[GbI.md](GbI.md)（设计）、[GbI_Experiment_Guide.md](GbI_Experiment_Guide.md)（复现指南）、[GbI_Improvement_Proposals.md](GbI_Improvement_Proposals.md)（根因诊断 × 文献借鉴 × v6 改进路线，09-03 新增）

## 摘要

| 阶段 | 状态 | 说明 |
|------|------|------|
| state_sac 候选快照池 (500k) | ⚠️ 快照可用，**成功率记录全废** | 10 个候选快照齐全（eval.log 仅 episode_reward 有效：均值 8.5→34.6@50k→77@500k；**success 恒 0——#6 bug 受害者，先前文档“8.5→85”实为 reward 均值，成功率从未被正确记录**）；决策场复测已用修复后代码重测 10 ckpt（见 §四） |
| benchmark 批次（09-02 08:06 启动，PARALLEL=1 并行 4 stage） | 🟢 **近尾声（09-03 15:40）** | `qmp:0` ✅ 299.8k（末期成功率 **0.60**，reward 91.1 超 indep 基线 400k 的 81.4）；`gbi_online:0` 299.8k（~COMPLETED，success 0.22@280k）；`gbi:0` 243.2k（81%，**success 0.51@240k 持续爬升**）；`guide:0` 267.4k（89%，success 0.35@260k）；`qmp_online:0` 258.2k（86%，success 0.34@240k）；无 NaN/崩溃（详见 §二批次节） |
| 批次核心发现 | 🔴 **R1 根因确诊（代码级证据）** | **QMP（λ≡0）全面强于 GbI 是候选池实现偏离设计的必然后果，不是方法失败**：候选池被实现为"同一训练轨迹的时间快照"（执行仍用任务 i 自己的 head，arbiter.py:73-108），而 GbI.md §2.1 与 CTPG 的候选池都是跨任务策略 → 候选间真实差异≈0 → κ_t 全噪声 → λ_t 归零（100k–137k 实测）→ GbI 退化成"迟钝版 QMP"（触发率 5–6% vs QMP 68%）。三根因 R1/R2/R3 证据链见 [GbI_Improvement_Proposals.md §2](GbI_Improvement_Proposals.md) |
| P0.1 跨任务候选池 | ✅ **已实现 + 验证（09-03）** | 候选 = 现役 actor 的 N 个任务 head（CTPG 同池语义），`cross_task_candidates.enabled` 开关，默认 False（现有 arm 行为逐位不变）；单元测试 20/20（含 legacy 回归）、py_compile、hydra 1.0.5 组装、启动脚本护栏全部通过；`scripts/run_cross_benchmark.sh` 就绪（Phase A 三臂 + SMOKE 模式 + 运行中批次检测，实测正确拒绝启动） |
| 高危 #7 actor 冻结 | ✅ **已修复** | Candidate 加 freeze 参数，own 传 freeze=False；单元验证通过 |
| 高危 #6 success 恒 0 | ✅ **已修复** | vec_env.py 按 key 解析 gymnasium 1.x infos（含掩码处理）；冒烟中 env_step 已正常显示 200 |
| 中危 #8 twohot 口径 | ✅ **已修复** | bins_to_reward 改 symlog 空间期望再 symexp（单元验证：尾部泄漏偏差 +2.0 → +0.1） |
| 冒烟验证 | ✅ **通过（09-02 06:00）** | 3000 步端到端：entropy 5.68→5.33→5.13 持续下降（冻结时恒 5.68；state_sac 同期 5.32 几乎一致）；α 0.995→0.887 正常收缩；`ES_i=200` 实测生效；冒烟数据已清理 |
| 三轮审查 + 优化 | ✅ 完成 | qmp 跳过想象（~4× 提速）；q_anchor 改 min 口径；详见 §四三轮审查节 |
| 在线自举候选池 | ✅ **已实现 + 冒烟通过** | GbI 免预训练直接开训（像 CTPG）；注册/FIFO 淘汰/段活跃延迟淘汰链路实测正常；gbi_online/qmp_online 两个 stage 已入流水线 |
| Phase 0 决策场 | ✅ **复测通过（09-02）** | 10 ckpt 真实 rollout 重测（state_sac 原 success 记录全废 #6）；200k-500k own 最优率 0.8-0.9 ≥0.5、gap 1.0>0.1（详见 §四决策场节） |
| CTPG 基线（guide_sac） | 🟡 排队中 | guide 冒烟 COMPLETED 已验证路径（ES_i=200、no_guide 门控在动、训练交互出现成功 Su_5=0.4）；`guide:0` 正式 run 将在 gbi_online 完成后自动补位（同 300k 预算同 eval 协议）；对比矩阵见 §四 CTPG 节 |
| 改进路线（v6） | ✅ **已成文（09-03）** | P0 对齐修复 → P1 反事实闭环 → P2 success 头 → P3 DR 校准 → P4 状态局部 λ_t(s) → P5 软仲裁 → P6 决策场密度；论文叙事 A/B/C 三选（NeurIPS 定位）；文献检索（CTPG/WPE/MoWM/DR-OPE 等）详见 [GbI_Improvement_Proposals.md](GbI_Improvement_Proposals.md) |
| 脚本 | ✅ 可用 | run_gbi_nohup.sh / run_gbi_then_qmp.sh 正常；run_full_benchmark.sh 已升级（并行调度 PARALLEL=1）；**新增 run_cross_benchmark.sh（Phase A 跨任务池，独立编排）**；代码同步至 github.com:Mayyoungyoung/gbi.git |

---

## 一、实验流水线（单 GPU 串行）

```
state_sac_indep (500k 训练) ──完成──> 快照候选池 (10 个 actor)
        │
        ▼
GbI 主实验 (300k, 想象裁决) ──完成──> 自动衔接
        │
        ▼
QMP 对照实验 (300k, λ≡0 消融)
```

总耗时估算：GbI ~15h + QMP ~15h ≈ 30h（串行）。

---

## 二、各阶段详情

### 1. state_sac 候选快照池 —— ✅ 完成

- **运行时间**：2026-08-31 09:54:56 → 13:44:41（500k 步，约 3h50m）
- **产物**：11 个 actor 快照（`actor_0.pt`、`actor_50000.pt` ~ `actor_500000.pt`），模型目录约 305MB
- **路径**：`gbi/experiments/runs/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_issue_57d4f43a4bf96214e8f7be83a0564528bff75e77_seed_0/model/`
- **训练末期指标**：reward 60~190（按任务），success 0%（独立 SAC 多任务早期水平）

### 2. benchmark 批次（09-02 08:06 启动，PARALLEL=1 并行）—— 🟢 进行中

> 编排：`xargs -P 3`（RUN_STAGES=`gbi:0 qmp:0 gbi_online:0 qmp_online:0 guide:0`），监控循环每 30 min 采样（`experiments/monitor/`）。历史注记：08-31 首批 GbI/QMP 因高危 #7 actor 冻结白跑（已归档 `legacy_20260902/`），本批为修复后重跑。

| stage | 配置 | 状态（09-03 15:40） | 关键结果 |
|---|---|---|---|
| `qmp:0` | λ≡0 / always / 快照池 | ✅ **~COMPLETED**（299.8k） | 成功率曲线 0.04→0.60（280k）；末期 reward 91.1 **超 indep 基线 400k 的 81.4**；7/10 任务 ≥50% |
| `gbi_online:0` | gbi / adaptive / 在线池 | ✅ **~COMPLETED**（299.8k） | 0.22（280k）；λ_t≈0.072 微弱非零；步速 ~12k/h |
| `guide:0` | CTPG guide_mhsac | 267.4k / 300k（89%） | 0.35（260k）；reward 65.7，稳步爬升中 |
| `gbi:0` | gbi / adaptive / 快照池 | 243.2k / 300k（81%） | **0.51（240k）**；reward 80.0；λ_t≈0.053 微弱非零；步速 ~7.8k/h |
| `qmp_online:0` | qmp / always / 在线池 | 258.2k / 300k（86%） | 0.34（240k）；entropy 1.05 偏低（观察项）；步速 ~20k/h |

**批次解读（详见 [GbI_Improvement_Proposals.md](GbI_Improvement_Proposals.md)）**：

1. **QMP 仍然最强**（0.60@280k），但 GbI 快照池版也在稳步学习（0.51@240k），差距在收窄（160k 时 gbi=0.29 vs qmp=0.42；240k 时 gbi=0.51 vs qmp=0.53）。
2. **λ_t 微弱非零**（gbi≈0.053、gbi_online≈0.072），相比此前精确归零已有改善，但仍远不足以主导裁决——R1（候选池结构偏离）仍是主因。
3. **触发率差异依旧**：gbi=adaptive 6% vs qmp=always 68%（R3 混杂），Phase A 的 `gbi_cross_fast`（gbi+always）将解耦此变量。
4. **Guide（CTPG）基线稳步爬升**（0.35@260k），表现介于 gbi_online 与 gbi 之间——CTPG 的 learned guide 在 300k 预算内不如 QMP 的零模型裁决。
5. **qmp_online entropy 偏低**（1.05 vs qmp 的 2.06），可能因在线候选池多样性不足导致过早收敛，列为观察项。

**数据注意事项**：
- `reports/benchmark_report.md` 为 09-02 01:10 基于已归档白跑数据的过期报告，批次结束后须重跑 `analyze_results.py`；
- `state_sac_indep` 的 success 记录全废（#6），与该基线对比只能用 episode_reward；
- 单 seed（RUN_STAGES 未含 `:1`），配对检验 n=1，最终报告以效应量为主。

---

## 三、卡死事故分析（2026-08-31 GbI 停摆 8 小时）

### 根因

启动方式 `python3 -u main.py ... 2>&1 | tee train.log` 依赖 SSH 终端：

```
python 主进程 ──stdout──> 管道(64KB) ──> tee ──> /dev/pts/1 (SSH 终端)
```

1. SSH 会话断开 → pty 缓冲区写满 → `tee` 写阻塞
2. 管道缓冲区积满 → python 主进程阻塞在 `pipe_write`
3. 训练循环每 200 步 `Logger.dump()` 都会 `print()`（`mtrl/logger.py` L121），stdout 阻塞即整体停死

### 实证

- 进程 `wchan = pipe_write`、State=S，CPU 10 秒采样仅 +1 tick，GPU 0%
- 最后一条日志 17:14:02（step 60200），此后 8 小时零输出
- fd/1 → `pipe:[2124368470]`，tee 进程（PID 277907）fd/1 → `/dev/pts/1`，tee CPU 时间 0:00

### 教训

- 长时间训练禁止 `| tee` / 前台终端启动；必须 nohup + 重定向文件
- 完成状态必须落在文件（log.jsonl COMPLETED），不能依赖进程存在或终端输出

---

## 四、代码审查结果（2026-09-01 首轮 / 09-02 二轮深度审查）

### 高危（已修复）

| # | 问题 | 修复 |
|---|------|------|
| 1 | `\| tee` 前台启动，SSH 断连即死锁（本次事故根因） | 新建 `run_gbi_nohup.sh`：nohup + 文件重定向 |
| 2 | `run_gbi_then_qmp.sh` 硬编码 GbI run 路径，重启后失效 | v2 自动发现最新 run 目录（`ls -td`） |
| 3 | 完成标记 `[gbi] done` 在代码中不存在，脚本实际靠进程消失判断，异常退出也会误触发 QMP | v2 改轮询 log.jsonl `status=COMPLETED`（run.py 自动写），并支持 `CONTINUE_ON_ABNORMAL_EXIT=0` 中止 |

### 高危（待处理）

| # | 问题 | 根因 | 影响 | 修复 |
|---|------|------|------|------|
| 6 | **success / env_step 指标恒 0**（2026-09-02 实测确认） | gymnasium 1.3.0 `AsyncVectorEnv.step` 返回**按 key 组织的批处理 dict**（`{'success': array, 'env_step': array, ...}`），而 `_repack_gymnasium_step`（vec_env.py）仍按旧 gym 0.21 的 `infos[env_id]` 格式解析 → `i in infos` 恒 False → 每个 env 的 info 都是空 dict，`setdefault('success', 0.0)` 兜底 | **GbI / QMP / state_sac 三个实验的 success 全部恒 0.0、env_step 全 0**（train.log / eval.log 中 `env_step_*: 0.0` 为铁证）；成功率指标完全失真，真实成功率未知（reward 不受影响，仍可信）；`guide_sac.update_no_guide(success=...)` 的 success 门控同样被破坏（后续 guide 实验必踩） | 约 5 行，改为按 key 解析（见下）；**✅ 已修复（09-02），冒烟已验证 env_step 正常** |

```python
# vec_env.py _repack_gymnasium_step 修复（gymnasium 1.3 格式）
info_list = []
for i in range(vec_env.num_envs):
    info = {k: v[i] for k, v in infos.items()
            if not k.startswith("_") and np.ndim(v) > 0}
    info.setdefault("success", 0.0)
    info.setdefault("env_step", 0)
    info_list.append(info)
```

| # | 问题 | 根因 | 影响 | 修复 |
|---|------|------|------|------|
| 7 | **own 候选冻结现役 actor → 训练整体失效**（2026-09-02 二轮审查实锤，**比 #6 更严重：不是指标失真，是学习本身没发生**） | `gbi_sac.__init__`（L114-116）把 `self.actor` **本体引用**传入 `Candidate("own", self.actor, ...)`，而 `Candidate.__init__`（arbiter.py）执行 `for p in self.actor.parameters(): p.requires_grad_(False)`——同一对象，**现役 actor 全部参数梯度被永久关闭**；Adam 对 grad=None 的参数直接跳过 → actor 300k 步从未被更新 | **铁证链**（GbI 与 QMP 一致，state_sac 正常对照）：① train/entropy 全程恒 5.68（= log_std 恒 0 冻死在初始值；state_sac 正常 5.32→2.11）；② α 单调塌缩至 1.8e-9（熵恒 5.68 > target −4 → alpha_loss 恒正 → α 被压到浮点下界；此前“QMP α=0.025 正常”系误判，实为塌缩进行时，已塌至 0.00425）；③ eval 平线 9.0→9.2（初始随机策略原地踏步）；④ 动作 99.45% 不变。**QMP（PID 10834）同构造路径，正在白跑** | ~3 行：`Candidate.__init__` 加 `freeze` 参数，own 传 `freeze=False`（见下）；快照候选从磁盘 load 到新对象，冻结行为不变。**✅ 已修复并单元验证（09-02）** |

```python
# arbiter.py Candidate.__init__ 修复（own 候选不得冻结现役 actor）
class Candidate:
    def __init__(self, name, actor, action_range, device, freeze: bool = True):
        self.name = name
        self.actor = actor.to(device).eval()   # own 时 .to(device) 为 no-op，引用保持
        if freeze:  # 仅快照候选冻结；own 引用现役 actor，必须保持可训练
            for p in self.actor.parameters():
                p.requires_grad_(False)

# gbi_sac.py L114-116 改为：
self.candidates: List[Candidate] = [
    Candidate("own", self.actor, self.action_range, device, freeze=False)
]
```

冒烟验收：修复后训练 ~2k 步，train/entropy 应显著偏离 5.68（对照 state_sac 同期已降至 ~5.3）；α 不应单调衰减至 0。

### 中危（待处理）

| # | 问题 | 建议 |
|---|------|------|
| 4 | ~~QMP 执行段强制 K_min=3 步~~ **已实测澄清（09-02）**：always 模式 `step_segment` 在 K_min 检查前即 stop，实测 QMP 段长均值 1.99 步 ≈ 单步 argmax Q，代码行为正确，此前描述过时 | 无需修改 |
| 5 | `_dump_source_log(final=True)` | **已确认修复**：multitask.py 训练循环尾部显式调用（`if hasattr(self.agent, "_dump_source_log"): self.agent._dump_source_log(final=True)`），`gbi_source_final.npz` 已落盘 ✓ |
| 8 | twohot `bins_to_reward` 用 **raw 期望口径**：`Σ probs·symexp(bin_center)`，bin 中心被 symexp 拉到 ±400，softmax 尾部 ~0.5% 概率泄漏到极端 bin 即造成 ±2 期望偏差（真实 scaled reward 仅 ±1）→ 想象增益 Î 噪声极大，是 λ_t 校准（κ_t≤0→λ=0）的**机制性根因嫌疑** | 改为 symlog 空间期望再 symexp 整体变换（DreamerV3 做法）：`symexp(Σ probs·bin_centers)`；与 #7 同批修复，否则重跑后 λ_t 仍难打开 |

### 低危（记录备查）

- `arbiter.metrics()` 中 `trigger_rate` 死代码
- `adapt_reward_head` 的 before-loss 只取 head[0]，更新用 5 头均值，诊断口径不一致
- 衔接脚本 `pgrep -f "main.py.*setup.alg=gbi_sac"` 已规避 QMP 误匹配（QMP 用 `setup.alg=qmp`），但多 seed 并行时仍可能互相干扰

### GbI reward 现状（2026-09-03 更新，修复 #7 后重跑）

**核心结论：GbI 300k 步修复后学习正常（9.0→80.0@240k），但仍弱于 QMP（91.1@280k）。**

| 实验 | eval reward（step 0 → 最新） | eval success | 备注 |
|------|------------------------------|--------------|------|
| state_sac_indep | 8.5 → **85.0**（500k） | (恒 0，#6) | 同代码同任务，学习正常 |
| QMP（快照池） | 9.0 → **91.1**（280k） | **0.600** | λ≡0 + always，最强 |
| GbI（快照池） | 9.0 → **80.0**（240k） | **0.510** | 学习正常，差距收窄 |
| Guide（CTPG） | 9.0 → **65.7**（260k） | 0.350 | learned guide，稳步爬升 |
| QMP（在线池） | 9.0 → **62.6**（240k） | 0.340 | 在线池多样性不足 |
| GbI（在线池） | 9.0 → **53.8**（280k） | 0.220 | λ_t 微弱非零 |

关键观察（均来自 seed0 实测日志，修复 #7 后重跑数据）：

- **eval 只测裸 own 策略**：`act(sample=False)` 不走仲裁（gbi_sac.py 设计如此），而 GbI 的产品是"own + 仲裁"整体，当前评估口径只反映 own；
- **λ_t 微弱非零**（09-03 更新）：gbi≈0.053、gbi_online≈0.072，相比此前精确归零已改善，但想象通道仍未主导裁决——R1 候选池结构偏离是主因；
- **alpha 正常收缩**：修复 #7 后 α_0 正常收缩（gbi≈0.037、qmp≈0.017），不再塌缩到浮点下界；
- **GbI 稳步学习**：eval reward 9.0→80.0@240k，success 0.04→0.51@240k——与旧版“全程平线”完全相反，证明 #7 修复有效；
- **最终 eval 各任务**（gbi step 240k）：push **179.9** / window-close **139.9** / window-open **109.2** / drawer-close **54.6** / reach **25.3** / button-press **46.8** / door-open **61.7** / drawer-open **9.7** / peg-insert **0.3** / pick-place **0.1**；

---

### 已排查排除（无问题）

- `log.jsonl` 只有 metadata 行：正常（run.py 仅写 RUNNING/COMPLETED 两行，训练指标走 train.log/eval.log）
- 混合任务 batch 的想象 rollout：`close_loop_rollout` 按样本独立 task one-hot，正确（arbiter 中"批内必须同任务"注释已过时）
- act/update/buffer 归集时序：与 gbi_sac 头部协议一致，段收尾 reward 归集正确
- ReplayBuffer idx 回绕与空 buffer 保护：正确
- 候选快照加载：结构与 gbi actor 一致，10 个全部加载成功
- `mtrl_gbi.yaml` metric 登记与代码 log key 完全对齐
- 60k 步时 eval reward 8.7 属正常早期水平（对照 state_sac 同期 eval 34.6）；**success 0% 当时误判为正常，实为 info 解析 bug（高危 6），需修正**

### 二轮审查确认符合设计/无问题（2026-09-02）

- **训练动作路由链**：`multitask.sample_action → sac.sample_action → act(sample=True) → gbi_sample` 接通无误，仲裁确实在训练路径上生效；
- **critic 的 not_done 掩码**（sac.py update_critic）：gymnasium autoreset 边界转移的 next_obs 污染被正确屏蔽，无白跑风险；
- **actor_loss.backward() 的 critic 梯度泄漏**：冻结 bug 下梯度会流经 critic 参数，但 update_critic 开头 zero_grad 兜住，时序上无实际污染；
- **候选执行用任务 i 自己的 head**：multi-head 由 task_obs 经 moe_masks 选择（actor.model_forward），候选池=“不同训练阶段快照”**符合 GbI.md L208 的 v5 设计**（决策场改造方案）；但与 §1“其他任务策略临时接管”（对标 CTPG）的叙事存在张力——实际是“自己早期版本接管”（时间维指导），论文写作时需注意表述；
- GbI.md 自带警示（L199）依然有效：Phase 0 决策场验收（own-policy 最优率 ≥0.5、gap 中位数 >0.1）在重跑前建议补测，避免再次空转。

### Phase 0 决策场复测（2026-09-02 通过，GbI.md L208 验收线）

**背景**：state_sac seed0 run 的 success 信号全程恒 0（#6 gymnasium infos bug 受害者——含 eval 记录；先前文档「eval 8.5→85」实为 episode_reward 均值，**成功率从未被正确记录**）；验收线（own 最优率 ≥0.5 + gap 中位数 >0.1）需要 success → 用修复后代码重测。

**方法**：`scripts/alg/eval_decision_field.py`（手工复现 multi-head actor 前向，10 ckpt × 10 任务 × 20 eps，3 分片并行 ~25 分钟）+ `analyze_decision_field.py` 报告。关键修正：跨模型必须同初始 seed（早期 seed0=100+step 致 reach 200k=1.0/500k=0.0 非单调污染，已修正为固定 42）。结果在 `gbi/experiments/results/phase0/`。

**结果（同 seed=42，20 eps/任务）**：

| own | 最优率 | 有效最优率* | 被超越 | 正向 gap |
|-----|--------|------------|--------|----------|
| 50k-150k | 0.50 | 0.17 | 5/10 | 1.00 |
| 200k-300k | 0.80 | 0.67 | 2/10 | 1.00 |
| 350k-450k | 0.90 | 0.83 | 1/10 | 1.00 |
| 500k | 0.80 | 0.67 | 2/10 | 1.00 |

*有效最优率：剔除 3 个全池 0 任务（pick-place/drawer-open/peg-insert，300k 内未学会，与训练 reward 低一致）后的重算。

**判定：✅ 决策场非空，验收线通过**（own 最优率 0.8-0.9 ≥0.5；被超越任务 gap 1.0 >0.1）。训练中段（200k-450k）own 最优率持续达标；早期（50k-150k）own 有效最优率仅 0.17（被全面超越——Gbi 场景下 own 早期弱是正常态，正是裁决/护栏用武之地）。

**额外发现（重要）**：① reach-v3 出现非单调：200k=1.0 后 250k 起恒 0（共享 trunk 被多数任务梯度扰动后丢失）——快照池「时间维指导」的真实机会；② 成功率绝对值低（mean 0.4-0.5@500k）是随机变体 + 200 步截断所致，训练 eval 同协议 → 跨算法相对可比；③ 注意 pick-place/drawer-open/peg-insert 三难任务 300k 预算内所有算法都难学会，对比表需标注。

### 四轮审查（2026-09-02 下午：stage 参数一致性 + 模块盘点 + 并行能力）
**核查项（均通过）**：

- stage 参数与冒烟/历史 run 逐一对比：COMMON 数组含全部 8 个 multitask 参数且 gbi/qmp/gbi_online/qmp_online/guide 五个 stage 均引用；guide 的 task_onehot 由 COMMON 覆盖为 False（yaml 默认 True）；encoder 默认即 identity；**无参数漂移**（历史上 03-54-24 run 与 state_sac 快照池实际生效配置 multi_head=True/identity 与 COMMON 一致，候选加载兼容）；
- 在线候选池代码（_register/_evict/_dump）：FIFO 从 index 1 pop（保留 own）、段活跃延迟淘汰、source_name 归档（L355）正确；
- arbiter qmp 分支：I/U 填 0、S=Q、护栏与 gain 门仅 gbi/imag 生效——与「唯一变量」对照设计一致；
- `--one` 单 stage 入口 + xargs 并行调度链路验证（未知 stage 返回 2、-P 并发正确补位）；并行代码复审：run_dir/日志/FAILED 标记按 stage 隔离无竞态；失败聚合 rc=123 其余 stage 不受影响；续跑断点语义保留；唯一待修正：注释/默认值按 10 vCPU 实测校准（MAX_PARALLEL 默认 3，见下）;

**模块盘点**：mtenv/env builder/ds(MTObs)/wrapper/abstract/video/optimizer 均被引用（必要）；冗余仅：gym_extensions（HalfCheetah 系，MT10 不用但保留无害）、scripts/alg 旧单跑脚本若干（被 benchmark 取代）、tmp_obs_check.py。

**步速澄清（1 快照 ~10 步/秒 vs 10 快照 3.9 步/秒）**：两者不是同一东西。~10 步/秒是 **qmp 模式**（S=Q 每步 argmax，**无想象**）冒烟实测（qmp 模式 10 快照同样 ~9-10 步/秒，para_probe 实测 9.1）；3.9 步/秒是 **gbi 模式 + 10 磁盘快照**——imagine() 每步对所有候选做 H=5 闭环想象，成本与候选数成正比（gbi 模式 1-3 在线候选 ~9 步/秒，10 候选 3.9）。**慢的不是快照多，是想象裁决**。

**并行实验（2026-09-02 实测校准，10 vCPU + T4 16GB）**：

- GPU 占用低的根因：单进程 qmp 训练 GPU 利用率 ~40%、显存仅 **493 MiB**——CPU 配额（10 vCPU）限流下环境交互是瓶颈；
- **3 进程并行实测**（gbi 10 快照 + qmp 10 快照 + guide，3000 步）：各任务步速仅退化 ~10%（gbi 3.9→3.5、qmp 10→9.1、guide ~10→9.1）；GPU 100%、显存合计 1.1GB/16GB、CPU load ~3/10——GPU 是新瓶颈；总吞吐 ~21.7 步/秒 ≈ 单进程 2.2×；
- 时间预估：核心 5 线（seed0）串行 ~55h → 3 并发 **~20-24h**；全量 10 stage 串行 ~104h → 3 并发 ~44-48h；
- run_full_benchmark.sh 已按实测校准：`PARALLEL=1 MAX_PARALLEL=3`（默认 3；含轻任务多时可 4；>5 无收益）。

### 三轮审查（2026-09-02 下午：未覆盖路径 + 新增优化）
**已修复（本轮）**：

| 问题 | 严重度 | 修复 |
|------|--------|------|
| qmp 模式浪费想象算力：`score()` 无条件 imagine，但 S=Q 不用 I/U（u_trigger 仅归档） | 性能（~4×） | mode=qmp 时跳过 close-loop rollout，I/U 填 0（arbiter.py，09-02） |
| q_anchor 用 (q1+q2)/2 而非 min：与 SAC 值学习口径（target 用 min）不一致，打分系统性偏高且偏向过乐观候选 | 中（口径） | 改 torch.min（arbiter.py，09-02） |

**确认无问题**：imagine() 闭包变量绑定（每次迭代立即调用，无晚绑定陷阱）；cancel_segment/reset_at_begin 的段终止链路；wm update 冷启动保护与残差缓存刷新；TTA 冻结主干只调末层；段内 reward 归集/TTA 留档排除 done 步（autoreset 边界处理正确）。

**记录备查**：① gbi/adaptive 模式段长均值 13.7 ≈ K_max=15，提前终止（m_low×U<0.7τ_on）几乎不触发（U 低）——可考虑缩短 K_max；② TTA 在 qmp 短段（~2 步）下只有 ≤2 条转移，适配方差大；③ guide（CTPG）路径从未在本环境验证过——已补冒烟（见下）。

### 白跑实验中仍有效的信号（2026-09-02 分析，不受 actor 冻结影响）

> 世界模型学的是环境动力学+奖励（与策略质量弱相关，数据来自探索性策略）；τ/U 统计有效；κ/λ 在旧 raw_expect 口径下失真，重跑后重评。

| 信号 | 首个 log 点 → 尾部（300k） | 结论 |
|------|--------------------------|------|
| wm_reward_loss | 2.37 → **0.41** | 奖励头收敛良好 |
| residual_stream_reward_loss | 2.20 → 0.55 | 残差加权流收敛良好 |
| wm_dyn_obs_mae | **0.001**（全程） | 动力学近乎完美，印证 Phase 0「坏的是奖励头」——#8 修复（打分口径）是对症的 |
| uscore_ema / τ_on | 2.43 → 0.0025 / 1.60 → 0.0037 | **U 与 τ 同步收缩**：分位触发仍工作，但绝对分歧极小 → 想象增益信噪比恶化，#8 去噪是 λ_t 打开的关键 |
| gbi_trigger_rate | 0.62% → 3.8% | 自适应触发在激活 |
| gbi_seg_len_mean | 13.3 → 13.7 ≈ K_max | 段几乎跑满 K_max（见上「记录备查」①） |
| kappa_t | 0 → −0.04 | 旧口径下无意义（#8 修复后重评） |
| state_sac eval（参照） | 8.5 → 34.6@50k → 74.3@300k → 85@500k | 300k 内 own 基线可达 74+，想象通道有加分空间 |

**优化决策（已实施）**：qmp 跳过想象（4×提速）；在线候选池（见下节）同时把 gbi 模式候选数从 11 降到 ≤5，步速预期从 3.9 → ~8 步/秒。

### 在线自举候选池（2026-09-02 新增，GbI 免预训练直接开训）

用户需求：GbI 应像 CTPG 一样不需要提前准备数据、可直接开训。实现（gbi_sac.py + gbi_sac.yaml）：

- `agent.gbi.online_candidates.enabled=True` 时，每 `interval`（缺省 25k）步将**现役 actor** deepcopy+冻结后入池，FIFO 保留最近 `max_count`（缺省 4）个；`warmup`（缺省 20k）前不注册；
- 与磁盘快照池可叠加（叠加时先淘汰更老的磁盘快照）；段活跃期间延迟淘汰，避免执行段引用的候选 index 因 FIFO 平移错位；
- source 归档新增 `source_name` 字段（候选名），分析时按名聚合即可无视 index 漂移；
- 语义与 GbI.md L208「混合不同训练阶段快照」的决策场改造方案一致，且从离线混合升级为在线滚动——候选永远「新鲜」，无过期问题（GbI.md §1「推算永远基于现役最新策略」的同源优势）；
- 流水线已加 `gbi_online` / `qmp_online` 两个 stage（`runs/gbi_online/`、`runs/qmp_online/`），qmp_online 用于隔离「在线候选池」与「想象增益」两个变量；

**冒烟验收（smoke_gbi_online，3000 步压缩参数 warmup=1500/interval=500/max_count=2，已清理）**：

- 注册/淘汰链路实测正确：`online_1500` 注册 → `online_2000` 注册 → FIFO 淘汰 `online_1500` → `online_2500` 注册（池=own+最近 2 个，max_count 生效）；stdout 可见每次池状态；
- `candidates_dir=null` 时启动池=`['own']`——免预训练直接开训成立；
- actor 学习正常（#7 无回归）：entropy 5.335→5.163、α 0.995→0.887，与 state_sac 同期一致；
- `success_5=1.0` 直接出现在 train 交互统计（#6 在 gbi 路径生效）；arbiter 指标（λ/κ/τ）正常记录；
- 步速 ~9 步/秒（含初始化 5.5 分钟/3000 步，gbi 模式 1–3 候选 + 想象）；注：step 3000 处的注册因循环边界不触达（同 eval/save 边界语义），正式 300k 无影响。

### CTPG（guide_sac）基线对比方案（2026-09-02）

- 基础设施已就绪：`run_full_benchmark.sh` 的 `guide:0/guide:1` 阶段（guide_mhsac = MHSAC+CTPG，与 GbI 同环境同 300k 预算同 eval 协议）；
- **guide 路径首次环境冒烟通过（smoke_guide，3000 步，已清理）**：COMPLETED 不崩；`ES_i=200`（#6 修复对 guide 生效——`update_no_guide(success=...)` 门控依赖的信号链路恢复）；no_guide 门控实测在动（0/1 混合采样）；训练交互出现成功（Su_5=0.4，mode=train）；步速 ~10 步/秒。CTPG 原版路径不写 actor 训练指标（train.log 仅 env 统计），正式跑用 eval.log 成功率曲线对比即可；
- 意义：#6 修复前 success 恒 0 会破坏 guide 的 `update_no_guide(success=...)` 门控——旧环境下 guide 结果本会不可用；现信号链路已恢复，guide 可作为有效基线；
- 公平性说明：CTPG 官方 MT10 预算未在本地仓库标明（原 mtrl 系工作常用 1M–2M）；本项目统一 300k 同预算对比（同代码同环境同 seed，隔离算法变量），若需对齐论文量级可另跑 guide 1M 长跑作参考点；
- 预期对比矩阵（300k）：`indep`（MTSAC 无指导）/ `guide`（CTPG：学出来的指导）/ `qmp`/`qmp_online`（算出来的指导，无想象）/ `gbi`/`gbi_online`（算出来的指导，含想象裁决）——后四者共享同一裁决框架，唯一变量为候选池来源与 λ_t。

| 问题 | 文件 | 修改 | 验证 |
|------|------|------|------|
| #7 actor 冻结 | `mtrl/agent/arbiter.py` | `Candidate.__init__` 加 `freeze: bool = True` 参数，仅在 freeze=True 时关闭梯度 | 单元验证：freeze=False 保持 requires_grad，freeze=True 冻结 ✓ |
| #7 actor 冻结 | `mtrl/agent/gbi_sac.py` | own 候选改传 `freeze=False`（快照候选不变） | 同上 |
| #6 infos 解析 | `mtrl/env/vec_env.py` | `_repack_gymnasium_step` 改按 key 解析批处理 dict，跳过 `_` 前缀掩码 key，掩码 False 的 env 走 setdefault 兕底 | 单元验证：success/env_step 正确取出（含掩码 False 场景）✓；冒烟中 `ES_i: 200` 正常显示（修复前恒 0） |
| #8 twohot 口径 | `mtrl/agent/components/twohot.py` | `bins_to_reward` 改 symlog 空间期望再 symexp 整体变换 | 单元验证：0.5% 尾部泄漏时偏差 +2.0（旧）→ +0.1（新） |
| 低危 adapt 口径 | `mtrl/agent/components/world_model.py` | `adapt_reward_head` 的 before/after loss 改用 5 头均值（与训练 loss 同口径，此前只取 head[0]） | py_compile ✓（行为验证随冒烟） |
| 低危 trigger_rate 死代码 | — | 现版 `arbiter.metrics()` 已无此死代码（仅 6 个诊断键），此前记录过时 | 复查确认 |

单元验证脚本（跑完即删）：构造 0.5% softmax 尾部泄漏 → 旧口径 E[r]=3.0（真值 1.0）、新口径 E[r]=1.1；三种 infos 场景解析全对。

---

## 五、脚本修复记录（v2，2026-09-01）

### 新建 `CTPG-main/CTPG-main/scripts/run_gbi_nohup.sh`（121 行，可执行）

- `bash scripts/run_gbi_nohup.sh` 后台启动（默认 nohup）；`foreground` 参数用于调试
- stdout/stderr → `gbi_stdout.log`，PID → `gbi.pid`，无管道无 tee
- 参数与 2026-08-31-14-02-10 GbI 主实验完全一致（300k 步、10 候选快照）

### 重写 `CTPG-main/CTPG-main/scripts/run_gbi_then_qmp.sh`（v2，85 行，可执行）

- 自动发现最新 GbI run 目录，轮询 `log.jsonl` 的 `"status": "COMPLETED"`
- GbI 进程消失但无标记 → 判定异常：默认打印警告后继续 QMP，`CONTINUE_ON_ABNORMAL_EXIT=0` 时中止
- QMP 启动同样去掉管道，stdout → `qmp_stdout.log`
- 用法：`nohup bash scripts/run_gbi_then_qmp.sh > runner.log 2>&1 &`

### 验证结果

- `bash -n` 语法检查：两个脚本均通过
- 异常退出分支实测：`CONTINUE_ON_ABNORMAL_EXIT=0` 下正确 abort（exit=1，未启动 QMP）
- 自动发现逻辑实测：能正确定位最新 run 目录

---

## 六、当前系统状态（2026-09-03 15:40 UTC）

| 项 | 状态 |
|----|------|
| GPU | T4 100% 利用率，~1.5 GB / 15 GB 显存（算力瓶颈非显存） |
| 训练进程 | 3 个运行中：`gbi:0`（243.2k，81%）、`guide:0`（267.4k，89%）、`qmp_online:0`（258.2k，86%）；`qmp:0` 与 `gbi_online:0` 已近 300k |
| 编排器 | `run_full_benchmark.sh` xargs -P 3 存活（PID 103510）；guide:0 已自动补位运行中 |
| 监控 | 30 min 循环存活（`experiments/monitor/`，samples.jsonl 60+ 条，最新 15:38 无警告） |
| 候选快照 | 11 个 actor（含 actor_0），10 个候选齐全 |
| P0.1 代码 | 已落盘（默认关闭）；运行中进程不受影响（模块已加载进内存） |
| 预计全批次完成 | qmp_online/gbi/guide 约 09-03 18:00–22:00 陆续结束 |

### 批次完整对比表（最新 eval 点，09-03 15:40 UTC）

| 算法 | eval@step | reward | success | entropy | λ_t | 触发率 | 换手率 |
|------|-----------|--------|---------|---------|-----|--------|--------|
| **qmp**（快照池，λ≡0，always） | 280k | **91.1** | **0.600** | 2.06 | 0.0 | 68.3% | 31.2% |
| **gbi**（快照池，adaptive） | 240k | 80.0 | 0.510 | 1.82 | 0.053 | 6.2% | 0.4% |
| **guide**（CTPG MHSAC） | 260k | 65.7 | 0.350 | 1.54 | — | — | — |
| **qmp_online**（在线池，always） | 240k | 62.6 | 0.340 | **1.05** | 0.0 | 76.0% | 24.0% |
| **gbi_online**（在线池，adaptive） | 280k | 53.8 | 0.220 | 2.06 | 0.072 | 6.9% | 0.3% |
| **indep**（state_sac 基线） | 500k | 85.0 | (恒 0，#6) | — | — | — | — |

**关键发现**：
- QMP > GbI > Guide > QMP_online > GBI_online（按 reward/success 排序）
- GbI 修复 #7 后学习正常（9.0→80.0@240k），但仍弱于 QMP——R1 根因的结构性限制
- 裁决框架本身有效：QMP 的 91.1 超过 indep 500k 的 85.0（尽管 indep success 不可比）
- 在线池整体弱于快照池——候选多样性不足（在线池仅 4 个近期快照 vs 快照池 10 个跨 50k–500k）
- Guide（CTPG）在 300k 预算内不敌纯 Q 锚裁决（0.35 vs 0.60），但仍在爬升中

---

## 七、后续行动计划（09-03 更新）

1. ~~立即止损 QMP~~ ✅ 已执行（kill 10834 + 编排器）
2. ~~修复高危 #7 / #6 / 中危 #8~~ ✅ 均已修复并验证
3. ~~冒烟验收~~ ✅ 已通过（09-02 06:00，熵下降/α 收缩/ES_i=200 全部反转白跑铁证）
4. ~~Phase 0 决策场复测~~ ✅ 已通过（09-02）
5. ~~重启全量流水线~~ 🟢 **近尾声**：benchmark 批次 09-02 08:06 启动（qmp:0 ~COMPLETED；gbi_online:0 ~COMPLETED；guide:0 89%；gbi:0 81%；qmp_online:0 86%）
6. ~~R1 根因诊断~~ ✅ **已确诊（09-03）**：候选池实现偏离设计（时间快照 ≠ 跨任务策略）+ R2 校准标签非配对 + R3 触发模式混杂，证据链见 [GbI_Improvement_Proposals.md §2](GbI_Improvement_Proposals.md)
7. ~~P0.1 跨任务候选池实现~~ ✅ **已落地并验证（09-03）**：`cross_task_candidates.enabled` 开关 + `run_cross_benchmark.sh`，单元测试 20/20（含 legacy 回归）
8. **批次结束后的 Phase A**（全部 COMPLETED 后，预计 09-03 22:00 前后）：
   - `SMOKE=1 bash scripts/run_cross_benchmark.sh`（3000 步冒烟，验收口径：池构建日志 / gain 分布非退化 / κ_t 信噪比 / SWR>0 / 无 NaN）
   - 通过后正式三臂（`gbi_cross` / `qmp_cross` / `gbi_cross_fast`，300k，PARALLEL=1 MAX_PARALLEL=3）
   - go/no-go：跨任务决策场 gap 中位数 >0.1；gbi_cross 的 λ_t 不再长区间归零；gbi_cross ≥ qmp_cross 且 ≥ guide
9. **批次完成后分析**：重跑 `analyze_results.py`（当前 reports/ 为过期白跑数据）；对比各 arm 的 train.log / eval.log / `gbi_source_*.npz`；qmp_online entropy 偏低（1.05）列为观察项
10. **多 seed**：Phase B 起 ≥3 seeds（当前批 n=1 只报效应量）

---

## 附录：关键路径速查

| 对象 | 路径 |
|------|------|
| GbI 启动脚本 | `CTPG-main/CTPG-main/scripts/run_gbi_nohup.sh` |
| 衔接脚本 | `CTPG-main/CTPG-main/scripts/run_gbi_then_qmp.sh` |
| benchmark 编排（运行中） | `CTPG-main/CTPG-main/scripts/run_full_benchmark.sh`（批次日志 `gbi/experiments/runs/benchmark_runner.log`） |
| Phase A 跨任务池编排 | `CTPG-main/CTPG-main/scripts/run_cross_benchmark.sh`（批次结束后 `SMOKE=1` 先冒烟） |
| 跨任务池开关 | `config/agent/gbi_sac.yaml` → `agent.gbi.cross_task_candidates.enabled` |
| GbI 运行日志（新） | `gbi/experiments/runs/gbi/metaworld/mt10/seed0/logs/metaworld-mt10/gbi_sac/<run>/` |
| GbI stdout | `gbi/experiments/runs/gbi/metaworld/mt10/seed0/gbi:0_stdout.log` |
| QMP 运行日志 | `gbi/experiments/runs/qmp/metaworld/mt10/seed0/logs/metaworld-mt10/qmp/<run>/` |
| 训练监控 | `gbi/experiments/monitor/`（monitor_loop.log / samples.jsonl，30 min 采样） |
| 候选快照池 | `gbi/experiments/runs/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_.../model/` |
| 作废旧运行 | `CTPG-main/CTPG-main/logs/metaworld-mt10/`；已归档批次 `gbi/experiments/runs/legacy_20260902/` |
| 设计文档 | `GbI.md` |
| 复现指南 | `GbI_Experiment_Guide.md` |
| 改进提案（根因×文献×v6 路线） | `GbI_Improvement_Proposals.md` |
