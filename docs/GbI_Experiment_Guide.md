# GbI 实验复现指南

> **目标读者**：拿到另一台 T4 服务器上复现 GbI 实验的研究者。本文档自包含，所有配置参数已解析为实际值。
>
> **工作目录**（本文档中所有相对路径的基准）：`<REPO_ROOT>` = 代码仓库根目录（即 `CTPG-main/CTPG-main/`）

---

## 1. 项目概述

### 1.1 一句话定位

**The first imagination-based behavior-policy arbiter for multi-task RL——共享放到动力学层，分化放到行为层。**

世界模型的动力学跨任务共享（MT10 同一机械臂、物理一致、数据可池化），但每任务保留独立策略；运行时由共享世界模型在想象空间中前前推算各候选策略的预期回报，动态裁决"此刻谁该执行"。**指导不再是被学出来的记忆，而是被算出来的选择**（selection-as-search vs. selection-as-memorization）。

### 1.2 核心设计要点

| 组件 | 设计 |
|---|---|
| **共享世界模型** | state-based RSSM（TD-MPC2 同构），ensemble×5 共享动力学、仅奖励头分叉；奖励头 two-hot(101 bins) + symlog + 交叉熵；双流采样（均匀流训动力学 + 残差加权流训奖励头） |
| **裁决器** | 一元统一公式：S_j(s_t) = Q_φi(s_t, a_j) + λ_t × (Î_j − Î_i)；Q 锚复用现役 SAC critic（零新增网络）；λ_t = max(0, κ_t) 由 L2 标注缓冲在线校准 |
| **自适应触发** | U_score = Var_k(Î^k_j)（5 奖励头对同一轨迹打分分歧）；τ_on = 近 W=10^4 步滑窗 0.9 分位；替代 CTPG 固定 K |
| **TTA** | 冻结 GRU/encoder/state/obs head，仅对奖励头最后层做 1-3 步梯度更新，lr = adapt_lr = 1e-5 |
| **护栏** | U > τ_reject（0.99 分位）→ 保持自身策略；换手要求相对增益 > 0；K_min=3 / K_max=15 执行段 |

### 1.3 与 CTPG 的关键差异

| 维度 | CTPG | GbI (v5) |
|---|---|---|
| 选择信号来源 | learned guide network（记忆） | 想象前瞻推算（search） |
| guide 训练 | 真实交互训练，非平稳 → hindsight correction | 无可学习选择器，零训练成本 |
| 拒绝坏指导 | policy-filter gate + comparable Q（两套机制） | 候选含自身 + Q 锚涌现表达拒绝 |
| 引导时长 | 固定 K 逐环境调参 | U_score 自适应触发 + [K_min, K_max] |
| 额外网络 | guide network + comparable Q | 共享世界模型（+ 5 奖励头）；Q 锚复用现役 critic |
| 失败模式 | selector 非平稳 / 记忆冷启动 | 奖励头偏置 / 空裁决场 |

---

## 2. 代码库结构

### 2.1 关键文件清单

```
<REPO_ROOT>/
├── main.py                              # Hydra 入口，config_name="config"
├── config/
│   ├── config.yaml                      # 顶层 Hydra defaults
│   ├── setup/mtrl.yaml                  # 基础路径、device、seed
│   ├── agent/
│   │   ├── state_sac.yaml               # 独立 SAC agent 配置
│   │   ├── gbi_sac.yaml                 # GbI agent 配置（含裁决参数）
│   │   ├── guide_sac.yaml               # CTPG guide SAC 配置
│   │   └── components/
│   │       ├── world_model.yaml          # 世界模型超参
│   │       ├── actor.yaml                # Actor 网络
│   │       ├── critic.yaml               # Critic 网络
│   │       ├── encoder.yaml              # 编码器（identity/feedforward/MoE 等）
│   │       ├── multitask.yaml            # 多任务策略配置
│   │       └── mask.yaml                 # 多任务 mask
│   ├── env/
│   │   ├── metaworld-mt10.yaml           # MetaWorld MT10 环境
│   │   └── metaworld-mt50.yaml           # MetaWorld MT50 环境
│   ├── experiment/mtrl.yaml              # 训练步数、评估/保存频率
│   ├── metrics/
│   │   ├── mtrl.yaml                     # 基础 SAC 指标
│   │   ├── mtrl_gbi.yaml                 # GbI 专属指标（含 wm_loss, arbiter/* 等）
│   │   └── mtrl_guide.yaml               # Guide 基线指标
│   └── replay_buffer/mtrl.yaml           # Replay buffer 配置
├── mtrl/
│   ├── agent/
│   │   ├── sac.py                        # 基础 SAC agent
│   │   ├── gbi_sac.py                    # GbI agent（SAC + 裁决通道）
│   │   ├── guide_sac.py                  # CTPG guide agent
│   │   ├── arbiter.py                    # 裁决器（打分/触发/护栏/λ_t 校准）
│   │   └── components/
│   │       ├── world_model.py            # 共享任务条件 RSSM 世界模型
│   │       └── twohot.py                 # two-hot + symlog 工具
│   ├── app/run.py                        # 实验循环
│   └── experiment/metaworld.py           # MetaWorld 实验构建器
├── scripts/
│   ├── alg/
│   │   ├── state_sac_indep.sh            # P0.1: 独立 SAC 500k 训快照池
│   │   ├── gbi.sh                        # GbI 主实验
│   │   ├── gbi_qmp.sh                    # QMP 消融（λ≡0, always 触发）
│   │   ├── guide_mtsac.sh                # CTPG guide 基线（MTSAC 变体）
│   │   └── guide_mhsac.sh                # CTPG guide 基线（MHSAC 变体）
│   ├── smoke.sh                          # 冒烟测试
│   ├── run_phase0.sh                     # Phase 0 全流程
│   └── env_setup.sh                      # 环境验证脚本
└── requirements.txt                      # Python 依赖
```

### 2.2 核心模块说明

#### world_model.py — 共享任务条件 RSSM

- **结构**：obs → encoder(LayerNorm+SiLU, 512-d) → z；[z; a] → GRU(512) → h；h → state_head(残差 Δz) → z_next；h → obs_head(残差 Δs) → s_next
- **奖励头**：5 个 `SymLogTwoHotHead`，输入 [h; onehot(task_id)]，输出 101 bins logits
- **ensemble 共享动力学**：5 成员共享 encoder/GRU/state_head/obs_head，仅奖励头分叉 → 分歧信号归因干净
- **训练**：双流——均匀流训动力学+奖励（bootstrap 0.7-0.9 子采样）；残差加权流只进奖励头梯度（quantile-sigmoid 权重，上限截断 5.0）
- **TTA**：`adapt_reward_head()` 冻结 GRU/encoder/state/obs head，只对奖励头最后层做 1-3 步梯度更新
- **编码器 EMA**：tau=0.995，提供 next-z 目标防表征坍缩
- **obs 归一化**：running mean/std，动力学预测与目标都在归一化空间

#### arbiter.py — 一元统一裁决器

- **Q 锚**：Q_φi(s, π_j(s))，用现役 SAC critic 对每个候选的均值动作打分，返回 (B, N)；Q 用均值 (q1+q2)/2 而非 min
- **想象打分**：对每个候选 π_j 在共享动力学上闭环 rollout H=5 步，5 个奖励头打分取均值 Î_j，同时算 Var_k 分歧 U
- **统一公式**：S_j = Q_j + λ_t × (Î_j − Î_i)；模式：gbi / qmp(λ≡0) / imag(纯想象)
- **触发**：adaptive（U 滑窗 0.9 分位）/ fixed_K / always
- **执行段状态机**：K_min=3 防振荡、K_max=15 硬上限、连续 m=3 步 U < 0.7τ_on 提前终止
- **λ_t 校准**：段结束后算真实增益 delta_real，与预测增益 delta_pred 做 Spearman 秩相关 κ_t，λ_t = max(0, κ_t)

#### gbi_sac.py — GbI Agent 集成层

- 继承 `mtrl.agent.sac.Agent`，在 `act()` 训练分支走 `gbi_sample()` 裁决
- `update()` 尾挂：世界模型训练 → 段收尾（L2 标注 + TTA）→ metric 落盘 → 来源归档
- 候选集：index 0 = 现役自身策略（同一 actor 引用）；其余为 state_sac 训练快照
- 时序协议：update_t → act_t → env.step_t → buffer.add → update_{t+1}

#### twohot.py — two-hot + symlog 工具

- `symlog(x) = sign(x) × ln(|x| + 1)`：压缩跨任务奖励尺度差（3 个数量级 → ±6）
- `TwoHotSymlog`：101 bins 铺在 symlog 空间 [symlog(-400), symlog(400)] ≈ [-5.99, 5.99]
- `twohot_loss()`：交叉熵损失，two-hot soft-label
- `bins_to_reward()`：softmax 加权原始空间分箱中心 → 期望奖励

---

## 3. 环境搭建

### 3.1 Python 版本与关键依赖

- **Python**: 3.8+
- **PyTorch**: `torch==1.13.1`（CUDA 可用）
- **MuJoCo**: `mujoco==2.1.5` + `mujoco-py==2.1.2.14`
- **MetaWorld**: 从源码安装（`requirements.txt` 中 `metaworld==0.1.0` 被注释，需手动安装）
- **Hydra**: `hydra-core==1.0.5` + `omegaconf==2.0.6`
- **其他关键包**：`numpy==1.26.2`, `scipy==1.11.4`, `gym-notices==0.0.8`, `cloudpickle==1.6.0`, `termcolor==1.1.0`, `tensorboard==2.4.0`

### 3.2 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n gbi python=3.8 -y
conda activate gbi

# 2. 安装 PyTorch（CUDA 11.x T4）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  -f https://download.pytorch.org/whl/cu117/torch_stable.html

# 3. 安装 MuJoCo 系统依赖（Ubuntu）
sudo apt-get install -y libgl1-mesa-glx libglew-dev libosmesa6-dev libglfw3-dev

# 4. 安装 Python 依赖
cd <REPO_ROOT>
pip install -r requirements.txt

# 5. 安装 MetaWorld（从源码）
pip install git+https://github.com/Farama-Foundation/MetaWorld.git@master#egg=metaworld

# 6. 设置 PYTHONPATH
export PYTHONPATH=<REPO_ROOT>:$PYTHONPATH
```

### 3.3 验证安装

```bash
# 运行环境验证脚本
bash scripts/env_setup.sh

# 或手动验证关键 import
python3 -c "
import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')
import metaworld; print('metaworld OK')
import hydra; print('hydra OK')
"
```

---

## 4. 配置详解

### 4.1 顶层配置 config/config.yaml

Hydra defaults 组合列表，决定加载哪些子配置。关键项：

| 项 | 值 | 说明 |
|---|---|---|
| setup | mtrl | 基础路径、device(cuda:0)、seed |
| experiment | mtrl | 训练步数、评估/保存频率 |
| agent | 按脚本覆盖 | sac / gbi_sac / guide_sac |
| env | 按脚本覆盖 | metaworld-mt10 / metaworld-mt50 |
| replay_buffer | mtrl | capacity=1000000, batch_size=128 |
| metrics | 按脚本覆盖 | mtrl / mtrl_gbi / mtrl_guide |

### 4.2 Setup 配置 config/setup/mtrl.yaml

```yaml
seed: null           # 由脚本覆盖
base_path:           # 由脚本覆盖（结果落盘根目录）
save_dir: ${setup.base_path}/logs/${setup.id}
device: cuda:0
id: sample
alg:                 # 由脚本覆盖
```

### 4.3 环境配置

#### MetaWorld MT10 (config/env/metaworld-mt10.yaml)

```yaml
name: metaworld-mt10
num_envs: 10
benchmark: metaworld.MT10
random_goal: True
wrappers:
  scale_reward_wrapper:
    reward_scale: 0.1        # 奖励缩放因子
  max_step_wrapper:
    max_step: 200             # 每 episode 最大步数
network:
  hidden_dim: 400
  num_layers: 5               # Actor/Critic MLP 层数
```

**10 个任务**：reach-v2, push-v2, pick-place-v2, door-open-v2, drawer-open-v2, drawer-close-v2, button-press-topdown-v2, peg-insert-side-v2, window-open-v2, window-close-v2

#### MetaWorld MT50 (config/env/metaworld-mt50.yaml)

同 MT10 结构，`num_envs: 50`，包含全部 50 个 MetaWorld 任务。

### 4.4 Agent 配置

#### state_sac (config/agent/state_sac.yaml)

```yaml
name: state_sac
encoder_feature_dim: 50
num_layers: 0
num_filters: 0
builder:
  _target_: mtrl.agent.sac.Agent
  discount: 0.99
  init_temperature: 1.0
  actor_update_freq: 1
  critic_tau: 0.01
  critic_target_update_freq: 1
  encoder_tau: 0.05
  loss_reduction: alpha_weight
```

#### gbi_sac (config/agent/gbi_sac.yaml)

```yaml
name: gbi_sac
encoder_feature_dim: 50

gbi:
  enabled: True
  arbiter_mode: gbi           # gbi | qmp | imag
  trigger_mode: adaptive      # adaptive | fixed_K | always
  H: 5                        # 想象 rollout 步数
  gamma: 0.99
  W: 10000                    # U 滑窗长度
  rho_on: 0.9                 # τ_on 分位数
  rho_reject: 0.99            # τ_reject 分位数
  tau_off_ratio: 0.7          # τ_off = 0.7 × τ_on
  K_min: 3                    # 最小执行段长度
  K_max: 15                   # 最大执行段长度
  m_low: 3                    # 连续低 U 步数触发提前终止
  fixed_K: 10                 # fixed_K 模式下的固定步长
  calibrate_window: 500       # λ_t 校准缓冲大小
  wm_update_every: 1          # 世界模型训练频率（每 N 步）
  wm_batch_size: 1024         # 世界模型训练 batch
  tta_enabled: True
  tta_recent_n: 5             # TTA recent-N 缓冲
  candidates_dir: null        # 候选快照目录（运行时覆盖）
  candidate_steps: []         # 候选快照步数列表（运行时覆盖）
  source_log_dump_every: 10000

builder:
  _target_: mtrl.agent.gbi_sac.Agent
  discount: 0.99
  init_temperature: 1.0
  actor_update_freq: 1
  critic_tau: 0.01
  critic_target_update_freq: 1
  encoder_tau: 0.05
  loss_reduction: alpha_weight
```

#### guide_sac (config/agent/guide_sac.yaml)

```yaml
name: guide_sac
encoder_feature_dim: 50
guide_encoder_feature_dim: 50
builder:
  _target_: mtrl.agent.guide_sac.Agent
  discount: 0.99
  init_temperature: 1.0
  guide_init_temperature: 1.0
  actor_update_freq: 1
  critic_tau: 0.01
  critic_target_update_freq: 1
  encoder_tau: 0.05
  guide_actor_update_freq: 1
  guide_critic_tau: 0.01
  guide_critic_target_update_freq: 1
  guide_encoder_tau: 0.05
  loss_reduction: alpha_weight
  guide_loss_reduction: mean
  guide_hindsight: False
```

### 4.5 世界模型配置 config/agent/components/world_model.yaml

```yaml
_target_: mtrl.agent.components.world_model.WorldModel
hidden_dim: 512                # GRU/encoder/state_head 隐层维度
num_heads: 5                   # 奖励头成员数
reward_bins: 101               # two-hot 分箱数
reward_min: -400.0             # 原始奖励下界
reward_max: 400.0              # 原始奖励上界
encoder_ema_tau: 0.995         # 编码器 EMA 衰减率
lr: 1e-4                       # 世界模型学习率
adapt_lr: 1e-5                 # TTA 适配学习率（0.1 × wm_lr）
adapt_scope: last_layer         # TTA 范围：last_layer | all_heads
grad_clip: 10.0
reward_weight: 1.0
obs_weight: 1.0
state_weight: 1.0
cache_size: 4096               # 残差加权缓存大小
cache_refresh_every: 512       # 缓存刷新间隔（步）
residual_quantile: 0.8         # quantile-sigmoid 拐点
residual_temperature: 5.0      # quantile-sigmoid 温度
residual_max_weight: 5.0       # 残差权重上限截断
```

### 4.6 多任务配置 config/agent/components/multitask.yaml

```yaml
num_envs: 10                   # 由 env.num_envs 决定（MT10=10, MT50=50）
should_use_disentangled_alpha: False   # 脚本中覆盖为 True
should_use_task_encoder: False
should_use_task_onehot: False
should_use_multi_head_policy: False    # 脚本中覆盖为 True
should_use_disjoint_policy: False
should_use_pcgrad: False
clip_grad_norm: 0.1
use_loss_threshold: True
mask_loss_step: 100000
mask_loss_threshold: 3000
max_alpha: 1.0
```

### 4.7 Actor / Critic / Encoder 配置

- **Actor**：`mtrl.agent.components.actor.Actor`，hidden_dim=400，num_layers=5（来自 env.network），log_std_bounds=[-20, 2]
- **Critic**：`mtrl.agent.components.critic.Critic`，hidden_dim=400，num_layers=5（与 actor 同构）
- **Encoder**：type=identity（脚本中选择），feature_dim=50

### 4.8 实验配置 config/experiment/mtrl.yaml

```yaml
name: ???                      # 由脚本覆盖（metaworld）
init_steps: 1500               # 随机探索初始步数
num_train_steps: 1000000       # 默认 1M，脚本中覆盖
eval_freq: 10000               # 评估频率（也等于保存频率）
save_freq: 100000
num_eval_episodes: 32
should_resume: False
save.model.should_save: False  # state_sac 脚本覆盖为 True
save.model.retain_last_n: 1
save.buffer.size_per_chunk: 10000
save_video: False
use_guide: False
```

### 4.9 Replay Buffer 配置 config/replay_buffer/mtrl.yaml

```yaml
_target_: mtrl.replay_buffer.ReplayBuffer
capacity: 1000000              # 100 万
batch_size: 128
```

### 4.10 Hydra 覆盖写法速查

```bash
# 切换 agent
agent=gbi_sac          # GbI
agent=state_sac        # 独立 SAC
agent=guide_sac        # CTPG guide

# 切换环境
env=metaworld-mt10
env=metaworld-mt50

# 切换指标
metrics=mtrl           # 基础 SAC
metrics=mtrl_gbi       # GbI（含 wm_loss, arbiter/* 等）
metrics=mtrl_guide     # Guide 基线

# 切换裁决模式
agent.gbi.arbiter_mode=gbi     # 完整公式
agent.gbi.arbiter_mode=qmp     # λ≡0 消融
agent.gbi.arbiter_mode=imag    # 纯想象消融

# 切换触发模式
agent.gbi.trigger_mode=adaptive   # U_score 自适应
agent.gbi.trigger_mode=fixed_K    # 固定步频
agent.gbi.trigger_mode=always     # 每步重判
```

---

## 5. 实验脚本与命令

### 5.1 冒烟测试

**目的**：验证 env 构建/训练/评估/日志链路全通（15k 步）。

```bash
# state_sac 冒烟
bash scripts/smoke.sh state_sac metaworld mt10

# GbI 冒烟
bash scripts/smoke.sh gbi metaworld mt10

# QMP 冒烟
bash scripts/smoke.sh qmp metaworld mt10
```

**smoke.sh 实际执行参数**：
- `num_train_steps=15000`
- `eval_freq=5000`
- `replay_buffer.capacity=20000`
- `replay_buffer.batch_size=512`
- `init_steps=1500`

**QMP 冒烟额外覆盖**：`agent.gbi.arbiter_mode=qmp`, `agent.gbi.trigger_mode=always`

### 5.2 state_sac 独立训练（产出候选快照池）

**脚本**：`scripts/alg/state_sac_indep.sh`

```bash
# 用法
bash scripts/alg/state_sac_indep.sh [seed] [map]
# 默认 seed=0, map=mt10

# 实际命令
bash scripts/alg/state_sac_indep.sh 0
```

**完整参数**：

```bash
python3 -u main.py \
  setup.alg=state_sac \
  setup.id=seed0 \
  setup.seed=0 \
  setup.base_path=/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0 \
  env=metaworld-mt10 \
  agent=state_sac \
  metrics=mtrl \
  experiment.name=metaworld \
  experiment.num_train_steps=500100 \
  experiment.eval_freq=50000 \
  experiment.save_freq=50000 \
  experiment.save.model.should_save=True \
  experiment.save.model.retain_last_n=-1 \
  experiment.num_eval_episodes=10 \
  replay_buffer.capacity=1000000 \
  replay_buffer.batch_size=1280 \
  agent.encoder.type_to_select=identity \
  agent.multitask.should_use_disentangled_alpha=True \
  agent.multitask.should_use_task_encoder=False \
  agent.multitask.should_use_task_onehot=False \
  agent.multitask.should_use_multi_head_policy=True \
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False
```

**关键参数说明**：
- `num_train_steps=500100`：训练 50 万步
- `save_freq=50000`：每 5 万步保存一次快照（产出 50k, 100k, ..., 500k 共 10 个快照）
- `retain_last_n=-1`：保留所有快照
- `batch_size=1280`：较大 batch 加速训练
- `multi_head_policy=True`：每任务独立 actor head（与 GbI 结构同构，才能互相加载 actor state_dict）
- `disentangled_alpha=True`：每任务独立温度系数

**预期产物**：
- 模型快照：`<base_path>/model/actor_{step}.pt`, `critic_{step}.pt` 等
- 训练日志：`<base_path>/train.log`
- 评估日志：`<base_path>/eval.log`

### 5.3 GbI 主实验

**脚本**：`scripts/alg/gbi.sh`

```bash
# 用法
bash scripts/alg/gbi.sh metaworld mt10 <capacity> <batch> <num_train_steps> [seed]

# 正式训练示例
bash scripts/alg/gbi.sh metaworld mt10 1000000 128 300000 0
```

**完整参数**（以 capacity=1000000, batch=128, steps=300000, seed=0 为例）：

```bash
python3 -u main.py \
  setup.alg=gbi_sac \
  setup.id=seed0 \
  setup.seed=0 \
  setup.base_path=/root/rivermind-data/lost+found/gbi/experiments/runs/gbi/metaworld/mt10/seed0 \
  env=metaworld-mt10 \
  agent=gbi_sac \
  metrics=mtrl_gbi \
  experiment.name=metaworld \
  experiment.num_train_steps=300000 \
  experiment.eval_freq=20000 \
  experiment.num_eval_episodes=10 \
  replay_buffer.capacity=1000000 \
  replay_buffer.batch_size=128 \
  agent.encoder.type_to_select=identity \
  agent.multitask.should_use_disentangled_alpha=True \
  agent.multitask.should_use_task_encoder=False \
  agent.multitask.should_use_task_onehot=False \
  agent.multitask.should_use_multi_head_policy=True \
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False
```

**加载候选快照**（Hydra 覆盖）：

```bash
# 在 gbi.sh 基础上追加：
agent.gbi.candidates_dir=/path/to/state_sac_indep/mt10/seed0/model \
agent.gbi.candidate_steps='[50000,100000,200000,300000,500000]'
```

### 5.4 QMP 对照实验

**脚本**：`scripts/alg/gbi_qmp.sh`

```bash
bash scripts/alg/gbi_qmp.sh metaworld mt10 1000000 128 300000 0
```

**与 GbI 的唯一差异**（其余参数完全相同）：

```bash
agent.gbi.arbiter_mode=qmp      # λ_t ≡ 0，纯 Q 锚裁决
agent.gbi.trigger_mode=always    # 每步重判（不依赖 U_score 触发）
```

### 5.5 Guide 基线（CTPG 复现）

#### guide_mtsac（shared encoder + task one-hot）

```bash
bash scripts/alg/guide_mtsac.sh metaworld mt10 1000000 128 300000
```

关键：`should_use_task_onehot=True`, `should_use_multi_head_policy=False`, `guide_hindsight=True`

#### guide_mhsac（multi-head policy）

```bash
bash scripts/alg/guide_mhsac.sh metaworld mt10 1000000 128 300000
```

关键：`should_use_multi_head_policy=True`, `should_use_task_onehot=False`, `guide_hindsight=True`

---

## 6. 实验流程

### 6.1 完整实验顺序

```
Phase 0.1: state_sac 独立训练 500k → 产出候选快照池
    ↓
Phase 1: GbI 主实验 + QMP 对照（+ 可选 Guide 基线）
    ↓
Phase 2: 消融实验（纯想象 / 固定 K / 不同 H 等）
    ↓
Phase 3: 规模外推（MT50 / HalfCheetah MT8）
```

### 6.2 各阶段详细说明

#### Phase 0.1：候选快照池

```bash
bash scripts/alg/state_sac_indep.sh 0    # seed=0
# 可选：多 seed
bash scripts/alg/state_sac_indep.sh 1
bash scripts/alg/state_sac_indep.sh 2
```

- **训练步数**：500,100 步
- **快照间隔**：50,000 步（产出 50k, 100k, ..., 500k 共 10 个快照）
- **快照路径**：`/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0/model/actor_{step}.pt`

#### Phase 1：GbI + QMP 主实验

```bash
# GbI（加载候选快照）
bash scripts/alg/gbi.sh metaworld mt10 1000000 128 300000 0
# 需通过 Hydra 覆盖设置 candidates_dir 和 candidate_steps

# QMP 对照
bash scripts/alg/gbi_qmp.sh metaworld mt10 1000000 128 300000 0

# Guide 基线（可选）
bash scripts/alg/guide_mhsac.sh metaworld mt10 1000000 128 300000
```

#### Phase 2：消融实验

| 消融项 | Hydra 覆盖 |
|---|---|
| 纯想象 | `agent.gbi.arbiter_mode=imag` |
| 固定 K=10 | `agent.gbi.trigger_mode=fixed_K agent.gbi.fixed_K=10` |
| H=3 | `agent.gbi.H=3` |
| H=8 | `agent.gbi.H=8` |
| ensemble=3 | 需改 `world_model.num_heads=3` |
| 无 TTA | `agent.gbi.tta_enabled=False` |

### 6.3 日志与指标说明

**日志文件**：
- `train.log`：训练过程 stdout 完整记录
- `eval.log`：评估结果
- `log.jsonl`：JSON Lines 格式的详细指标
- `config.json`：本次运行的完整解析后配置

**关键训练指标**（`mtrl_gbi` 指标集）：

| 指标 | 缩写 | 说明 |
|---|---|---|
| episode_reward | R | 训练 episode 平均奖励 |
| success | Su | 训练 episode 成功率 |
| wm_loss | WL | 世界模型总损失 |
| wm_state_loss | WSL | 动力学状态损失（MSE） |
| wm_reward_loss | WRL | 奖励头损失（交叉熵） |
| wm_dyn_obs_mae | DMAE | 动力学 obs MAE（诊断） |
| arbiter/lambda_t | LT | 想象通道信任权重 |
| arbiter/kappa_t | KT | Spearman 秩相关 |
| arbiter/tau_on | TON | 触发阈值 |
| gbi_trigger_rate | TGR | 触发率 |
| gbi_switch_rate | SWR | 换手率 |
| gbi_uscore_ema | UEM | U_score EMA |
| gbi_seg_len_mean | SLM | 执行段平均长度 |
| gbi_adapt_loss_before | ALB | TTA 前损失 |
| gbi_adapt_loss_after | ALA | TTA 后损失 |

**来源归档**：每 10000 步 dump 一次 `gbi_source_step{N}.npz`，包含逐条决策记录（step, env, task, source, triggered, u, s_chosen）。

---

## 7. 硬件需求与建议

### 7.1 最低配置（T4 级别）

| 资源 | 要求 |
|---|---|
| GPU | NVIDIA T4 (16GB) 或同级 |
| CPU | 8+ 核（需设 OMP_NUM_THREADS=1 等避免线程空转） |
| RAM | 32GB+ |
| 磁盘 | 50GB+（快照 + buffer + 日志） |

### 7.2 推荐配置

| 资源 | 推荐 |
|---|---|
| GPU | NVIDIA A100 (40GB) / V100 (32GB) |
| CPU | 16+ 核 |
| RAM | 64GB+ |
| 磁盘 | 100GB+ SSD |

### 7.3 关键参数在不同硬件下的调整建议

| 参数 | T4 (16GB) | V100 (32GB) | A100 (40GB) |
|---|---|---|---|
| replay_buffer.capacity | 1,000,000 | 1,000,000 | 2,000,000 |
| replay_buffer.batch_size | 128 | 256 | 512 |
| world_model.hidden_dim | 512 | 512 | 512 |
| wm_batch_size | 1024 | 2048 | 4096 |
| num_heads | 5 | 5 | 5 |
| state_sac batch_size | 1280 | 1280 | 2560 |

**T4 特别注意**：
- 必须设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
- 如内存不足，可降低 `replay_buffer.capacity` 至 500,000
- `wm_batch_size` 可降至 512

---

## 8. 已知问题与修复记录

### 8.1 已修复的 Bug

| 问题 | 原因 | 修复 |
|---|---|---|
| Device mismatch | 世界模型 buffer/tensor 未跟随 device 迁移 | world_model.to(device) 中同步迁移 _obs_sum, _obs_sqsum, _task_onehot_eye |
| Action shape 不匹配 | actor forward 约定 action_shape[0]*2，加载快照时需对齐 | _load_snapshot_candidates 中 action_shape=action_shape[0]*2 |
| Logger key 冲突 | GbI 新增 metric key 未在 mtrl_gbi.yaml 中登记 | 新增 wm_loss, arbiter/*, gbi_* 等 metric 条目 |
| Config duplicate | config.yaml defaults 中重复列出某些组件 | 清理 defaults 列表 |
| 线程空转步速坍塌 | 23 核 cgroup 配额下 spawn worker 每进程开 192 线程 | 所有脚本开头设置 OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 |
| 奖励头状态依赖重尾偏置 | MSE 回归对饱和平台/重尾奖励失效 | two-hot + symlog + 残差加权双流采样（v5 核心修复） |

### 8.2 已知的非阻塞问题

| 问题 | 影响 | 状态 |
|---|---|---|
| Q 锚用均值而非 min | q_anchor 返回 (q1+q2)/2 而非 min(q1,q2)；实测均值表现更好 | 保留现状，作为设计选择 |
| lambda_ramp 未实现 | Arbiter 接受 lambda_ramp 参数但未使用；λ_t 完全由 Spearman κ_t 校准 | 预留接口，不影响运行 |
| Phase 0 评估 harness 未实现 | run_phase0.sh 中 P0.2-P0.4 评估脚本为 TODO 注释 | 不阻塞后续实验 |
| Phase 0 决策场为空 | 同期同质快照下反事实差异≈0 | 已通过混合不同训练阶段快照解决 |

---

## 9. 当前实验状态

### 9.1 state_sac 500k（已完成）

- **路径**：`/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0/`
- **日志**：`/root/rivermind-data/lost+found/CTPG-main/CTPG-main/logs/metaworld-mt10/state_sac/2026-08-31-07-06-43_issue_d8c28e93bded171421846ce8e996c50ea4523015_seed_0/`
- **快照**：每 50k 步保存，共 10 个快照（50k ~ 500k）

### 9.2 GbI 300k（运行中）

- **路径**：`/root/rivermind-data/lost+found/gbi/experiments/runs/gbi/metaworld/mt10/seed0/`
- **训练步数**：300,000

### 9.3 QMP 自动衔接

- **路径**：`/root/rivermind-data/lost+found/gbi/experiments/runs/qmp/`
- **已有日志**：多个 `gbi_source_step*.npz` 文件已生成

---

## 附录 A：完整脚本内容（方便重建）

### A.1 smoke.sh

```bash
#!/usr/bin/env bash
set -e

ALG="$1"
ENV="$2"
MAP="$3"
if [ -z "$ALG" ] || [ -z "$ENV" ] || [ -z "$MAP" ]; then
    echo "usage: bash scripts/smoke.sh <alg> <env> <map>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SMOKE_STEPS=15000
EVAL_FREQ=5000

METRICS=mtrl
AGENT_CFG="$ALG"
if [ "$ALG" == "gbi" ] || [ "$ALG" == "gbi_sac" ]; then
    METRICS=mtrl_gbi
    AGENT_CFG=gbi_sac
elif [ "$ALG" == "qmp" ]; then
    METRICS=mtrl_gbi
    AGENT_CFG=gbi_sac
fi

if [ "$ENV" == "metaworld" ] && [ "$MAP" == "mt10" ]; then
    ENV_NAME="metaworld-mt10"
    RB_CAPACITY=20000
    RB_BATCH=512
elif [ "$ENV" == "gym_extensions" ]; then
    ENV_NAME="gym_extensions-$MAP"
    RB_CAPACITY=20000
    RB_BATCH=512
else
    echo "Error: unsupported env/map: $ENV/$MAP"
    exit 1
fi

cmd=(
  python3 -u main.py
  "setup.alg=$ALG"
  "setup.id=smoke_${ALG}_${MAP}_s0"
  "setup.seed=0"
  "env=$ENV_NAME"
  "agent=$AGENT_CFG"
  "metrics=$METRICS"
  "experiment.name=$ENV"
  "experiment.num_train_steps=$SMOKE_STEPS"
  "experiment.eval_freq=$EVAL_FREQ"
  "experiment.init_steps=1500"
  "replay_buffer.capacity=$RB_CAPACITY"
  "replay_buffer.batch_size=$RB_BATCH"
)

if [ "$ALG" == "qmp" ]; then
    cmd+=( "agent.gbi.arbiter_mode=qmp" "agent.gbi.trigger_mode=always" )
fi

echo "[smoke] launching: ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}" 2>&1 | tee "smoke_${ALG}_${MAP}.log"
echo "[smoke] done (exit=$?)"
```

### A.2 state_sac_indep.sh

```bash
#!/usr/bin/env bash
set -e

seed=${1:-0}
map=${2:-mt10}
env_name="metaworld-$map"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_BASE="/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep"
RUN_DIR="${RUN_BASE}/${map}/seed${seed}"
mkdir -p "$RUN_DIR"

cmd=(
  python3 -u main.py
  setup.alg=state_sac
  "setup.id=seed${seed}"
  "setup.seed=$seed"
  "setup.base_path=$RUN_DIR"
  env=$env_name
  agent=state_sac
  metrics=mtrl
  experiment.name=metaworld
  experiment.num_train_steps=500100
  experiment.eval_freq=50000
  experiment.save_freq=50000
  experiment.save.model.should_save=True
  experiment.save.model.retain_last_n=-1
  experiment.num_eval_episodes=10
  replay_buffer.capacity=1000000
  replay_buffer.batch_size=1280
  agent.encoder.type_to_select=identity
  agent.multitask.should_use_disentangled_alpha=True
  agent.multitask.should_use_task_encoder=False
  agent.multitask.should_use_task_onehot=False
  agent.multitask.should_use_multi_head_policy=True
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False
)

echo "[p0.1] launching: ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
echo "[p0.1] done (exit=$?)"
```

### A.3 gbi.sh

```bash
#!/usr/bin/env bash
set -e

env=$1
map=$2
replay_buffer_capacity=$3
replay_buffer_batch_size=$4
num_train_steps=$5
seed=${6:-0}

env_name="$env-$map"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_BASE="/root/rivermind-data/lost+found/gbi/experiments/runs/gbi"
RUN_DIR="${RUN_BASE}/${env}/${map}/seed${seed}"
mkdir -p "$RUN_DIR"

cmd=(
  python3 -u main.py
  setup.alg=gbi_sac
  "setup.id=seed${seed}"
  "setup.seed=$seed"
  "setup.base_path=$RUN_DIR"
  env=$env_name
  agent=gbi_sac
  metrics=mtrl_gbi
  experiment.name=$env
  experiment.num_train_steps=$num_train_steps
  experiment.eval_freq=20000
  experiment.num_eval_episodes=10
  replay_buffer.capacity=$replay_buffer_capacity
  replay_buffer.batch_size=$replay_buffer_batch_size
  agent.encoder.type_to_select=identity
  agent.multitask.should_use_disentangled_alpha=True
  agent.multitask.should_use_task_encoder=False
  agent.multitask.should_use_task_onehot=False
  agent.multitask.should_use_multi_head_policy=True
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False
)

echo "[gbi] launching: ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
echo "[gbi] done (exit=$?)"
```

### A.4 gbi_qmp.sh

```bash
#!/usr/bin/env bash
set -e

env=$1
map=$2
replay_buffer_capacity=$3
replay_buffer_batch_size=$4
num_train_steps=$5
seed=${6:-0}

env_name="$env-$map"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_BASE="/root/rivermind-data/lost+found/gbi/experiments/runs/qmp"
RUN_DIR="${RUN_BASE}/${env}/${map}/seed${seed}"
mkdir -p "$RUN_DIR"

cmd=(
  python3 -u main.py
  setup.alg=qmp
  "setup.id=seed${seed}"
  "setup.seed=$seed"
  "setup.base_path=$RUN_DIR"
  env=$env_name
  agent=gbi_sac
  metrics=mtrl_gbi
  experiment.name=$env
  experiment.num_train_steps=$num_train_steps
  experiment.eval_freq=20000
  experiment.num_eval_episodes=10
  replay_buffer.capacity=$replay_buffer_capacity
  replay_buffer.batch_size=$replay_buffer_batch_size
  agent.encoder.type_to_select=identity
  agent.multitask.should_use_disentangled_alpha=True
  agent.multitask.should_use_task_encoder=False
  agent.multitask.should_use_task_onehot=False
  agent.multitask.should_use_multi_head_policy=True
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False
  agent.gbi.arbiter_mode=qmp
  agent.gbi.trigger_mode=always
)

echo "[qmp] launching: ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
echo "[qmp] done (exit=$?)"
```

### A.5 guide_mtsac.sh

```bash
#!/usr/bin/env bash
env=$1
map=$2
replay_buffer_capacity=$3
replay_buffer_batch_size=$4
num_train_steps=$5

env_name="$env-$map"

PYTHONPATH=. python3 -u main.py \
setup.alg=guide_mtsac \
metrics=mtrl_guide \
env=$env_name \
agent=guide_sac \
experiment.name=$env \
experiment.num_train_steps=$num_train_steps \
experiment.use_guide=True \
replay_buffer.capacity=$replay_buffer_capacity \
replay_buffer.batch_size=$replay_buffer_batch_size \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.should_use_task_encoder=False \
agent.multitask.should_use_task_onehot=True \
agent.multitask.should_use_multi_head_policy=False \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.guide_encoder.type_to_select=identity \
agent.builder.guide_hindsight=True
```

### A.6 guide_mhsac.sh

```bash
#!/usr/bin/env bash
env=$1
map=$2
replay_buffer_capacity=$3
replay_buffer_batch_size=$4
num_train_steps=$5

env_name="$env-$map"

PYTHONPATH=. python3 -u main.py \
setup.alg=guide_mhsac \
metrics=mtrl_guide \
env=$env_name \
agent=guide_sac \
experiment.name=$env \
experiment.num_train_steps=$num_train_steps \
experiment.use_guide=True \
replay_buffer.capacity=$replay_buffer_capacity \
replay_buffer.batch_size=$replay_buffer_batch_size \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.should_use_task_encoder=False \
agent.multitask.should_use_task_onehot=False \
agent.multitask.should_use_multi_head_policy=True \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.guide_encoder.type_to_select=identity \
agent.builder.guide_hindsight=True
```

### A.7 env_setup.sh

```bash
#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/root/rivermind-data/lost+found/gbi"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "[env_setup] python: $(python3 --version 2>&1)"
echo "[env_setup] pip freeze -> $LOG_DIR/env_setup.log"
python3 -m pip list 2>/dev/null | sort > "$LOG_DIR/env_setup.log" || \
    python3 -m pip freeze 2>/dev/null | sort > "$LOG_DIR/env_setup.log"

echo "[env_setup] gpu info -> $LOG_DIR/gpu_info.txt"
nvidia-smi > "$LOG_DIR/gpu_info.txt" 2>&1 || echo "no nvidia-smi" | tee -a "$LOG_DIR/gpu_info.txt"

echo "[env_setup] verifying imports"
VERIFY=$(python3 - <<'PY'
import importlib, torch
for mod, pkg in [("gymnasium","gymnasium"), ("metaworld","metaworld"),
                 ("hydra","hydra-core"), ("ml_logger","ml-logger"),
                 ("omegaconf","omegaconf"), ("termcolor","termcolor")]:
    try:
        m = importlib.import_module(mod)
        print(f"OK  {pkg:<18} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"FAIL {pkg:<18} {e}")
print(f"OK  torch {torch.__version__} cuda={torch.cuda.is_available()}")
PY
)
echo "$VERIFY" | tee -a "$LOG_DIR/env_setup.log"
if echo "$VERIFY" | grep -q FAIL; then
    echo "[env_setup] some imports failed — abort"
    exit 1
fi
echo "[env_setup] DONE (log: $LOG_DIR/env_setup.log)"
```

### A.8 run_phase0.sh

```bash
#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "$0")/.." && pwd)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== Phase 0.1: state_sac 500k 候选快照池 ==="
bash scripts/alg/state_sac_indep.sh 0

# --- P0.2-P0.4 评估 harness 待实现（不阻塞后续实验） ---

echo "[phase0] P0.1 done. P0.2-P0.4 harness 未启用（见注释）。"
```

---

## 附录 B：快速启动命令（TL;DR）

```bash
# 0. 环境搭建
conda create -n gbi python=3.8 -y && conda activate gbi
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
pip install -r requirements.txt
pip install git+https://github.com/Farama-Foundation/MetaWorld.git@master#egg=metaworld
export PYTHONPATH=<REPO_ROOT>:$PYTHONPATH

# 1. 冒烟测试（验证环境）
bash scripts/smoke.sh state_sac metaworld mt10

# 2. 训练候选快照池（500k 步，约 4-6 小时 T4）
bash scripts/alg/state_sac_indep.sh 0

# 3. GbI 主实验（300k 步，加载候选快照）
bash scripts/alg/gbi.sh metaworld mt10 1000000 128 300000 0
# 注：需通过 Hydra 覆盖设置 candidates_dir 和 candidate_steps

# 4. QMP 对照
bash scripts/alg/gbi_qmp.sh metaworld mt10 1000000 128 300000 0
```
