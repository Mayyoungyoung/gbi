#!/usr/bin/env bash
# QMP 消融基线：GbI 裁决公式去掉想象项（λ_t ≡ 0），触发改为每步重判（always）。
# 与 gbi.sh 同参数协议；其余配置一致，保证唯一变量是裁决公式/触发方式。
# （M6 修复：补齐候选快照池与 checkpoint 参数；arbiter.py 已保证 qmp 模式
#   无 guardrail_reject/K_min 惯性，λ_t ≡ 0 口径纯净。）
# 注意：本脚本前台 `| tee` 启动仅用于调试；正式长跑请用 run_gbi_then_qmp.sh。
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

# 候选快照池（与 gbi.sh 同源，可用环境变量覆盖）
CANDIDATES_DIR="${GBI_CANDIDATES_DIR:-/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_issue_57d4f43a4bf96214e8f7be83a0564528bff75e77_seed_0/model}"
CANDIDATE_STEPS="${GBI_CANDIDATE_STEPS:-[50000,100000,150000,200000,250000,300000,350000,400000,450000,500000]}"

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
  "agent.gbi.candidates_dir=$CANDIDATES_DIR"
  "agent.gbi.candidate_steps=$CANDIDATE_STEPS"
  # checkpoint 留档（与 run_gbi_then_qmp.sh 同参数）
  experiment.save.model.should_save=True
  experiment.save_freq=20000
  experiment.save.model.retain_last_n=5
)

echo "[qmp] launching: ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
echo "[qmp] done (exit=$?)"
