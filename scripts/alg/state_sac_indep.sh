#!/usr/bin/env bash
# P0.1：独立 SAC（multi-head + disentangled alpha）MT10 训 500k，
# save_freq=50k 保留快照 → early/mid/late 候选池（供 GbI 裁决器复用）。
# 用法: bash scripts/alg/state_sac_indep.sh [seed]
set -e

seed=${1:-0}
map=${2:-mt10}

env_name="metaworld-$map"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_BASE="${GBI_RUN_BASE:-/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep}"
RUN_DIR="${RUN_BASE}/${map}/seed${seed}"
mkdir -p "$RUN_DIR"

# 500k 步 / eval+save 每 50k（快照: 50k,100k,...,500k）
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
