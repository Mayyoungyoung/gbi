#!/usr/bin/env bash
# CTPG guide_mhsac 基线（multi-head policy + guide）—— MT10 对比参照。
# （M7 扩展：支持 seed/base_path/checkpoint 留档，供 run_full_benchmark.sh 串行调用）
# 用法: bash scripts/alg/guide_mhsac.sh metaworld mt10 <capacity> <batch> <num_train_steps> [seed]
set -e

env=$1
map=$2
replay_buffer_capacity=$3
replay_buffer_batch_size=$4
num_train_steps=$5
seed=${6:-0}

env_name="$env-$map"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_BASE="/root/rivermind-data/lost+found/gbi/experiments/runs/guide"
RUN_DIR="${RUN_BASE}/${map}/seed${seed}"
mkdir -p "$RUN_DIR"

PYTHONPATH=. python3 -u main.py \
setup.alg=guide_mhsac \
"setup.id=seed${seed}" \
"setup.seed=$seed" \
"setup.base_path=$RUN_DIR" \
metrics=mtrl_guide \
env=$env_name \
agent=guide_sac \
experiment.name=$env \
experiment.num_train_steps=$num_train_steps \
experiment.eval_freq=20000 \
experiment.num_eval_episodes=10 \
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
agent.builder.guide_hindsight=True \
experiment.save.model.should_save=True \
experiment.save_freq=20000 \
experiment.save.model.retain_last_n=5
