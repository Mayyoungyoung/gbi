#!/usr/bin/env bash
# GbI（Guide-by-Imagination v5）主算法启动脚本 —— MT10。
# （M6 修复：补齐候选快照池与 checkpoint 参数，此前缺 candidates_dir/
#   candidate_steps 导致裁决退化为仅 own，不可用作 GbI 实验。）
# 注意：本脚本前台 `| tee` 启动，仅用于调试/短跑；正式 300k 长跑请用
# scripts/run_gbi_nohup.sh（防 SSH 断连管道死锁，2026-08-31 事故）。
# 用法: bash scripts/alg/gbi.sh metaworld mt10 <capacity> <batch> <num_train_steps> [seed]
set -e

env=$1
map=$2
replay_buffer_capacity=$3
replay_buffer_batch_size=$4
num_train_steps=$5
seed=${6:-0}

env_name="$env-$map"

# 23 核 cgroup 配额：单线程化 env worker（否则 spawn worker 每进程开 192 线程，
# 线程空转导致步速坍塌 ~4 it/s）。世界模型/仲裁打分在 GPU 主进程不受影响。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 结果落盘目录（计划 §7: experiments/runs/{alg}/{env}/{seed}）
RUN_BASE="/root/rivermind-data/lost+found/gbi/experiments/runs/gbi"
RUN_DIR="${RUN_BASE}/${env}/${map}/seed${seed}"
mkdir -p "$RUN_DIR"

# 候选快照池（state_sac_indep 独立源策略），可用环境变量覆盖
CANDIDATES_DIR="${GBI_CANDIDATES_DIR:-/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_issue_57d4f43a4bf96214e8f7be83a0564528bff75e77_seed_0/model}"
CANDIDATE_STEPS="${GBI_CANDIDATE_STEPS:-[50000,100000,150000,200000,250000,300000,350000,400000,450000,500000]}"

# 源策略 multitask 配置与 CTPG guide_mhsac 的 source 一致（multi-head +
# disentangled alpha + identity encoder）——候选快照池（state_sac_indep.sh）
# 与 gbi_sac 结构同构，才能互相加载 actor state_dict。
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
  "agent.gbi.candidates_dir=$CANDIDATES_DIR"
  "agent.gbi.candidate_steps=$CANDIDATE_STEPS"
  # checkpoint 留档（与 run_gbi_nohup.sh 同参数）
  experiment.save.model.should_save=True
  experiment.save_freq=20000
  experiment.save.model.retain_last_n=5
)

echo "[gbi] launching: ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
echo "[gbi] done (exit=$?)"
