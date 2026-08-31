#!/usr/bin/env bash
# smoke.sh — 冒烟测试: 在 MT10 上跑少量步数, 验证 env 构建/训练/评估/日志链路全通.
# 用法: bash scripts/smoke.sh <alg> <env: metaworld|gym_extensions> <map>
#   bash scripts/smoke.sh state_sac metaworld mt10     # 15k 步
#   bash scripts/smoke.sh gbi       metaworld mt10     # 15k 步 (GbI agent)
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

SMOKE_STEPS=15000
EVAL_FREQ=5000

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
  "agent=$ALG"
  "metrics=mtrl"
  "experiment.name=$ENV"
  "experiment.num_train_steps=$SMOKE_STEPS"
  "experiment.eval_freq=$EVAL_FREQ"
  "experiment.init_steps=1500"
  "replay_buffer.capacity=$RB_CAPACITY"
  "replay_buffer.batch_size=$RB_BATCH"
)

echo "[smoke] launching: ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}" 2>&1 | tee "/root/rivermind-data/lost+found/gbi/smoke_${ALG}_${MAP}.log"
echo "[smoke] done (exit=$?)"