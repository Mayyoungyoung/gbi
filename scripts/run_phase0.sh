#!/usr/bin/env bash
# Phase 0 全流程（MT10，seed=0）：P0.1 候选池 → P0.2 决策场 → P0.3 世界模型 → P0.4 双轨仲裁。
# 用法: bash scripts/run_phase0.sh
set -e

cd "$(cd "$(dirname "$0")/.." && pwd)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== Phase 0.1: state_sac 500k 候选快照池 ==="
bash scripts/alg/state_sac_indep.sh 0

# --- P0.2 决策场验收（harness 完成后启用） ---
# TODO: Phase 0 评估 harness 待实现（不阻塞后续实验）
# python3 -m mtrl.experiment.phase0.eval_decision_field \
#     --run_dir /root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0 \
#     --out_dir  /root/rivermind-data/lost+found/gbi/experiments/results/phase0/p02

# --- P0.3 世界模型验收（奖励头偏置表，harness 完成后启用） ---
# TODO: Phase 0 评估 harness 待实现（不阻塞后续实验）
# python3 -m mtrl.experiment.phase0.eval_world_model \
#     --run_dir ... --out_dir /root/rivermind-data/lost+found/gbi/experiments/results/phase0/p03

# --- P0.4 双轨仲裁验收（harness 完成后启用） ---
# TODO: Phase 0 评估 harness 待实现（不阻塞后续实验）
# python3 -m mtrl.experiment.phase0.eval_arbiter_agreement \
#     --run_dir ... --out_dir /root/rivermind-data/lost+found/gbi/experiments/results/phase0/p04

echo "[phase0] P0.1 done. P0.2-P0.4 harness 未启用（见注释）。"
