#!/usr/bin/env bash
# smoke_guide.sh — CTPG（guide_sac）基线路径首次环境验证（2026-09-02）
# 验收：不崩；train.log entropy 在动（guide 主 actor 也在学习）；
#       ES_i=200（#6 修复对 guide 生效）；update_no_guide 的 success 门控恢复
set -e
cd /root/rivermind-data/lost+found/CTPG-main/CTPG-main
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

exec python3 -u main.py \
  setup.alg=guide_mhsac "setup.id=smoke" "setup.seed=0" \
  "setup.base_path=/root/rivermind-data/lost+found/gbi/experiments/runs/smoke_guide/metaworld/mt10/seed0" \
  env=metaworld-mt10 agent=guide_sac metrics=mtrl_guide experiment.name=metaworld \
  experiment.num_train_steps=3000 experiment.eval_freq=2000 experiment.num_eval_episodes=5 \
  experiment.use_guide=True \
  replay_buffer.capacity=100000 replay_buffer.batch_size=1024 \
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
  experiment.save.model.should_save=False
