#!/usr/bin/env bash
# smoke_gbi_online.sh — 在线自举候选池链路验证（2026-09-02）
# 压缩参数：warmup=1500 interval=500 max_count=2 → 3000 步内注册 4 个快照、
# 淘汰 >=2 个（触发 FIFO + 段活跃延迟淘汰路径）
# 验收：stdout 出现 >=4 次 "online candidate registered"、>=1 次 "evicted"；
#       不崩；entropy 在动
set -e
cd /root/rivermind-data/lost+found/CTPG-main/CTPG-main
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

exec python3 -u main.py \
  setup.alg=gbi_sac "setup.id=smoke" "setup.seed=0" \
  "setup.base_path=/root/rivermind-data/lost+found/gbi/experiments/runs/smoke_gbi_online/metaworld/mt10/seed0" \
  env=metaworld-mt10 agent=gbi_sac metrics=mtrl_gbi experiment.name=metaworld \
  experiment.num_train_steps=3000 experiment.eval_freq=2000 experiment.num_eval_episodes=5 \
  replay_buffer.capacity=100000 replay_buffer.batch_size=1024 \
  agent.encoder.type_to_select=identity \
  agent.multitask.should_use_disentangled_alpha=True \
  agent.multitask.should_use_task_encoder=False \
  agent.multitask.should_use_task_onehot=False \
  agent.multitask.should_use_multi_head_policy=True \
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
  agent.gbi.candidates_dir=null \
  agent.gbi.online_candidates.enabled=True \
  agent.gbi.online_candidates.interval=500 \
  agent.gbi.online_candidates.warmup=1500 \
  agent.gbi.online_candidates.max_count=2 \
  agent.gbi.arbiter_mode=gbi agent.gbi.trigger_mode=adaptive \
  experiment.save.model.should_save=False
