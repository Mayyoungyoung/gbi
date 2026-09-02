#!/usr/bin/env bash
# smoke_fix.sh — 2026-09-02 三处修复（#7 freeze / #6 infos / #8 twohot）的端到端冒烟
# 3000 步（init_steps=1500，后 1500 步有 update），save_freq=1500 保存 2 个 ckpt。
# 验收（对照高危 #7 冻结铁证）：
#   1) train.log entropy 应显著低于 5.68（冻结时 1500 次 update 后仍恒 5.680；
#      state_sac 同期 1500 次 update 已降至 ~5.32）
#   2) actor_1500.pt 与 actor_3000.pt 参数必须有差异（冻结时完全不变）
#   3) train.log env_step_*/success 解析路径生效（#6）
set -e
cd /root/rivermind-data/lost+found/CTPG-main/CTPG-main
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

exec python3 -u main.py \
  setup.alg=qmp "setup.id=smoke" "setup.seed=0" \
  "setup.base_path=/root/rivermind-data/lost+found/gbi/experiments/runs/smoke_fix/metaworld/mt10/seed0" \
  env=metaworld-mt10 agent=gbi_sac metrics=mtrl_gbi \
  experiment.name=metaworld \
  experiment.num_train_steps=3000 \
  experiment.eval_freq=3000 \
  experiment.num_eval_episodes=5 \
  replay_buffer.capacity=100000 replay_buffer.batch_size=1024 \
  agent.encoder.type_to_select=identity \
  agent.multitask.should_use_disentangled_alpha=True \
  agent.multitask.should_use_task_encoder=False \
  agent.multitask.should_use_task_onehot=False \
  agent.multitask.should_use_multi_head_policy=True \
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
  "agent.gbi.candidates_dir=/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_issue_57d4f43a4bf96214e8f7be83a0564528bff75e77_seed_0/model" \
  "agent.gbi.candidate_steps=[500000]" \
  agent.gbi.arbiter_mode=qmp agent.gbi.trigger_mode=always \
  experiment.save.model.should_save=True \
  experiment.save_freq=1500 \
  experiment.save.model.retain_last_n=2
