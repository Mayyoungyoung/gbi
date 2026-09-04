#!/usr/bin/env bash
# bench60k_aux.sh — 与 run_cross_benchmark.sh 并列的辅助臂（indep / guide），
# 参数镜像 scripts/alg/state_sac_indep.sh 与 scripts/alg/guide_mhsac.sh，
# 训练预算 60k（正式 300k 的 20%，新机缩短预算的关键流程验证，见实验声明）。
# 用法: bash gbi_out/setup/bench60k_aux.sh <indep|guide>
set -e
REPO=/root/rivermind-data/lost+found/gbi
RUNS_ROOT=/root/rivermind-data/lost+found/gbi_out/runs
ARM="$1"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$REPO"

COMMON=(
  agent.encoder.type_to_select=identity
  agent.multitask.should_use_disentangled_alpha=True
  agent.multitask.should_use_task_encoder=False
  agent.multitask.should_use_task_onehot=False
  agent.multitask.should_use_multi_head_policy=True
  agent.multitask.actor_cfg.should_condition_model_on_task_info=False
  agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False
  agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False
)

case "$ARM" in
  indep)
    RUN_DIR="$RUNS_ROOT/state_sac_indep/mt10/seed0"
    cmd=(python3 -u main.py
      setup.alg=state_sac "setup.id=seed0" "setup.seed=0" "setup.base_path=$RUN_DIR"
      env=metaworld-mt10 agent=state_sac metrics=mtrl experiment.name=metaworld
      experiment.num_train_steps=60100 experiment.eval_freq=10000 experiment.save_freq=20000
      experiment.save.model.should_save=True experiment.save.model.retain_last_n=-1
      experiment.num_eval_episodes=10
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1280
      "${COMMON[@]}")
    ;;
  guide)
    RUN_DIR="$RUNS_ROOT/guide/mt10/seed0"
    cmd=(python3 -u main.py
      setup.alg=guide_mhsac "setup.id=seed0" "setup.seed=0" "setup.base_path=$RUN_DIR"
      env=metaworld-mt10 agent=guide_sac metrics=mtrl_guide experiment.name=metaworld
      experiment.num_train_steps=60100 experiment.eval_freq=10000 experiment.num_eval_episodes=10
      experiment.use_guide=True
      experiment.save.model.should_save=True experiment.save_freq=20000 experiment.save.model.retain_last_n=5
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1280
      agent.guide_encoder.type_to_select=identity
      agent.builder.guide_hindsight=True
      "${COMMON[@]}")
    ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

mkdir -p "$RUN_DIR"
echo "[bench60k:$ARM] launching: ${cmd[*]}"
exec "${cmd[@]}"
