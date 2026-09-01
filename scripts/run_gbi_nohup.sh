#!/usr/bin/env bash
# run_gbi_nohup.sh — GbI 主实验启动脚本（nohup 后台启动，防 SSH 断连死锁）
#
# 背景：之前用 `2>&1 | tee` 前台启动，SSH 会话断开后 pty 写阻塞 → 管道级联阻塞
#       → 训练停死（2026-08-31 事故，GbI 卡在 step 60k/300k 共 8 小时）。
#       本脚本不依赖任何终端/管道：stdout/stderr 直接重定向到文件。
#
# 用法:
#   bash scripts/run_gbi_nohup.sh [nohup|foreground]
#     nohup      : 后台启动（默认），日志 -> $BASE_PATH/gbi_stdout.log，PID -> gbi.pid
#     foreground : 前台启动（调试用），输出到当前终端
#
# 完成检测（供 run_gbi_then_qmp.sh 衔接）:
#   run.py 在实验结束时向 <run>/log.jsonl 追加 {"status": "COMPLETED", ...}，
#   衔接脚本轮询该标记即可，无需依赖进程或终端。
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 23 核 cgroup 配额限制: spawn worker 未受限时 torch/OpenBLAS 每进程开 192 线程,
# 20 个 worker 会线程空转导致步速坍塌 (~4 it/s)。单线程化 env worker 后恢复正常。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MODE="${1:-nohup}"

# ---- 实验参数（与 2026-08-31-14-02-10 GbI 主实验一致） ----
BASE_PATH="/root/rivermind-data/lost+found/gbi/experiments/runs/gbi/metaworld/mt10/seed0"
CANDIDATES_DIR="/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_issue_57d4f43a4bf96214e8f7be83a0564528bff75e77_seed_0/model"
CANDIDATE_STEPS="[50000,100000,150000,200000,250000,300000,350000,400000,450000,500000]"
NUM_TRAIN_STEPS=300000

cmd=(
  python3 -u main.py
  setup.alg=gbi_sac
  "setup.id=seed0"
  "setup.seed=0"
  "setup.base_path=$BASE_PATH"
  env=metaworld-mt10
  agent=gbi_sac
  metrics=mtrl_gbi
  experiment.name=metaworld
  "experiment.num_train_steps=$NUM_TRAIN_STEPS"
  experiment.eval_freq=20000
  experiment.num_eval_episodes=10
  replay_buffer.capacity=1000000
  replay_buffer.batch_size=1024
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
  # ---- checkpoint 留档（H2 修复）：每 20k 步存一次（与 eval_freq 同步，
  #      save 逻辑挂在 step%eval_freq==0 分支内，save_freq 必须是 eval_freq 倍数），
  #      retain_last_n=5 保留最近 5 个；断点续训需另开 buffer 保存 ----
  experiment.save.model.should_save=True
  experiment.save_freq=20000
  experiment.save.model.retain_last_n=5
)

mkdir -p "$BASE_PATH"
STDOUT_LOG="$BASE_PATH/gbi_stdout.log"

if [ "$MODE" == "foreground" ]; then
    echo "[gbi] foreground launch: ${cmd[*]}"
    PYTHONPATH=. "${cmd[@]}"
else
    echo "[gbi] background launch (nohup), stdout -> $STDOUT_LOG"
    # 无管道无 tee：stdout/stderr 直接写文件，SSH 断连不影响训练
    PYTHONPATH=. nohup "${cmd[@]}" > "$STDOUT_LOG" 2>&1 &
    echo $! > "$BASE_PATH/gbi.pid"
    echo "[gbi] PID=$(cat "$BASE_PATH/gbi.pid") (saved to $BASE_PATH/gbi.pid)"
    echo "[gbi] monitor: tail -f $STDOUT_LOG"
    echo "[gbi] run dir will appear under $BASE_PATH/logs/metaworld-mt10/gbi_sac/"
fi
