#!/usr/bin/env bash
# run_gbi_then_qmp.sh — GbI 完成后自动衔接 QMP 对照实验（v2）。
# v2 修复（2026-09-01）：
#   1) 自动发现最新 GbI run 目录，不再硬编码路径
#   2) 完成检测用 log.jsonl 的 {"status": "COMPLETED"}（run.py 自动写），
#      不再依赖不存在的 "[gbi] done" 打印
#   3) QMP 启动去掉 `| tee` 管道，stdout 直接重定向文件（防断连死锁）
# 用法:
#   nohup bash scripts/run_gbi_then_qmp.sh > runner.log 2>&1 &
#   CONTINUE_ON_ABNORMAL_EXIT=0 可禁止 GbI 异常退出时继续 QMP（默认继续）
set -e

GBI_BASE="/root/rivermind-data/lost+found/gbi/experiments/runs/gbi/metaworld/mt10/seed0"
QMP_BASE="/root/rivermind-data/lost+found/gbi/experiments/runs/qmp/metaworld/mt10/seed0"
CANDIDATES_DIR="/root/rivermind-data/lost+found/gbi/experiments/runs/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_issue_57d4f43a4bf96214e8f7be83a0564528bff75e77_seed_0/model"

echo "[sequential_runner] waiting for GbI completion (polling log.jsonl status=COMPLETED)..."

GBI_LOG_DIR=""
while true; do
    # 自动发现最新 GbI run 目录（hydra 按启动时间命名）
    latest=$(ls -td "$GBI_BASE"/logs/metaworld-mt10/gbi_sac/*/ 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        GBI_LOG_DIR="${latest%/}"
        if grep -q '"status": "COMPLETED"' "$GBI_LOG_DIR/log.jsonl" 2>/dev/null; then
            echo "[sequential_runner] GbI completed: $GBI_LOG_DIR"
            break
        fi
    fi
    # GbI 进程消失但无 COMPLETED 标记 → 异常退出
    if ! pgrep -f "main.py.*setup.alg=gbi_sac" > /dev/null 2>&1; then
        if [ -n "$GBI_LOG_DIR" ] && grep -q '"status": "COMPLETED"' "$GBI_LOG_DIR/log.jsonl" 2>/dev/null; then
            echo "[sequential_runner] GbI completed: $GBI_LOG_DIR"
            break
        fi
        if [ "${CONTINUE_ON_ABNORMAL_EXIT:-1}" != "1" ]; then
            echo "[sequential_runner] ERROR: GbI exited without COMPLETED; abort."
            exit 1
        fi
        echo "[sequential_runner] WARNING: GbI exited without COMPLETED; continuing to QMP."
        break
    fi
    sleep 60
done

echo "[sequential_runner] Starting QMP experiment..."
cd /root/rivermind-data/lost+found/CTPG-main/CTPG-main
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$QMP_BASE"
QMP_STDOUT="$QMP_BASE/qmp_stdout.log"

qmp_cmd=(
  python3 -u main.py
  setup.alg=qmp
  "setup.id=seed0"
  "setup.seed=0"
  "setup.base_path=$QMP_BASE"
  env=metaworld-mt10
  agent=gbi_sac
  metrics=mtrl_gbi
  experiment.name=metaworld
  experiment.num_train_steps=300000
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
  agent.gbi.arbiter_mode=qmp
  agent.gbi.trigger_mode=always
  "agent.gbi.candidates_dir=$CANDIDATES_DIR"
  "agent.gbi.candidate_steps=[50000,100000,150000,200000,250000,300000,350000,400000,450000,500000]"
  # ---- checkpoint 留档（H2 修复，与 GbI 主实验同参数）----
  experiment.save.model.should_save=True
  experiment.save_freq=20000
  experiment.save.model.retain_last_n=5
)

# 无管道无 tee：stdout 直接写文件，防断连死锁
PYTHONPATH=. "${qmp_cmd[@]}" > "$QMP_STDOUT" 2>&1
echo "[sequential_runner] QMP done (exit=$?)"
