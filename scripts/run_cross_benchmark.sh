#!/usr/bin/env bash
# run_cross_benchmark.sh — Phase A 跨任务候选池实验（P0.1，2026-09-03）
#
# 前提：GbI_Improvement_Proposals.md §4 P0.1（候选池 = 现役 actor 的 N 个任务
# head，与 CTPG 同池语义）。本脚本与 run_full_benchmark.sh 完全独立——
# 不改动后者（其正被运行中的批次解释执行，就地编辑有破坏运行中 bash 的风险）。
#
# 阶段（RUN_STAGES 可覆盖，默认 seed0 三臂）：
#   gbi_cross:0       gbi/adaptive/cross（主臂，协议对齐旧 gbi:0）
#   qmp_cross:0       qmp/always/cross（λ≡0 消融，协议对齐旧 qmp:0）
#   gbi_cross_fast:0  gbi/always/cross（P0.2 触发解耦：与 qmp_cross 同触发频率，
#                     唯一变量 = λ_t；与 gbi_cross 比，唯一变量 = 触发模式）
#   （qmp+adaptive 组合无意义：qmp 跳过想象 → U 恒 0 → adaptive 永不触发）
#
# 启动（先确认现有批次已结束——脚本自带运行中训练检测，FORCE=1 可越过）：
#   nohup bash scripts/run_cross_benchmark.sh \
#     > /root/rivermind-data/lost+found/gbi/experiments/runs/cross_runner.log 2>&1 &
# 并行：PARALLEL=1 MAX_PARALLEL=3 nohup bash scripts/run_cross_benchmark.sh ... &
# 冒烟（3000 步快速验收：候选池构建日志/λ_t/κ_t/换手率）：SMOKE=1 bash scripts/run_cross_benchmark.sh
# 断点续跑：重复执行，已完成（log.jsonl 含 COMPLETED）的阶段自动跳过。
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$REPO_ROOT"

RUNS_ROOT="/root/rivermind-data/lost+found/gbi/experiments/runs"
REPORT_DIR="$RUNS_ROOT/../reports"
mkdir -p "$REPORT_DIR"

DEFAULT_STAGES="gbi_cross:0 qmp_cross:0 gbi_cross_fast:0"
STAGES="${RUN_STAGES:-$DEFAULT_STAGES}"

if [ "${SMOKE:-0}" = "1" ]; then
    NUM_STEPS=3000 EVAL_FREQ=1000 SAVE_FREQ=3000
else
    NUM_STEPS=300000 EVAL_FREQ=20000 SAVE_FREQ=20000
fi

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

log() { echo "[cross-benchmark][$(date '+%F %T')] $*"; }

check_completed() {
    local tag="$1" run_dir="$2" sub="$3"
    local run="$(ls -td "$run_dir"/logs/$sub/*/ 2>/dev/null | head -1)"
    if [ -z "$run" ]; then
        echo "[$tag] 无 run 目录，待运行: $run_dir/logs/$sub/"
        return 1
    fi
    if grep -q '"status": "COMPLETED"' "$run/log.jsonl" 2>/dev/null; then
        echo "[$tag] 已完成，跳过: $run"
        return 0
    fi
    echo "[$tag] 存在未完成 run（残留/中断），将重新运行: $run"
    return 1
}

run_stage() {
    local tag="$1" run_dir="$2" sub="$3"; shift 3
    local fail_mark="$run_dir/${tag}_FAILED"
    if check_completed "$tag" "$run_dir" "$sub"; then return 0; fi
    rm -f "$fail_mark"
    mkdir -p "$run_dir"
    log "=== stage $tag START（stdout → $run_dir/${tag}_stdout.log） ==="
    "$@" >> "$run_dir/${tag}_stdout.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "!!! stage $tag 退出码 rc=$rc → 标记 FAILED 并中止流水线"
        touch "$fail_mark"
        return 1
    fi
    if ! check_completed "$tag" "$run_dir" "$sub"; then
        log "!!! stage $tag 缺少 COMPLETED 标记 → 标记 FAILED 并中止流水线"
        touch "$fail_mark"
        return 1
    fi
    log "=== stage $tag DONE ==="
    return 0
}

# run_cross_stage <alg标签> <seed> <arbiter_mode> <trigger_mode> <log子目录>
run_cross_stage() {
    local alg="$1" seed="$2" mode="$3" trig="$4" sub="$5"
    local run_dir="$RUNS_ROOT/$alg/metaworld/mt10/seed$seed"
    local cmd=(python3 -u main.py
      "setup.alg=$alg" "setup.id=seed$seed" "setup.seed=$seed" "setup.base_path=$run_dir"
      env=metaworld-mt10 agent=gbi_sac metrics=mtrl_gbi experiment.name=metaworld
      "experiment.num_train_steps=$NUM_STEPS" "experiment.eval_freq=$EVAL_FREQ" experiment.num_eval_episodes=10
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1024
      "${COMMON[@]}"
      agent.gbi.candidates_dir=null
      agent.gbi.online_candidates.enabled=False
      agent.gbi.cross_task_candidates.enabled=True
      "agent.gbi.arbiter_mode=$mode" "agent.gbi.trigger_mode=$trig"
      experiment.save.model.should_save=True "experiment.save_freq=$SAVE_FREQ"
      experiment.save.model.retain_last_n=5)
    run_stage "${alg}:${seed}" "$run_dir" "metaworld-mt10/$sub" "${cmd[@]}"
}

run_stage_entry() {
    local entry="$1"
    local alg="${entry%%:*}"
    local seed="${entry##*:}"
    case "$alg" in
        gbi_cross)       run_cross_stage gbi_cross "$seed" gbi adaptive gbi_cross ;;
        qmp_cross)       run_cross_stage qmp_cross "$seed" qmp always qmp_cross ;;
        gbi_cross_fast)  run_cross_stage gbi_cross_fast "$seed" gbi always gbi_cross_fast ;;
        *) log "未知阶段 $entry，中止"; return 2 ;;
    esac
}

if [ "${1:-}" = "--one" ]; then
    run_stage_entry "${2:-}"
    exit $?
fi

# 安全护栏：现有批次仍在跑时拒绝启动（FORCE=1 越过；--one 子进程不做检测）
if [ "${FORCE:-0}" != "1" ]; then
    if pgrep -f "python3 -u main.py setup.alg" > /dev/null 2>&1; then
        log "检测到训练进程仍在运行（现有批次未结束），拒绝启动。"
        log "确认批次结束后重试，或 FORCE=1 强制并行（不建议：T4 已是瓶颈）。"
        exit 1
    fi
fi

log "Phase A 跨任务候选池流水线启动；阶段列表: $STAGES"
log "SMOKE=${SMOKE:-0} NUM_STEPS=$NUM_STEPS EVAL_FREQ=$EVAL_FREQ"

if [ "${PARALLEL:-0}" = "1" ]; then
    MAX_PARALLEL="${MAX_PARALLEL:-3}"
    log "并行模式启动（并发=$MAX_PARALLEL）: $STAGES"
    echo "$STAGES" | tr ' ' '\n' | grep -v '^$' | \
        xargs -P "$MAX_PARALLEL" -I{} bash "$0" --one {}
    rc=$?
    if [ $rc -ne 0 ]; then
        log "并行流水线存在失败 stage（rc=$rc），检查各 run 目录的 *_FAILED 标记"
    fi
    log "=== 全部阶段完成 ==="
    exit $rc
fi

for entry in $STAGES; do
    run_stage_entry "$entry"
    if [ $? -ne 0 ]; then
        log "流水线在阶段 $entry 中止。修复后重新执行本脚本可断点续跑。"
        exit 1
    fi
done
log "=== 全部阶段完成 ==="
