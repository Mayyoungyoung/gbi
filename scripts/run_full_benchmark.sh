#!/usr/bin/env bash
# run_full_benchmark.sh — GbI 完整对比实验串行流水线（单 GPU，T4）
#
# 阶段（默认；可用 RUN_STAGES 环境变量覆盖）：
#   gbi:0    GbI 主实验 seed0（300k, gbi/adaptive, 候选快照池, checkpoint 留档）
#   qmp:0    QMP 消融 seed0（300k, λ≡0/每步重判，其余参数与 gbi 完全一致）
#   indep:1  state_sac_indep seed1（500k, 参考基线，快照留存）
#   gbi:1 / qmp:1 / indep:2
#   guide:0 / guide:1   CTPG guide_mhsac 基线（可选对照）
#
# 启动（防 SSH 断连：nohup + 文件重定向，无管道无 tee）：
#   nohup bash scripts/run_full_benchmark.sh \
#     > /root/rivermind-data/lost+found/gbi/experiments/runs/benchmark_runner.log 2>&1 &
#
# 断点续跑：脚本可重复执行，已完成（log.jsonl 含 COMPLETED）的阶段自动跳过。
# 失败策略：任一阶段异常退出或缺少 COMPLETED 标记 → 写 <RUN_DIR>/<tag>_FAILED
#           并中止流水线（异常时不启动后续阶段，避免坏状态级联）。
# 报告：每阶段结束后自动刷新 gbi/experiments/reports/benchmark_report.md。
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$REPO_ROOT"

RUNS_ROOT="/root/rivermind-data/lost+found/gbi/experiments/runs"
REPORT_DIR="$RUNS_ROOT/../reports"
mkdir -p "$REPORT_DIR"

# 候选快照池：state_sac_indep seed0 的 10 个快照（所有 gbi/qmp run 固定同池，
# 保证跨 seed/跨算法的候选集唯一变量）
CAND_DIR="$RUNS_ROOT/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/2026-08-31-09-54-56_issue_57d4f43a4bf96214e8f7be83a0564528bff75e77_seed_0/model"
CAND_STEPS="[50000,100000,150000,200000,250000,300000,350000,400000,450000,500000]"

DEFAULT_STAGES="gbi:0 qmp:0 gbi_online:0 qmp_online:0 indep:1 gbi:1 qmp:1 indep:2 guide:0 guide:1"
STAGES="${RUN_STAGES:-$DEFAULT_STAGES}"

# gbi/qmp 共用的多任务配置（与 2026-08-31-14-02-10 GbI 主实验完全一致）
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

log() { echo "[benchmark][$(date '+%F %T')] $*"; }

# check_completed <tag> <RUN_DIR> <log子目录> —— 输出并返回 0 表示已完成
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

# run_stage <tag> <RUN_DIR> <log子目录> <命令...>
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

refresh_report() {
    python3 scripts/alg/analyze_results.py --out "$REPORT_DIR/benchmark_report.md" \
        >> "$REPORT_DIR/analyzer.log" 2>&1 \
        && log "报告已刷新: $REPORT_DIR/benchmark_report.md" \
        || log "报告生成失败（部分数据缺失属正常），详见 $REPORT_DIR/analyzer.log"
}

stage_gbi() {
    local seed="$1"
    local run_dir="$RUNS_ROOT/gbi/metaworld/mt10/seed$seed"
    local cmd=(python3 -u main.py
      setup.alg=gbi_sac "setup.id=seed$seed" "setup.seed=$seed" "setup.base_path=$run_dir"
      env=metaworld-mt10 agent=gbi_sac metrics=mtrl_gbi experiment.name=metaworld
      experiment.num_train_steps=300000 experiment.eval_freq=20000 experiment.num_eval_episodes=10
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1024
      "${COMMON[@]}"
      "agent.gbi.candidates_dir=$CAND_DIR" "agent.gbi.candidate_steps=$CAND_STEPS"
      agent.gbi.arbiter_mode=gbi agent.gbi.trigger_mode=adaptive
      experiment.save.model.should_save=True experiment.save_freq=20000
      experiment.save.model.retain_last_n=5)
    run_stage "gbi:$seed" "$run_dir" "metaworld-mt10/gbi_sac" "${cmd[@]}"
}

stage_qmp() {
    local seed="$1"
    local run_dir="$RUNS_ROOT/qmp/metaworld/mt10/seed$seed"
    local cmd=(python3 -u main.py
      setup.alg=qmp "setup.id=seed$seed" "setup.seed=$seed" "setup.base_path=$run_dir"
      env=metaworld-mt10 agent=gbi_sac metrics=mtrl_gbi experiment.name=metaworld
      experiment.num_train_steps=300000 experiment.eval_freq=20000 experiment.num_eval_episodes=10
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1024
      "${COMMON[@]}"
      "agent.gbi.candidates_dir=$CAND_DIR" "agent.gbi.candidate_steps=$CAND_STEPS"
      agent.gbi.arbiter_mode=qmp agent.gbi.trigger_mode=always
      experiment.save.model.should_save=True experiment.save_freq=20000
      experiment.save.model.retain_last_n=5)
    run_stage "qmp:$seed" "$run_dir" "metaworld-mt10/qmp" "${cmd[@]}"
}

# 在线自举变体（2026-09-02）：像 CTPG 一样免预训练直接开训——
# 候选池不再用磁盘快照，而是每 25k 步在线 snapshot 现役 actor（FIFO 保留 4 个）。
# gbi_online = 完整裁决；qmp_online = 其 λ≡0 消融（隔离"在线候选池"与"想象增益"两个变量）
stage_gbi_online() {
    local seed="$1"
    local run_dir="$RUNS_ROOT/gbi_online/metaworld/mt10/seed$seed"
    local cmd=(python3 -u main.py
      setup.alg=gbi_sac "setup.id=seed$seed" "setup.seed=$seed" "setup.base_path=$run_dir"
      env=metaworld-mt10 agent=gbi_sac metrics=mtrl_gbi experiment.name=metaworld
      experiment.num_train_steps=300000 experiment.eval_freq=20000 experiment.num_eval_episodes=10
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1024
      "${COMMON[@]}"
      agent.gbi.candidates_dir=null
      agent.gbi.online_candidates.enabled=True
      agent.gbi.arbiter_mode=gbi agent.gbi.trigger_mode=adaptive
      experiment.save.model.should_save=True experiment.save_freq=20000
      experiment.save.model.retain_last_n=5)
    run_stage "gbi_online:$seed" "$run_dir" "metaworld-mt10/gbi_sac" "${cmd[@]}"
}

stage_qmp_online() {
    local seed="$1"
    local run_dir="$RUNS_ROOT/qmp_online/metaworld/mt10/seed$seed"
    local cmd=(python3 -u main.py
      setup.alg=qmp_online "setup.id=seed$seed" "setup.seed=$seed" "setup.base_path=$run_dir"
      env=metaworld-mt10 agent=gbi_sac metrics=mtrl_gbi experiment.name=metaworld
      experiment.num_train_steps=300000 experiment.eval_freq=20000 experiment.num_eval_episodes=10
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1024
      "${COMMON[@]}"
      agent.gbi.candidates_dir=null
      agent.gbi.online_candidates.enabled=True
      agent.gbi.arbiter_mode=qmp agent.gbi.trigger_mode=always
      experiment.save.model.should_save=True experiment.save_freq=20000
      experiment.save.model.retain_last_n=5)
    run_stage "qmp_online:$seed" "$run_dir" "metaworld-mt10/qmp_online" "${cmd[@]}"
}

stage_indep() {
    local seed="$1"
    local run_dir="$RUNS_ROOT/state_sac_indep/mt10/seed$seed"
    local cmd=(python3 -u main.py
      setup.alg=state_sac "setup.id=seed$seed" "setup.seed=$seed" "setup.base_path=$run_dir"
      env=metaworld-mt10 agent=state_sac metrics=mtrl experiment.name=metaworld
      experiment.num_train_steps=500100 experiment.eval_freq=50000 experiment.num_eval_episodes=10
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1280
      "${COMMON[@]}"
      experiment.save.model.should_save=True experiment.save_freq=50000
      experiment.save.model.retain_last_n=-1)
    run_stage "indep:$seed" "$run_dir" "metaworld-mt10/state_sac" "${cmd[@]}"
}

stage_guide() {
    local seed="$1"
    local run_dir="$RUNS_ROOT/guide/mt10/seed$seed"
    local cmd=(python3 -u main.py
      setup.alg=guide_mhsac "setup.id=seed$seed" "setup.seed=$seed" "setup.base_path=$run_dir"
      env=metaworld-mt10 agent=guide_sac metrics=mtrl_guide experiment.name=metaworld
      experiment.num_train_steps=300000 experiment.eval_freq=20000 experiment.num_eval_episodes=10
      experiment.use_guide=True
      replay_buffer.capacity=1000000 replay_buffer.batch_size=1024
      "${COMMON[@]}"
      agent.guide_encoder.type_to_select=identity
      agent.builder.guide_hindsight=True
      experiment.save.model.should_save=True experiment.save_freq=20000
      experiment.save.model.retain_last_n=5)
    run_stage "guide:$seed" "$run_dir" "metaworld-mt10/guide_mhsac" "${cmd[@]}"
}

log "pipeline 启动；阶段列表: $STAGES"
log "候选快照池: $CAND_DIR"

# 单 stage 入口（并行调度用）：bash run_full_benchmark.sh --one gbi:0
run_stage_entry() {
    local entry="$1"
    local alg="${entry%%:*}"
    local seed="${entry##*:}"
    case "$alg" in
        gbi)         stage_gbi "$seed" ;;
        qmp)         stage_qmp "$seed" ;;
        gbi_online)  stage_gbi_online "$seed" ;;
        qmp_online)  stage_qmp_online "$seed" ;;
        indep)       stage_indep "$seed" ;;
        guide)       stage_guide "$seed" ;;
        *) log "未知阶段 $entry，中止"; return 2 ;;
    esac
}

if [ "$1" = "--one" ]; then
    run_stage_entry "$2"
    exit $?
fi

if [ "${PARALLEL:-0}" = "1" ]; then
    # 并行模式（2026-09-02）：10 个 stage 无相互依赖（候选池固定为 seed0 快照），
    # 实测单进程显存 <1GB（qmp）、GPU 利用率 ~40%（瓶颈在 CPU 环境交互），
    # 单卡 T4 15GB + 96 核 CPU 可安全并行；MAX_PARALLEL 控制并发（建议 4-6）
    MAX_PARALLEL="${MAX_PARALLEL:-4}"
    log "并行模式启动（并发=$MAX_PARALLEL）: $STAGES"
    echo "$STAGES" | tr ' ' '\n' | grep -v '^$' | \
        xargs -P "$MAX_PARALLEL" -I{} bash "$0" --one {}
    rc=$?
    if [ $rc -ne 0 ]; then
        log "并行流水线存在失败 stage（rc=$rc），检查各 run 目录的 *_FAILED 标记"
    fi
    refresh_report
    log "=== 全部阶段完成。最终报告: $REPORT_DIR/benchmark_report.md ==="
    exit $rc
fi

for entry in $STAGES; do
    run_stage_entry "$entry"
    if [ $? -ne 0 ]; then
        log "流水线在阶段 $entry 中止。修复后重新执行本脚本可断点续跑。"
        exit 1
    fi
    refresh_report
done

log "=== 全部阶段完成。最终报告: $REPORT_DIR/benchmark_report.md ==="
