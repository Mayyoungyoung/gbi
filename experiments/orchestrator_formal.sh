#!/usr/bin/env bash
# orchestrator_formal.sh — 正式长时训练编排（标准 scripts/alg 启动脚本 + 标准预算）
#
# 臂参数声明（全部 seed=0、num_eval_episodes=10）：
#   state_sac_indep: 500100 步, eval_freq=50000, save_freq=50000   （脚本默认）
#   guide  (CTPG)   : 300100 步, eval_freq=20000, save_freq=20000
#   gbi             : 300100 步, eval_freq=20000, save_freq=20000,
#                     快照池=[20k,40k,60k]（60k 短训池；500k 标准池待
#                     state_sac_indep 完成后可另跑补充臂，见实验说明）
#   qmp             : 300100 步, 同 gbi；gbi COMPLETED 后自动补位（2 路并发上限）
# 并发：F1 三路（state_sac/guide/gbi），F2 一路（qmp）——不超过 3 路，防步速坍塌。
set -u
REPO=/root/rivermind-data/lost+found/gbi
RUNS_ROOT=/root/rivermind-data/lost+found/gbi_out/runs
POOL_DIR=$(ls -td "$RUNS_ROOT"/state_sac_indep/mt10/seed0/logs/metaworld-mt10/state_sac/*/model/ 2>/dev/null | head -1)
cd "$REPO"

echo "[formal] $(date '+%F %T') 前置检查"
# 注意：用 main[.]py 字符类写法，避免 pgrep 匹配到启动命令自身的 shell 包装
# （其 cmdline 里含同一字符串时的经典自匹配问题）。
if pgrep -f "python3 -u main[.]py setup[.]alg" >/dev/null 2>&1; then
    echo "[formal] 检测到训练进程仍在运行，中止。"; exit 1
fi
if [ -z "$POOL_DIR" ]; then echo "[formal] 快照池不存在，中止。"; exit 1; fi

# 清冒烟残留（上一批 60k 的 ad-hoc 目录，避免 analyze 混淆）
rm -rf "$RUNS_ROOT/smoke_fix" "$RUNS_ROOT/smoke_fix2" "$RUNS_ROOT/smoke_mini"

export PATH=/opt/conda/envs/gbi/bin:$PATH
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

echo "[formal] $(date '+%F %T') F1: 三路并行启动（state_sac_indep 500k / guide 300k / gbi 300k）"
GBI_RUN_BASE="$RUNS_ROOT/state_sac_indep" nohup bash scripts/alg/state_sac_indep.sh 0 \
    > "$RUNS_ROOT/formal_state_sac_stdout.log" 2>&1 &
echo "  state_sac pid=$!"
sleep 3
GBI_RUN_BASE="$RUNS_ROOT/guide" nohup bash scripts/alg/guide_mhsac.sh metaworld mt10 1000000 1280 300100 0 \
    > "$RUNS_ROOT/formal_guide_stdout.log" 2>&1 &
echo "  guide pid=$!"
sleep 3
GBI_RUN_BASE="$RUNS_ROOT/gbi" GBI_CANDIDATES_DIR="$POOL_DIR" GBI_CANDIDATE_STEPS="[20000,40000,60000]" \
  nohup bash scripts/alg/gbi.sh metaworld mt10 1000000 1280 300100 0 \
    > "$RUNS_ROOT/formal_gbi_stdout.log" 2>&1 &
echo "  gbi pid=$!"

# ---- F2: gbi COMPLETED 后自动接 qmp（轮询 log.jsonl）----
echo "[formal] $(date '+%F %T') F2 watcher: 等 gbi COMPLETED 后启动 qmp"
while true; do
    sleep 300
    grun=$(ls -td "$RUNS_ROOT/gbi/metaworld/mt10/seed0/logs/metaworld-mt10/gbi_sac/"*/ 2>/dev/null | head -1)
    if [ -n "$grun" ] && grep -q '"status": "COMPLETED"' "$grun/log.jsonl" 2>/dev/null; then
        echo "[formal] $(date '+%F %T') gbi COMPLETED → 启动 qmp（同快照池）"
        GBI_RUN_BASE="$RUNS_ROOT/qmp" GBI_CANDIDATES_DIR="$POOL_DIR" GBI_CANDIDATE_STEPS="[20000,40000,60000]" \
          nohup bash scripts/alg/gbi_qmp.sh metaworld mt10 1000000 1280 300100 0 \
            > "$RUNS_ROOT/formal_qmp_stdout.log" 2>&1 &
        echo "[formal] qmp pid=$! 启动完成，watcher 退出"
        exit 0
    fi
    if ! pgrep -f "python3 -u main[.]py setup[.]alg=gbi_sac" >/dev/null 2>&1; then
        # gbi 进程已消失但无 COMPLETED → 异常退出，不自动接 qmp
        echo "[formal] $(date '+%F %T') WARN: gbi 进程消失且无 COMPLETED 标记，watcher 退出（qmp 不启动）"
        exit 2
    fi
done
