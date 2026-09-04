#!/usr/bin/env bash
# orchestrator60k.sh — 正式关键流程实验（缩短预算 60k，声明见实验说明）
# 顺序：等冒烟完成 → 清冒烟残留（Migration §6 必做）→ 3 cross 臂并行 + indep/guide 并行
set -u
REPO=/root/rivermind-data/lost+found/gbi
RUNS_ROOT=/root/rivermind-data/lost+found/gbi_out/runs
cd "$REPO"

echo "[orch] $(date '+%F %T') waiting for smoke to finish..."
# 冒烟完成标志：run_cross_benchmark.sh 末尾打印「=== 全部阶段完成 ===」；
# 若冒烟失败（出现 FAILED 标记）则中止编排。
while true; do
    if grep -q '全部阶段完成' "$RUNS_ROOT/smoke_runner.log" 2>/dev/null; then
        break
    fi
    if grep -q 'FAILED' "$RUNS_ROOT/smoke_runner.log" 2>/dev/null && \
       ! pgrep -f "scripts/run_cross_benchmark.sh" >/dev/null 2>&1; then
        echo "[orch] smoke failed, aborting"
        exit 1
    fi
    sleep 30
done
echo "[orch] $(date '+%F %T') smoke done. clearing smoke residues..."
rm -rf "$RUNS_ROOT/gbi_cross" "$RUNS_ROOT/qmp_cross" "$RUNS_ROOT/gbi_cross_fast"

export PATH=/opt/conda/envs/gbi/bin:$PATH
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

echo "[orch] $(date '+%F %T') launching cross arms (3-way parallel, 60k steps)..."
NUM_STEPS=60100 EVAL_FREQ=10000 SAVE_FREQ=20000 PARALLEL=1 MAX_PARALLEL=3 \
  RUNS_ROOT="$RUNS_ROOT" FORCE=1 \
  nohup bash scripts/run_cross_benchmark.sh > "$RUNS_ROOT/cross_runner.log" 2>&1 &

sleep 5
echo "[orch] $(date '+%F %T') launching indep arm (60k)..."
nohup bash "$RUNS_ROOT/../setup/bench60k_aux.sh" indep > "$RUNS_ROOT/indep_stdout.log" 2>&1 &

sleep 5
echo "[orch] $(date '+%F %T') launching guide arm (60k)..."
nohup bash "$RUNS_ROOT/../setup/bench60k_aux.sh" guide > "$RUNS_ROOT/guide_stdout.log" 2>&1 &

echo "[orch] $(date '+%F %T') all launched. polling until every arm has COMPLETED..."
while true; do
    sleep 60
    done_cnt=0
    total=5
    for tag in gbi_cross qmp_cross gbi_cross_fast; do
        run=$(ls -td "$RUNS_ROOT/$tag"/metaworld/mt10/seed0/logs/metaworld-mt10/*/ 2>/dev/null | head -1)
        [ -n "$run" ] && grep -q '"status": "COMPLETED"' "$run/log.jsonl" 2>/dev/null && done_cnt=$((done_cnt+1))
    done
    for tag in state_sac_indep guide; do
        run=$(ls -td "$RUNS_ROOT/$tag"/mt10/seed0/logs/metaworld-mt10/*/ 2>/dev/null | head -1)
        [ -n "$run" ] && grep -q '"status": "COMPLETED"' "$run/log.jsonl" 2>/dev/null && done_cnt=$((done_cnt+1))
    done
    echo "[orch] $(date '+%F %T') completed $done_cnt/$total"
    if [ "$done_cnt" -ge "$total" ]; then
        echo "[orch] $(date '+%F %T') ALL ARMS COMPLETED" 
        break
    fi
done
echo "[orch] done"
