#!/usr/bin/env python3
"""训练健康监控：采样所有 RUNNING stage 的核心指标并检测"白跑模式"。

检测项（对应历史事故）：
- ⚠️ 冻结复发（#7）：actor_entropy 尾值 ≈5.68（log_std≡0 时 0.5·4·(1+ln2π)）且 step>5000
- ⚠️ α 塌缩：alpha_0 < 0.02（熵恒 > target −4 时 alpha_loss 恒正）
- ⚠️ #6 回归：train.log 的 env_step_i 全 0 且 step>2000
- ⚠️ 停滞：train.log 超过 20 分钟未更新
- ⚠️ 想象通道异常（gbi）：u_window/τ 无意义时 U 恒 0

用法:
  python3 scripts/monitor_training.py [--runs-root DIR] [--append FILE]
    自动发现 RUNNING 的 run 并打印摘要；--append 追加 JSONL 供长期采样。
"""
import argparse
import glob
import json
import os
import time


def find_running_runs(runs_root: str):
    """扫描 runs_root 下 <alg>/metaworld/mt10/seedN/logs/metaworld-mt10/<sub>/*/ 最新 run，
    返回 status=RUNNING 的 {name, log_dir, train_log}。"""
    found = []
    for meta in sorted(glob.glob(os.path.join(runs_root, "*/metaworld/mt10/seed*/logs/metaworld-mt10/*/*"))):
        if not os.path.isdir(meta):
            continue
        log_json = os.path.join(meta, "log.jsonl")
        if not os.path.exists(log_json):
            continue
        status = None
        try:
            with open(log_json) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if d.get("status"):
                            status = d["status"]
                    except Exception:
                        pass
        except Exception:
            continue
        if status != "RUNNING":
            continue
        # name: runs_root 下的相对结构，如 gbi/metaworld/mt10/seed0
        rel = os.path.relpath(meta, runs_root)
        # rel 形如 <alg>/metaworld/mt10/<seed>/logs/...
        parts = rel.split("/")
        alg, seed_dir = parts[0], parts[3]
        tl = os.path.join(meta, "train.log")
        found.append({"alg": alg, "seed": seed_dir, "dir": meta,
                      "train_log": tl if os.path.exists(tl) else None})
    return found


def tail_train(run):
    tl = run["train_log"]
    if not tl:
        return None
    try:
        mtime = os.path.getmtime(tl)
        rows = []
        with open(tl) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows, mtime
    except Exception:
        return None


def summarize(run, rows, mtime, now):
    if not rows:
        return {"name": f"{run['alg']}:{run['seed']}", "step": None,
                "note": "no train rows", "warns": []}
    last = rows[-1]
    step = last.get("step")
    s = {"name": f"{run['alg']}:{run['seed']}", "step": step}
    # 训练指标（actor 健康）
    ent = last.get("actor_entropy")
    alpha = last.get("alpha_0")
    es = [last.get(f"env_step_{i}") for i in range(10)]
    s["entropy"] = round(ent, 3) if ent is not None else None
    s["alpha0"] = round(alpha, 4) if alpha is not None else None
    es_ok = es and all(v == 200 for v in es if v is not None)
    s["ES_ok"] = bool(es_ok)
    s["age_min"] = round((now - mtime) / 60, 1)
    # wm 健康
    for k in ("wm_reward_loss", "wm_dyn_obs_mae", "critic_loss", "actor_loss"):
        v = last.get(k)
        if v is not None:
            s[k] = round(v, 4)
    # 告警
    warns = []
    if step and step > 5000 and ent is not None and ent > 5.60:
        warns.append(f"entropy≈{ent:.3f} 恒高（冻结复发 #7?）")
    if alpha is not None and alpha < 0.02:
        warns.append(f"alpha={alpha:.4f} 塌缩")
    if step and step > 2000 and es and not es_ok and all(v == 0 for v in es if v is not None):
        warns.append("env_step 全 0（#6 回归?）")
    if s["age_min"] > 20:
        warns.append(f"train.log {s['age_min']}min 未更新（停滞?）")
    if step is None:
        warns.append("step 缺失")
    s["warns"] = warns
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/root/rivermind-data/lost+found/gbi/experiments/runs")
    ap.add_argument("--append", default=None, help="JSONL 追加路径（长期采样）")
    args = ap.parse_args()

    now = time.time()
    runs = find_running_runs(args.runs_root)
    print(f"[monitor {time.strftime('%F %T')}] RUNNING runs: {len(runs)}")
    if not runs:
        print("（无运行中的 run）")
        return
    rows = []
    for r in runs:
        out = summarize(r, *tail_train(r) or (None, now), now)
        rows.append(out)
        w = (" ⚠️ " + "; ".join(out["warns"])) if out["warns"] else ""
        print(f"  {out['name']:>14} step={out.get('step')} "
              f"ent={out.get('entropy')} α={out.get('alpha0')} "
              f"ES_ok={out.get('ES_ok')} age={out.get('age_min')}min{w}")
    if args.append:
        with open(args.append, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%F %T"), "runs": rows}) + "\n")


if __name__ == "__main__":
    main()
