#!/usr/bin/env python3
# analyze_results.py — GbI 对比实验报告生成器
#
# 对比对象（数据全部来自各算法 eval.log / train.log）：
#   gbi    GbI 完整版（arbiter gbi/adaptive, 300k, 候选池）
#   qmp    QMP 消融（λ_t≡0 / always 触发, 300k, 同候选池）
#   indep  state_sac_indep（500k, 参考基线）
#   guide  CTPG guide_mhsac（可选对照, 300k）
#
# 用法:
#   python3 scripts/alg/analyze_results.py [--seeds 0 1 2] [--out path/report.md]
# 缺失的数据阶段会标记为 PENDING，不影响其余部分生成。
#
# 输出:
#   - 成功率/回报 学习曲线均值±方差（含 PNG 图）
#   - 各任务独立成功率对照表
#   - GbI vs QMP / GbI vs indep / GbI vs guide 的配对比较
#   - 仲裁指标（TGR/SWR/U/LT/KT/SLM 等）合理性分析
#   - 明确结论：「GbI 是否有效」及依据

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from scipy import stats as sstats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

RUNS_ROOT = "/root/rivermind-data/lost+found/gbi/experiments/runs"

# 各算法 → (runs 子路径模板, log 子目录)
ALGS = {
    "gbi": ("gbi/metaworld/mt10/seed{seed}", "metaworld-mt10/gbi_sac"),
    "qmp": ("qmp/metaworld/mt10/seed{seed}", "metaworld-mt10/qmp"),
    "indep": ("state_sac_indep/mt10/seed{seed}", "metaworld-mt10/state_sac"),
    "guide": ("guide/mt10/seed{seed}", "metaworld-mt10/guide_mhsac"),
    # ---- Phase A 跨任务候选池臂（P0.1，run_cross_benchmark.sh 布局）----
    "gbi_cross": ("gbi_cross/metaworld/mt10/seed{seed}", "metaworld-mt10/gbi_cross"),
    "qmp_cross": ("qmp_cross/metaworld/mt10/seed{seed}", "metaworld-mt10/qmp_cross"),
    "gbi_cross_fast": ("gbi_cross_fast/metaworld/mt10/seed{seed}", "metaworld-mt10/gbi_cross_fast"),
}
ALG_LABEL = {
    "gbi": "GbI 完整版",
    "qmp": "QMP 消融 (λ≡0)",
    "indep": "state_sac_indep",
    "guide": "CTPG guide_mhsac",
    "gbi_cross": "GbI 跨任务池",
    "qmp_cross": "QMP 跨任务池 (λ≡0)",
    "gbi_cross_fast": "GbI 跨任务池+always",
}
ALG_COLOR = {"gbi": "tab:blue", "qmp": "tab:orange", "indep": "tab:green", "guide": "tab:red",
             "gbi_cross": "tab:purple", "qmp_cross": "tab:brown", "gbi_cross_fast": "tab:pink"}

# train.log 中需要提取的仲裁/世界模型指标（key 前缀 + 输出标签）
METRIC_KEYS = [
    ("TGR", "gbi_trigger_rate", "触发率"),
    ("SWR", "gbi_switch_rate", "换手率"),
    ("UEM", "gbi_uscore_ema", "U_score EMA"),
    ("LT", "arbiter/lambda_t", "λ_t 信任权重"),
    ("KT", "arbiter/kappa_t", "Spearman κ_t"),
    ("TON", "arbiter/tau_on", "触发阈值 τ_on"),
    ("TREJ", "arbiter/tau_reject", "拒绝阈值 τ_reject"),
    ("SLM", "gbi_seg_len_mean", "执行段平均长度"),
    ("SGC", "gbi_seg_count", "执行段累计数"),
    ("ALB", "gbi_adapt_loss_before", "TTA 前损失"),
    ("ALA", "gbi_adapt_loss_after", "TTA 后损失"),
    ("WL", "wm_loss", "世界模型总损失"),
    ("WRL", "wm_reward_loss", "奖励头损失"),
    ("DMAE", "wm_dyn_obs_mae", "动力学 MAE"),
    # ---- G2 奖励分箱健康度 + 决策场密度（2026-09-03 新增指标）----
    ("RRK", "wm_reward_rank_corr", "奖励头排序保真度 Spearman"),
    ("RABS", "wm_reward_absmax", "奖励分箱 absmax"),
    ("RBU", "wm_reward_bin_utilization", "分箱利用率"),
    ("REXP", "wm_reward_range_expansions", "分箱扩容次数"),
    ("GAM", "arbiter/gain_abs_median", "|gain| 中位数（决策场密度）"),
    ("GMM", "arbiter/gain_max_median", "每行 max_j gain 中位数"),
    ("GPF", "arbiter/gain_pos_frac", "gain>0 占比"),
    ("QSM", "arbiter/q_spread_median", "Q 锚极差中位数"),
]

NUM_TASKS = 10


def _run_budget(run_dir):
    """从 log.jsonl 的 metadata 中取 experiment.num_train_steps。"""
    path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(path):
        return None
    try:
        for line in open(path):
            d = json.loads(line)
            exp = d.get("experiment", {})
            if exp is not None and exp.get("num_train_steps") is not None:
                return int(exp["num_train_steps"])
    except Exception:
        return None
    return None


def newest_run(run_base, subdir, max_budget=None, min_budget=None):
    """(alg, seed) → 最新 run 目录；不存在返回 None。

    同一布局下可能存在多个批次（如 60k 短测与 300k 正式），目录名最新者
    未必是目标批次——按 log.jsonl 里的训练预算过滤，避免新批次遮蔽旧批次
    （2026-09-03 实测：60k 分析被刚启动的 300k run 遮蔽，indep/guide 的
    eval.log 只剩 step=0 一行）。
    """
    pattern = os.path.join(run_base, "logs", subdir, "*")
    dirs = sorted(glob.glob(pattern), reverse=True)
    for d in dirs:
        budget = _run_budget(d)
        if budget is None:
            continue
        if max_budget is not None and budget > max_budget:
            continue
        if min_budget is not None and budget < min_budget:
            continue
        return d
    return None


def load_eval(run_dir):
    """eval.log → (steps, episode_reward, success, per_task_success, env_step)。"""
    path = os.path.join(run_dir, "eval.log")
    if not os.path.exists(path):
        return None
    steps, reward, success, per_task, env_step = [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" not in d or "episode_reward" not in d:
                continue
            steps.append(int(d["step"]))
            reward.append(float(d["episode_reward"]))
            success.append(float(d["success"]))
            per_task.append([float(d.get(f"success_{i}", np.nan)) for i in range(NUM_TASKS)])
            env_step.append([float(d.get(f"env_step_{i}", np.nan)) for i in range(NUM_TASKS)])
    if not steps:
        return None
    order = np.argsort(steps)
    return {
        "steps": np.asarray(steps)[order],
        "reward": np.asarray(reward)[order],
        "success": np.asarray(success)[order],
        "per_task": np.asarray(per_task)[order],
        "env_step": np.asarray(env_step)[order],
    }


def load_train_metrics(run_dir):
    """train.log → {metric_key: {step: value}}。"""
    path = os.path.join(run_dir, "train.log")
    out = {key: {} for key, _, _ in METRIC_KEYS}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            m = re.search(r'"step":\s*(\d+)', line)
            if not m:
                continue
            step = int(m.group(1))
            for key, full, _ in METRIC_KEYS:
                mm = re.search(r'"%s":\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)' % re.escape(full), line)
                if mm:
                    out[key][step] = float(mm.group(1))
    return out


def final_window(eval_data, n=3):
    """最后 n 个评估点的均值（末期性能）。"""
    return float(np.mean(eval_data["success"][-n:])), float(np.mean(eval_data["reward"][-n:]))


def curve_mean_std(eval_list):
    """多 seed 曲线对齐后的 mean±std。steps 取交集。"""
    if not eval_list:
        return None
    common = set(eval_list[0]["steps"].tolist())
    for e in eval_list[1:]:
        common &= set(e["steps"].tolist())
    common = np.asarray(sorted(common))
    if len(common) == 0:
        return None
    succ = np.asarray([e["success"][np.isin(e["steps"], common)] for e in eval_list])
    rew = np.asarray([e["reward"][np.isin(e["steps"], common)] for e in eval_list])
    return {
        "steps": common,
        "succ_mean": succ.mean(0),
        "succ_std": succ.std(0),
        "rew_mean": rew.mean(0),
        "rew_std": rew.std(0),
        "n_seeds": len(eval_list),
    }


def value_at(eval_data, step):
    idx = np.where(eval_data["steps"] == step)[0]
    return float(eval_data["success"][idx[0]]) if len(idx) else np.nan


def paired_compare(name_a, name_b, evals_a, evals_b, match_step):
    """配对比较（seed 对齐、在 match_step 处比较末期成功率）。"""
    a_vals, b_vals, seeds = [], [], []
    for seed in sorted(set(evals_a) & set(evals_b)):
        va = value_at(evals_a[seed], match_step)
        vb = value_at(evals_b[seed], match_step)
        if not np.isnan(va) and not np.isnan(vb):
            a_vals.append(va)
            b_vals.append(vb)
            seeds.append(seed)
    if not a_vals:
        return None
    a_vals, b_vals = np.asarray(a_vals), np.asarray(b_vals)
    diff = a_vals - b_vals
    p = None
    if HAS_SCIPY and len(a_vals) >= 2:
        _, p = sstats.ttest_rel(a_vals, b_vals)
    auc_a = np.mean([np.mean(evals_a[s]["success"]) for s in seeds])
    auc_b = np.mean([np.mean(evals_b[s]["success"]) for s in seeds])
    return {
        "seeds": seeds,
        "a_mean": a_vals.mean(), "b_mean": b_vals.mean(),
        "diff_mean": diff.mean(), "diff_std": diff.std(ddof=1) if len(diff) > 1 else 0.0,
        "p_value": p, "auc_a": auc_a, "auc_b": auc_b,
    }


def render_curve(ax, curve, label, color):
    if curve is None:
        return
    ax.plot(curve["steps"], curve["succ_mean"], label=label, color=color, lw=2)
    ax.fill_between(
        curve["steps"],
        np.clip(curve["succ_mean"] - curve["succ_std"], 0, 1),
        np.clip(curve["succ_mean"] + curve["succ_std"], 0, 1),
        alpha=0.2, color=color,
    )


def fmt_p(p):
    if p is None:
        return "n/a"
    return f"p={p:.3f}" + (" *" if p < 0.05 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--out", default="/root/rivermind-data/lost+found/gbi/experiments/reports/benchmark_report.md")
    ap.add_argument("--max-budget", type=int, default=None,
                    help="只分析 num_train_steps <= 该值的 run（用于短测批次分析，避免被长批次遮蔽）")
    ap.add_argument("--min-budget", type=int, default=None,
                    help="只分析 num_train_steps >= 该值的 run（用于正式批次分析）")
    args = ap.parse_args()

    lines = []
    add = lines.append
    add("# GbI 对比实验验证报告")
    add("")
    add(f"> 自动生成时间（UTC）：{datetime.utcnow().strftime('%Y-%m-%d %H:%M')}　"
        f"分析脚本：scripts/alg/analyze_results.py")
    add("")

    # ---------- 数据收集 ----------
    eval_data, train_metrics = {}, {}
    for alg, (sub_tpl, log_sub) in ALGS.items():
        eval_data[alg] = {}
        train_metrics[alg] = {}
        for seed in args.seeds:
            run_dir = newest_run(os.path.join(args.root, sub_tpl.format(seed=seed)), log_sub,
                                 max_budget=args.max_budget, min_budget=args.min_budget)
            if run_dir is None:
                continue
            ev = load_eval(run_dir)
            if ev is not None:
                eval_data[alg][seed] = ev
            train_metrics[alg][seed] = load_train_metrics(run_dir)

    curves = {alg: curve_mean_std(list(eval_data[alg].values())) for alg in ALGS}

    # ---------- 总体概况 ----------
    add("## 1. 数据概况")
    add("")
    add("| 算法 | seeds | 末期成功率（均值±std，最后 3 个评估点） | 末期平均回报 |")
    add("|---|---|---|---|")
    final = {}
    for alg in ALGS:
        seeds_done = sorted(eval_data[alg])
        if not seeds_done:
            add(f"| {ALG_LABEL[alg]} | — | PENDING | PENDING |")
            continue
        succs, rews = [], []
        for s in seeds_done:
            sc, rw = final_window(eval_data[alg][s])
            succs.append(sc)
            rews.append(rw)
        final[alg] = (float(np.mean(succs)), float(np.std(succs)))
        add(f"| {ALG_LABEL[alg]} | {seeds_done} | "
            f"{np.mean(succs):.3f} ± {np.std(succs):.3f} | {np.mean(rews):.2f} |")
    add("")

    # ---------- 学习曲线图 ----------
    if HAS_MPL and any(curves.values()):
        report_dir = os.path.dirname(args.out)
        os.makedirs(report_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for alg in ALGS:
            if curves[alg]:
                render_curve(axes[0], curves[alg], ALG_LABEL[alg], ALG_COLOR[alg])
                c = curves[alg]
                axes[1].plot(c["steps"], c["rew_mean"], label=ALG_LABEL[alg], color=ALG_COLOR[alg], lw=2)
                axes[1].fill_between(
                    c["steps"], c["rew_mean"] - c["rew_std"], c["rew_mean"] + c["rew_std"],
                    alpha=0.2, color=ALG_COLOR[alg])
        axes[0].set_xlabel("env steps"); axes[0].set_ylabel("success rate")
        axes[0].set_title("MT10 成功率曲线（均值±std，跨 seed）"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("env steps"); axes[1].set_ylabel("episode reward")
        axes[1].set_title("MT10 平均回报曲线"); axes[1].legend(); axes[1].grid(alpha=0.3)
        png = os.path.join(report_dir, "benchmark_curves.png")
        fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
        add(f"![学习曲线](benchmark_curves.png)")
        add("")

    # ---------- 各任务成功率对照表 ----------
    add("## 2. 各任务独立成功率对照表（末期，均值跨 seed）")
    add("")
    header = "| 任务 | " + " | ".join(ALG_LABEL[a] for a in ALGS) + " | GbI−QMP |"
    add(header)
    add("|" + "---|" * (len(ALGS) + 2))
    task_mean = {alg: None for alg in ALGS}
    for alg in ALGS:
        seeds_done = sorted(eval_data[alg])
        if not seeds_done:
            continue
        mats = np.asarray([eval_data[alg][s]["per_task"][-3:] for s in seeds_done])
        task_mean[alg] = mats.mean(axis=(0, 1))
    # GbI−QMP 列：优先跨任务池配对（Phase A），否则回退 legacy 配对
    diff_pair = ("gbi_cross", "qmp_cross")
    if task_mean["gbi_cross"] is None or task_mean["qmp_cross"] is None:
        diff_pair = ("gbi", "qmp")
    for t in range(NUM_TASKS):
        vals = {a: (f"{task_mean[a][t]:.3f}" if task_mean[a] is not None else "PENDING") for a in ALGS}
        diff = ""
        if task_mean[diff_pair[0]] is not None and task_mean[diff_pair[1]] is not None:
            d = task_mean[diff_pair[0]][t] - task_mean[diff_pair[1]][t]
            diff = f"{d:+.3f}"
        add(f"| task {t} | " + " | ".join(vals[a] for a in ALGS) + f" | {diff} |")
    add("")

    # ---------- 关键对比 ----------
    add("## 3. 关键对比（配对，按 seed 对齐）")
    add("")
    sec_idx = 1
    for name_a, name_b in [("gbi_cross", "qmp_cross"), ("gbi_cross", "indep"),
                            ("gbi_cross", "guide"), ("gbi", "qmp"), ("gbi", "indep"), ("gbi", "guide")]:
        ev_a, ev_b = eval_data[name_a], eval_data[name_b]
        common_seeds = sorted(set(ev_a) & set(ev_b))
        if not common_seeds:
            add(f"### 3.{sec_idx} {ALG_LABEL[name_a]} vs {ALG_LABEL[name_b]} —— PENDING（数据缺失）")
            add("")
            sec_idx += 1
            continue
        # 对齐步数：取两算法共有的评估步（gbI/qmp=20k 间隔；indep=50k 间隔 → 交集）
        steps_a, steps_b = ev_a[common_seeds[0]]["steps"], ev_b[common_seeds[0]]["steps"]
        common_steps = sorted(set(steps_a.tolist()) & set(steps_b.tolist()))
        match_step = max(common_steps)
        res = paired_compare(name_a, name_b, ev_a, ev_b, match_step)
        add(f"### 3.{sec_idx} {ALG_LABEL[name_a]} vs {ALG_LABEL[name_b]}")
        add("")
        if res is None:
            add("（无可配对数据）")
        else:
            add(f"- 对齐评估步：{match_step}；配对 seeds：{res['seeds']}")
            add(f"- 末期成功率：{ALG_LABEL[name_a]} **{res['a_mean']:.3f}** vs "
                f"{ALG_LABEL[name_b]} **{res['b_mean']:.3f}**，差 **{res['diff_mean']:+.3f}**"
                f"（跨 seed std={res['diff_std']:.3f}，配对 t 检验 {fmt_p(res['p_value'])}）")
            add(f"- 学习曲线 AUC（成功率均值）：{res['auc_a']:.3f} vs {res['auc_b']:.3f}")
        add("")
        sec_idx += 1

    # ---------- 仲裁指标 ----------
    add("## 4. 仲裁/世界模型指标分析（train.log）")
    add("")
    for alg in ["gbi_cross", "qmp_cross", "gbi_cross_fast", "gbi", "qmp"]:
        seeds_done = sorted(train_metrics[alg])
        if not seeds_done:
            continue
        add(f"### {ALG_LABEL[alg]}")
        add("")
        add("| 指标 | 全程均值 | 末期均值（后 20%） | 判定 |")
        add("|---|---|---|---|")
        for key, _, label in METRIC_KEYS:
            all_vals, late_vals = [], []
            for s in seeds_done:
                tm = train_metrics[alg][s]
                if tm and tm[key]:
                    steps = sorted(tm[key])
                    vals = np.asarray([tm[key][st] for st in steps])
                    all_vals.extend(vals.tolist())
                    cut = int(len(vals) * 0.8)
                    late_vals.extend(vals[cut:].tolist())
            if not all_vals:
                add(f"| {label} | — | — | — |")
                continue
            m_all, m_late = np.mean(all_vals), np.mean(late_vals)
            verdict = ""
            if key == "TGR":
                verdict = ("正常" if 0.02 < m_all < 0.5 else "异常（触发过少/过多）")
            elif key == "SWR":
                verdict = ("有换手" if m_all > 0 else "无换手（裁决从未切换）")
            elif key == "LT":
                verdict = ("想象通道被信任" if m_late > 0.05 else "λ≈0（想象通道未被信任）")
            elif key == "DMAE":
                verdict = ("动力学良好" if m_all < 0.01 else "动力学偏差偏大")
            elif key == "ALB":
                verdict = ("TTA 有效" if m_all > 0 and (m_late < m_all) else "—")
            elif key == "RRK":
                verdict = ("排序保真度良好" if m_late > 0.2 else "排序能力不足（奖励头不可用于裁决）")
            elif key == "RABS":
                verdict = ("分箱范围合理" if 0.5 < m_late < 8.0 else "分箱范围异常（过窄/过宽）")
            elif key == "RBU":
                verdict = ("分辨率可接受" if m_late > 0.05 else "bin 浪费严重")
            elif key == "GAM":
                verdict = ("决策场非退化" if m_all > 1e-3 else "决策场空（候选间无真实差异）")
            elif key == "GMM":
                verdict = ("gap 达标(>0.1)" if m_all > 0.1 else "gap 未达标(≤0.1)")
            add(f"| {label} | {m_all:.4f} | {m_late:.4f} | {verdict} |")
        add("")

    # ---------- 结论 ----------
    add("## 5. 结论：GbI 是否有效？")
    add("")
    # 优先用跨任务池配对（Phase A 设计口径），缺失时回退 legacy 快照池配对
    if eval_data["gbi_cross"] and eval_data["qmp_cross"]:
        pair_a, pair_b = "gbi_cross", "qmp_cross"
    else:
        pair_a, pair_b = "gbi", "qmp"
    res_gq = paired_compare(pair_a, pair_b, eval_data[pair_a], eval_data[pair_b],
                            max_steps(pair_a, pair_b, eval_data))
    res_gi = paired_compare(pair_a, "indep", eval_data[pair_a], eval_data["indep"],
                            max_steps(pair_a, "indep", eval_data))
    if res_gq is None:
        add(f"- **裁决公式判定：PENDING**（{pair_a} 或 {pair_b} 数据尚未完成）")
    else:
        d, p = res_gq["diff_mean"], res_gq["p_value"]
        if d > 0.05 and (p is None or p < 0.05):
            verdict = f"**想象增益项 λ_t·(Î_j−Î_i) 有效**：{ALG_LABEL[pair_a]} 末期成功率高出 {ALG_LABEL[pair_b]} {d:+.3f}"
            if p is not None:
                verdict += f"（配对 t 检验 p={p:.3f}）"
            verdict += "。"
        elif abs(d) <= 0.05:
            verdict = (f"**想象增益项未发挥作用**：{ALG_LABEL[pair_a]} 与 {ALG_LABEL[pair_b]} 末期成功率差 {d:+.3f} "
                       f"（≤0.05 判定阈值），裁决主要由 Q 锚主导，λ_t·(Î_j−Î_i) 贡献微弱。")
        else:
            verdict = (f"**想象增益项呈负贡献**：{ALG_LABEL[pair_a]} 低于 {ALG_LABEL[pair_b]} {d:+.3f}，想象打分干扰了 Q 锚裁决，"
                       f"需结合 λ_t 校准门与 U_score 护栏诊断。")
        add(f"- **裁决公式判定（{pair_a} vs {pair_b}）**：{verdict}")
    if res_gi is None:
        add("- **与独立训练对比判定：PENDING**（indep 数据尚未完成）")
    else:
        d = res_gi["diff_mean"]
        add(f"- **多任务裁决 vs 独立训练**：{ALG_LABEL[pair_a]} 末期成功率与 state_sac_indep 差 {d:+.3f}"
            + ("，共享多任务框架 + 裁决优于独立训练。" if d > 0.05
               else ("，两者接近，裁决机制未带来整体提升。" if abs(d) <= 0.05
                     else "，多任务共享框架劣于独立训练，需排查负迁移。")))
    add("")
    add("> 判据说明：末期成功率 = 最后 3 个评估点均值；配对比较按相同 seed 对齐、在共同最大"
        "评估步上做配对 t 检验（n=seeds 数，小样本时以效应量为主）；GbI vs indep 在同一"
        "训练步数（300k）上比较，注意 indep 额外训练到 500k 的信息仅供参考。")
    add("")
    add("> 仲裁指标合理性参考：TGR（触发率）应处于 2%–50% 区间；SWR（换手率）>0 说明裁决"
        "确实在候选间切换；λ_t 由 Spearman κ_t 校准，末期 >0 表示想象增益与真实增益正相关；"
        "DMAE（动力学一步 MAE）<0.01 表示世界模型动力学可信。")
    add("")

    report_dir = os.path.dirname(args.out)
    os.makedirs(report_dir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"[analyze] 报告已写入 {args.out}")
    print(f"[analyze] gbi seeds: {sorted(eval_data['gbi'])} | qmp: {sorted(eval_data['qmp'])} | "
          f"indep: {sorted(eval_data['indep'])} | guide: {sorted(eval_data['guide'])} | "
          f"gbi_cross: {sorted(eval_data['gbi_cross'])} | qmp_cross: {sorted(eval_data['qmp_cross'])} | "
          f"gbi_cross_fast: {sorted(eval_data['gbi_cross_fast'])}")


def max_steps(name_a, name_b, eval_data):
    common = set(eval_data[name_a]) & set(eval_data[name_b])
    if not common:
        return 0
    s = common.pop()
    steps_a = set(eval_data[name_a][s]["steps"].tolist())
    steps_b = set(eval_data[name_b][s]["steps"].tolist())
    inter = steps_a & steps_b
    return max(inter) if inter else 0


if __name__ == "__main__":
    main()
