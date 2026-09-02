#!/usr/bin/env python3
"""决策场分析：从 eval_decision_field.py 的 rollout 结果构造 GbI.md L208 验收指标。

- own 最优率：own 在多少任务上 ≥ 池中最强（平局计 own，与仲裁 argmax 语义一致）
- 真实 gap：每任务 best - own（仅在有真实差异的任务上统计正向 gap）
- 有效最优率：剔除"全池无差异"（全 0 或全 1）的平局任务后重算，防虚增

用法: python3 scripts/alg/analyze_decision_field.py <shard_dir> [--out report.md]
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir", help="含 field_shard*.json 的目录")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    steps = ["50000", "100000", "150000", "200000", "250000",
             "300000", "350000", "400000", "450000", "500000"]
    V = {}
    for f in sorted(glob.glob(os.path.join(args.shard_dir, "field_shard*.json"))):
        for k, v in json.load(open(f)).items():
            V[k] = np.array(v["succ"])
    missing = [s for s in steps if s not in V]
    if missing:
        print(f"缺 ckpt: {missing}")
    M = np.stack([V[k] for k in steps])  # (10, 10)
    best = M.max(0)
    lines = []
    lines.append("## Phase 0 决策场复测（rollout success，20 eps/任务，同 seed）\n")
    lines.append("| own | 最优率 | 有效最优率* | 被超越任务 | 正向 gap 中位 |")
    lines.append("|-----|--------|------------|-----------|--------------|")
    for r, k in enumerate(steps):
        own = M[r]
        opt = np.mean(own >= best - 1e-9)
        varied = best > 0  # 有差异任务（池中有人成功）
        eff_opt = np.mean(own[varied] >= best[varied] - 1e-9) if varied.any() else np.nan
        gaps = best - own
        pos = gaps[gaps > 0.05]
        lines.append(
            f"| {int(k)//1000}k | {opt:.2f} | {eff_opt:.2f} | "
            f"{int(np.sum(gaps > 0.05))}/10 | "
            f"{np.median(pos) if len(pos) else 0:.2f} (n={len(pos)}) |"
        )
    lines.append(
        "\n*有效最优率：剔除全池无差异（best=0）的平局任务后重算，防虚增。"
    )
    lines.append(
        "\n验收线（GbI.md L208）：own 最优率 ≥0.5 且真实 gap 中位数 >0.1。"
    )
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
