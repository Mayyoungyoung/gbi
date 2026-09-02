#!/usr/bin/env python3
"""Phase 0 决策场复测：对 state_sac 候选快照做真实 rollout eval（success 率）。

背景：state_sac seed0 run 全程 success 恒 0（#6 gymnasium infos bug 受害者），
eval.log 只余 episode_reward。决策场验收（GbI.md L208：own 真实最优率 ≥0.5、
真实 gap 中位数 >0.1）需要 success 信号 → 用修复后的代码重测 10 个 ckpt。

actor 结构（从 config.json/ckpt 实测恢复，无需实例化完整 agent）：
  trunk = MLP(39→400→400→400→400→400, 5 层 Linear + ReLU 间隔)
  head  = (10, 400, 8) 每任务一 head（8 = 2×act_dim，mu+log_std）
  act   = tanh(mu), eval 用 mean action

用法:
  python3 scripts/alg/eval_decision_field.py \
      --model_dir <dir> --steps 50000,...,500000 \
      [--eps 20] [--shard 0/3] [--device cuda:0] [--out results.json]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

OBS_DIM = 39
ACT_DIM = 4
HIDDEN = 400
NUM_LAYERS = 5
NUM_HEADS = 10
MAX_EP_STEPS = 200  # 与训练 eval 的 max_step_wrapper 一致


def build_weights(sd: dict) -> dict:
    """把 ckpt state_dict 转成易用的张量（在 device 上）。"""
    w = {}
    for i in range(NUM_LAYERS):
        w[f"L{i}"] = (sd[f"model.0.{2*i}.weight"], sd[f"model.0.{2*i}.bias"])
    w["head"] = (sd["model.2._model.0.weight"], sd["model.2._model.0.bias"])
    return w


def act_deterministic(w: dict, obs: torch.Tensor, task_idx: int, device) -> np.ndarray:
    """obs: (B, 39) → mean action (B, 4)（tanh squashed）。"""
    x = obs
    for i in range(NUM_LAYERS):
        W, b = w[f"L{i}"]
        x = F.linear(x, W, b)
        if i < NUM_LAYERS - 1:
            x = F.relu(x)
    x = F.relu(x)  # trunk 与 head 之间的 ReLU（model = Sequential(trunk, ReLU, heads)）
    Wh, bh = w["head"]
    # 专家 Linear: x.matmul(W) 输出 (num_experts, B, O)；bias (E,1,O) 广播 dim1
    out = torch.einsum("bi,tio->tbo", x, Wh) + bh  # (10, B, 8)
    out = out[task_idx]  # (B, 8)
    mu, _ = out.chunk(2, dim=-1)
    return torch.tanh(mu).detach().cpu().numpy()


def make_env(task_name: str, task, seed: int):
    import metaworld

    env = metaworld.MT10().train_classes[task_name]()
    env.set_task(task)
    env._freeze_rand_vec = True  # 与训练一致：冻结随机目标
    env.reset(seed=seed)
    return env


def eval_ckpt(sd: dict, task_names: list, tasks: list, eps: int, device, seed0: int):
    """一个 ckpt × 10 任务 × eps episode → (succ[10], rew[10])"""
    w = build_weights(sd)
    succ = np.zeros(len(task_names))
    rew = np.zeros(len(task_names))
    for t, (name, task) in enumerate(zip(task_names, tasks)):
        env = make_env(name, task, seed=seed0)
        n_succ = 0
        rews = []
        for e in range(eps):
            obs, _ = env.reset()
            done = False
            total = 0.0
            steps = 0
            while not done and steps < MAX_EP_STEPS:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act = act_deterministic(w, obs_t, t, device)[0]
                obs, r, terminated, truncated, info = env.step(act)
                done = terminated or truncated
                total += r
                steps += 1
                if info.get("success"):
                    n_succ += 1
                    break
            rews.append(total)
        succ[t] = n_succ / eps
        rew[t] = float(np.mean(rews))
        print(f"  task {t} {name}: succ={succ[t]:.2f} rew={rew[t]:.1f}", flush=True)
    return succ, rew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--steps", required=True, help="逗号分隔 ckpt step 列表")
    ap.add_argument("--eps", type=int, default=20)
    ap.add_argument("--shard", default=None, help="i/n，多进程分片")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        steps = steps[i::n]

    import metaworld

    mt10 = metaworld.MT10()
    task_names = list(mt10.train_classes.keys())  # 10 任务（顺序与训练 env 一致）
    tasks = []
    for name in task_names:
        tasks.append(next(t for t in mt10.train_tasks if t.env_name == name))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} 任务={task_names}", flush=True)

    results = {}
    for step in steps:
        path = os.path.join(args.model_dir, f"actor_{step}.pt")
        sd = torch.load(path, map_location=device)
        print(f"== ckpt {step}: {path} ==", flush=True)
        succ, rew = eval_ckpt(sd, task_names, tasks, args.eps, device, seed0=100 + step)
        results[str(step)] = {"succ": succ.tolist(), "rew": rew.tolist(),
                              "task_names": task_names}
        # 增量落盘，崩溃可续
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[partial saved] {args.out}", flush=True)

    print(f"ALL_DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
