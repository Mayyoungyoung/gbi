# GbI 服务器迁移指南（换机继续实验）

> 更新：2026-09-03。适用场景：把 GbI 实验从旧机（10 vCPU + Tesla T4 16GB）迁往新服务器继续执行。
> 前置事实：仓库 `master` 已含 P0.1 跨任务候选池与四份文档（5451fbf）；旧机 09-02 批次
> （qmp/gbi_online 已完成，guide/gbi/qmp_online 进行中）的运行数据是**机器本地产物，不入 git**——
> 本指南只负责"代码 + 环境 + 流程"的迁移，结果数据如需带走见 §8。

## 1. 仓库与目录约定

| 对象 | 旧机位置 | git 内？ |
|---|---|---|
| 代码仓库（main.py / mtrl / config / scripts / docs） | `lost+found/CTPG-main/CTPG-main/`（= 仓库根，clone 后无此嵌套） | ✅ `git@github.com:Mayyoungyoung/gbi.git` |
| 设计 / 指南 / 状态 / 提案 文档 | `lost+found/` 根（工作副本）+ `docs/`（入库副本） | ✅ `docs/GbI_*.md` |
| benchmark 运行数据（run 目录/模型/日志/曲线） | `lost+found/gbi/experiments/runs/` | ❌（巨大且持续变化，勿入库） |
| 报告 / 分析产物 | `lost+found/gbi/experiments/reports/` | ❌ |
| 监控采样 | `lost+found/gbi/experiments/monitor/` | ❌ |
| 任务文本嵌入（roberta_small） | 仓库 `metadata/task_embedding/` | ✅ 已入库 |

新机建议布局（`$WORK` 为任意工作根目录）：

```bash
$WORK/gbi        # git clone 的仓库（代码）
$WORK/gbi_out/runs       # 等价旧机 gbi/experiments/runs
$WORK/gbi_out/reports
$WORK/gbi_out/monitor
```

## 2. GitHub 访问

```bash
ssh-keygen -t ed25519 -C "your@email" -f ~/.ssh/id_ed25519   # 已有可跳过
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub   # 加到 github.com → Settings → SSH keys
ssh -T git@github.com       # 预期输出 Hi Mayyoungyoung!
git clone git@github.com:Mayyoungyoung/gbi.git && cd gbi
```

## 3. 环境复现（以旧机实测组合为准，勿盲装 requirements.txt）

> ⚠️ `requirements.txt` 复制自 CTPG 上游，其中 `gym==0.18`、`torch==1.13.1`、
> `metaworld==0.1.0`（已注释）、mujoco-py 等**均非旧机实测版本**。实测可用组合
> （Python 3.10.13 / conda）：**torch 2.1.0 + gymnasium 1.3.0 + metaworld 3.1.1 + mujoco 3.3.0**。
> 代码已针对 gymnasium 1.x 修复（#6，`vec_env.py` 按 key 解析 infos）——
> **切勿降级到 gym 0.21 或老 gymnasium**，否则 success 恒 0 回归。

```bash
conda create -n gbi python=3.10 -y && conda activate gbi

# PyTorch（按新机 CUDA 驱动选 cu118/cu121 均可，旧机实测 cu 组合 torch 2.1.0）
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 核心依赖（只装代码真正用到的；无需整表安装 requirements.txt）
pip install hydra-core==1.0.5 omegaconf==2.0.6 ml-logger==0.7 numpy==1.26.2 \
  gymnasium==1.3.0 pandas matplotlib scipy pyyaml termcolor cloudpickle psutil \
  mujoco pytest tqdm

# MetaWorld（Farama 主线即可，旧机实测 3.1.1）
pip install git+https://github.com/Farama-Foundation/MetaWorld.git@master
```

系统依赖（Ubuntu，MuJoCo/OpenGL）：

```bash
sudo apt-get install -y libgl1 libgl1-mesa-glx libglew-dev libosmesa6-dev libglfw3-dev
```

验证（等价 `scripts/env_setup.sh` 的 import 检查，但该脚本 `LOG_DIR` 硬编码旧机路径，
新机直接跑下面的片段更省事）：

```bash
python3 - <<'PY'
import importlib, torch
for mod, pkg in [("gymnasium","gymnasium"),("metaworld","metaworld"),
                 ("hydra","hydra-core"),("ml_logger","ml-logger"),
                 ("omegaconf","omegaconf"),("termcolor","termcolor")]:
    m = importlib.import_module(mod)
    print(f"OK  {pkg:<18} {getattr(m, '__version__', '?')}")
print(f"OK  torch {torch.__version__} cuda={torch.cuda.is_available()}")
PY
```

## 4. 路径改造清单（换机必改的硬编码）

以下脚本硬编码了旧机路径，新机按需修改后**就地提交**（旧路径留在 git 历史可查）：

| 文件 | 变量/位置 | 新机改法 |
|---|---|---|
| `scripts/run_cross_benchmark.sh` | L29 `RUNS_ROOT="/root/.../gbi/experiments/runs"` | 改为 `$WORK/gbi_out/runs` |
| `scripts/run_full_benchmark.sh` | L27 `RUNS_ROOT` | 同上 |
| `scripts/run_full_benchmark.sh` | L33 `CAND_DIR`（state_sac 500k 快照池） | 沿用旧机快照则 rsync 该 model 目录到新机并改路径；全新则先跑 `state_sac_indep seed0`（500k，见 Guide §5.2）再改 |
| `scripts/run_gbi_nohup.sh` / `run_gbi_then_qmp.sh` / `scripts/alg/*.sh` | `BASE_PATH` / `CANDIDATES_DIR` 等 | legacy 参考脚本，跑之前同步改 |
| `scripts/env_setup.sh` | L7 `LOG_DIR` | 改为 `$WORK/gbi_out` |
| `scripts/alg/analyze_results.py` | `--root` 有默认值但**支持参数覆盖** | 报告生成一律 `--root $WORK/gbi_out/runs --out $WORK/gbi_out/reports/...` |
| `scripts/monitor_training.py` | 支持 `--runs-root` 参数 | 见 §6 监控命令 |

`metadata/task_embedding/`、`config/` 无路径依赖，无需改。

## 5. 环境冒烟验收（按顺序，每步看指标再进下一步）

```bash
export PYTHONPATH=$WORK/gbi
cd $WORK/gbi

# 1) 单 agent 冒烟（15k 步，~10 分钟）：验证 env/训练/eval/日志链路
bash scripts/smoke.sh gbi metaworld mt10

# 2) 冒烟判据（对应历史白跑事故，缺一不可）：
#    - train.log: actor_entropy 持续下降（≠ 恒 5.68 → 无 #7 冻结）
#    - alpha_0 收缩（≠ 恒 0.995）
#    - eval.log: env_step_i 全 200、success 非恒 0（→ 无 #6 回归）
#    - 进程退出码 0
```

## 6. Phase A（跨任务候选池，P0.1）启动流程

前提：旧机 09-02 批次已结束（`run_cross_benchmark.sh` 自带运行中训练检测，有训练在跑会拒绝启动，`FORCE=1` 可越过但 T4 并发无收益）。

```bash
cd $WORK/gbi
mkdir -p $WORK/gbi_out/runs

# 1) SMOKE 冒烟（3000 步，单 stage 串行，~10 分钟）
SMOKE=1 bash scripts/run_cross_benchmark.sh

# 冒烟判据（stdout / run 目录 train.log）：
#   候选池构建日志 "cross-task candidate pool: N live task heads"（N=10）
#   λ_t / κ_t / 换手率(SWR) 摆脱全零；entropy 正常下降；无 NaN
```

**⚠️ 冒烟后必须清残留**：冒烟 run 的 `log.jsonl` 带 `"status": "COMPLETED"`，
`check_completed` 会把最新 run 视为已完成 → 正式启动会被跳过。清法：

```bash
rm -rf $WORK/gbi_out/runs/gbi_cross $WORK/gbi_out/runs/qmp_cross $WORK/gbi_out/runs/gbi_cross_fast
# （或删除对应 run 目录下的 log.jsonl）
```

```bash
# 2) 正式三臂（300k；并发 3 为 T4 实测安全上限；nohup 防 SSH 断连，禁止 tee/前台）
PARALLEL=1 MAX_PARALLEL=3 nohup bash scripts/run_cross_benchmark.sh \
  > $WORK/gbi_out/runs/cross_runner.log 2>&1 &

# 3) 监控循环（30 分钟采样；nohup 后台）
nohup bash -c 'while true; do sleep 1800; \
  python3 scripts/monitor_training.py --runs-root $WORK/gbi_out/runs \
  --append $WORK/gbi_out/monitor/samples.jsonl; done' \
  > $WORK/gbi_out/monitor/monitor_loop.log 2>&1 &

# 查看进度
tail -f $WORK/gbi_out/runs/gbi_cross/metaworld/mt10/seed0/gbi_cross:0_stdout.log
```

go/no-go（引用 GbI_Improvement_Proposals.md / GbI_Experiment_Status.md）：
跨任务决策场 gap 中位数 > 0.1；gbi_cross 的 λ_t 不出现长区间归零；`gbi_cross ≥ qmp_cross ≥ guide`。

若需多 seed（Phase B 起 ≥3 seeds）：`RUN_STAGES="gbi_cross:1 qmp_cross:1 gbi_cross_fast:1 gbi_cross:2 ..."` 覆盖后重跑即可，已完成 stage 自动跳过。

## 7. 旧式快照池实验（可选，Phase B 若需对照 gbi/qmp legacy）

快照池臂（gbi:0/qmp:0）依赖 state_sac_indep 的磁盘快照。新机流程：
1. 跑 `bash scripts/alg/state_sac_indep.sh 0`（500k，~4h）或 rsync 旧机 `.../state_sac_indep/mt10/seed0/logs/.../model/`（305MB）；
2. 把 model 目录绝对路径写入 `run_full_benchmark.sh` 的 `CAND_DIR`；
3. `RUN_STAGES="gbi:1 qmp:1" PARALLEL=1 MAX_PARALLEL=2 nohup bash scripts/run_full_benchmark.sh ... &`。

在线池臂（gbi_online/qmp_online）与 guide 无外部依赖，直接按 §4 改路径后跑。

## 8. 旧机结果数据迁移（可选）

```bash
# 旧机 → 新机（示例；video/buffer 可剔除减量）
rsync -av --exclude video --exclude buffer \
  root@OLD_HOST:/root/rivermind-data/lost+found/gbi/experiments/runs/ \
  $WORK/gbi_out/runs/

# 新机生成/刷新报告（--root 覆盖硬编码默认）
python3 scripts/alg/analyze_results.py \
  --root $WORK/gbi_out/runs --out $WORK/gbi_out/reports/benchmark_report.md
```

## 9. 换机常见坑清单

| 坑 | 说明 |
|---|---|
| numpy ≥ 2.0 | 旧机 1.26.x；numpy 2.x 与 torch 1.x/旧 omegaconf 生态不兼容，务必钉 <2 |
| gymnasium 降级 / 混装 gym | #6 修复针对 gymnasium 1.x 按 key 的 infos；gym 0.21 会令 success 恒 0（假收敛判断） |
| MetaWorld 旧版指南（mujoco-py） | README 的 v2.0.0 + mujoco-py 流程**过时**；实测 Farama master（3.1.1）+ mujoco 3.x |
| torch/CUDA 不匹配 | 先 `nvidia-smi` 看驱动支持的 CUDA 再选 cu 版本；装完 `torch.cuda.is_available()` 必须 True |
| 长训练前台 / `\| tee` | 2026-08-31 停摆事故根因；一律 nohup + 重定向文件，完成判据看 `log.jsonl` 的 `"status": "COMPLETED"` |
| 并发 > 3-4 | T4 实测 3 并发即 GPU 100%；并发 5+ 只摊薄步速 |
| 忘设线程环境变量 | 每个训练进程必须 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ...`（run_full/run_cross 脚本已内置，手跑 main.py 时记得） |
| SMOKE 残留导致正式被跳过 | 见 §6，正式前清空三臂 run 目录 |
| 训练中跑 cross 脚本 | 护栏会拒绝；确认旧批次结束再启动 |
| 改完路径忘提交 | 硬编码路径改动就地 commit，避免新机重蹈；提交用 `git -c user.name=Mayyoungyoung -c user.email=Mayyoungyoung@users.noreply.github.com commit -m "..."` |

## 10. 文档同步习惯（沿用旧机）

`docs/` 是入库副本；日常编辑建议在工作区根保留一份同名工作副本，改完 `cp` 回 `docs/`
再提交。本文档如被新机改造（路径/环境差异），请在文首"更新"行追加记录并推送，
保持单一事实来源可追溯。
