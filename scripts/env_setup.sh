#!/usr/bin/env bash
# env_setup.sh — M0 环境建设: 记录依赖清单到可复现目录 (环境本身已在此前交互中装好).
# 用法: bash scripts/env_setup.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/root/rivermind-data/lost+found/gbi"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "[env_setup] python: $(python3 --version 2>&1)"
echo "[env_setup] pip freeze -> $LOG_DIR/env_setup.log"
python3 -m pip list 2>/dev/null | sort > "$LOG_DIR/env_setup.log" || \
    python3 -m pip freeze 2>/dev/null | sort > "$LOG_DIR/env_setup.log"

echo "[env_setup] gpu info -> $LOG_DIR/gpu_info.txt"
nvidia-smi > "$LOG_DIR/gpu_info.txt" 2>&1 || echo "no nvidia-smi" | tee -a "$LOG_DIR/gpu_info.txt"

echo "[env_setup] verifying imports (metaworld / gymnasium / hydra / ml_logger / torch)"
VERIFY=$(python3 - <<'PY'
import importlib, torch
for mod, pkg in [("gymnasium","gymnasium"), ("metaworld","metaworld"),
                 ("hydra","hydra-core"), ("ml_logger","ml-logger"),
                 ("omegaconf","omegaconf"), ("termcolor","termcolor")]:
    try:
        m = importlib.import_module(mod)
        print(f"OK  {pkg:<18} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"FAIL {pkg:<18} {e}")
print(f"OK  torch {torch.__version__} cuda={torch.cuda.is_available()}")
PY
)
echo "$VERIFY" | tee -a "$LOG_DIR/env_setup.log"
if echo "$VERIFY" | grep -q FAIL; then
    echo "[env_setup] some imports failed — abort"
    exit 1
fi
echo "[env_setup] DONE (log: $LOG_DIR/env_setup.log)"