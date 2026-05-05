#!/bin/bash
# Llama N=1000 rerun for activation-level experiments.
# - per_layer_same_pos: 1000 benign (OpenOrca) + 800 malicious (HarmBench+AdvBench)
# - typo_decay_preamble_control / multi_typo_vanilla / punct_decay_control: OpenOrca only
set -u

# Resolve paths relative to this script's location so the runner works
# from any clone path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
LOG="$REPO_ROOT/results/_N1000_runlog.txt"
mkdir -p "$(dirname "$LOG")"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

{
  echo "=== START $(ts) ==="

  echo "[$(ts)] === 1/4: per_layer_same_pos (on-site rotation, mal/ben mix) ==="
  $PY activation_robustness/experiments/per_layer_same_pos.py --n-prompts 1000 --n-malicious 800
  echo "[$(ts)] EXIT: $?"

  echo "[$(ts)] === 2/4: typo_decay_preamble_control (spatial decay) ==="
  $PY activation_robustness/experiments/typo_decay_preamble_control.py --n-prompts 1000
  echo "[$(ts)] EXIT: $?"

  echo "[$(ts)] === 3/4: multi_typo_vanilla (two-typo interaction) ==="
  $PY activation_robustness/experiments/multi_typo_vanilla.py --n-prompts 1000
  echo "[$(ts)] EXIT: $?"

  echo "[$(ts)] === 4/4: punct_decay_control (punct-decay appendix) ==="
  $PY activation_robustness/experiments/punct_decay_control.py --n-prompts 1000
  echo "[$(ts)] EXIT: $?"

  echo "=== END $(ts) ==="
} > "$LOG" 2>&1
