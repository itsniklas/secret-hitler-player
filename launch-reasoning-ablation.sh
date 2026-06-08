#!/bin/bash
# DeepSeek V3.1 Terminus reasoning ON vs OFF ablation.
#
# This script runs the OFF condition only. The ON condition is already
# captured by runsF2-DEEPSEEK31TERMINUS/ (100 games), made after the
# hard-coded reasoning bundle was added.
#
# Cluster layout (4 nodes, 16 GPU):
#   - DeepSeek anchor:  3 nodes (head + 2 workers, TP=4, PP=3) on port 8080
#   - Llama opponents:  1 node  (TP=4)                          on port 8081
#
# Phase order:
#   1. On DeepSeek head node:    bash vllm-deepseek.sh <head_node> 0
#      On DeepSeek worker #1:    bash vllm-deepseek.sh <head_node> 1
#      On DeepSeek worker #2:    bash vllm-deepseek.sh <head_node> 2
#   2. On opponent node:         bash vllm.sh   (defaults to Llama 3.3 70B on 8081)
#   3. From login node:          bash launch-reasoning-ablation.sh <ds_head_node> <opp_node>
#
# REASONING_ENABLED=0 flips simulator/players/hitler_player.py to send
# {reasoning_effort:none, reasoning.enabled:false, chat_template_kwargs.thinking:false}
# instead of the default ON bundle.
#
# Usage:
#   bash launch-reasoning-ablation.sh <ds_head_node> <opp_node> [<anchor_port>] [<opp_port>]

set -euo pipefail

if [ $# -lt 2 ]; then
  cat >&2 <<USAGE
usage: bash launch-reasoning-ablation.sh <ds_head_node> <opp_node> [<anchor_port>] [<opp_port>]
  ds_head_node:  rank-0 node of the DeepSeek V3.1 vLLM (3-node TP=4 PP=3)
  opp_node:      node running the Llama 3.3 70B opponent vLLM
USAGE
  exit 2
fi

ANCHOR_NODE="$1"
OPP_NODE="$2"
ANCHOR_PORT="${3:-8080}"
OPP_PORT="${4:-8081}"

CFG="config-reasoning-off-local.yaml"
SUMMARY="runsT-DEEPSEEK-NOREASON"

if [ ! -f "$CFG" ]; then
  echo "Config not found: $CFG" >&2
  exit 2
fi

probe() {
  local node="$1" port="$2" label="$3"
  echo "[$(date +%T)] Probing $label endpoint: http://$node:$port/v1/models"
  curl -sf --max-time 10 "http://$node:$port/v1/models" \
    | python -c 'import json,sys; d=json.load(sys.stdin); print("  models:", [m["id"] for m in d.get("data",[])])' \
    || { echo "  ERROR: $label endpoint not responding" >&2; exit 1; }
}
probe "$ANCHOR_NODE" "$ANCHOR_PORT" "Anchor (DeepSeek V3.1 Terminus)"
probe "$OPP_NODE"   "$OPP_PORT"     "Opponent (Llama 3.3 70B Instruct via vllm.sh)"

# Patch base_url in the config (idempotent — first base_url is llm_player,
# second is basic_llm_player after the marker line).
python - "$CFG" "$ANCHOR_NODE" "$ANCHOR_PORT" "$OPP_NODE" "$OPP_PORT" <<'PY'
import sys, re, shutil, pathlib
cfg, an, ap, on, op = sys.argv[1:]
p = pathlib.Path(cfg)
shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
txt = p.read_text()
parts = re.split(r"(basic_llm_player:)", txt, maxsplit=1)
assert len(parts) == 3, "config does not contain a basic_llm_player block"
top, marker, rest = parts
top = re.sub(r'base_url:\s*"[^"]*"', f'base_url: "http://{an}:{ap}/v1/"', top, count=1)
rest = re.sub(r'base_url:\s*"[^"]*"', f'base_url: "http://{on}:{op}/v1/"', rest, count=1)
p.write_text(top + marker + rest)
print(f"  patched {cfg}: llm_player -> {an}:{ap}, basic_llm_player -> {on}:{op}")
PY
grep base_url "$CFG" | sed 's/^/   /'

mkdir -p "$SUMMARY" logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/launch-reasoning-off.$TS.log"
echo "[$(date +%T)] REASONING_ENABLED=0 — launching 100-game thinking-OFF run; log -> $LOG"
REASONING_ENABLED=0 LLM_API_KEY="${LLM_API_KEY:-EMPTY}" bash sim.sh "$CFG" 2>&1 | tee "$LOG"
echo "[$(date +%T)] Thinking-OFF arm complete. Summaries: $SUMMARY/"
