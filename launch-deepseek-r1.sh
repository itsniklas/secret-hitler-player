#!/bin/bash
# DeepSeek-R1 evaluated as Alice vs 4× Llama 3.3 70B opponents.
#
# Cluster shape (4 nodes, 16 A100):
#   - DeepSeek-R1 anchor:  3 nodes (head + 2 workers, TP=4, PP=3) on port 8080
#   - Llama 3.3 70B opps:  1 node  (TP=4)                          on port 8081
#
# Reasoning bundle is the default ON path in simulator/players/hitler_player.py
# (R1 is thinking-only, so the bundle is mostly a no-op; left ON for parity
# with the existing runsF2-DEEPSEEK31TERMINUS baseline).
#
# Phase order:
#   1. On R1 head node:    bash vllm-deepseek-r1.sh <head_node> 0
#      On R1 worker #1:    bash vllm-deepseek-r1.sh <head_node> 1
#      On R1 worker #2:    bash vllm-deepseek-r1.sh <head_node> 2
#   2. On opponent node:   bash vllm.sh   (defaults to Llama 3.3 70B on 8081)
#   3. From login node:    bash launch-deepseek-r1.sh <r1_head_node> <opp_node>

set -euo pipefail

if [ $# -lt 2 ]; then
  cat >&2 <<USAGE
usage: bash launch-deepseek-r1.sh <r1_head_node> <opp_node> [<anchor_port>] [<opp_port>]
  r1_head_node:  rank-0 node of the DeepSeek-R1 vLLM (3-node TP=4 PP=3)
  opp_node:      node running the Llama 3.3 70B opponent vLLM
USAGE
  exit 2
fi

ANCHOR_NODE="$1"
OPP_NODE="$2"
ANCHOR_PORT="${3:-8080}"
OPP_PORT="${4:-8081}"

CFG="config-deepseek-r1-local.yaml"
SUMMARY="runsN-DEEPSEEK-R1"

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
probe "$ANCHOR_NODE" "$ANCHOR_PORT" "Anchor (DeepSeek-R1)"
probe "$OPP_NODE"   "$OPP_PORT"     "Opponent (Llama 3.3 70B Instruct via vllm.sh)"

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
LOG="logs/launch-deepseek-r1.$TS.log"
echo "[$(date +%T)] Launching 100-game DeepSeek-R1 run; log -> $LOG"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}" bash sim.sh "$CFG" 2>&1 | tee "$LOG"
echo "[$(date +%T)] DeepSeek-R1 run complete. Summaries: $SUMMARY/"
