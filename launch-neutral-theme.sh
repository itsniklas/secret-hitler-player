#!/bin/bash
# Loaded-terms (neutral theme) ablation launcher, three arms.
#
# Each arm is a 100-game self-play against 4× Llama-3.3-70B opponents
# with the loaded-term rewriter (simulator/theme.py) active. The arms
# swap Alice's model:
#
#   gemma     → google/gemma-3-27b-it              → runsF2-GEMMA-NEUTRAL/
#   mistral   → mistralai/Mistral-Small-24B-2501   → runsF2-MISTRALSMALL-NEUTRAL/
#   gptoss    → openai/gpt-oss-120b                → runsF2-GPTOSS120B-NEUTRAL/
#
# We need TWO endpoints:
#   - Alice endpoint  (port 8080)  serves the focus model on one node
#   - Opponent endpoint (port 8081) serves Llama 3.3 70B on a different node
#
# All arms share the same Llama endpoint; only Alice's endpoint changes
# between arms. Arms run SEQUENTIALLY unless you spin up multiple Alice
# endpoints on different nodes.
#
# Usage:
#   bash launch-neutral-theme.sh <arm>  <alice_node> <opp_node> [<alice_port>] [<opp_port>]
#
#     arm           gemma | mistral | gptoss
#     alice_node    hostname of node running the focus-model vLLM
#     opp_node      hostname of node running the Llama 3.3 70B vLLM
#     alice_port    default 8080
#     opp_port      default 8081
#
# Example:
#   On node A (ggpu142):  bash vllm.sh   # serves Llama 3.3 70B on :8081
#   On node B (ggpu143):  MODEL=google/gemma-3-27b-it PORT=8080 bash vllm.sh
#                         (or vllm-gemma.sh — same script with MODEL/PORT overrides)
#
#   From login node:
#     bash launch-neutral-theme.sh gemma   ggpu143 ggpu142
#     (then later, swap node B to mistral:)
#     bash vllm-mistral.sh   # on node B (sets --tokenizer-mode auto for Mistral)
#     bash launch-neutral-theme.sh mistral ggpu143 ggpu142
#     (then later, swap node B to gpt-oss-120b:)
#     bash vllm-gptoss.sh   # on node B — DO NOT use vllm.sh for gpt-oss.
#                            # vllm-gptoss.sh adds --language-model-only and
#                            # the TIKTOKEN_RS_CACHE_DIR apptainer env that
#                            # gpt-oss-120b requires; the stock vllm.sh lacks
#                            # both and the server will fail to start.
#     bash launch-neutral-theme.sh gptoss  ggpu143 ggpu142

set -euo pipefail

if [ $# -lt 3 ]; then
  cat >&2 <<USAGE
usage: bash launch-neutral-theme.sh <arm> <alice_node> <opp_node> [<alice_port>] [<opp_port>]
  arm:        gemma | mistral | gptoss
  alice_node: vLLM serving focus model (use vllm-gptoss.sh for gptoss arm)
  opp_node:   vLLM serving Llama-3.3-70B (opponents)
USAGE
  exit 2
fi

ARM="$1"
ALICE_NODE="$2"
OPP_NODE="$3"
ALICE_PORT="${4:-8080}"
OPP_PORT="${5:-8081}"

case "$ARM" in
  gemma)
    CFG="config-neutral-gemma-local.yaml"
    ALICE_MODEL="google/gemma-3-27b-it"
    SUMMARY="runsF2-GEMMA-NEUTRAL"
    ;;
  mistral)
    CFG="config-neutral-mistral-local.yaml"
    ALICE_MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
    SUMMARY="runsF2-MISTRALSMALL-NEUTRAL"
    ;;
  gptoss|gpt-oss|gpt-oss-120b)
    CFG="config-neutral-gptoss-local.yaml"
    ALICE_MODEL="openai/gpt-oss-120b"
    SUMMARY="runsF2-GPTOSS120B-NEUTRAL"
    ;;
  *) echo "Unknown arm: $ARM (expected 'gemma' | 'mistral' | 'gptoss')" >&2; exit 2 ;;
esac

probe() {
  local node="$1" port="$2" label="$3"
  echo "[$(date +%T)] Probing $label endpoint: http://$node:$port/v1/models"
  curl -sf --max-time 5 "http://$node:$port/v1/models" \
    | python -c 'import json,sys; d=json.load(sys.stdin); print("  models:", [m["id"] for m in d.get("data",[])])' \
    || { echo "  ERROR: $label endpoint not responding" >&2; exit 1; }
}
probe "$ALICE_NODE" "$ALICE_PORT" "Alice ($ALICE_MODEL)"
probe "$OPP_NODE"   "$OPP_PORT"   "Opponent (Llama-3.3-70B)"

# Patch base_url in the config (idempotent).
#   llm_player.base_url       -> alice_node:alice_port
#   basic_llm_player.base_url -> opp_node:opp_port
python - "$CFG" "$ALICE_NODE" "$ALICE_PORT" "$OPP_NODE" "$OPP_PORT" <<'PY'
import sys, re, shutil, pathlib
cfg, an, ap, on, op = sys.argv[1:]
p = pathlib.Path(cfg)
shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
txt = p.read_text()
# llm_player block: first base_url
# basic_llm_player block: second base_url
parts = re.split(r"(basic_llm_player:)", txt, maxsplit=1)
assert len(parts) == 3, "config does not contain a basic_llm_player block"
top, marker, rest = parts
top = re.sub(r'base_url:\s*"[^"]*"', f'base_url: "http://{an}:{ap}/v1/"', top, count=1)
rest = re.sub(r'base_url:\s*"[^"]*"', f'base_url: "http://{on}:{op}/v1/"', rest, count=1)
p.write_text(top + marker + rest)
print(f"  patched {cfg}: llm_player -> {an}:{ap}, basic_llm_player -> {on}:{op}")
PY
grep base_url "$CFG" | sed 's/^/   /'

grep -q '^[[:space:]]*theme:[[:space:]]*"neutral"' "$CFG" \
  || { echo "$CFG missing 'theme: \"neutral\"'" >&2; exit 1; }

mkdir -p "$SUMMARY" logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/launch-neutral-theme-$ARM.$TS.log"
echo "[$(date +%T)] Launching $ARM arm; log -> $LOG"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}" bash sim.sh "$CFG" 2>&1 | tee "$LOG"
echo "[$(date +%T)] $ARM arm complete. Summaries: $SUMMARY/"
