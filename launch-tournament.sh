#!/bin/bash
# Anchor-vs-opponent tournament launcher — Qwen 3.5 397B A17B anchor.
#
# 50 games per opponent class, three classes, one anchor. Three different
# model families for opponent-class diversity (Google / OpenAI MoE / Mistral):
#
#   gemma     Qwen 3.5 397B A17B  vs  Gemma 3 27B Instruct      → runsA-QWEN-vs-GEMMA/
#   gptoss    Qwen 3.5 397B A17B  vs  GPT-OSS 120B              → runsA-QWEN-vs-GPTOSS120B/
#   mistral   Qwen 3.5 397B A17B  vs  Mistral Small 24B         → runsA-QWEN-vs-MISTRALSMALL/
#
# Substituted Gemma 3 27B for the originally-planned Llama 3.1 70B — the
# latter is architecturally too close to the Llama 3.3 70B baseline.
#
# Cluster layout (16 GPU, 4 nodes):
#   - Qwen anchor:   3 nodes  (head + 2 workers, TP=4, PP=3)  on port 8080
#   - Opponents:     1 node   (TP=4)                           on port 8081
#
# (Qwen needs 3 nodes — see the comment block in vllm-qwen.sh for why
# 2 nodes OoM.)
#
# Three arms share the SAME Qwen endpoint and only swap the opponent endpoint
# between phases. Do not unload Qwen — it costs a 12-GPU reload each time.
#
# Phase order (any order is fine, but we run sequentially):
#   1. On head Qwen node:   bash vllm-qwen.sh <head_node> 0
#      On worker node #1:    bash vllm-qwen.sh <head_node> 1
#      On worker node #2:    bash vllm-qwen.sh <head_node> 2
#   2. On opponent node:  MODEL=google/gemma-3-27b-it bash vllm.sh  (for gemma — port 8081)
#                         bash vllm-gptoss.sh  (for gptoss  — port 8081, GPT-OSS quirks)
#                         bash vllm-mistral.sh (for mistral — port 8081, --tokenizer-mode auto)
#   3. From login node:   bash launch-tournament.sh <arm> <qwen_head_node> <opp_node>
#
# Usage:
#   bash launch-tournament.sh <arm> <qwen_head_node> <opp_node> [<anchor_port>] [<opp_port>]
#
#     arm           gemma | gptoss | mistral
#     qwen_head_node  hostname of the rank-0 Qwen node
#     opp_node      hostname of the node running the opponent vLLM
#     anchor_port   default 8080
#     opp_port      default 8081
#
# Example sequence:
#   On ggpu152 (Qwen head):     bash vllm-qwen.sh ggpu152 0
#   On ggpu154 (Qwen worker 1): bash vllm-qwen.sh ggpu152 1
#   On ggpu195 (Qwen worker 2): bash vllm-qwen.sh ggpu152 2
#   On ggpu197 (opponent):      MODEL=google/gemma-3-27b-it bash vllm.sh
#   From login node:            bash launch-tournament.sh gemma ggpu152 ggpu197
#   (then later, swap opponent endpoint:)
#   On ggpu195:                bash vllm-gptoss.sh
#   From login node:           bash launch-tournament.sh gptoss ggpu152 ggpu195
#   (then later:)
#   On ggpu195:                bash vllm-mistral.sh
#   From login node:           bash launch-tournament.sh mistral ggpu152 ggpu195

set -euo pipefail

if [ $# -lt 3 ]; then
  cat >&2 <<USAGE
usage: bash launch-tournament.sh <arm> <qwen_head_node> <opp_node> [<anchor_port>] [<opp_port>]
  arm:            gemma | gptoss | mistral
  qwen_head_node: vLLM head node serving the Qwen anchor (rank 0)
  opp_node:       vLLM serving the opponent class
USAGE
  exit 2
fi

ARM="$1"
ANCHOR_NODE="$2"
OPP_NODE="$3"
ANCHOR_PORT="${4:-8080}"
OPP_PORT="${5:-8081}"

# Anchor model selected via env var (defaults to qwen for backward compat).
#   ANCHOR=qwen  -> config-tournament-qwen-vs-<arm>-local.yaml,  runsA-QWEN-vs-*
#   ANCHOR=kimi  -> config-tournament-kimi-vs-<arm>-local.yaml,  runsA-KIMI-vs-*
ANCHOR="${ANCHOR:-qwen}"
case "$ANCHOR" in
  qwen)      ANCHOR_LABEL="Qwen 3.5 397B A17B";    ANCHOR_TAG="QWEN" ;;
  kimi)      ANCHOR_LABEL="Kimi K2.5";             ANCHOR_TAG="KIMI" ;;
  deepseek)  ANCHOR_LABEL="DeepSeek V3.1 Terminus"; ANCHOR_TAG="DEEPSEEK" ;;
  *) echo "Unknown ANCHOR: $ANCHOR (expected 'qwen' | 'kimi' | 'deepseek')" >&2; exit 2 ;;
esac

case "$ARM" in
  gemma|gemma3)
    ARM_KEY="gemma"
    OPP_MODEL="google/gemma-3-27b-it"
    OPP_TAG="GEMMA"
    OPP_LAUNCHER="vllm.sh (with MODEL=google/gemma-3-27b-it)"
    ;;
  gptoss|gpt-oss|gpt-oss-120b)
    ARM_KEY="gptoss"
    OPP_MODEL="openai/gpt-oss-120b"
    OPP_TAG="GPTOSS120B"
    OPP_LAUNCHER="vllm-gptoss.sh"
    ;;
  mistral|mistral-small)
    ARM_KEY="mistral"
    OPP_MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
    OPP_TAG="MISTRALSMALL"
    OPP_LAUNCHER="vllm-mistral.sh"
    ;;
  *) echo "Unknown arm: $ARM (expected 'gemma' | 'gptoss' | 'mistral')" >&2; exit 2 ;;
esac

CFG="config-tournament-${ANCHOR}-vs-${ARM_KEY}-local.yaml"
SUMMARY="runsA-${ANCHOR_TAG}-vs-${OPP_TAG}"
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
probe "$ANCHOR_NODE" "$ANCHOR_PORT" "Anchor ($ANCHOR_LABEL)"
probe "$OPP_NODE"   "$OPP_PORT"     "Opponent ($OPP_MODEL — launched via $OPP_LAUNCHER)"

# Patch base_url in the config (idempotent).
#   llm_player.base_url       -> ANCHOR_NODE:ANCHOR_PORT
#   basic_llm_player.base_url -> OPP_NODE:OPP_PORT
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
LOG="logs/launch-tournament-$ARM.$TS.log"
echo "[$(date +%T)] Launching $ARM arm; log -> $LOG"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}" bash sim-50.sh "$CFG" 2>&1 | tee "$LOG"
echo "[$(date +%T)] $ARM arm complete. Summaries: $SUMMARY/"
