#!/bin/bash
# Llama 3.3 70B launcher.
#
# Usage:
#   bash launch-llama33.sh [<vllm_node>] [<port>]
#
# If no node is given we try to discover one of the user's Slurm jobs and
# probe its hostname for a vLLM server on port 8081. If a node is given we
# skip discovery and use it directly.

set -euo pipefail

PORT="${2:-8081}"
NODE="${1:-}"

discover_node() {
  local jobs node ip
  # List the user's Slurm jobs and their nodes.
  jobs="$(squeue -u "$USER" -h -o '%i %N' 2>/dev/null || true)"
  if [ -z "$jobs" ]; then
    return 1
  fi
  while read -r jid host; do
    [ -z "$host" ] && continue
    # Skip any "ggpuXXX,ggpuYYY" multi-node — we want the head node.
    host="${host%%,*}"
    # Try the health endpoint.
    if curl -sf --max-time 3 "http://$host:$PORT/v1/models" >/dev/null 2>&1; then
      echo "$host"
      return 0
    fi
  done <<< "$jobs"
  return 1
}

if [ -z "$NODE" ]; then
  echo "[$(date +%T)] Discovering vLLM node from Slurm jobs…"
  NODE="$(discover_node || true)"
  if [ -z "$NODE" ]; then
    echo "ERROR: no node serving on :$PORT found in any of your Slurm jobs." >&2
    echo "Pass the node hostname explicitly:  bash launch-llama33.sh <node> [<port>]" >&2
    exit 1
  fi
fi

echo "[$(date +%T)] Using vLLM endpoint http://$NODE:$PORT"
curl -sf --max-time 5 "http://$NODE:$PORT/v1/models" \
  | python -c 'import json,sys; d=json.load(sys.stdin); print("  models:", [m["id"] for m in d.get("data",[])])' \
  || { echo "ERROR: endpoint not responding"; exit 1; }

# Patch base_url in the config (idempotent).
sed -i.bak -E "s|base_url: \"http://[^/]+/v1/\"|base_url: \"http://$NODE:$PORT/v1/\"|g" config-llama33-local.yaml
echo "[$(date +%T)] Patched config-llama33-local.yaml -> http://$NODE:$PORT/v1/"
grep base_url config-llama33-local.yaml | sed 's/^/    /'

mkdir -p runsF2-LLAMA3370B logs
echo "[$(date +%T)] Launching 100 games via sim.sh"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}" bash sim.sh config-llama33-local.yaml 2>&1 | tee logs/launch-llama33.$(date +%Y%m%d_%H%M%S).log
echo "[$(date +%T)] Done. Summaries land in runsF2-LLAMA3370B/"
