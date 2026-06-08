#!/bin/bash
# Kimi K2.5 vLLM launcher (anchor model).
#
# 3 nodes × 4 GPUs = 12 GPUs total (TP=4 within a node, PP=3 across nodes).
# Kimi K2.5 is 555 GB on disk; same 3-node layout as the main-repo
# vllmA-Kimi.sh (which this mirrors). This therefore needs 4 nodes total:
# 3 for Kimi + 1 for the opponent.
#
# Kimi-specific vLLM flags vs the generic launcher:
#   --tool-call-parser kimi_k2 / --reasoning-parser kimi_k2  (Kimi formats)
#   --language-model-only + --limit-mm-per-prompt.image 0    (skip the MM head)
#   --mm-encoder-tp-mode data
# TIKTOKEN_RS_CACHE_DIR is injected through --cleanenv (container $HOME is RO).
#
# Usage on the HEAD node (rank 0):
#   bash vllm-kimi.sh <master_node_ip> 0
# Usage on the two WORKER nodes (rank 1 and rank 2):
#   bash vllm-kimi.sh <master_node_ip> 1
#   bash vllm-kimi.sh <master_node_ip> 2
#
set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $(hostname)"

if [ $# -lt 2 ]; then
  echo "usage: bash vllm-kimi.sh <master_node_ip> <node_rank>" >&2
  echo "  node_rank: 0 on the head node, 1/2 on the worker nodes" >&2
  exit 2
fi

MASTER_ADDR="$1"
NODE_RANK="$2"

if [ "$NODE_RANK" -ne 0 ]; then
  HEADLESS="--headless"
else
  HEADLESS=""
fi

export HF_HOME=$PROJECT/hf
export MODEL=${MODEL:-moonshotai/Kimi-K2.5}
export PORT=${PORT:-8080}
export LLM_API_KEY=${LLM_API_KEY:-EMPTY}
export LLM_BASE_URL=http://localhost:${PORT}/v1/

export SIF=$PROJECT/vllm-container/vllm-21.0.sif

module load gcc cuda apptainer nvhpc openmpi
mkdir -p $HF_HOME logs
#  --env GLOO_SOCKET_IFNAME=ib0 \ #  --env NCCL_SOCKET_IFNAME=ib0 \

echo "Starting VLLM server with model $MODEL (master=$MASTER_ADDR, rank=$NODE_RANK, port=$PORT)"
apptainer exec \
  --nv \
  --cleanenv \
  --env TIKTOKEN_RS_CACHE_DIR="$HF_HOME" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=1 \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 3 \
    --nnodes 3 \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
    --disable-custom-all-reduce \
    --async-scheduling \
    --limit-mm-per-prompt.image 0 \
    --mm-encoder-tp-mode data \
    --language-model-only \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    ${HEADLESS}
