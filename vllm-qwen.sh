#!/bin/bash
# Qwen 3.5 397B A17B vLLM launcher (anchor model).
#
# 3 nodes × 4 GPUs = 12 GPUs total (TP=4 within a node, PP=3 across nodes).
#
# Why 3 nodes and not 2: Qwen 3.5 397B A17B weights are 752 GB on disk
# (bf16 as shipped — Qwen has not released an FP8 build). 2 nodes give
# 8 × 80 GB = 640 GB raw VRAM, which is less than the weight size before
# any KV cache is allocated, so the model OoMs at load time on 2 nodes.
# 3 nodes give 960 GB raw → 864 GB at 0.9 util → fits weights with ~110 GB
# of KV-cache headroom, comfortable for our batch sizes.
#
# The existing main-repo vllmA-Qwen.sh is also TP=4 PP=3 nnodes=3 for the
# same reason.
#
# Total cluster cost: 3 anchor nodes + 1 opponent node = 4 nodes = 16 GPUs.
#
# Usage on the HEAD node (rank 0):
#   bash vllm-qwen.sh <master_node_ip> 0
# Usage on the two WORKER nodes (rank 1 and rank 2):
#   bash vllm-qwen.sh <master_node_ip> 1
#   bash vllm-qwen.sh <master_node_ip> 2
#
# master_node_ip is the head node's hostname (resolvable on the cluster).
#
set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $(hostname)"

if [ $# -lt 2 ]; then
  echo "usage: bash vllm-qwen.sh <master_node_ip> <node_rank>" >&2
  echo "  node_rank: 0 on the head node, 1 on the worker node" >&2
  exit 2
fi

MASTER_ADDR="$1"
NODE_RANK="$2"

# Worker ranks run vLLM in headless mode (no HTTP listener; head node serves).
if [ "$NODE_RANK" -ne 0 ]; then
  HEADLESS="--headless"
else
  HEADLESS=""
fi

export HF_HOME=$PROJECT/hf
export MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B}
export PORT=${PORT:-8080}
export LLM_API_KEY=${LLM_API_KEY:-EMPTY}
export LLM_BASE_URL=http://localhost:${PORT}/v1/

export SIF=$PROJECT/vllm-container/vllm-21.0.sif

module load gcc cuda apptainer nvhpc openmpi
mkdir -p $HF_HOME logs

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
    --language-model-only \
    --reasoning-parser qwen3 \
    ${HEADLESS}
