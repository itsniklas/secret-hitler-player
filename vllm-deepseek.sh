#!/bin/bash
# DeepSeek V3.1 Terminus vLLM launcher (anchor model).
#
# 3 nodes × 4 GPUs = 12 GPUs total (TP=4 within a node, PP=3 across nodes).
# Same layout as the main-repo vllmA-DeepSeek.sh. This therefore needs
# 4 nodes total: 3 for DeepSeek + 1 for the opponent.
#
# DeepSeek-specific vLLM flags vs the generic launcher:
#   --enable-expert-parallel              (MoE expert parallelism)
#   --distributed-executor-backend mp     (multi-proc executor required for EP)
#   --reasoning-parser deepseek_v3        (DeepSeek-V3 reasoning channel)
#   --decode-context-parallel-size 4      (was --dcp 4 in older vLLM; renamed
#                                          in v21 because --dcp is now ambiguous
#                                          with --dcp-comm-backend etc.)
#   --quantization fp8                    (in W3xJ)
#   --language-model-only                 (in W3xJ)
#
# Matches the W3xJ historical config that ran DeepSeek V3.1 successfully on
# this cluster, including --disable-custom-all-reduce. The marlin_gemm crash
# in MLA prefill_context is hypothesized to go away when one of:
# (a) decode-context-parallel-size=4 reshapes the kv_b_proj input;
# (b) --disable-custom-all-reduce routes the TP all-reduce through the
#     stock NCCL path, producing the dtype Marlin expects.
#
# Usage on the HEAD node (rank 0):
#   bash vllm-deepseek.sh <master_node_ip> 0
# Usage on the two WORKER nodes (rank 1 and rank 2):
#   bash vllm-deepseek.sh <master_node_ip> 1
#   bash vllm-deepseek.sh <master_node_ip> 2
#
set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $(hostname)"

if [ $# -lt 2 ]; then
  echo "usage: bash vllm-deepseek.sh <master_node_ip> <node_rank>" >&2
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
export MODEL=${MODEL:-deepseek-ai/DeepSeek-V3.1-Terminus}
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
  --env GLOO_SOCKET_IFNAME=ib0 \
  --env NCCL_SOCKET_IFNAME=ib0 \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --decode-context-parallel-size 4 \
    --pipeline-parallel-size 3 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --nnodes 3 \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
    --disable-custom-all-reduce \
    --async-scheduling \
    --quantization fp8 \
    --language-model-only \
    --reasoning-parser deepseek_v3 \
    ${HEADLESS}
