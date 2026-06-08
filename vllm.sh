#!/bin/bash

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $(hostname)"

# Set these environment variables or add them to your .env file
export HF_HOME=$PROJECT/hf
export MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
export PORT=${PORT:-8081}
export LLM_API_KEY=$LLM_API_KEY
export LLM_BASE_URL=http://localhost:${PORT}/v1/
# Extra arguments forwarded verbatim to `vllm serve` (e.g. --tokenizer-mode auto
# for Mistral models, or --max-model-len overrides).
export EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}

export SIF=$PROJECT/vllm-container/vllm-21.0.sif

# export VLLM_USE_MODELSCOPE=true

module load gcc cuda apptainer
mkdir -p $HF_HOME logs

echo "Starting VLLM server with model $MODEL"
apptainer exec \
  --nv \
  --cleanenv \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve $MODEL \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
    --disable-custom-all-reduce $EXTRA_VLLM_ARGS
