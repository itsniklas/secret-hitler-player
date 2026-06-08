#!/bin/bash
# GPT-OSS-120B-specific vLLM launcher (used by the neutral-theme gptoss arm).
#
# Two GPT-OSS quirks that the stock vllm.sh does NOT handle:
#
#   1. --language-model-only
#      gpt-oss-120b ships multimodal head/encoder weights that vLLM tries to
#      load by default. We only need the text head for the simulator, and
#      the multimodal load path 410s with vLLM 21.0. The flag skips it.
#
#   2. TIKTOKEN_RS_CACHE_DIR (injected through apptainer --cleanenv)
#      gpt-oss uses a tiktoken tokenizer that writes a small BPE cache on
#      first use. The container's default $HOME is read-only, so without
#      this var the tokenizer crashes mid-startup. We point it at $HF_HOME
#      which is already bind-mounted writable.
#
# Defaults to PORT 8080 (the Alice endpoint). Override with PORT=...
#
# Example:
#   bash vllm-gptoss.sh                     # PORT 8080, TP=4
#   PORT=8082 bash vllm-gptoss.sh           # different port
#
set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $(hostname)"

export HF_HOME=$PROJECT/hf
export MODEL=${MODEL:-openai/gpt-oss-120b}
export PORT=${PORT:-8080}
export LLM_API_KEY=${LLM_API_KEY:-EMPTY}
export LLM_BASE_URL=http://localhost:${PORT}/v1/

export SIF=$PROJECT/vllm-container/vllm-21.0.sif

module load gcc cuda apptainer
mkdir -p $HF_HOME logs

echo "Starting VLLM server with model $MODEL on port $PORT"
apptainer exec \
  --nv \
  --cleanenv \
  --env TIKTOKEN_RS_CACHE_DIR="$HF_HOME" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --language-model-only \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
    --disable-custom-all-reduce
