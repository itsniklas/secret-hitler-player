#!/bin/bash
# Mistral-specific vLLM launcher.
# vLLM 21.x auto-selects --tokenizer-mode mistral for Mistral models, which
# rejects every `chat_template_kwargs` extra body field. The simulator's
# get_completion path sends chat_template_kwargs={"thinking": True}
# (needed for reasoning-enabled anchors like DeepSeek / Kimi), so Mistral
# 415s on every request. Forcing --tokenizer-mode auto switches to the HF
# tokenizer which silently ignores unsupported extras.
#
# Default port is 8080 (the Alice endpoint port used by the neutral-theme
# configs); override with `PORT=...`.
#
# Example:
#   bash vllm-mistral.sh

export MODEL="${MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"
export PORT="${PORT:-8080}"
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-} --tokenizer-mode auto"
exec bash "$(dirname "$0")/vllm.sh"
