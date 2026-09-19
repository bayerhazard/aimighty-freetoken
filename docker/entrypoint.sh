#!/usr/bin/env bash
set -euo pipefail

export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

# FreeToken's ZeroMQ IPC and flashinfer JIT need a writable /tmp; and the linker
# needs a libcuda.so stub, which the NVIDIA container runtime injects at runtime.
for c in /usr/lib/x86_64-linux-gnu/libcuda.so \
         /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
         /usr/lib64/libcuda.so.1 \
         /usr/local/cuda/lib64/stubs/libcuda.so; do
  if [ -e "$c" ]; then
    ln -sfn "$c" /opt/cuda13/lib64/stubs/libcuda.so
    break
  fi
done

: "${MODEL_REPO:=nvidia/Qwen3.6-35B-A3B-NVFP4}"
: "${SERVED_MODEL_NAME:=Qwen3.6-35B-A3B}"
: "${MODELS_DIR:=/shared-models/llms}"

MODEL_DIR="${MODELS_DIR}/${MODEL_REPO##*/}"
mkdir -p "$MODELS_DIR"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "==> downloading ${MODEL_REPO} to ${MODEL_DIR} (this can take a while)"
  hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}" --max-workers 2
else
  echo "==> model already present at ${MODEL_DIR}"
fi

echo "==> starting FreeToken (offload MoE, ${NUM_TOKENS:-200000} KV, ${MAX_RUNNING_REQUESTS:-2} parallel)"
exec ft serve \
  --model "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 --port 1919 \
  --moe-strategy "${MOE_STRATEGY:-offload}" \
  --moe-cache-size "${MOE_CACHE_SIZE:-8800}" \
  --memory-ratio "${MEMORY_RATIO:-0.95}" \
  --num-tokens "${NUM_TOKENS:-200000}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-2}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-2}" \
  --attention-backend "${ATTENTION_BACKEND:-triton}" \
  --ple-backend "${PLE_BACKEND:-pinned}" \
  ${EXTRA_ARGS:-}
