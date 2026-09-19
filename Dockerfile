# AIM Qwen3.6 35B A3B FT — FreeToken serving image (CUDA 13, sm_120)
# Build (no GPU required): docker buildx build --platform linux/amd64 -t ghcr.io/bayerhazard/freetoken:26.9.1 .
FROM python:3.12-slim-bookworm

# Pinned FreeToken nightly pair (runtime + prebuilt kernel cache). Do not use the
# moving `nightly` tag in production; these URLs are the pinned build stamp.
ARG FT_RUNTIME_WHEEL="https://github.com/FlashML-org/FreeToken/releases/download/nightly/freetoken-0.1.3%2Bgcc1f5c2c9-cp312-cp312-linux_x86_64.whl"
ARG FT_KERNEL_WHEEL="https://github.com/FlashML-org/FreeToken/releases/download/nightly/freetoken_kernel_cache-0.1.3%2Bcu130.gcc1f5c2c9-py3-none-linux_x86_64.whl"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/freetoken/venv
ENV PATH=/opt/freetoken/venv/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "freetoken[accel] @ ${FT_RUNTIME_WHEEL}" \
        "${FT_KERNEL_WHEEL}" \
        "nvidia-cuda-nvcc==13.4.92" \
        "nvidia-cuda-crt==13.4.92" \
        "nvidia-cuda-runtime==13.4.92" \
        "nvidia-nvvm==13.4.92" \
        ninja \
        huggingface_hub

# Build a CUDA_HOME from the pip CUDA wheels (mirrors the validated host setup).
# The libcuda stub symlink is (re)pointed at runtime by the entrypoint.
RUN SP=/opt/freetoken/venv/lib/python3.12/site-packages/nvidia/cu13 \
    && mkdir -p /opt/cuda13/lib64/stubs \
    && ln -sfn "$SP/bin" /opt/cuda13/bin \
    && ln -sfn "$SP/include" /opt/cuda13/include \
    && ln -sfn "$SP/nvvm" /opt/cuda13/nvvm \
    && ln -sfn "$SP/lib/libcudart.so.13" /opt/cuda13/lib64/libcudart.so \
    && ln -sfn "$SP/lib/libcudart_static.a" /opt/cuda13/lib64/libcudart_static.a \
    && ln -sfn /opt/cuda13/lib64 /opt/cuda13/lib

ENV CUDA_HOME=/opt/cuda13
ENV PATH=/opt/cuda13/bin:/opt/freetoken/venv/bin:$PATH
ENV TMPDIR=/tmp
ENV HF_HOME=/shared-models/huggingface

COPY docker/entrypoint.sh /opt/freetoken/entrypoint.sh
RUN chmod +x /opt/freetoken/entrypoint.sh

ENTRYPOINT ["/opt/freetoken/entrypoint.sh"]
