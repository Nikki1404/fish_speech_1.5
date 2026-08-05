# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04


ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=1800 \
    HF_HUB_ETAG_TIMEOUT=120

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        portaudio19-dev \
        libasound2-dev \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone \
    --branch v1.5.1 \
    --depth 1 \
    https://github.com/fishaudio/fish-speech.git \
    /app/fish-speech

WORKDIR /app/fish-speech

RUN python3.10 -m venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}"

RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# Install a mutually compatible Torch stack for CUDA 12.4.
RUN python -m pip install \
    --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Keep later dependency installation from replacing the CUDA 12.4 wheels.
RUN printf '%s\n' \
    'torch==2.4.1' \
    'torchvision==0.19.1' \
    'torchaudio==2.4.1' \
    > /app/pytorch-constraints.txt

RUN python -m pip install \
    --no-cache-dir \
    --constraint /app/pytorch-constraints.txt \
    -e ".[stable]"

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --constraint /app/pytorch-constraints.txt \
    -r /app/requirements.txt

# Verify the Torch stack during the image build.
RUN python - <<'PY'
import torch
import torchaudio
import torchvision

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
print("torchvision:", torchvision.__version__)

assert torch.__version__.startswith("2.4.1")
assert torchaudio.__version__.startswith("2.4.1")
assert torch.version.cuda == "12.4"
PY

ARG MODEL_REPO=fishaudio/fish-speech-1.5
ARG MODEL_REVISION=main

RUN mkdir -p /app/checkpoints/fish-speech-1.5 && \
    hf download "${MODEL_REPO}" \
        --revision "${MODEL_REVISION}" \
        --local-dir /app/checkpoints/fish-speech-1.5 \
        --max-workers 1 && \
    test -f /app/checkpoints/fish-speech-1.5/model.pth && \
    test -f /app/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth

COPY app /app/fish-speech/app

WORKDIR /app/fish-speech

ENV DEVICE=cuda \
    HALF=1 \
    COMPILE=0 \
    MAX_TEXT_LENGTH=1000 \
    LLAMA_CHECKPOINT_PATH=/app/checkpoints/fish-speech-1.5 \
    DECODER_CHECKPOINT_PATH=/app/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
    DECODER_CONFIG_NAME=firefly_gan_vq \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

ENTRYPOINT []

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
