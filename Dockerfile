# Receipt extractor API.
#
# 1. The OCR baseline (small, the default):
#      docker build -t receipt-extractor .
#      docker run --rm -p 8080:8080 receipt-extractor
#
# 2. The fine tuned vision-language model, on a CPU. First put the LoRA adapter
#    folder from the Kaggle run (the one holding adapter_config.json) in
#    models/lora. Then:
#      docker build --build-arg VARIANT=vlm -t receipt-extractor-vlm .
#      docker run --rm -p 8080:8080 receipt-extractor-vlm
#    This image installs CPU only PyTorch and downloads the base model from the
#    Hugging Face hub when the container starts.
#
# 3. The same as 2, with the base model downloaded while building:
#      docker build --build-arg VARIANT=vlm \
#          --build-arg BASE_MODEL=HuggingFaceTB/SmolVLM-500M-Instruct \
#          -t receipt-extractor-vlm .
#    The image is about 1 GB larger, but it starts without network access. The
#    build also loads the model and the adapter once, exactly as the server does
#    at start up, so a wrong adapter folder fails the build and not the deployment.
#
# 4. The same as 3, with the adapter downloaded from a Hugging Face model repo
#    instead of read from models/lora (the repo has to be public):
#      docker build --build-arg VARIANT=vlm \
#          --build-arg BASE_MODEL=HuggingFaceTB/SmolVLM-500M-Instruct \
#          --build-arg ADAPTER_REPO=<user>/receipt-extractor-lora \
#          -t receipt-extractor-vlm .
#    ADAPTER_REVISION=<commit hash> pins the download to one version of the repo.
FROM python:3.13-slim

ARG VARIANT=ocr
ARG BASE_MODEL=
ARG ADAPTER_REPO=
ARG ADAPTER_REVISION=

# PREDICTOR, MODEL_ID and ADAPTER_PATH are what app/main.py reads. They can be
# overridden with docker run -e. The hub cache lives in /opt/hf so that it is
# the same place at build time and at run time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PREDICTOR=${VARIANT} \
    MODEL_ID=${BASE_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct} \
    ADAPTER_PATH=/srv/models/lora

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-vlm.txt ./
# CPU only PyTorch is installed first, from the PyTorch CPU index, so that pip
# cannot pick the much larger CUDA build from PyPI. The last check makes the
# build fail if a CUDA build got in anyway.
RUN case "$VARIANT" in ocr|vlm) ;; *) echo "VARIANT must be ocr or vlm, got $VARIANT" >&2; exit 1;; esac \
    && if [ -n "$BASE_MODEL" ] && [ "$VARIANT" != "vlm" ]; then echo "BASE_MODEL needs VARIANT=vlm" >&2; exit 1; fi \
    && if [ -n "$ADAPTER_REPO" ] && [ "$VARIANT" != "vlm" ]; then echo "ADAPTER_REPO needs VARIANT=vlm" >&2; exit 1; fi \
    && pip install -r requirements.txt \
    && if [ "$VARIANT" = "vlm" ]; then \
         pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
         && pip install -r requirements-vlm.txt \
         && python -c "import torch; assert torch.version.cuda is None, 'a CUDA build of PyTorch was installed'"; \
       fi
       
COPY receipts ./receipts
COPY app ./app
COPY models ./models

# Only when ADAPTER_REPO is given: replace models/lora with the adapter from the hub. This
# runs as root, before USER, because the models folder is owned by root.
RUN if [ -n "$ADAPTER_REPO" ]; then \
      rm -rf "$ADAPTER_PATH" \
      && python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2], revision=sys.argv[3] or None)" "$ADAPTER_REPO" "$ADAPTER_PATH" "$ADAPTER_REVISION"; \
    fi

RUN useradd --create-home appuser \
    && mkdir -p /opt/hf \
    && chown appuser /opt/hf
USER appuser

# Only when BASE_MODEL is given: download the base model now and load it with
# the adapter, through the same function the server uses at start up.
RUN if [ -n "$BASE_MODEL" ]; then \
      python -c "from app.main import predictor_from_env; print('loaded', predictor_from_env()[0])"; \
    fi
# From here on a baked image never asks the hub for anything.
ENV HF_HUB_OFFLINE=${BASE_MODEL:+1}

# Cloud Run sets PORT. The default is for running the container locally.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
