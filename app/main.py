"""FastAPI service: upload a receipt image, get vendor, date and total back.

Configuration through environment variables:
    PREDICTOR     "ocr" (default) or "vlm"
    MODEL_ID      Hugging Face model id (vlm only)
    ADAPTER_PATH  folder with LoRA weights (vlm only, optional). It must be a folder on this
                  machine holding adapter_config.json, not a hub id, as the image size
                  settings of our training run are read from the folder next to it.
    QUANTIZATION  "8bit", "4bit" or "dynamic int8" (vlm only, optional)
    DTYPE         "auto" (default), "bfloat16", "float16" or "float32" (vlm only)
    IMAGE_LONGEST_EDGE  pixels for the longest image side, SmolVLM style models (optional)
    IMAGE_SPLITTING     "on" or "off", SmolVLM style models (optional)
    MAX_UPLOAD_MB largest accepted upload, default 10
"""
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from receipts.predictors import make_ocr_predictor, make_vlm_predictor
from receipts.vlm_io import image_options_from_env

class ExtractionResult(BaseModel):
    vendor: Optional[str] = None
    date: Optional[str] = None
    total: Optional[str] = None
    predictor: str
    latency_ms: float

def predictor_from_env() -> tuple:
    name = os.environ.get("PREDICTOR", "ocr").lower()
    if name == "ocr":
        return name, make_ocr_predictor()
    if name == "vlm":
        model_id = os.environ.get("MODEL_ID")
        if not model_id:
            raise RuntimeError("MODEL_ID must be set when PREDICTOR=vlm")
        longest_edge, splitting = image_options_from_env(os.environ)
        adapter_path = os.environ.get("ADAPTER_PATH") or None
        if adapter_path and not (Path(adapter_path) / "adapter_config.json").is_file():
            # Without this check a wrong path would be treated as a hub id by PEFT, which gives a
            # error at best, or the adapter's image settings would be skipped at worst.
            raise RuntimeError(
                f"ADAPTER_PATH={adapter_path!r} is not an adapter folder (no adapter_config.json in it). "
                "It must be a folder on this machine. In Docker, put the adapter in models/lora before "
                "building, or mount it with -v. Set ADAPTER_PATH to an empty value to serve the base model."
            )
        return name, make_vlm_predictor(
            model_id,
            adapter_path=adapter_path,
            quantization=os.environ.get("QUANTIZATION") or None,
            dtype=os.environ.get("DTYPE") or "auto",
            image_longest_edge=longest_edge,
            image_splitting=splitting,
        )
    raise RuntimeError(f"unknown PREDICTOR {name!r}")


def create_app(predictor: Optional[Callable] = None, predictor_name: str = "custom") -> FastAPI:
    """Build the app. Pass a predictor to override the environment (used by tests)."""
    state = {}
    lock = threading.Lock()
    max_bytes = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # The model is loaded once here
        if predictor is not None:
            state["predict"], state["name"] = predictor, predictor_name
        else:
            state["name"], state["predict"] = predictor_from_env()
        yield
        state.clear()

    app = FastAPI(title="Receipt extractor", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "predictor": state.get("name")}

    # A def endpoint runs in FastAPI's thread pool, so slow model calls
    # don't  block event loop.
    @app.post("/extract", response_model=ExtractionResult)
    def extract(file: UploadFile = File(...)):
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="file is not a readable image")

        start = time.perf_counter()
        try:
            with lock:
                record = state["predict"](image)
        except Exception:
            raise HTTPException(status_code=500, detail="prediction failed")
        latency_ms = (time.perf_counter() - start) * 1000

        if record is None:
            raise HTTPException(status_code=422, detail="model produced no usable output")
        return ExtractionResult(
            vendor=record.get("vendor"),
            date=record.get("date"),
            total=record.get("total"),
            predictor=state["name"],
            latency_ms=round(latency_ms, 2),
        )

    return app
app = create_app()
