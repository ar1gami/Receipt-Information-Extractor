"""Shared model loading and input preparation for the vision-language model.

Everything that imports torch or transformers does so inside a function, so the
rest of the project (and its tests) works on a machine without them.
"""
from typing import Optional

from .vlm_output import PROMPT

DTYPE_NAMES = ("auto", "bfloat16", "float16", "float32")
# 8bit and 4bit use bitsandbytes and need a CUDA GPU. dynamic-int8 runs on CPU.
QUANT_NAMES = (None, "8bit", "4bit", "dynamic-int8")


def build_messages(target: Optional[str] = None) -> list:
    """Chat messages for one receipt. With a target, the answer turn is included."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def auto_model_class():
    """The image-plus-text auto class, under whichever name this version of transformers uses."""
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError(
        "this version of transformers has neither AutoModelForImageTextToText "
        "nor AutoModelForVision2Seq"
    )


def resolve_dtype(name: str):
    """Turn a dtype name into a torch dtype. "auto" picks the best the hardware supports."""
    if name not in DTYPE_NAMES:
        raise ValueError(f"dtype must be one of {DTYPE_NAMES}, got {name!r}")
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            # bfloat16 is fast from compute capability 8.0 (Ampere) up. Older
            # GPUs such as the T4 only emulate it, slowly, so they get float16.
            major, _minor = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    return getattr(torch, name)


def _from_pretrained(cls, model_id, torch_dtype, **kwargs):
    # Newer transformers calls the argument dtype, older ones torch_dtype.
    try:
        return cls.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)


def _image_processor(processor):
    """The image processor if it is SmolVLM style (Idefics3), otherwise None."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "do_image_splitting"):
        return image_processor
    return None


def apply_image_options(processor, longest_edge: Optional[int] = None, splitting: Optional[bool] = None):
    """Set how much of each image the model sees. Returns the processor.

    SmolVLM style models resize an image so its longest side is longest_edge
    pixels, cut it into square tiles (512 pixels for SmolVLM) and add one
    small whole-image view. Every tile costs 64 tokens and a pass through the
    image encoder, so a large longest_edge uses a lot of memory and time. A
    smaller one is cheaper but makes small print harder to read.
    splitting=False skips the tiles and keeps only the small whole-image view,
    which is too coarse for receipts.

    The same settings must be used for training and for inference. They are
    saved with the processor, so a merged model folder remembers them.
    """
    if longest_edge is None and splitting is None:
        return processor
    image_processor = _image_processor(processor)
    if image_processor is None:
        raise ValueError(
            "the image size options only work with SmolVLM style models, whose processor "
            "splits images into tiles. This model's processor has no image splitting."
        )
    if longest_edge is not None:
        if int(longest_edge) < 1:
            raise ValueError("image longest edge must be at least 1")
        image_processor.size = {"longest_edge": int(longest_edge)}
    if splitting is not None:
        image_processor.do_image_splitting = bool(splitting)
    return processor


def image_settings(processor) -> Optional[dict]:
    """The current image settings of a SmolVLM style processor, or None for other models."""
    image_processor = _image_processor(processor)
    if image_processor is None:
        return None
    size = image_processor.size
    longest = size.get("longest_edge") if isinstance(size, dict) else getattr(size, "longest_edge", None)
    tile = getattr(image_processor, "max_image_size", None)
    tile = tile.get("longest_edge") if isinstance(tile, dict) else getattr(tile, "longest_edge", None)
    return {
        "longest_edge": longest,
        "tile_size": tile,
        "splitting": bool(image_processor.do_image_splitting),
    }


def add_image_args(parser):
    """The two command line flags every script that loads a model shares."""
    parser.add_argument(
        "--image-longest-edge", type=int, metavar="PIXELS",
        help="SmolVLM style models only: resize each image so its longest side is this many pixels "
             "(the model's default is 2048). Lower uses less memory.",
    )
    parser.add_argument(
        "--no-image-splitting", action="store_true",
        help="SmolVLM style models only: do not cut the image into tiles (too coarse for receipts)",
    )


def image_options_from_args(args):
    """(longest_edge, splitting) from parsed arguments. splitting is None when not changed."""
    return args.image_longest_edge, (False if args.no_image_splitting else None)


def image_options_from_env(environ):
    """(longest_edge, splitting) from IMAGE_LONGEST_EDGE and IMAGE_SPLITTING ("on" or "off")."""
    raw = (environ.get("IMAGE_LONGEST_EDGE") or "").strip()
    try:
        longest = int(raw) if raw else None
    except ValueError:
        raise RuntimeError(f"IMAGE_LONGEST_EDGE must be a whole number, got {raw!r}")
    flag = (environ.get("IMAGE_SPLITTING") or "").strip().lower()
    if flag not in ("", "on", "off"):
        raise RuntimeError(f"IMAGE_SPLITTING must be on or off, got {flag!r}")
    return longest, (None if not flag else flag == "on")


def adapter_image_options(adapter_path, longest_edge=None, splitting=None):
    """Fill in image options from the adapter's training run when none were given.

    scripts/finetune.py writes run_config.json next to the adapter. Using the
    same image size as training avoids a silent mismatch, which lowers accuracy
    without any error. Options given explicitly always win.
    """
    if longest_edge is not None or splitting is not None or not adapter_path:
        return longest_edge, splitting
    import json
    from pathlib import Path

    path = Path(adapter_path) / "run_config.json"
    if not path.is_file():
        return longest_edge, splitting
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return longest_edge, splitting
    return config.get("image_longest_edge"), (False if config.get("no_image_splitting") else None)


def load_model(
    model_id: str,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    image_longest_edge: Optional[int] = None,
    image_splitting: Optional[bool] = None,
):
    """Return (processor, model).

    model_id can be a Hugging Face hub id or a local folder.
    quantization: None, "8bit" or "4bit" (bitsandbytes, CUDA only), or
    "dynamic-int8" (PyTorch dynamic quantization of the linear layers, CPU).
    image_longest_edge, image_splitting: see apply_image_options.
    """
    if quantization not in QUANT_NAMES:
        raise ValueError(f"quantization must be one of {QUANT_NAMES}, got {quantization!r}")

    import torch
    from transformers import AutoProcessor

    torch_dtype = torch.float32 if quantization == "dynamic-int8" else resolve_dtype(dtype)
    kwargs = {}
    if quantization in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        if quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        kwargs["device_map"] = "auto"

    processor = AutoProcessor.from_pretrained(model_id)
    apply_image_options(processor, image_longest_edge, image_splitting)
    model = _from_pretrained(auto_model_class(), model_id, torch_dtype, **kwargs)

    if quantization == "dynamic-int8":
        model, backend = quantize_int8_cpu(model)
        model.quantization_backend = backend
    elif quantization is None:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def quantize_int8_cpu(model):
    """Dynamic int8 quantization of the linear layers, for CPU. Returns (model, backend name).

    PyTorch's own quantize_dynamic is used when it exists, because it was the
    fastest option measured for generating one token at a time (about 4.6x
    faster than float32 on a stack of linear layers, against 1.04x for torchao's
    dynamic int8). PyTorch has deprecated it, so torchao is the fallback for
    versions where it is gone. torchao still shrinks the model, but it gives
    little speedup when generating token by token.
    """
    import torch

    if hasattr(torch.ao.quantization, "quantize_dynamic"):
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
        return model, "torch.ao"
    try:
        import torchao
        import torchao.quantization as tq
    except ImportError:
        tq = None
    if tq is not None:
        config = getattr(tq, "Int8DynamicActivationInt8WeightConfig", None) or getattr(
            tq, "int8_dynamic_activation_int8_weight", None
        )
        if config is not None:
            tq.quantize_(model, config())
            return model, f"torchao {torchao.__version__}"
    raise RuntimeError(
        "this PyTorch no longer has torch.ao.quantization.quantize_dynamic. "
        "Install torchao (pip install torchao) to quantize on CPU, or use "
        "--quantization 8bit on a CUDA GPU."
    )


def move_to(batch, device, dtype):
    """Move processor output to a device. Floating point tensors (the pixel
    values) are also cast to the model's dtype, so a bfloat16 model does not
    receive float32 images."""
    out = {}
    for key, value in batch.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            out[key] = value.to(device=device, dtype=dtype)
        elif hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def model_size_mb(model) -> float:
    """Size of the model weights in megabytes, measured by saving the state dict.

    Serializing is slower than adding up parameter sizes, but it is the one
    measurement that is fair across float, bitsandbytes and dynamic int8 models.
    """
    import os
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "weights.pt")
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / 1e6
