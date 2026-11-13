"""Predictors share one interface: predict(PIL image) -> dict or None.

None means the model produced no usable output. A predictor may also set a
last_text attribute holding the raw text it produced for the latest image, a
size_mb attribute holding the model's weight size, a quantization_backend
attribute naming the library that did the quantizing, and an image_settings
attribute with the image size options in use. The evaluation script records
them when they exist.
"""
from typing import Callable, Optional

from .ocr_baseline import ocr_predict
from .vlm_output import parse_model_output

Predictor = Callable[[object], Optional[dict]]

def make_ocr_predictor() -> Predictor:
    return ocr_predict

def make_vlm_predictor(
    model_id: str,
    adapter_path: Optional[str] = None,
    quantization: Optional[str] = None,
    dtype: str = "auto",
    max_new_tokens: int = 128,
    image_longest_edge: Optional[int] = None,
    image_splitting: Optional[bool] = None,
) -> Predictor:
    """Load a vision language model and return a predict function.

    model_id: hub id or local folder (for example the output of merge_adapter.py).
    adapter_path: folder holding LoRA weights from scripts/finetune.py.
    quantization: see receipts.vlm_io.load_model.
    image_longest_edge, image_splitting: see receipts.vlm_io.apply_image_options.
    When neither is given and an adapter is used, the settings of the training
    run saved next to the adapter are used.
    """
    if adapter_path and quantization == "dynamic-int8":
        raise ValueError(
            "dynamic-int8 cannot be combined with an adapter. Merge the adapter first "
            "with scripts/merge_adapter.py, then load the merged folder with dynamic-int8."
        )

    from .vlm_io import adapter_image_options, load_model

    image_longest_edge, image_splitting = adapter_image_options(
        adapter_path, image_longest_edge, image_splitting
    )
    processor, model = load_model(model_id, dtype, quantization, image_longest_edge, image_splitting)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return predictor_from_model(processor, model, max_new_tokens)

def predictor_from_model(processor, model, max_new_tokens: int = 128) -> Predictor:
    """Wrap a processor and a model that are already loaded as a predictor.
    Generation is greedy, so the same model and image always give the same text.
    """
    import torch

    from .vlm_io import build_messages, image_settings, model_size_mb, move_to

    model.eval()
    prompt = processor.apply_chat_template(
        build_messages(), add_generation_prompt=True, tokenize=False
    )

    def predict(image) -> Optional[dict]:
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = move_to(inputs, model.device, model.dtype)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        predict.last_text = text
        return parse_model_output(text)

    predict.last_text = None
    predict.image_settings = image_settings(processor)
    predict.quantization_backend = getattr(model, "quantization_backend", None)
    try:
        predict.size_mb = model_size_mb(model)
    except Exception:  # a size is nice to have, never worth failing the load
        predict.size_mb = None
    return predict
