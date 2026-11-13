"""Merge LoRA weights into the base model and save a standalone model folder.

    python scripts/merge_adapter.py --model-id <hub id> --adapter outputs/lora --output outputs/merged

The merged folder can be used anywhere a model id is accepted, and it can be
quantized at load time:

    python scripts/evaluate.py data/test.jsonl --predictor vlm --model-id outputs/merged --quantization dynamic-int8

Merging removes the adapter's extra computation at inference time. Merge into
the full precision model, not a quantized one. The image size used in training
is read from the adapter folder and saved with the merged model's processor.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receipts.vlm_io import DTYPE_NAMES, add_image_args, image_options_from_args  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=DTYPE_NAMES, default="auto")
    add_image_args(p)
    args = p.parse_args(argv)

    from peft import PeftModel

    from receipts.vlm_io import adapter_image_options, load_model

    # With no image flags, use the image size the adapter was trained with.
    longest_edge, splitting = adapter_image_options(args.adapter, *image_options_from_args(args))
    processor, model = load_model(
        args.model_id, args.dtype, image_longest_edge=longest_edge, image_splitting=splitting
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"saved merged model to {out}")
    return out


if __name__ == "__main__":
    main()
