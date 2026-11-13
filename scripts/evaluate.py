"""Evaluate a predictor on a JSONL dataset and print per field accuracy.

Examples:
    python scripts/evaluate.py data/test.jsonl --predictor ocr
    python scripts/evaluate.py data/test.jsonl --predictor vlm --model-id <id> --limit 5 --show-raw 5
    python scripts/evaluate.py data/test.jsonl --predictor vlm --model-id <id> \
        --adapter outputs/lora --quantization 8bit --latency --out reports/int8.json

Run a new model with --show-raw 5 first and read what it actually wrote. A low
score is usually a formatting problem you can see there.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receipts.benchmark import measure_latency  # noqa: E402
from receipts.data import load_dataset  # noqa: E402
from receipts.metrics import evaluate, format_table  # noqa: E402
from receipts.predictors import make_ocr_predictor, make_vlm_predictor  # noqa: E402
from receipts.vlm_io import DTYPE_NAMES, QUANT_NAMES, add_image_args, image_options_from_args  # noqa: E402


def run(predict, dataset, day_first=False):
    """Predict every receipt. Returns (predictions, metrics, raw_texts)."""
    truths = [t for _, t in dataset]
    preds, raws = [], []
    for image, _ in dataset:
        preds.append(predict(image))
        raws.append(getattr(predict, "last_text", None))
    return preds, evaluate(preds, truths, day_first=day_first), raws

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset", help="JSONL file")
    parser.add_argument("--predictor", choices=["ocr", "vlm"], required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--adapter")
    parser.add_argument("--quantization", choices=[q for q in QUANT_NAMES if q])
    parser.add_argument("--dtype", choices=DTYPE_NAMES, default="auto")
    add_image_args(parser)
    parser.add_argument("--day-first", action="store_true", help="read 12/10/2026 as 12 October")
    parser.add_argument("--limit", type=int, metavar="N", help="use only the first N receipts (for quick trials)")
    parser.add_argument("--latency", action="store_true", help="also measure latency")
    parser.add_argument(
        "--latency-images", type=int, default=50, metavar="N",
        help="time only the first N receipts (default 50), since timing repeats every prediction",
    )
    parser.add_argument("--show-raw", type=int, default=0, metavar="N", help="print the raw output for the first N receipts")
    parser.add_argument("--out", help="write predictions and metrics to this JSON file")
    args = parser.parse_args(argv)

    size_mb = None
    backend = None
    settings = None
    if args.predictor == "vlm":
        if not args.model_id:
            parser.error("--model-id is required for the vlm predictor")
        longest_edge, splitting = image_options_from_args(args)
        predict = make_vlm_predictor(
            args.model_id, args.adapter, args.quantization, args.dtype,
            image_longest_edge=longest_edge, image_splitting=splitting,
        )
        size_mb = getattr(predict, "size_mb", None)
        backend = getattr(predict, "quantization_backend", None)
        settings = getattr(predict, "image_settings", None)
    else:
        predict = make_ocr_predictor()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.latency_images < 1:
        parser.error("--latency-images must be at least 1")
    dataset = load_dataset(args.dataset, limit=args.limit)
    preds, result, raws = run(predict, dataset, day_first=args.day_first)
    print(format_table(args.predictor, result))
    if size_mb is not None:
        print(f"  model size      : {size_mb:.0f} MB")
    if backend:
        print(f"  quantized with  : {backend}")
    if settings:
        print(f"  image settings  : {settings}")
    for i in range(min(args.show_raw, len(raws))):
        print(f"--- raw output, receipt {i} ---\n{raws[i]}\n--- parsed: {preds[i]}")

    report = {"metrics": result, "predictions": preds, "raw_outputs": raws}
    if size_mb is not None:
        report["model_size_mb"] = size_mb
    if backend:
        report["quantization_backend"] = backend
    if settings:
        report["image_settings"] = settings

    def save():
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2))

    # Save scores before timing, so stopping the slow timing step keeps them.
    save()
    if args.latency:
        images = [img for img, _ in dataset][: args.latency_images]
        print(f"  timing {len(images)} receipts (press Ctrl+C to skip; scores are already saved)...")
        try:
            latency = measure_latency(predict, images)
        except KeyboardInterrupt:
            print("  latency skipped (interrupted). Scores were saved without latency.")
        else:
            report["latency"] = latency
            print(
                f"  latency         : median {latency['median_ms']:.0f} ms, "
                f"p95 {latency['p95_ms']:.0f} ms, {latency['images_per_s']:.2f} img/s"
            )
            save()
    return report

if __name__ == "__main__":
    main()
