"""LoRA fine-tuning of a vision-language model for receipt extraction.

Run this on a GPU (Colab, Kaggle or similar), not on a laptop CPU.

    pip install -r requirements-train.txt
    python scripts/finetune.py data/train.jsonl --model-id <hub id> \
        --output outputs/lora --epochs 2 --wandb

Use a training file that does NOT contain your test receipts. A validation
split is carved out of the training file automatically and the test set is
never read here.

Start with --max-examples 20 --epochs 1 to check that the loss moves before a
full run. The loop was checked on a tiny random model on CPU. It has not been
run on a real model, so model specific details such as the chat template are
the most likely thing to need adjusting.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receipts.data import load_jsonl  # noqa: E402
from receipts.training_utils import split_rows  # noqa: E402
from receipts.vlm_io import DTYPE_NAMES, add_image_args, image_options_from_args, image_settings  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("train_jsonl")
    p.add_argument("--model-id", required=True)
    p.add_argument("--output", default="outputs/lora")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8, help="receipts per optimizer step (batch size is 1)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--dtype", choices=DTYPE_NAMES, default="auto",
                   help="auto uses bfloat16 where the GPU supports it, otherwise float16, and float32 on CPU")
    add_image_args(p)
    p.add_argument("--freeze-vision", action="store_true",
                   help="do not train the image encoder: much less memory and faster, LoRA only on the language model")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="trade speed for memory by recomputing activations in the backward pass")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-examples", type=int, help="use only the first N training rows (for smoke runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model

    from receipts.training import freeze_vision, prompt_lengths, train
    from receipts.vlm_io import load_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = Path(args.train_jsonl).parent
    rows = load_jsonl(args.train_jsonl)
    if args.max_examples:
        rows = rows[: args.max_examples]
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    print(f"train {len(train_rows)} receipts, validation {len(val_rows)} receipts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found, training will be extremely slow", file=sys.stderr)
    longest_edge, splitting = image_options_from_args(args)
    processor, model = load_model(
        args.model_id, args.dtype, image_longest_edge=longest_edge, image_splitting=splitting
    )
    settings = image_settings(processor)
    if settings:
        print(f"image settings {settings}")
    print(f"prompt tokens for the first receipts: {prompt_lengths(processor, train_rows, base)}")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    if args.freeze_vision:
        print(f"froze {freeze_vision(model):,} image encoder parameters")
    model.print_trainable_parameters()

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="receipt-extractor", config=vars(args))

    def log(metrics, step):
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()))
        if run:
            run.log(metrics, step=step)

    history = train(
        model,
        processor,
        train_rows,
        val_rows,
        base,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        device=device,
        seed=args.seed,
        log=log,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved adapter to {out}")
    if run:
        run.finish()
    return history


if __name__ == "__main__":
    main()
