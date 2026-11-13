"""LoRA fine-tuning loop for the receipt model.

Kept apart from scripts/finetune.py so it can be tested with a tiny model.
torch is imported inside the functions that need it.
"""
import math
import random
from pathlib import Path
from typing import Callable, List, Optional

from .data import load_image
from .training_utils import build_target, mask_prompt_labels
from .vlm_io import build_messages, move_to


def encode_example(processor, image, target: str):
    """Tokenize one example and mask the prompt so the loss covers the answer only.

    Returns (batch, labels). The prompt length comes from tokenizing the prompt
    alone, so the check below confirms the chat template really starts the full
    text with those same tokens. If it does not, masking would hide the wrong
    tokens and training would quietly learn the wrong thing.
    """
    prompt_text = processor.apply_chat_template(
        build_messages(), add_generation_prompt=True, tokenize=False
    )
    full_text = processor.apply_chat_template(
        build_messages(target), add_generation_prompt=False, tokenize=False
    )
    prompt_ids = processor(text=prompt_text, images=[image], return_tensors="pt")["input_ids"][0].tolist()
    batch = processor(text=full_text, images=[image], return_tensors="pt")
    full_ids = batch["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    if full_ids[:prompt_len] != prompt_ids:
        raise ValueError(
            "the chat template changes the prompt tokens when the answer is added, "
            "so the prompt cannot be masked by length. Check the model's chat template."
        )
    if len(full_ids) <= prompt_len:
        raise ValueError("the answer produced no tokens")
    return batch, mask_prompt_labels(full_ids, prompt_len)


def prompt_lengths(processor, rows: List[dict], base_dir, n: int = 5) -> List[int]:
    """Token count of the prompt (image tokens included) for the first n rows.

    A SmolVLM style model turns each image tile into many tokens, so this is the
    quickest way to see how heavy an image size setting is before training.
    """
    prompt = processor.apply_chat_template(build_messages(), add_generation_prompt=True, tokenize=False)
    lengths = []
    for row in rows[:n]:
        image = load_image(Path(base_dir), row)
        lengths.append(int(processor(text=prompt, images=[image], return_tensors="pt")["input_ids"].shape[1]))
    return lengths


def freeze_vision(model) -> int:
    """Turn off training for every parameter of the image encoder. Returns how many were frozen.

    With nothing in the image encoder to train, PyTorch does not keep its
    activations for the backward pass, which saves a lot of memory and time.
    The LoRA layers added to the language model still train.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if "vision" in name.lower() and param.requires_grad:
            param.requires_grad = False
            frozen += param.numel()
    if frozen == 0:
        raise ValueError("found no trainable image encoder parameters to freeze (no parameter name contains 'vision')")
    return frozen


def example_loss(model, processor, row: dict, base_dir: Path, device):
    import torch

    image = load_image(Path(base_dir), row)
    batch, labels = encode_example(processor, image, build_target(row))
    inputs = move_to(batch, device, model.dtype)
    inputs["labels"] = torch.tensor([labels], device=device)
    return model(**inputs).loss


def accumulation_sizes(n_examples: int, grad_accum: int) -> List[int]:
    """Size of the accumulation group each example belongs to, in order.

    The last group is smaller when n_examples is not a multiple of grad_accum,
    and its loss is divided by its real size, not by grad_accum.
    """
    sizes = []
    for start in range(0, n_examples, grad_accum):
        size = min(grad_accum, n_examples - start)
        sizes.extend([size] * size)
    return sizes


def train(
    model,
    processor,
    train_rows: List[dict],
    val_rows: List[dict],
    base_dir,
    *,
    epochs: int,
    lr: float,
    grad_accum: int,
    device,
    seed: int = 0,
    max_grad_norm: float = 1.0,
    log: Optional[Callable[[dict, int], None]] = None,
) -> List[dict]:
    """Train the trainable parameters (the LoRA weights) and return the history.

    Batch size is one receipt, with gradient accumulation giving the effective
    batch. Stops with an error if the loss becomes NaN or infinite.
    """
    import torch

    if not train_rows:
        raise ValueError("no training rows")
    if epochs < 1 or grad_accum < 1:
        raise ValueError("epochs and grad_accum must be at least 1")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("the model has no trainable parameters")

    optimizer = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    sizes = accumulation_sizes(len(train_rows), grad_accum)
    history = []
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        running = 0.0
        for n, idx in enumerate(order):
            group = sizes[n]
            loss = example_loss(model, processor, train_rows[idx], base_dir, device)
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"loss was {value} at epoch {epoch}, receipt {n + 1}. Try --dtype float32 "
                    "or a lower --lr."
                )
            (loss / group).backward()
            running += value / group
            if (n + 1) % grad_accum == 0 or n + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                record = {"epoch": epoch, "step": step, "train_loss": running}
                history.append(record)
                if log:
                    log({"train_loss": running, "epoch": epoch}, step)
                running = 0.0

        if val_rows:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    example_loss(model, processor, r, base_dir, device).item() for r in val_rows
                ) / len(val_rows)
            history.append({"epoch": epoch, "step": step, "val_loss": val_loss})
            if log:
                log({"val_loss": val_loss, "epoch": epoch}, step)
    return history
