"""Checks of the SmolVLM code path on a tiny random model of the same classes.

SmolVLM is built from the Idefics3 classes. It cuts each image into square
tiles, adds a whole-image view, and expands every tile into a run of image
tokens, so one receipt becomes hundreds of tokens. tests/toy_vlm.py builds a
small random model with that structure. These tests run the project's real
code on it, offline and on CPU: the chat template, prompt masking, the image
size options, training, saving, merging and evaluation. They show the plumbing
works. They say nothing about how well the real model reads receipts.
Skipped when torch, transformers or peft are not installed.
"""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

from receipts.data import load_jsonl, write_jsonl  # noqa: E402
from receipts.training import encode_example, freeze_vision, prompt_lengths  # noqa: E402
from receipts.training_utils import IGNORE_INDEX, build_target, split_rows  # noqa: E402
from receipts.vlm_io import (  # noqa: E402
    adapter_image_options,
    apply_image_options,
    image_options_from_env,
    image_settings,
    load_model,
)
from tests.toy_vlm import build_toy_idefics3, build_toy_vlm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGE_TOKENS_PER_TILE = 4  # the toy's image_seq_len

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 200, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 230, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}
ROWS = [
    {"vendor": "Red Co", "date": "2026-01-01", "total": "1.00"},
    {"vendor": "Green Co", "date": "2026-02-02", "total": "2.00"},
    {"vendor": "Blue Co", "date": "2026-03-03", "total": "3.00"},
    {"vendor": "Yellow Co", "date": "2026-04-04", "total": "4.00"},
    {"vendor": "Black Co", "date": "2026-05-05", "total": "5.00"},
    {"vendor": "White Co", "date": "2026-06-06", "total": "6.00"},
]
TRAIN_ROWS = 4
VAL_FRACTION = 0.25
TRAIN_EDGE = 128  # differs from the toy's default of 64, so adoption can be told apart


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def toy3_dir(tmp_path_factory):
    return build_toy_idefics3(tmp_path_factory.mktemp("toy_idefics3"))


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    """Six tall solid color images (receipt shaped) with different answers."""
    folder = tmp_path_factory.mktemp("toy3_data")
    (folder / "images").mkdir()
    rows = []
    for (name, rgb), truth in zip(COLORS.items(), ROWS):
        Image.new("RGB", (40, 120), rgb).save(folder / "images" / f"{name}.png")
        rows.append({"image": f"images/{name}.png", **truth})
    write_jsonl(folder / "data.jsonl", rows)
    return folder


def finetune_args(toy3_dir, dataset_dir, out_dir, *extra, epochs=300, lr=2e-3):
    return [
        str(dataset_dir / "data.jsonl"),
        "--model-id", str(toy3_dir),
        "--output", str(out_dir),
        "--max-examples", str(TRAIN_ROWS),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--grad-accum", "2",
        "--lora-r", "16",
        "--val-fraction", str(VAL_FRACTION),
        "--dtype", "float32",
        "--seed", "0",
        *extra,
    ]


@pytest.fixture(scope="session")
def adapter_dir(toy3_dir, dataset_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("adapter3") / "lora"
    load_script("finetune").main(
        finetune_args(toy3_dir, dataset_dir, out, "--image-longest-edge", str(TRAIN_EDGE))
    )
    return out


def seen_rows(dataset_dir):
    rows = load_jsonl(dataset_dir / "data.jsonl")[:TRAIN_ROWS]
    train_rows, _ = split_rows(rows, VAL_FRACTION, seed=0)
    return train_rows


def tall_image():
    return Image.new("RGB", (40, 200), (250, 10, 10))


def n_image_tokens(processor, image):
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = processor(text=prompt, images=[image], return_tensors="pt")
    image_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    return int((batch["input_ids"] == image_id).sum()), int(batch["pixel_values"].shape[1])


# ---- loading, templates and masking on the Idefics3 classes ----

def test_toy_is_built_from_the_smolvlm_classes(toy3_dir):
    processor, model = load_model(str(toy3_dir), "float32")
    assert type(model).__name__ == "Idefics3ForConditionalGeneration"
    assert type(processor).__name__ == "Idefics3Processor"


def test_chat_template_follows_the_smolvlm_layout(toy3_dir):
    processor, _ = load_model(str(toy3_dir), "float32")
    from receipts.vlm_io import build_messages

    prompt = processor.apply_chat_template(build_messages(), add_generation_prompt=True, tokenize=False)
    full = processor.apply_chat_template(build_messages('{"a": 1}'), add_generation_prompt=False, tokenize=False)
    assert "User:<image>" in prompt
    assert prompt.endswith("<end_of_utterance>\nAssistant:")
    assert full.startswith(prompt)  # the property the masking relies on
    assert full.rstrip().endswith("<end_of_utterance>")


def test_each_tile_becomes_a_run_of_image_tokens(toy3_dir):
    processor, _ = load_model(str(toy3_dir), "float32")
    tokens, tiles = n_image_tokens(processor, tall_image())
    assert tiles >= 2
    assert tokens == IMAGE_TOKENS_PER_TILE * tiles


def test_encode_example_masks_prompt_and_image_tokens(toy3_dir, dataset_dir):
    processor, _ = load_model(str(toy3_dir), "float32")
    row = load_jsonl(dataset_dir / "data.jsonl")[0]
    image = Image.open(dataset_dir / row["image"]).convert("RGB")
    target = build_target(row)
    batch, labels = encode_example(processor, image, target)

    ids = batch["input_ids"][0].tolist()
    assert len(labels) == len(ids)
    answer_positions = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
    first = answer_positions[0]
    assert answer_positions == list(range(first, len(ids)))
    decoded = processor.tokenizer.decode([ids[i] for i in answer_positions], skip_special_tokens=True)
    assert json.loads(decoded.strip()) == json.loads(target)
    image_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_positions = [i for i, t in enumerate(ids) if t == image_id]
    assert image_positions and max(image_positions) < first
    assert all(labels[i] == IGNORE_INDEX for i in image_positions)
    assert "pixel_attention_mask" in batch  # passed on to the model with the pixels


# ---- image size options ----

def test_lower_longest_edge_means_fewer_image_tokens(toy3_dir):
    counts = {}
    for edge in (32, 64, 128, 256):
        processor, _ = load_model(str(toy3_dir), "float32", image_longest_edge=edge)
        counts[edge] = n_image_tokens(processor, tall_image())[0]
    assert counts[32] < counts[64] < counts[128] < counts[256]


def test_no_splitting_keeps_only_the_whole_image_view(toy3_dir):
    processor, _ = load_model(str(toy3_dir), "float32", image_splitting=False)
    tokens, tiles = n_image_tokens(processor, tall_image())
    assert tiles == 1
    assert tokens == IMAGE_TOKENS_PER_TILE


def test_image_settings_report_what_was_applied(toy3_dir):
    processor, _ = load_model(str(toy3_dir), "float32")
    assert image_settings(processor) == {"longest_edge": 64, "tile_size": 32, "splitting": True}
    apply_image_options(processor, 160, False)
    assert image_settings(processor) == {"longest_edge": 160, "tile_size": 32, "splitting": False}


def test_options_left_unset_change_nothing(toy3_dir):
    processor, _ = load_model(str(toy3_dir), "float32")
    before = n_image_tokens(processor, tall_image())
    assert apply_image_options(processor) is processor
    assert n_image_tokens(processor, tall_image()) == before
    # one option alone leaves the other as it was
    apply_image_options(processor, longest_edge=96)
    assert image_settings(processor)["splitting"] is True
    apply_image_options(processor, splitting=False)
    assert image_settings(processor)["longest_edge"] == 96


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_longest_edge_is_rejected(toy3_dir, bad):
    processor, _ = load_model(str(toy3_dir), "float32")
    with pytest.raises(ValueError, match="at least 1"):
        apply_image_options(processor, longest_edge=bad)


def test_options_on_a_model_without_tiles_give_a_clear_error(tmp_path):
    llava_dir = build_toy_vlm(tmp_path / "llava")
    processor, _ = load_model(str(llava_dir), "float32")
    assert image_settings(processor) is None
    with pytest.raises(ValueError, match="SmolVLM style"):
        apply_image_options(processor, longest_edge=512)
    with pytest.raises(ValueError, match="SmolVLM style"):
        load_model(str(llava_dir), "float32", image_splitting=False)
    # asking for nothing is always fine
    assert load_model(str(llava_dir), "float32")[0] is not None


def test_settings_survive_saving_and_reloading_the_processor(toy3_dir, tmp_path):
    processor, _ = load_model(str(toy3_dir), "float32", image_longest_edge=96)
    processor.save_pretrained(tmp_path / "saved")
    from transformers import AutoProcessor

    reloaded = AutoProcessor.from_pretrained(tmp_path / "saved")
    assert image_settings(reloaded)["longest_edge"] == 96
    image = tall_image()
    assert n_image_tokens(reloaded, image) == n_image_tokens(processor, image)


def test_prompt_lengths_shrink_with_the_image_size(toy3_dir, dataset_dir):
    rows = load_jsonl(dataset_dir / "data.jsonl")
    big, _ = load_model(str(toy3_dir), "float32", image_longest_edge=256)
    small, _ = load_model(str(toy3_dir), "float32", image_longest_edge=32)
    long_lengths = prompt_lengths(big, rows, dataset_dir, n=3)
    short_lengths = prompt_lengths(small, rows, dataset_dir, n=3)
    assert len(long_lengths) == len(short_lengths) == 3
    assert all(a > b for a, b in zip(long_lengths, short_lengths))


# ---- reading the options from the environment and from a training run ----

def test_environment_options():
    assert image_options_from_env({}) == (None, None)
    assert image_options_from_env({"IMAGE_LONGEST_EDGE": "1024", "IMAGE_SPLITTING": "off"}) == (1024, False)
    assert image_options_from_env({"IMAGE_LONGEST_EDGE": " 768 ", "IMAGE_SPLITTING": "ON"}) == (768, True)
    assert image_options_from_env({"IMAGE_LONGEST_EDGE": "", "IMAGE_SPLITTING": ""}) == (None, None)
    with pytest.raises(RuntimeError, match="whole number"):
        image_options_from_env({"IMAGE_LONGEST_EDGE": "big"})
    with pytest.raises(RuntimeError, match="on or off"):
        image_options_from_env({"IMAGE_SPLITTING": "maybe"})


def test_adapter_options_come_from_its_run_config(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps({"image_longest_edge": 1536, "no_image_splitting": False}))
    assert adapter_image_options(tmp_path) == (1536, None)
    assert adapter_image_options(tmp_path, 512, None) == (512, None)  # explicit wins
    assert adapter_image_options(tmp_path, None, False) == (None, False)
    (tmp_path / "run_config.json").write_text(json.dumps({"image_longest_edge": None, "no_image_splitting": True}))
    assert adapter_image_options(tmp_path) == (None, False)


def test_adapter_without_a_usable_run_config_gives_no_options(tmp_path):
    assert adapter_image_options(None) == (None, None)
    assert adapter_image_options(tmp_path) == (None, None)  # no file
    (tmp_path / "run_config.json").write_text("not json")
    assert adapter_image_options(tmp_path) == (None, None)


# ---- training on the Idefics3 classes ----

def test_finetune_learns_and_records_the_image_size(adapter_dir):
    config = json.loads((adapter_dir / "run_config.json").read_text())
    assert config["image_longest_edge"] == TRAIN_EDGE
    assert config["no_image_splitting"] is False
    history = json.loads((adapter_dir / "history.json").read_text())
    losses = [h["train_loss"] for h in history if "train_loss" in h]
    assert losses[-1] < 0.25 * losses[0]
    # the processor saved with the adapter carries the training image size
    from transformers import AutoProcessor

    assert image_settings(AutoProcessor.from_pretrained(adapter_dir))["longest_edge"] == TRAIN_EDGE


def test_predictor_with_adapter_uses_the_training_image_size(toy3_dir, adapter_dir):
    from receipts.predictors import make_vlm_predictor

    adopted = make_vlm_predictor(str(toy3_dir), str(adapter_dir), dtype="float32")
    assert adopted.image_settings["longest_edge"] == TRAIN_EDGE
    explicit = make_vlm_predictor(str(toy3_dir), str(adapter_dir), dtype="float32", image_longest_edge=64)
    assert explicit.image_settings["longest_edge"] == 64
    plain = make_vlm_predictor(str(toy3_dir), dtype="float32")
    assert plain.image_settings["longest_edge"] == 64  # nothing to adopt without an adapter


def test_adapter_reloaded_by_evaluate_reproduces_training_answers(toy3_dir, dataset_dir, adapter_dir, tmp_path):
    evaluate = load_script("evaluate")
    seen_path = dataset_dir / "seen.jsonl"
    write_jsonl(seen_path, seen_rows(dataset_dir))
    out = tmp_path / "report.json"
    report = evaluate.main(
        [
            str(seen_path),
            "--predictor", "vlm",
            "--model-id", str(toy3_dir),
            "--adapter", str(adapter_dir),
            "--dtype", "float32",
            "--latency",
            "--out", str(out),
        ]
    )
    assert report["metrics"]["n"] == 3
    assert report["metrics"]["all_fields_accuracy"] >= 2 / 3
    assert report["image_settings"]["longest_edge"] == TRAIN_EDGE
    assert json.loads(out.read_text())["image_settings"]["longest_edge"] == TRAIN_EDGE
    assert report["latency"]["n_calls"] == 3


def test_mismatched_image_size_is_visible_in_the_report(toy3_dir, dataset_dir, adapter_dir, tmp_path):
    """Forcing another size works, and the report says which size was used."""
    evaluate = load_script("evaluate")
    report = evaluate.main(
        [
            str(dataset_dir / "data.jsonl"),
            "--predictor", "vlm",
            "--model-id", str(toy3_dir),
            "--adapter", str(adapter_dir),
            "--dtype", "float32",
            "--image-longest-edge", "32",
            "--limit", "2",
        ]
    )
    assert report["image_settings"]["longest_edge"] == 32


def test_untrained_model_scores_badly_so_the_metric_can_tell_the_difference(toy3_dir, dataset_dir):
    evaluate = load_script("evaluate")
    report = evaluate.main(
        [str(dataset_dir / "data.jsonl"), "--predictor", "vlm", "--model-id", str(toy3_dir), "--dtype", "float32"]
    )
    assert report["metrics"]["all_fields_accuracy"] == 0.0


def test_merge_keeps_the_training_image_size_and_the_answers(toy3_dir, dataset_dir, adapter_dir, tmp_path):
    from receipts.predictors import make_vlm_predictor

    merge = load_script("merge_adapter")
    merged = merge.main(
        ["--model-id", str(toy3_dir), "--adapter", str(adapter_dir), "--output", str(tmp_path / "merged"), "--dtype", "float32"]
    )
    standalone = make_vlm_predictor(str(merged), dtype="float32")
    assert standalone.image_settings["longest_edge"] == TRAIN_EDGE  # read from the merged folder itself
    with_adapter = make_vlm_predictor(str(toy3_dir), str(adapter_dir), dtype="float32")
    for row in seen_rows(dataset_dir):
        image = Image.open(dataset_dir / row["image"]).convert("RGB")
        with_adapter(image), standalone(image)
        assert with_adapter.last_text == standalone.last_text


def test_dynamic_int8_runs_on_the_merged_idefics3_model(toy3_dir, dataset_dir, adapter_dir, tmp_path):
    from receipts.predictors import make_vlm_predictor

    merge = load_script("merge_adapter")
    merged = merge.main(
        ["--model-id", str(toy3_dir), "--adapter", str(adapter_dir), "--output", str(tmp_path / "merged"), "--dtype", "float32"]
    )
    full = make_vlm_predictor(str(merged), dtype="float32")
    quant = make_vlm_predictor(str(merged), quantization="dynamic-int8")
    assert quant.size_mb < full.size_mb
    image = Image.open(dataset_dir / seen_rows(dataset_dir)[0]["image"]).convert("RGB")
    quant(image)
    assert isinstance(quant.last_text, str) and quant.last_text.strip()


# ---- memory savers ----

def test_freeze_vision_leaves_only_language_lora_weights_trainable(toy3_dir):
    from peft import LoraConfig, get_peft_model

    _, model = load_model(str(toy3_dir), "float32")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable_before = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("vision" in n.lower() for n in trainable_before)  # all-linear reaches the image encoder
    frozen = freeze_vision(model)
    assert frozen > 0
    trainable_after = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable_after and not any("vision" in n.lower() for n in trainable_after)
    assert all("lora" in n for n in trainable_after)


def test_freeze_vision_errors_when_there_is_nothing_to_freeze(tmp_path):
    from peft import LoraConfig, get_peft_model

    llava_dir = build_toy_vlm(tmp_path / "llava")
    _, model = load_model(str(llava_dir), "float32")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"))
    # The LLaVA toy's image encoder is named "vision_tower", so this one does freeze.
    assert freeze_vision(model) > 0
    with pytest.raises(ValueError, match="no trainable image encoder"):
        freeze_vision(model)  # a second call finds nothing left


@pytest.mark.parametrize("flags", [["--freeze-vision"], ["--gradient-checkpointing"], ["--freeze-vision", "--gradient-checkpointing"]])
def test_memory_saving_flags_still_train(toy3_dir, dataset_dir, tmp_path, flags):
    history = load_script("finetune").main(
        finetune_args(toy3_dir, dataset_dir, tmp_path / "lora", *flags, epochs=60)
    )
    losses = [h["train_loss"] for h in history if "train_loss" in h]
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[-1] < 0.5 * losses[0]


def test_frozen_vision_weights_do_not_change_during_training(toy3_dir, dataset_dir, tmp_path):
    """The image encoder still gets (inactive) LoRA layers, but training must not move them.

    A LoRA layer starts with its B matrix at zero, so a B matrix that is still
    zero after training was never updated. The language model's must have moved.
    """
    out = tmp_path / "lora"
    load_script("finetune").main(finetune_args(toy3_dir, dataset_dir, out, "--freeze-vision", epochs=5))
    from safetensors.torch import load_file

    saved = load_file(str(next(out.glob("adapter_model*.safetensors"))))
    b_matrices = {name: tensor for name, tensor in saved.items() if "lora_B" in name}
    vision = {n: t for n, t in b_matrices.items() if "vision" in n.lower()}
    language = {n: t for n, t in b_matrices.items() if "vision" not in n.lower()}
    assert vision and language
    assert all(float(t.abs().max()) == 0.0 for t in vision.values()), "image encoder LoRA weights were trained"
    assert any(float(t.abs().max()) > 0.0 for t in language.values()), "language model LoRA weights did not train"


def test_without_the_flag_the_image_encoder_does_train(toy3_dir, dataset_dir, tmp_path):
    """The check above only means something if the same measurement sees movement without the flag."""
    out = tmp_path / "lora"
    load_script("finetune").main(finetune_args(toy3_dir, dataset_dir, out, epochs=5))
    from safetensors.torch import load_file

    saved = load_file(str(next(out.glob("adapter_model*.safetensors"))))
    vision = [t for n, t in saved.items() if "lora_B" in n and "vision" in n.lower()]
    assert vision and any(float(t.abs().max()) > 0.0 for t in vision)


# ---- bitsandbytes quantization on the Idefics3 classes ----
#
# The real use is 8 bit and 4 bit on a CUDA GPU. Recent bitsandbytes also runs
# on a CPU, which lets these tests exercise the project's loading, size and
# evaluation code against the real quantization library. They skip when
# bitsandbytes is missing or cannot quantize anything on this machine, which a
# probe with a plain tiny Llama model (no project code) decides.

@pytest.fixture(scope="session")
def wide_toy_dir(tmp_path_factory):
    return build_toy_idefics3(tmp_path_factory.mktemp("toy_idefics3_wide"), vision_hidden=64)


def bitsandbytes_works(mode, folder):
    try:
        from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        )
        LlamaForCausalLM(config).save_pretrained(folder)
        if mode == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32)
        model = LlamaForCausalLM.from_pretrained(folder, quantization_config=quant, device_map="auto")
        model(torch.zeros(1, 4, dtype=torch.long))
        return True
    except Exception:
        return False


@pytest.fixture(scope="session", params=["8bit", "4bit"])
def bnb_mode(request, tmp_path_factory):
    pytest.importorskip("bitsandbytes")
    if not bitsandbytes_works(request.param, tmp_path_factory.mktemp(f"probe_{request.param}")):
        pytest.skip(f"bitsandbytes cannot run {request.param} quantization on this machine")
    return request.param


def test_bitsandbytes_model_loads_generates_and_is_smaller(wide_toy_dir, dataset_dir, bnb_mode):
    from receipts.predictors import make_vlm_predictor

    full = make_vlm_predictor(str(wide_toy_dir), dtype="float32")
    quant = make_vlm_predictor(str(wide_toy_dir), quantization=bnb_mode, dtype="float32")
    image = Image.open(dataset_dir / seen_rows(dataset_dir)[0]["image"]).convert("RGB")
    quant(image)
    assert isinstance(quant.last_text, str)
    assert quant.size_mb is not None and quant.size_mb < full.size_mb
    assert quant.image_settings == full.image_settings


def test_merged_model_is_evaluated_quantized_by_the_script(wide_toy_dir, dataset_dir, tmp_path, bnb_mode):
    """The exact steps the Kaggle notebook runs: train, merge, then evaluate the merged folder quantized."""
    load_script("finetune").main(
        finetune_args(wide_toy_dir, dataset_dir, tmp_path / "lora", "--freeze-vision", epochs=5)
    )
    merged = load_script("merge_adapter").main(
        ["--model-id", str(wide_toy_dir), "--adapter", str(tmp_path / "lora"),
         "--output", str(tmp_path / "merged"), "--dtype", "float32"]
    )
    out = tmp_path / "report.json"
    report = load_script("evaluate").main(
        [str(dataset_dir / "data.jsonl"), "--predictor", "vlm", "--model-id", str(merged),
         "--quantization", bnb_mode, "--dtype", "float32", "--latency", "--latency-images", "2",
         "--out", str(out)]
    )
    assert report["model_size_mb"] > 0
    assert report["latency"]["n_calls"] == 2
    assert report["metrics"]["n"] == 6
    saved = json.loads(out.read_text())
    assert saved["image_settings"]["splitting"] is True
    assert saved["model_size_mb"] == report["model_size_mb"]
