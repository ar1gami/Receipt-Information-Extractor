"""End to end checks of the model code on a tiny random model, on CPU, offline.

These run the real Transformers and PEFT code paths: loading, chat templates,
LoRA training, saving, merging, quantizing, generating and serving. They prove
the plumbing works with the installed library versions. They say nothing about
how well a real model will read receipts. Skipped when torch, transformers or
peft are not installed.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

from receipts.data import load_jsonl, write_jsonl  # noqa: E402
from receipts.training import encode_example, train  # noqa: E402
from receipts.training_utils import IGNORE_INDEX, build_target, split_rows  # noqa: E402
from receipts.vlm_io import load_model, model_size_mb, move_to  # noqa: E402
from tests.toy_vlm import build_toy_vlm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

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

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    return build_toy_vlm(tmp_path_factory.mktemp("toy_model"))

@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    """Six solid color images with different answers, plus a data.jsonl."""
    folder = tmp_path_factory.mktemp("toy_data")
    (folder / "images").mkdir()
    rows = []
    for (name, rgb), truth in zip(COLORS.items(), ROWS):
        Image.new("RGB", (48, 48), rgb).save(folder / "images" / f"{name}.png")
        rows.append({"image": f"images/{name}.png", **truth})
    write_jsonl(folder / "data.jsonl", rows)
    return folder


TRAIN_ROWS = 4  # the first four colors; one of them is held out for validation
VAL_FRACTION = 0.25

def trained_adapter(toy_dir, dataset_dir, out_dir, epochs=300, lr=2e-3):
    """Run the real fine-tuning script on the toy data and return the adapter folder.

    Four receipts, one held out, gives three to learn. With one answer per image,
    only the first letter of each answer depends on the image, so the model needs
    a good number of epochs before generation (not just the loss) is exact.
    """
    finetune = load_script("finetune")
    finetune.main(
        [
            str(dataset_dir / "data.jsonl"),
            "--model-id", str(toy_dir),
            "--output", str(out_dir),
            "--max-examples", str(TRAIN_ROWS),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--grad-accum", "2",
            "--lora-r", "16",
            "--val-fraction", str(VAL_FRACTION),
            "--dtype", "float32",
            "--seed", "0",
        ]
    )
    return out_dir

@pytest.fixture(scope="session")
def adapter_dir(toy_dir, dataset_dir, tmp_path_factory):
    return trained_adapter(toy_dir, dataset_dir, tmp_path_factory.mktemp("adapter") / "lora")


def train_rows_of(dataset_dir):
    """The receipts finetune.py trained on in trained_adapter (same split, same seed)."""
    rows = load_jsonl(dataset_dir / "data.jsonl")[:TRAIN_ROWS]
    train_rows, _ = split_rows(rows, VAL_FRACTION, seed=0)
    return train_rows


#  Loading and encoding:

def test_toy_model_loads_through_the_project_loader(toy_dir):
    processor, model = load_model(str(toy_dir), "float32")
    assert model.dtype == torch.float32
    assert str(model.device) == "cpu"
    assert hasattr(processor, "apply_chat_template")

def test_encode_example_masks_prompt_and_keeps_the_answer(toy_dir, dataset_dir):
    processor, _ = load_model(str(toy_dir), "float32")
    row = load_jsonl(dataset_dir / "data.jsonl")[0]
    image = Image.open(dataset_dir / row["image"]).convert("RGB")
    target = build_target(row)
    batch, labels = encode_example(processor, image, target)

    ids = batch["input_ids"][0].tolist()
    assert len(labels) == len(ids)
    answer_positions = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
    # Everything before the first answer token is masked, and the answer runs to the end.
    first = answer_positions[0]
    assert answer_positions == list(range(first, len(ids)))
    assert all(labels[i] == ids[i] for i in answer_positions)
    # The unmasked tokens decode to the JSON answer (plus the end of turn token).
    decoded = processor.tokenizer.decode([ids[i] for i in answer_positions], skip_special_tokens=True)
    assert json.loads(decoded.strip()) == json.loads(target)
    # The image placeholder was expanded and sits inside the masked prompt.
    image_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_positions = [i for i, t in enumerate(ids) if t == image_id]
    assert len(image_positions) == 4
    assert max(image_positions) < first

@pytest.mark.parametrize(
    "cuda, capability, expected",
    [
        (False, None, torch.float32),   # CPU
        (True, (8, 0), torch.bfloat16),  # A100 class
        (True, (8, 9), torch.bfloat16),  # L4 class
        (True, (7, 5), torch.float16),   # T4: bfloat16 is only emulated
        (True, (6, 0), torch.float16),   # P100
    ],
)
def test_auto_dtype_follows_the_hardware(monkeypatch, cuda, capability, expected):
    from receipts.vlm_io import resolve_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if capability:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert resolve_dtype("auto") == expected

def test_explicit_dtype_names_ignore_the_hardware(monkeypatch):
    from receipts.vlm_io import resolve_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_dtype("float16") == torch.float16
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype("float32") == torch.float32

def test_move_to_casts_only_floating_point_tensors():
    batch = {
        "pixel_values": torch.zeros(1, 3, 2, 2, dtype=torch.float32),
        "input_ids": torch.tensor([[1, 2, 3]]),
        "note": "not a tensor",
    }
    out = move_to(batch, "cpu", torch.bfloat16)
    assert out["pixel_values"].dtype == torch.bfloat16
    assert out["input_ids"].dtype == torch.int64
    assert out["note"] == "not a tensor"


#  Training:

def lora_model(toy_dir):
    from peft import LoraConfig, get_peft_model

    processor, model = load_model(str(toy_dir), "float32")
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    return processor, model

def test_only_lora_weights_are_trainable(toy_dir):
    _, model = lora_model(toy_dir)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)

def test_training_reduces_loss_and_reports_history(toy_dir, dataset_dir):
    processor, model = lora_model(toy_dir)
    rows = load_jsonl(dataset_dir / "data.jsonl")
    train_rows, val_rows = split_rows(rows, 0.17, seed=0)
    logged = []
    history = train(
        model, processor, train_rows, val_rows, dataset_dir,
        epochs=25, lr=5e-3, grad_accum=2, device="cpu", seed=0,
        log=lambda metrics, step: logged.append((step, metrics)),
    )
    train_losses = [h["train_loss"] for h in history if "train_loss" in h]
    val_losses = [h for h in history if "val_loss" in h]
    # 5 training receipts with accumulation 2 gives 3 optimizer steps per epoch.
    assert len(train_losses) == 25 * 3
    assert len(val_losses) == 25
    assert train_losses[-1] < 0.25 * train_losses[0]
    assert [s for s, _ in logged if "train_loss" in _] == list(range(1, 76))
    assert all(torch.isfinite(torch.tensor(h["val_loss"])) for h in val_losses)

def test_training_rejects_bad_arguments(toy_dir, dataset_dir):
    processor, model = lora_model(toy_dir)
    rows = load_jsonl(dataset_dir / "data.jsonl")
    common = dict(epochs=1, lr=1e-3, grad_accum=2, device="cpu")
    with pytest.raises(ValueError, match="no training rows"):
        train(model, processor, [], [], dataset_dir, **common)
    with pytest.raises(ValueError, match="at least 1"):
        train(model, processor, rows, [], dataset_dir, **{**common, "epochs": 0})
    # A model with nothing trainable must be refused, not silently "trained".
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError, match="no trainable parameters"):
        train(model, processor, rows, [], dataset_dir, **common)

def test_training_stops_on_non_finite_loss(toy_dir, dataset_dir, monkeypatch):
    import receipts.training as training

    processor, model = lora_model(toy_dir)
    rows = load_jsonl(dataset_dir / "data.jsonl")
    monkeypatch.setattr(
        training, "example_loss", lambda *a, **k: torch.tensor(float("nan"), requires_grad=True)
    )
    with pytest.raises(FloatingPointError, match="float32"):
        train(model, processor, rows, [], dataset_dir, epochs=1, lr=1e-3, grad_accum=2, device="cpu")


#  scripts, end to end:

def test_finetune_script_saves_adapter_and_history(adapter_dir):
    for name in ("adapter_config.json", "run_config.json", "history.json"):
        assert (adapter_dir / name).is_file(), name
    assert any(p.name.startswith("adapter_model") for p in adapter_dir.iterdir())
    history = json.loads((adapter_dir / "history.json").read_text())
    losses = [h["train_loss"] for h in history if "train_loss" in h]
    assert losses[-1] < 0.25 * losses[0]

def test_finetune_logs_a_run_to_wandb_in_offline_mode(toy_dir, dataset_dir, tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    finetune = load_script("finetune")
    history = finetune.main(
        [
            str(dataset_dir / "data.jsonl"),
            "--model-id", str(toy_dir),
            "--output", str(tmp_path / "lora"),
            "--max-examples", "4",
            "--epochs", "2",
            "--lr", "5e-3",
            "--grad-accum", "2",
            "--val-fraction", "0.25",
            "--dtype", "float32",
            "--wandb",
        ]
    )
    # 3 receipts with accumulation 2 is 2 optimizer steps an epoch, plus one validation entry.
    assert len(history) == 2 * (2 + 1)
    assert list((tmp_path / "wandb").glob("offline-run-*")), "no run was recorded"

def test_adapter_reloaded_by_the_predictor_reproduces_training_answers(toy_dir, dataset_dir, adapter_dir, tmp_path):
    evaluate = load_script("evaluate")
    seen = train_rows_of(dataset_dir)
    seen_path = dataset_dir / "seen.jsonl"
    write_jsonl(seen_path, seen)
    out = tmp_path / "report.json"
    report = evaluate.main(
        [
            str(seen_path),
            "--predictor", "vlm",
            "--model-id", str(toy_dir),
            "--adapter", str(adapter_dir),
            "--dtype", "float32",
            "--latency",
            "--show-raw", "1",
            "--out", str(out),
        ]
    )
    # The untrained model scores 0 on this (see the next test), so at least two
    # of the three trained receipts coming back exact shows fine tuning worked.
    assert len(seen) == 3
    assert report["metrics"]["all_fields_accuracy"] >= 2 / 3
    assert report["model_size_mb"] > 0
    assert "quantization_backend" not in report  # nothing was quantized
    assert all(isinstance(t, str) and t.strip() for t in report["raw_outputs"])
    assert report["latency"]["n_calls"] == len(seen)
    assert json.loads(out.read_text())["metrics"]["n"] == len(seen)

def test_saved_adapter_behaves_exactly_like_the_trained_model(toy_dir, dataset_dir, tmp_path):
    """Saving and reloading must not change the model, whether or not training went well."""
    from receipts.predictors import make_vlm_predictor, predictor_from_model

    processor, model = lora_model(toy_dir)
    rows = load_jsonl(dataset_dir / "data.jsonl")
    train(model, processor, rows[:4], [], dataset_dir, epochs=15, lr=5e-3, grad_accum=2, device="cpu")
    in_memory = predictor_from_model(processor, model)
    model.save_pretrained(tmp_path / "lora")
    reloaded = make_vlm_predictor(str(toy_dir), str(tmp_path / "lora"), dtype="float32")
    base = make_vlm_predictor(str(toy_dir), dtype="float32")

    changed = 0
    for row in rows:  # includes receipts the model never saw
        image = Image.open(dataset_dir / row["image"]).convert("RGB")
        in_memory(image), reloaded(image), base(image)
        assert in_memory.last_text == reloaded.last_text
        changed += in_memory.last_text != base.last_text
    # If training changed nothing, the comparison above would prove nothing.
    assert changed == len(rows)

def test_untrained_model_scores_badly_so_the_metric_can_tell_the_difference(toy_dir, dataset_dir, tmp_path):
    evaluate = load_script("evaluate")
    report = evaluate.main(
        [str(dataset_dir / "data.jsonl"), "--predictor", "vlm", "--model-id", str(toy_dir), "--dtype", "float32"]
    )
    assert report["metrics"]["all_fields_accuracy"] == 0.0
    assert report["metrics"]["parse_failure_rate"] == 1.0  # random model writes no valid JSON

def test_merged_model_matches_the_adapter(toy_dir, dataset_dir, adapter_dir, tmp_path):
    from receipts.predictors import make_vlm_predictor

    merge = load_script("merge_adapter")
    merged = merge.main(
        ["--model-id", str(toy_dir), "--adapter", str(adapter_dir), "--output", str(tmp_path / "merged"), "--dtype", "float32"]
    )
    with_adapter = make_vlm_predictor(str(toy_dir), str(adapter_dir), dtype="float32")
    standalone = make_vlm_predictor(str(merged), dtype="float32")
    for row in train_rows_of(dataset_dir):
        image = Image.open(dataset_dir / row["image"]).convert("RGB")
        assert with_adapter(image) == standalone(image)
        assert with_adapter.last_text == standalone.last_text
    # Merging adds nothing: the standalone folder is a normal model with no adapter files.
    assert not (merged / "adapter_config.json").exists()

def test_dynamic_int8_runs_and_is_smaller(toy_dir, dataset_dir, adapter_dir, tmp_path):
    from receipts.predictors import make_vlm_predictor

    merge = load_script("merge_adapter")
    merged = merge.main(
        ["--model-id", str(toy_dir), "--adapter", str(adapter_dir), "--output", str(tmp_path / "merged"), "--dtype", "float32"]
    )
    full = make_vlm_predictor(str(merged), dtype="float32")
    quant = make_vlm_predictor(str(merged), quantization="dynamic-int8")
    assert quant.size_mb < 0.75 * full.size_mb
    assert quant.quantization_backend == "torch.ao"
    assert full.quantization_backend is None
    row = train_rows_of(dataset_dir)[0]
    image = Image.open(dataset_dir / row["image"]).convert("RGB")
    quant(image)
    assert isinstance(quant.last_text, str) and quant.last_text.strip()

def test_evaluate_report_names_the_quantization_backend(toy_dir, dataset_dir, adapter_dir, tmp_path):
    merge = load_script("merge_adapter")
    evaluate = load_script("evaluate")
    merged = merge.main(
        ["--model-id", str(toy_dir), "--adapter", str(adapter_dir), "--output", str(tmp_path / "merged"), "--dtype", "float32"]
    )
    seen_path = dataset_dir / "seen_int8.jsonl"
    write_jsonl(seen_path, train_rows_of(dataset_dir))
    report = evaluate.main(
        [str(seen_path), "--predictor", "vlm", "--model-id", str(merged), "--quantization", "dynamic-int8"]
    )
    assert report["quantization_backend"] == "torch.ao"
    assert report["model_size_mb"] > 0

def test_int8_falls_back_to_torchao_when_torch_ao_is_gone(toy_dir, monkeypatch):
    pytest.importorskip("torchao")
    from receipts.vlm_io import quantize_int8_cpu

    _, model = load_model(str(toy_dir), "float32")
    before = model_size_mb(model)
    monkeypatch.delattr(torch.ao.quantization, "quantize_dynamic")
    model, backend = quantize_int8_cpu(model)
    assert backend.startswith("torchao ")
    assert model_size_mb(model) < 0.75 * before

def test_int8_says_what_to_do_when_no_backend_exists(toy_dir, monkeypatch):
    import sys

    from receipts.vlm_io import quantize_int8_cpu

    _, model = load_model(str(toy_dir), "float32")
    monkeypatch.delattr(torch.ao.quantization, "quantize_dynamic")
    monkeypatch.setitem(sys.modules, "torchao", None)  # makes "import torchao" fail
    with pytest.raises(RuntimeError, match="torchao"):
        quantize_int8_cpu(model)

def test_dynamic_int8_with_an_adapter_is_refused_with_a_useful_message(toy_dir, adapter_dir):
    from receipts.predictors import make_vlm_predictor

    with pytest.raises(ValueError, match="Merge the adapter first"):
        make_vlm_predictor(str(toy_dir), str(adapter_dir), quantization="dynamic-int8")

def test_model_size_matches_the_files_on_disk_roughly(toy_dir):
    _, model = load_model(str(toy_dir), "float32")
    on_disk = sum(p.stat().st_size for p in Path(toy_dir).glob("*.safetensors")) / 1e6
    assert model_size_mb(model) == pytest.approx(on_disk, rel=0.15)


# Serving:

def test_api_serves_the_fine_tuned_model(toy_dir, dataset_dir, adapter_dir, monkeypatch):
    import io

    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setenv("PREDICTOR", "vlm")
    monkeypatch.setenv("MODEL_ID", str(toy_dir))
    monkeypatch.setenv("ADAPTER_PATH", str(adapter_dir))
    row = train_rows_of(dataset_dir)[0]
    buf = io.BytesIO()
    Image.open(dataset_dir / row["image"]).convert("RGB").save(buf, format="PNG")

    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok", "predictor": "vlm"}
        response = client.post("/extract", files={"file": ("r.png", buf.getvalue(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predictor"] == "vlm"
    assert body["vendor"] == row["vendor"]
    assert body["date"] == row["date"]
    assert body["total"] == row["total"]

def test_api_returns_422_when_the_model_writes_no_json(toy_dir, monkeypatch):
    import io

    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setenv("PREDICTOR", "vlm")
    monkeypatch.setenv("MODEL_ID", str(toy_dir))
    monkeypatch.delenv("ADAPTER_PATH", raising=False)
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), "red").save(buf, format="PNG")
    with TestClient(create_app()) as client:
        response = client.post("/extract", files={"file": ("r.png", buf.getvalue(), "image/png")})
    assert response.status_code == 422

#  Dockerfile's bake step 

def run_dockerfile_bake_step(tmp_path, toy_dir, adapter_path):
    """Run the Dockerfile's own bake command, text and all, with the toy model in place of the hub model.

    Docker cannot be run where these tests were written. This is the closest check on the one step
    that loads a model while the image is being built: the same shell text, the same python
    command, the same environment variables (apart from the paths, which point at the toy files).
    """
    from tests.test_dockerfile_logic import final_env, step_containing

    _, body = step_containing("predictor_from_env")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "python"  # the image has a command called python, so this machine needs one too
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    env = {
        **os.environ,
        **final_env({"VARIANT": "vlm", "BASE_MODEL": str(toy_dir)}),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "BASE_MODEL": str(toy_dir),  # a build argument reaches RUN steps as an environment variable
        "MODEL_ID": str(toy_dir),
        "ADAPTER_PATH": str(adapter_path),
        "HF_HOME": str(tmp_path / "hf"),
    }
    env.pop("HF_HUB_OFFLINE", None)  # in the Dockerfile this is set after the bake step, not before it
    return subprocess.run(["/bin/sh", "-c", body], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)

@pytest.mark.skipif(os.name == "nt", reason="needs a POSIX shell")
def test_dockerfile_bake_step_loads_the_model_and_the_adapter(toy_dir, adapter_dir, tmp_path):
    result = run_dockerfile_bake_step(tmp_path, toy_dir, adapter_dir)
    assert result.returncode == 0, result.stderr
    assert "loaded vlm" in result.stdout

@pytest.mark.skipif(os.name == "nt", reason="needs a POSIX shell")
def test_dockerfile_bake_step_fails_the_build_when_the_adapter_folder_is_wrong(toy_dir, tmp_path):
    result = run_dockerfile_bake_step(tmp_path, toy_dir, tmp_path / "no-such-adapter")
    assert result.returncode != 0
    assert "not an adapter folder" in result.stderr
