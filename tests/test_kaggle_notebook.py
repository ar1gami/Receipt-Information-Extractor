"""Checks on notebooks/kaggle_finetune.ipynb.

The static checks catch a notebook that no longer matches the scripts it calls
(a renamed flag, a missing script) and a notebook saved with the long run
switched on. The last tests run the notebook's own code on a fake Kaggle folder
layout, with a tiny random model standing in for SmolVLM and a handful of
synthetic receipts in the SROIE folder layout. That proves the paths, commands
and bookkeeping work. It says nothing about the real model, which only a real
Kaggle run can show. The end to end test needs torch, transformers and peft and
takes about a minute.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "kaggle_finetune.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def code_sources():
    return ["".join(c["source"]) for c in cells() if c["cell_type"] == "code"]


def cell_containing(text):
    matches = [src for src in code_sources() if text in src]
    assert len(matches) == 1, f"expected exactly one code cell containing {text!r}, found {len(matches)}"
    return matches[0]


# ---- static checks ----

def test_notebook_is_a_valid_notebook_and_every_code_cell_compiles():
    data = json.loads(NOTEBOOK.read_text())
    assert data["nbformat"] == 4
    assert data["cells"]
    for cell in data["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None  # saved clean
            ast.parse("".join(cell["source"]))


def test_saved_with_the_safe_first_run_settings():
    settings = cell_containing("def setting")
    assert 'DO_CHECKS = setting("DO_CHECKS", True)' in settings
    assert 'DO_FULL_RUN = setting("DO_FULL_RUN", False)' in settings
    assert 'FORCE = setting("FORCE", False)' in settings
    assert 'REQUIRE_GPU = setting("REQUIRE_GPU", True)' in settings
    assert 'INSTALL = setting("INSTALL", True)' in settings


def test_scripts_the_notebook_calls_exist():
    names = set(re.findall(r"scripts/([a-z_]+\.py)", "\n".join(code_sources())))
    assert {"evaluate.py", "finetune.py", "merge_adapter.py", "prepare_sroie.py", "compare.py"} <= names
    for name in names:
        assert (ROOT / "scripts" / name).is_file(), name


def test_every_command_line_flag_in_the_notebook_is_defined_somewhere_in_the_project():
    flags = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', "\n".join(code_sources())))
    assert "--image-longest-edge" in flags and "--freeze-vision" in flags  # the pattern found something
    defined = "\n".join(p.read_text() for p in (ROOT / "scripts").glob("*.py"))
    defined += (ROOT / "receipts" / "vlm_io.py").read_text()
    # pip's own flag is the only one that is not the project's
    missing = sorted(f for f in flags if f != "--root-user-action" and f'"{f}"' not in defined)
    assert not missing, f"flags the notebook passes that no script defines: {missing}"


def test_notebook_has_no_dashes_that_read_like_ai_text():
    assert "\u2014" not in NOTEBOOK.read_text()


# ---- running the notebook's own code on a fake Kaggle ----

def make_fake_kaggle(base, n_train, n_test, code_as_zip=False):
    """Create input and working folders shaped like Kaggle's, return (input, working)."""
    from receipts.synth import make_receipt

    kaggle_input = base / "input"
    kaggle_working = base / "working"
    code = kaggle_input / "receipt-code" / "receipt-extractor"
    code.mkdir(parents=True)
    kaggle_working.mkdir()
    for name in ("receipts", "scripts"):
        shutil.copytree(ROOT / name, code / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "requirements-vlm.txt", code / "requirements-vlm.txt")
    (code / "data").mkdir()
    (code / "data" / "must_not_be_copied.txt").write_text("a data folder in the zip must be left out")

    sroie = kaggle_input / "sroie-datasetv2" / "SROIE2019"
    seed = 0
    for split, count in (("train", n_train), ("test", n_test)):
        for sub in ("img", "entities", "box"):
            (sroie / split / sub).mkdir(parents=True)
        for _ in range(count):
            image, truth = make_receipt(seed, date_style="slash")
            seed += 1
            stem = f"X{seed:011d}"
            image.save(sroie / split / "img" / f"{stem}.jpg")
            label = {"company": truth["vendor"], "date": truth["date"], "address": "1 Fake Street", "total": truth["total"]}
            (sroie / split / "entities" / f"{stem}.txt").write_text(json.dumps(label))
            (sroie / split / "box" / f"{stem}.txt").write_text("0,0,1,1,x")

    if code_as_zip:
        archive = kaggle_input / "receipt-code" / "receipt-extractor.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in code.rglob("*"):
                if path.is_file():
                    zf.write(path, Path("receipt-extractor") / path.relative_to(code))
        shutil.rmtree(code)
    return kaggle_input, kaggle_working


def run_cells(tmp_path, sources, env_extra, timeout=900):
    script = tmp_path / "notebook_cells.py"
    script.write_text("\n\n".join(sources))
    env = {**os.environ, **{f"RX_{k}": str(v) for k, v in env_extra.items()}}
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, f"notebook code failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result.stdout


def test_notebook_finds_the_code_inside_a_zip_and_leaves_the_data_folder_out(tmp_path):
    kaggle_input, kaggle_working = make_fake_kaggle(tmp_path, 2, 1, code_as_zip=True)
    out = run_cells(
        tmp_path,
        [cell_containing("def setting"), cell_containing("def find_project")],
        {"INPUT_DIR": kaggle_input, "WORK_DIR": kaggle_working},
    )
    project = kaggle_working / "receipt-extractor"
    assert (project / "scripts" / "finetune.py").is_file()
    assert (project / "receipts" / "vlm_io.py").is_file()
    assert not (project / "data").exists()
    assert "project code:" in out


def test_notebook_stops_with_a_clear_message_when_no_code_is_attached(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "working").mkdir()
    script = tmp_path / "cells.py"
    script.write_text("\n\n".join([cell_containing("def setting"), cell_containing("def find_project")]))
    env = {**os.environ, "RX_INPUT_DIR": str(tmp_path / "input"), "RX_WORK_DIR": str(tmp_path / "working")}
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "Attach the dataset that holds the project zip" in result.stderr


def bitsandbytes_runs_8bit(folder):
    try:
        import torch
        from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        )
        LlamaForCausalLM(config).save_pretrained(folder)
        model = LlamaForCausalLM.from_pretrained(
            folder, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
        )
        model(torch.zeros(1, 4, dtype=torch.long))
        return True
    except Exception:
        return False


def test_whole_notebook_runs_on_a_fake_kaggle_with_a_tiny_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("tokenizers")
    from tests.toy_vlm import build_toy_idefics3

    kaggle_input, kaggle_working = make_fake_kaggle(tmp_path, n_train=12, n_test=6)
    model = build_toy_idefics3(tmp_path / "toy_model", vision_hidden=64)
    quant_modes = "8bit" if bitsandbytes_runs_8bit(tmp_path / "probe") else ""

    out = run_cells(
        tmp_path,
        code_sources(),
        {
            "INPUT_DIR": kaggle_input,
            "WORK_DIR": kaggle_working,
            "MODEL_ID": model,
            "IMAGE_LONGEST_EDGE": 64,  # the toy's tiles are 32 pixels, SmolVLM's are 512
            "REQUIRE_GPU": 0,
            "INSTALL": 0,
            "EXPECTED_TRAIN": 12,
            "EXPECTED_TEST": 6,
            "DO_CHECKS": 1,
            "DO_FULL_RUN": 1,
            "EPOCHS": 2,
            "TINY_EXAMPLES": 6,
            "GRAD_ACCUM": 2,
            "LEARNING_RATE": "2e-3",
            "LATENCY_IMAGES": 2,
            "EVAL_DTYPE": "float32",
            "QUANT_MODES": quant_modes,
        },
    )

    project = kaggle_working / "receipt-extractor"
    assert "train receipts 12 (expected 12), test receipts 6 (expected 6)" in out
    assert "WARNING: the counts differ" not in out
    assert "OK, the loss went down." in out or "The loss did not go down" in out  # the check ran and reported
    for name in ("check_zero_shot", "check_tiny", "zero_shot", "tuned", "merge_check_adapter", "merge_check_merged"):
        assert (project / "reports" / f"{name}.json").is_file(), name
    assert (project / "outputs" / "lora" / "adapter_config.json").is_file()
    assert (project / "outputs" / "merged" / "config.json").is_file()
    tuned = json.loads((project / "reports" / "tuned.json").read_text())
    assert tuned["metrics"]["n"] == 6
    assert tuned["image_settings"]["longest_edge"] == 64
    assert tuned["latency"]["n_calls"] == 2
    if quant_modes:
        assert (project / "reports" / "int8.json").is_file()
        assert "int8" in (project / "reports" / "results.txt").read_text()
    settings = json.loads((project / "reports" / "run_settings.json").read_text())
    assert settings["image_longest_edge"] == 64 and settings["test_receipts"] == 6
    assert (kaggle_working / "receipt_results.zip").is_file()
    assert (kaggle_working / "logs" / "finetune.log").is_file()
    assert not (project / "data").exists()  # the image copies are removed at the end

    # Running it again must not redo finished work.
    again = run_cells(
        tmp_path,
        code_sources(),
        {
            "INPUT_DIR": kaggle_input, "WORK_DIR": kaggle_working, "MODEL_ID": model, "IMAGE_LONGEST_EDGE": 64,
            "REQUIRE_GPU": 0, "INSTALL": 0, "EXPECTED_TRAIN": 12, "EXPECTED_TEST": 6, "DO_CHECKS": 1,
            "DO_FULL_RUN": 1, "EPOCHS": 2, "TINY_EXAMPLES": 6, "GRAD_ACCUM": 2, "LEARNING_RATE": "2e-3",
            "LATENCY_IMAGES": 2, "EVAL_DTYPE": "float32", "QUANT_MODES": quant_modes,
        },
    )
    assert again.count("already exists, skipping this step") >= 5
