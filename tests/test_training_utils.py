import importlib.util
import json
import py_compile
from pathlib import Path

import pytest

from receipts.training_utils import IGNORE_INDEX, build_target, mask_prompt_labels, split_rows

ROOT = Path(__file__).resolve().parent.parent

def test_build_target_is_valid_json_with_fixed_keys():
    target = build_target({"image": "x.png", "vendor": "A", "total": "1.00", "junk": 5})
    assert json.loads(target) == {"vendor": "A", "date": None, "total": "1.00"}

def test_build_target_round_trips_through_the_output_parser():
    from receipts.vlm_output import parse_model_output

    row = {"vendor": 'Joe "The Fish" Co', "date": "2026-01-02", "total": "5.00"}
    assert parse_model_output(build_target(row)) == row

def test_mask_prompt_labels():
    labels = mask_prompt_labels([10, 11, 12, 13, 14], prompt_len=3)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 13, 14]

def test_mask_prompt_labels_edges():
    assert mask_prompt_labels([1, 2], 0) == [1, 2]
    assert mask_prompt_labels([1, 2], 2) == [IGNORE_INDEX, IGNORE_INDEX]
    with pytest.raises(ValueError):
        mask_prompt_labels([1, 2], 3)
    with pytest.raises(ValueError):
        mask_prompt_labels([1, 2], -1)

def test_split_is_deterministic_and_disjoint():
    rows = [{"image": f"{i}.png"} for i in range(50)]
    train1, val1 = split_rows(rows, 0.2, seed=1)
    train2, val2 = split_rows(rows, 0.2, seed=1)
    assert train1 == train2 and val1 == val2
    assert len(val1) == 10 and len(train1) == 40
    assert not {r["image"] for r in train1} & {r["image"] for r in val1}
    assert {r["image"] for r in train1} | {r["image"] for r in val1} == {r["image"] for r in rows}

def test_split_changes_with_seed():
    rows = [{"image": f"{i}.png"} for i in range(50)]
    assert split_rows(rows, 0.2, seed=1)[1] != split_rows(rows, 0.2, seed=2)[1]

def test_split_always_has_at_least_one_validation_row():
    train, val = split_rows([{"image": "a"}, {"image": "b"}, {"image": "c"}], 0.01, seed=0)
    assert len(val) == 1 and len(train) == 2

@pytest.mark.parametrize("frac", [0, 1, -0.5, 1.5])
def test_split_validates_fraction(frac):
    with pytest.raises(ValueError):
        split_rows([{"image": "a"}, {"image": "b"}], frac, seed=0)

def test_split_needs_two_rows():
    with pytest.raises(ValueError):
        split_rows([{"image": "a"}], 0.5, seed=0)

def test_every_script_compiles():
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    expected = {
        "compare.py", "error_report.py", "evaluate.py", "finetune.py",
        "make_synthetic.py", "merge_adapter.py", "prepare_sroie.py", "split_dataset.py",
    }
    assert expected <= {p.name for p in scripts}
    for script in scripts:
        py_compile.compile(str(script), doraise=True)

def test_finetune_argument_parsing_works_without_torch():
    spec = importlib.util.spec_from_file_location("finetune", ROOT / "scripts" / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["train.jsonl", "--model-id", "some/model", "--epochs", "3"])
    assert args.epochs == 3 and args.grad_accum == 8 and args.seed == 0 and args.dtype == "auto"
    with pytest.raises(SystemExit):
        module.parse_args(["train.jsonl", "--model-id", "m", "--dtype", "float8"])
    with pytest.raises(SystemExit):
        module.parse_args(["train.jsonl"])  # --model id is required

def test_build_messages_shape():
    from receipts.vlm_io import build_messages
    from receipts.vlm_output import PROMPT

    prompt_only = build_messages()
    assert [m["role"] for m in prompt_only] == ["user"]
    assert prompt_only[0]["content"] == [{"type": "image"}, {"type": "text", "text": PROMPT}]
    with_answer = build_messages('{"vendor": null}')
    assert [m["role"] for m in with_answer] == ["user", "assistant"]
    assert with_answer[1]["content"] == [{"type": "text", "text": '{"vendor": null}'}]
    # Building the training messages must not mutate the prompt-only ones.
    assert len(build_messages()) == 1

@pytest.mark.parametrize(
    "n, accum, expected",
    [
        (8, 4, [4] * 8),
        (10, 4, [4] * 8 + [2] * 2),
        (3, 8, [3] * 3),
        (1, 1, [1]),
        (5, 1, [1] * 5),
        (0, 4, []),
    ],
)
def test_accumulation_sizes(n, accum, expected):
    from receipts.training import accumulation_sizes

    sizes = accumulation_sizes(n, accum)
    assert sizes == expected
    # Each example's weight 1/size sums to exactly one per optimizer step.
    assert sum(1 / s for s in sizes) == pytest.approx(len(set(range(0, n, accum))) if n else 0)

def test_bad_options_are_rejected_before_any_heavy_import():
    from receipts.vlm_io import load_model, resolve_dtype

    with pytest.raises(ValueError, match="dtype"):
        resolve_dtype("float8")
    with pytest.raises(ValueError, match="quantization"):
        load_model("some/model", "auto", "2bit")
    from receipts.predictors import make_vlm_predictor

    with pytest.raises(ValueError, match="merge"):
        make_vlm_predictor("some/model", adapter_path="x", quantization="dynamic-int8")
