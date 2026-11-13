import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

def load_compare():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def report(vendor, date, total, all_fields, parse_fail, latency=None, size_mb=None):
    r = {
        "metrics": {
            "field_accuracy": {"vendor": vendor, "date": date, "total": total},
            "all_fields_accuracy": all_fields,
            "parse_failure_rate": parse_fail,
        }
    }
    if latency:
        r["latency"] = latency
    if size_mb is not None:
        r["model_size_mb"] = size_mb
    return r

def test_table_values_and_blank_latency(tmp_path):
    compare = load_compare()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(report(0.5, 0.75, 0.25, 0.1, 0.0)))
    b.write_text(
        json.dumps(report(0.9, 0.95, 0.85, 0.8, 0.05, {"median_ms": 123.4, "p95_ms": 200.6}, size_mb=512.4))
    )
    text = compare.main([f"ocr={a}", f"tuned={b}"])
    lines = text.splitlines()
    assert lines[0].split() == [
        "model", "vendor", "date", "total", "all", "fields", "parse", "fail",
        "median", "ms", "p95", "ms", "size", "MB",
    ]
    assert lines[2].split() == ["ocr", "50.0%", "75.0%", "25.0%", "10.0%", "0.0%"]
    assert lines[3].split() == ["tuned", "90.0%", "95.0%", "85.0%", "80.0%", "5.0%", "123", "201", "512"]

def test_rows_keep_given_order(tmp_path):
    compare = load_compare()
    paths = []
    for name in ("x", "y", "z"):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(report(1, 1, 1, 1, 0)))
        paths.append(f"{name}={p}")
    text = compare.main(list(reversed(paths)))
    assert [l.split()[0] for l in text.splitlines()[2:]] == ["z", "y", "x"]

def test_bad_argument_rejected():
    compare = load_compare()
    with pytest.raises(ValueError, match="label=path"):
        compare.parse_pairs(["no-equals-sign"])

def test_no_arguments_exits():
    compare = load_compare()
    with pytest.raises(SystemExit):
        compare.main([])

def test_works_on_a_real_evaluate_report(tmp_path):
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    spec = importlib.util.spec_from_file_location("make_synthetic", ROOT / "scripts" / "make_synthetic.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)

    jsonl = ms.build(tmp_path / "ds", 4, start_seed=900)
    out = tmp_path / "ocr.json"
    ev.main([str(jsonl), "--predictor", "ocr", "--latency", "--out", str(out)])
    text = load_compare().main([f"ocr={out}"])
    assert "ocr" in text and "%" in text
