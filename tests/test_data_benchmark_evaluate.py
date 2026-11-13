import importlib.util
import json
from pathlib import Path

import pytest

from receipts.benchmark import measure_latency, percentile
from receipts.data import load_dataset, load_jsonl, truth_of, write_jsonl

ROOT = Path(__file__).resolve().parent.parent

def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Data Loading:

def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"image": "a.png", "vendor": "A"}\n\n{"image": "b.png"}\n')
    rows = load_jsonl(p)
    assert [r["image"] for r in rows] == ["a.png", "b.png"]

def test_load_jsonl_reports_line_number_for_bad_json(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"image": "a.png"}\n{oops}\n')
    with pytest.raises(ValueError, match=":2:"):
        load_jsonl(p)

def test_load_jsonl_requires_image_key(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"vendor": "A"}\n')
    with pytest.raises(ValueError, match="image"):
        load_jsonl(p)

def test_load_jsonl_rejects_empty_file(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="no rows"):
        load_jsonl(p)

def test_truth_of_keeps_only_fields():
    assert truth_of({"image": "x", "vendor": "A", "extra": 1}) == {
        "vendor": "A",
        "date": None,
        "total": None,
    }

def test_synthetic_dataset_round_trip(tmp_path):
    make_synthetic = _load_script("make_synthetic")
    jsonl = make_synthetic.build(tmp_path / "ds", 3, start_seed=5)
    data = load_dataset(jsonl)
    assert len(data) == 3
    image, truth = data[0]
    assert image.mode == "RGB"
    assert set(truth) == {"vendor", "date", "total"}
    # Written rows match regenerated truths.
    rows = load_jsonl(jsonl)
    assert rows[1]["vendor"] == data[1][1]["vendor"]

def test_write_jsonl_round_trip(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"image": "a", "total": "1.00"}])
    assert json.loads(p.read_text().strip()) == {"image": "a", "total": "1.00"}


# Benchmark:
def test_percentile_known_values():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == 3
    assert percentile(values, 100) == 5
    assert percentile(values, 25) == 2
    assert percentile([7], 95) == 7

def test_percentile_interpolates():
    assert percentile([0, 10], 50) == pytest.approx(5)
    assert percentile([0, 10], 95) == pytest.approx(9.5)

def test_percentile_validation():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)

def test_measure_latency_counts_calls_and_skips_warmup():
    calls = []

    def predict(x):
        calls.append(x)

    stats = measure_latency(predict, ["a", "b", "c"], warmup=2, repeats=2)
    assert len(calls) == 2 + 3 * 2
    assert stats["n_calls"] == 6
    assert stats["median_ms"] >= 0
    assert stats["p95_ms"] >= stats["median_ms"]

def test_measure_latency_reflects_slow_calls():
    import time

    def slow(_):
        time.sleep(0.02)

    stats = measure_latency(slow, [1, 2, 3], warmup=0)
    assert stats["median_ms"] >= 15
    assert stats["images_per_s"] < 80

def test_measure_latency_requires_images():
    with pytest.raises(ValueError):
        measure_latency(lambda x: x, [])

# Evaluating Scripts:
def test_run_with_fake_predictor_gives_known_accuracy():
    evaluate_script = _load_script("evaluate")
    from PIL import Image

    img = Image.new("RGB", (4, 4))
    truths = [
        {"vendor": "A", "date": "2026-01-01", "total": "1.00"},
        {"vendor": "B", "date": "2026-01-02", "total": "2.00"},
    ]
    dataset = [(img, t) for t in truths]
    answers = iter([dict(truths[0]), None])
    def fake(_img):
        fake.last_text = "raw text"
        return next(answers)

    preds, result, raws = evaluate_script.run(fake, dataset)
    assert preds[1] is None
    assert result["all_fields_accuracy"] == 0.5
    assert result["parse_failure_rate"] == 0.5
    assert raws == ["raw text", "raw text"]

def test_run_without_last_text_attribute_gives_none_raws():
    evaluate_script = _load_script("evaluate")
    from PIL import Image

    truth = {"vendor": "A", "date": "2026-01-01", "total": "1.00"}
    _, _, raws = evaluate_script.run(lambda _img: dict(truth), [(Image.new("RGB", (2, 2)), truth)])
    assert raws == [None]

def test_evaluate_cli_ocr_on_synthetic_set(tmp_path, capsys):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 8, start_seed=300)
    out = tmp_path / "report.json"
    report = evaluate_script.main([str(jsonl), "--predictor", "ocr", "--latency", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "total accuracy" in printed and "latency" in printed
    assert report["metrics"]["field_accuracy"]["total"] >= 0.85
    saved = json.loads(out.read_text())
    assert len(saved["predictions"]) == 8
    assert saved["latency"]["n_calls"] == 8
    # Raw OCR text is kept so rule failures can be debugged later.
    assert len(saved["raw_outputs"]) == 8
    assert all(isinstance(t, str) and t.strip() for t in saved["raw_outputs"])

def test_show_raw_prints_raw_text(tmp_path, capsys):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 3, start_seed=500)
    evaluate_script.main([str(jsonl), "--predictor", "ocr", "--show-raw", "2"])
    printed = capsys.readouterr().out
    assert printed.count("--- raw output, receipt") == 2
    assert "TOTAL" in printed

def test_limit_uses_only_the_first_receipts(tmp_path, capsys):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 6, start_seed=700)
    report = evaluate_script.main([str(jsonl), "--predictor", "ocr", "--limit", "2", "--latency"])
    assert report["metrics"]["n"] == 2
    assert len(report["predictions"]) == 2 and len(report["raw_outputs"]) == 2
    assert report["latency"]["n_calls"] == 2
    assert "(n=2)" in capsys.readouterr().out
    # A limit larger than the data just uses everything.
    assert evaluate_script.main([str(jsonl), "--predictor", "ocr", "--limit", "99"])["metrics"]["n"] == 6

@pytest.mark.parametrize("bad", ["0", "-3"])
def test_limit_must_be_positive(tmp_path, bad):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 2, start_seed=710)
    with pytest.raises(SystemExit):
        evaluate_script.main([str(jsonl), "--predictor", "ocr", "--limit", bad])
    with pytest.raises(ValueError, match="at least 1"):
        load_dataset(jsonl, limit=int(bad))

def test_out_creates_missing_parent_folder(tmp_path):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 2, start_seed=600)
    out = tmp_path / "new" / "deeper" / "report.json"
    evaluate_script.main([str(jsonl), "--predictor", "ocr", "--out", str(out)])
    assert out.is_file()

def test_evaluate_cli_requires_model_id_for_vlm(tmp_path):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 1)
    with pytest.raises(SystemExit):
        evaluate_script.main([str(jsonl), "--predictor", "vlm"])

def test_latency_images_limits_what_is_timed(tmp_path):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 5, start_seed=800)
    report = evaluate_script.main([str(jsonl), "--predictor", "ocr", "--latency", "--latency-images", "2"])
    assert report["metrics"]["n"] == 5  # all five scored
    assert report["latency"]["n_calls"] == 2  # only two timed

def test_latency_images_must_be_positive(tmp_path):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 2, start_seed=810)
    with pytest.raises(SystemExit):
        evaluate_script.main([str(jsonl), "--predictor", "ocr", "--latency", "--latency-images", "0"])

def test_scores_are_saved_even_if_timing_is_interrupted(tmp_path, monkeypatch, capsys):
    make_synthetic = _load_script("make_synthetic")
    evaluate_script = _load_script("evaluate")
    jsonl = make_synthetic.build(tmp_path / "ds", 2, start_seed=820)
    out = tmp_path / "report.json"

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluate_script, "measure_latency", interrupted)
    report = evaluate_script.main([str(jsonl), "--predictor", "ocr", "--latency", "--out", str(out)])
    assert "latency" not in report
    assert "skipped" in capsys.readouterr().out
    saved = json.loads(out.read_text())
    assert saved["metrics"]["n"] == 2 and "latency" not in saved
