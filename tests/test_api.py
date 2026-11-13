import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app, predictor_from_env
from receipts.synth import make_receipt

def png_bytes(image=None):
    image = image or Image.new("RGB", (20, 20), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def client_for(predictor, **kwargs):
    # The context manager runs the lifespan, which installs the predictor.
    return TestClient(create_app(predictor, **kwargs))

def post(client, data, name="r.png", ctype="image/png"):
    return client.post("/extract", files={"file": (name, data, ctype)})

def test_health():
    with client_for(lambda img: None, predictor_name="fake") as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "predictor": "fake"}

def test_extract_returns_prediction_and_latency():
    fake = lambda img: {"vendor": "Shop", "date": "2026-01-02", "total": "9.99"}
    with client_for(fake, predictor_name="fake") as c:
        r = post(c, png_bytes())
        assert r.status_code == 200
        body = r.json()
        assert body["vendor"] == "Shop"
        assert body["date"] == "2026-01-02"
        assert body["total"] == "9.99"
        assert body["predictor"] == "fake"
        assert body["latency_ms"] >= 0

def test_missing_fields_come_back_as_null():
    with client_for(lambda img: {"vendor": "Shop"}) as c:
        body = post(c, png_bytes()).json()
        assert body["vendor"] == "Shop"
        assert body["date"] is None and body["total"] is None

def test_predictor_receives_rgb_image():
    seen = {}

    def spy(img):
        seen["mode"] = img.mode
        return {}

    gray = Image.new("L", (10, 10))
    with client_for(spy) as c:
        assert post(c, png_bytes(gray)).status_code == 200
    assert seen["mode"] == "RGB"

def test_not_an_image_is_400():
    with client_for(lambda img: {}) as c:
        r = post(c, b"this is not an image", name="x.txt", ctype="text/plain")
        assert r.status_code == 400
        assert "image" in r.json()["detail"]

def test_truncated_image_is_400():
    data = png_bytes()[:30]
    with client_for(lambda img: {}) as c:
        assert post(c, data).status_code == 400

def test_empty_file_is_400():
    with client_for(lambda img: {}) as c:
        assert post(c, b"").status_code == 400

def test_missing_file_field_is_422():
    with client_for(lambda img: {}) as c:
        assert c.post("/extract").status_code == 422

def test_oversized_upload_is_413(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")  # about 1 KB
    big = png_bytes(Image.effect_noise((200, 200), 80).convert("RGB"))
    assert len(big) > 1048
    with client_for(lambda img: {}) as c:
        assert post(c, big).status_code == 413

def test_model_returning_none_is_422():
    with client_for(lambda img: None) as c:
        r = post(c, png_bytes())
        assert r.status_code == 422
        assert "no usable output" in r.json()["detail"]

def test_predictor_exception_is_500_without_leaking_details():
    def boom(img):
        raise RuntimeError("secret internal path /models/abc")

    with client_for(boom) as c:
        r = post(c, png_bytes())
        assert r.status_code == 500
        assert "secret" not in r.text

def test_predictor_is_never_called_concurrently():
    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def slow(img):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with guard:
            active["now"] -= 1
        return {}

    with client_for(slow) as c:
        data = png_bytes()
        results = []

        def call():
            results.append(post(c, data).status_code)

        threads = [threading.Thread(target=call) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert results == [200] * 6
    assert active["max"] == 1

def test_real_ocr_predictor_through_the_api(monkeypatch):
    monkeypatch.setenv("PREDICTOR", "ocr")
    from fastapi import FastAPI  # noqa: F401

    image, truth = make_receipt(11)
    with TestClient(create_app()) as c:
        r = post(c, png_bytes(image))
        assert r.status_code == 200
        body = r.json()
        assert body["predictor"] == "ocr"
        assert body["total"] == truth["total"]
        assert body["date"] == truth["date"]

def test_predictor_from_env_rejects_bad_config(monkeypatch):
    monkeypatch.setenv("PREDICTOR", "banana")
    with pytest.raises(RuntimeError, match="unknown PREDICTOR"):
        predictor_from_env()
    monkeypatch.setenv("PREDICTOR", "vlm")
    monkeypatch.delenv("MODEL_ID", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_ID"):
        predictor_from_env()

def test_predictor_from_env_rejects_an_adapter_path_that_is_not_an_adapter_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDICTOR", "vlm")
    monkeypatch.setenv("MODEL_ID", "some/model")
    for bad in (str(tmp_path / "missing"), str(tmp_path), "someone/hub-adapter"):
        monkeypatch.setenv("ADAPTER_PATH", bad)
        with pytest.raises(RuntimeError, match="not an adapter folder"):
            predictor_from_env()
