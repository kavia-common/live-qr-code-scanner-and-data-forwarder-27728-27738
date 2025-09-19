import numpy as np  # type: ignore
import pytest

from src.api import main as api_main


def test_decode_zxing_returns_none_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_main, "_ZXING_AVAILABLE", False)
    monkeypatch.setattr(api_main, "zxingcpp_read_barcode", None)
    img = np.zeros((10, 10), dtype=np.uint8)
    assert api_main.decode_zxing(img) is None


def test_decode_zxing_with_single_result_structure(monkeypatch: pytest.MonkeyPatch):
    class Result:
        def __init__(self, text):
            self.text = text

    def fake_read(img):
        return Result("HELLO-ZXING")

    monkeypatch.setattr(api_main, "_ZXING_AVAILABLE", True)
    monkeypatch.setattr(api_main, "zxingcpp_read_barcode", fake_read)
    img = np.zeros((10, 10), dtype=np.uint8)
    assert api_main.decode_zxing(img) == "HELLO-ZXING"


def test_decode_zxing_with_list_structure_and_decoded_text(monkeypatch: pytest.MonkeyPatch):
    class R1:
        decoded_text = ""

    class R2:
        decoded_text = "QR2"

    def fake_read(img):
        return [R1(), R2()]

    monkeypatch.setattr(api_main, "_ZXING_AVAILABLE", True)
    monkeypatch.setattr(api_main, "zxingcpp_read_barcode", fake_read)
    img = np.zeros((10, 10), dtype=np.uint8)
    assert api_main.decode_zxing(img) == "QR2"
