from typing import Any, Dict, Tuple
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


def test_process_video_success_scan_complete(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    # Mock download to temp to avoid network and OpenCV
    def fake_download(url: str) -> Tuple[str, tempfile.TemporaryDirectory]:
        # Provide a real temp file path to satisfy downstream open
        ctx = tempfile.TemporaryDirectory()
        path = ctx.name + "/video.mp4"
        # Create empty file to pass existence check; _detect_qr_from_video_file is mocked anyway
        with open(path, "wb") as f:
            f.write(b"")
        return path, ctx

    monkeypatch.setattr(api_main, "_download_video_to_temp", fake_download)

    # Mock detection result
    def fake_detect(
        path: str,
        max_frames: int = 1500,
        frame_stride: int = 5,
        stop_after_first: bool = True,
        grayscale: bool = False,
        adaptive_threshold: bool = False,
        denoise: bool = False,
        decoder_backend: str = "auto",
        use_opencv: bool = True,
        use_pyzbar: bool = True,
        use_zxing: bool = True,
    ) -> Dict[str, Any]:
        # Validate flags mapping flowed through
        assert max_frames == 50
        assert frame_stride == 2
        assert stop_after_first is False
        assert grayscale is True
        assert adaptive_threshold is True
        assert denoise is True
        assert decoder_backend == "opencv"
        assert use_opencv is True
        assert use_pyzbar is False
        assert use_zxing is False
        return {"detected": ["QR1"], "frames_scanned": 7}

    monkeypatch.setattr(api_main, "_detect_qr_from_video_file", fake_detect)

    payload = {
        "url": "https://www.dropbox.com/s/abc/video.mp4?dl=0",
        "max_frames": 50,
        "frame_stride": 2,
        "stop_after_first": False,
        "grayscale": True,
        "adaptive_threshold": True,
        "denoise": True,
        "decoder_backend": "opencv",
        "use_opencv": True,
        "use_pyzbar": False,
        "use_zxing": False,
    }
    resp = client.post("/process_video", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "Scan complete"
    assert body["detected"] == ["QR1"]
    assert body["frames_scanned"] == 7


def test_process_video_alias_flags_mapping(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    # No explicit use_pyzbar/use_zxing provided; provide aliases and ensure mapping
    def fake_download(url: str) -> Tuple[str, tempfile.TemporaryDirectory]:
        ctx = tempfile.TemporaryDirectory()
        path = ctx.name + "/video.mp4"
        open(path, "wb").close()
        return path, ctx

    monkeypatch.setattr(api_main, "_download_video_to_temp", fake_download)

    recorded: Dict[str, Any] = {}

    def fake_detect(
        path: str,
        max_frames: int = 1500,
        frame_stride: int = 5,
        stop_after_first: bool = True,
        grayscale: bool = False,
        adaptive_threshold: bool = False,
        denoise: bool = False,
        decoder_backend: str = "auto",
        use_opencv: bool = True,
        use_pyzbar: bool = True,
        use_zxing: bool = True,
    ) -> Dict[str, Any]:
        recorded.update(
            dict(
                max_frames=max_frames,
                frame_stride=frame_stride,
                stop_after_first=stop_after_first,
                use_opencv=use_opencv,
                use_pyzbar=use_pyzbar,
                use_zxing=use_zxing,
                decoder_backend=decoder_backend,
            )
        )
        return {"detected": [], "frames_scanned": 1}

    monkeypatch.setattr(api_main, "_detect_qr_from_video_file", fake_detect)

    payload = {
        "url": "https://example.com/v.mp4",
        "decoder_backend": "auto",
        # aliases supplied; should map to use_pyzbar/use_zxing
        "use_pyzbar_fallback": False,
        "use_zxingcpp": False,
    }
    resp = client.post("/process_video", json=payload)
    assert resp.status_code == 200
    assert recorded["decoder_backend"] == "auto"
    assert recorded["use_pyzbar"] is False
    assert recorded["use_zxing"] is False
    # Defaults preserved for others
    assert recorded["use_opencv"] is True
    assert resp.json()["message"] == "No QR codes detected"


def test_process_video_handles_download_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_download(url: str):
        raise RuntimeError("network issue")

    monkeypatch.setattr(api_main, "_download_video_to_temp", fake_download)

    resp = client.post("/process_video", json={"url": "https://example.com/file.mp4"})
    assert resp.status_code == 500
    assert "Failed to process video URL" in resp.json()["detail"]
