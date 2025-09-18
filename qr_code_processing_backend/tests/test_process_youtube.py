from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# We import from the app module under test
from src.api import main as api_main


def test_process_youtube_success_with_detected_codes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that POST /process_youtube returns the expected structure when QR codes are detected.
    We mock out pytube.YouTube and the internal video processing function to avoid network and heavy CPU.
    """
    # 1) Mock pytube.YouTube to bypass network access and file download.
    class FakeStream:
        def download(self, output_path: str, filename: str) -> str:
            # Simulate a download that would output a file path.
            # The path value is not used directly by the test because the detection function is also mocked.
            return f"{output_path}/{filename}"

    class FakeStreams:
        def filter(self, progressive: bool = None, file_extension: str = None):
            return self

        def order_by(self, key: str):
            return self

        def first(self):
            return FakeStream()

    class FakeYouTube:
        def __init__(self, url: str) -> None:
            self.url = url
            self.streams = FakeStreams()

    monkeypatch.setattr(api_main, "YouTube", FakeYouTube)

    # 2) Mock the internal detection to simulate finding a QR code quickly.
    def fake_detect(path: str, max_frames: int = 1500, frame_stride: int = 5, stop_after_first: bool = True) -> Dict[str, Any]:
        assert isinstance(path, str)
        # Simulate that some frames were scanned and a QR code was found
        return {"detected": ["QR-HELLO-WORLD"], "frames_scanned": 3}

    monkeypatch.setattr(api_main, "_detect_qr_from_video_file", fake_detect)

    # 3) Call the endpoint with a plausible YouTube URL
    payload = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "max_frames": 10,
        "frame_stride": 2,
        "stop_after_first": True,
    }
    resp = client.post("/process_youtube", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    # 4) Validate response structure and contents
    assert "message" in body
    assert "detected" in body
    assert "frames_scanned" in body

    assert body["message"] == "Scan complete"
    assert isinstance(body["detected"], list)
    assert body["detected"] == ["QR-HELLO-WORLD"]
    assert isinstance(body["frames_scanned"], int)
    assert body["frames_scanned"] == 3


def test_process_youtube_success_no_codes_found(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that POST /process_youtube returns 'No QR codes detected' when none are found.
    """
    # Mock YouTube the same as above to avoid network access
    class FakeStream:
        def download(self, output_path: str, filename: str) -> str:
            return f"{output_path}/{filename}"

    class FakeStreams:
        def filter(self, progressive: bool = None, file_extension: str = None):
            return self

        def order_by(self, key: str):
            return self

        def first(self):
            return FakeStream()

    class FakeYouTube:
        def __init__(self, url: str) -> None:
            self.url = url
            self.streams = FakeStreams()

    monkeypatch.setattr(api_main, "YouTube", FakeYouTube)

    # Mock detection: no codes found
    def fake_detect_none(path: str, max_frames: int = 1500, frame_stride: int = 5, stop_after_first: bool = True) -> Dict[str, Any]:
        return {"detected": [], "frames_scanned": 5}

    monkeypatch.setattr(api_main, "_detect_qr_from_video_file", fake_detect_none)

    resp = client.post("/process_youtube", json={"url": "https://www.youtube.com/watch?v=abc123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "No QR codes detected"
    assert body["detected"] == []
    assert body["frames_scanned"] == 5


def test_process_youtube_rejects_invalid_url(client: TestClient) -> None:
    """
    Verify validation error is raised for invalid URL format.
    """
    resp = client.post("/process_youtube", json={"url": "not-a-valid-url"})
    assert resp.status_code == 422
