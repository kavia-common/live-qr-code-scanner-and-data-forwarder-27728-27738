import io

import httpx
import pytest

# Import target module to access internal helpers
from src.api import main as api_main


def test_rewrite_dropbox_url_sets_dl_1_when_missing_or_zero():
    original = "https://www.dropbox.com/scl/fi/abc/file.mp4?rlkey=xyz&dl=0"
    rewritten = api_main._rewrite_dropbox_url_if_needed(original)
    assert "dropbox.com" in rewritten
    assert "dl=1" in rewritten

    original2 = "https://www.dropbox.com/s/abc/file.mp4"
    rewritten2 = api_main._rewrite_dropbox_url_if_needed(original2)
    assert "dl=1" in rewritten2


def test_rewrite_dropbox_url_keeps_non_dropbox_untouched():
    url = "https://example.com/video.mp4?token=abc"
    assert api_main._rewrite_dropbox_url_if_needed(url) == url


def test_download_video_to_temp_success(monkeypatch: pytest.MonkeyPatch):
    # Simulate a small streaming response via httpx.stream
    class DummyStreamCtx:
        def __init__(self, content: bytes):
            self._content = content
            self.status_code = 200
            self._io = io.BytesIO(content)
            self.headers = {"Content-Type": "video/mp4"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_bytes(self):
            # yield in two chunks
            yield self._content[: len(self._content) // 2]
            yield self._content[len(self._content) // 2 :]

    def fake_stream(method: str, url: str, timeout: float, follow_redirects: bool):
        assert method == "GET"
        # Ensure Dropbox links are rewritten
        assert "dl=1" in api_main._rewrite_dropbox_url_if_needed(url)
        return DummyStreamCtx(b"\x00\x01\x02\x03\x04")

    monkeypatch.setattr(httpx, "stream", fake_stream)

    path, tmpctx = api_main._download_video_to_temp(
        "https://www.dropbox.com/s/abc/video.mp4?dl=0"
    )
    try:
        # File should exist and be non-empty
        with open(path, "rb") as f:
            data = f.read()
            assert len(data) == 5
    finally:
        tmpctx.cleanup()


def test_download_video_to_temp_http_error(monkeypatch: pytest.MonkeyPatch):
    class DummyBadStreamCtx:
        def __init__(self):
            self.status_code = 404

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_bytes(self):
            # No content yielded in error
            if False:
                yield b""

    def fake_stream(method: str, url: str, timeout: float, follow_redirects: bool):
        return DummyBadStreamCtx()

    monkeypatch.setattr(httpx, "stream", fake_stream)

    with pytest.raises(RuntimeError) as ei:
        _path, ctx = api_main._download_video_to_temp("https://www.dropbox.com/s/abc/video.mp4")
    assert "Failed to fetch video from URL" in str(ei.value)
