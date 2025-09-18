import json
from typing import Generator

import httpx
import pytest
from fastapi.testclient import TestClient

# Import app-level objects
from src.api.main import app, ServiceConfig, SCANNER


@pytest.fixture(autouse=True)
def _reset_runtime_config(monkeypatch) -> Generator[None, None, None]:
    """
    Auto-used fixture to reset CONFIG to safe deterministic defaults for each test.
    Ensures we do not perform real network calls or use real webcams by default.
    """
    # Set deterministic, fake endpoints (we won't actually call them; httpx calls are mocked)
    new_config = ServiceConfig(
        external_api_url="https://example.com/external",
        webhook_url="https://example.com/webhook",
        external_api_key="test-key",
        camera_index=0,
        scan_interval_ms=250,
        send_unique_only=True,
    )
    # Replace CONFIG in module under test
    monkeypatch.setattr("src.api.main.CONFIG", new_config, raising=True)
    # Also ensure the global SCANNER isn't running between tests
    SCANNER.stop()
    yield
    # Ensure scanner stopped at end of test
    SCANNER.stop()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient for API integration-style tests.
    """
    with TestClient(app) as c:
        yield c


class DummyResponse(httpx.Response):
    """
    Helper subclass to simplify creating httpx.Response with JSON body in tests.
    """

    @classmethod
    def json_response(cls, status_code: int, data: dict) -> "DummyResponse":
        content = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        return cls(status_code=status_code, content=content, headers=headers)
