from typing import Any, Dict

import httpx

# Import the module dynamically so CONFIG is referenced at runtime (supports monkeypatch)
from src.api import main as api_main


def test_send_to_external_includes_auth_and_payload(monkeypatch):
    """
    Verify that _send_to_external sends JSON body with qr_data and includes Authorization header when API key is set.
    """
    sent: Dict[str, Any] = {}

    class FakeClient:
        def post(self, url, json=None, headers=None):
            sent["url"] = url
            sent["json"] = json
            sent["headers"] = headers or {}
            # Simulate HTTP 200 response
            return httpx.Response(status_code=200, json={"ok": True})

    client = FakeClient()
    resp = api_main.QRScannerService._send_to_external(client, "HELLO-QR")  # type: ignore[arg-type]
    assert resp.status_code == 200
    assert sent["url"] == str(api_main.CONFIG.external_api_url)
    assert sent["json"] == {"qr_data": "HELLO-QR"}
    assert "Authorization" in sent["headers"]
    assert sent["headers"]["Authorization"] == f"Bearer {api_main.CONFIG.external_api_key}"


def test_forward_to_webhook_sends_summary(monkeypatch):
    """
    Verify that _forward_to_webhook posts summarized payload of the external response to webhook URL.
    """
    # Create a fake external response
    external_resp = httpx.Response(status_code=201, json={"external": "payload", "status": "created"})

    captured: Dict[str, Any] = {}

    class FakeClient:
        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers or {}
            return httpx.Response(status_code=202, json={"webhook": "accepted"})

    client = FakeClient()
    wh_resp = api_main.QRScannerService._forward_to_webhook(client, external_resp)  # type: ignore[arg-type]
    assert wh_resp.status_code == 202
    assert captured["url"] == str(api_main.CONFIG.webhook_url)

    payload = captured["json"]
    assert payload["source"] == "qr-code-processing-backend"
    assert payload["external_status_code"] == 201
    assert isinstance(payload["external_headers"], dict)
    assert payload["external_body"] == {"external": "payload", "status": "created"}
    assert captured["headers"]["Content-Type"] == "application/json"
