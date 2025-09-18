import httpx
from src.api.main import QRScannerService


def test_process_success_happy_path(client, monkeypatch):
    """
    Process QR with mocked external and webhook calls; verify response aggregation.
    """

    # Mock external
    def fake_send(client_obj, data: str):
        assert data == "TEST-QR"
        return httpx.Response(status_code=200, json={"received": True, "data": data})

    # Mock webhook
    def fake_forward(client_obj, external_resp: httpx.Response):
        # Typical webhook 204/no content is OK; we'll return 204 with empty json
        return httpx.Response(status_code=204, content=b"")

    monkeypatch.setattr(QRScannerService, "_send_to_external", staticmethod(fake_send))
    monkeypatch.setattr(QRScannerService, "_forward_to_webhook", staticmethod(fake_forward))

    resp = client.post("/process", json={"data": "TEST-QR"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["message"] == "Processed and forwarded"
    assert payload["external_status"] == 200
    assert payload["forwarded_status"] == 204
    assert payload["external_response_echo"] == {"received": True, "data": "TEST-QR"}


def test_process_handles_non_json_external_body(client, monkeypatch):
    """
    When external returns non-JSON body, the service should include raw text in echo.
    """
    def fake_send(client_obj, data: str):
        return httpx.Response(status_code=200, content=b"plain text response", headers={"Content-Type": "text/plain"})

    def fake_forward(client_obj, external_resp: httpx.Response):
        return httpx.Response(status_code=200, json={"ok": True})

    monkeypatch.setattr(QRScannerService, "_send_to_external", staticmethod(fake_send))
    monkeypatch.setattr(QRScannerService, "_forward_to_webhook", staticmethod(fake_forward))

    resp = client.post("/process", json={"data": "ANY"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["external_status"] == 200
    assert payload["external_response_echo"] == {"raw": "plain text response"}
