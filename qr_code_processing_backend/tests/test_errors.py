import httpx

from src.api.main import QRScannerService


def test_process_returns_502_on_request_error_to_external(client, monkeypatch):
    def fake_send(client_obj, data: str):
        raise httpx.RequestError("network down")

    # _forward_to_webhook won't be called, but mock anyway
    def fake_forward(client_obj, external_resp: httpx.Response):
        return httpx.Response(status_code=200)

    monkeypatch.setattr(QRScannerService, "_send_to_external", staticmethod(fake_send))
    monkeypatch.setattr(QRScannerService, "_forward_to_webhook", staticmethod(fake_forward))

    resp = client.post("/process", json={"data": "X"})
    assert resp.status_code == 502
    assert "Network error while contacting external/webhook" in resp.json()["detail"]


def test_process_returns_500_on_unexpected_exception(client, monkeypatch):
    def fake_send(client_obj, data: str):
        # Simulate unexpected bug
        raise RuntimeError("boom")

    def fake_forward(client_obj, external_resp: httpx.Response):
        return httpx.Response(status_code=200)

    monkeypatch.setattr(QRScannerService, "_send_to_external", staticmethod(fake_send))
    monkeypatch.setattr(QRScannerService, "_forward_to_webhook", staticmethod(fake_forward))

    resp = client.post("/process", json={"data": "X"})
    assert resp.status_code == 500
    assert "Processing failed" in resp.json()["detail"]
