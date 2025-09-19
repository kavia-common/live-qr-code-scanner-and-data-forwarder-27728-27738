from src.api.main import SCANNER


def test_scanner_status_initial(client):
    resp = client.get("/scanner/status")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["running"] is False


def test_scanner_start_success(client, monkeypatch):
    # Mock SCANNER methods
    monkeypatch.setattr(SCANNER, "is_running", lambda: False)
    started = {"called": False}

    def fake_start(camera_index: int, scan_interval_ms: int):
        started["called"] = True
        assert isinstance(camera_index, int)
        assert isinstance(scan_interval_ms, int)

    monkeypatch.setattr(SCANNER, "start", fake_start)

    resp = client.post("/scanner/start", json={"camera_index": 1, "scan_interval_ms": 300})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "started"
    assert started["called"] is True


def test_scanner_start_defaults_when_no_body(client, monkeypatch):
    # If no body provided, defaults from CONFIG should be used and start called
    monkeypatch.setattr(SCANNER, "is_running", lambda: False)
    called = {"args": None}

    def fake_start(camera_index: int, scan_interval_ms: int):
        called["args"] = (camera_index, scan_interval_ms)

    monkeypatch.setattr(SCANNER, "start", fake_start)
    resp = client.post("/scanner/start")
    assert resp.status_code == 202
    assert resp.json()["status"] == "started"
    assert isinstance(called["args"][0], int)
    assert isinstance(called["args"][1], int)


def test_scanner_start_already_running(client, monkeypatch):
    monkeypatch.setattr(SCANNER, "is_running", lambda: True)
    resp = client.post("/scanner/start")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "already_running"


def test_scanner_start_handles_failure(client, monkeypatch):
    monkeypatch.setattr(SCANNER, "is_running", lambda: False)

    def fake_start(camera_index: int, scan_interval_ms: int):
        raise RuntimeError("camera error")

    monkeypatch.setattr(SCANNER, "start", fake_start)

    resp = client.post("/scanner/start")
    assert resp.status_code == 500
    assert "Failed to start scanner" in resp.json()["detail"]


def test_scanner_stop_returns_stopped(client, monkeypatch):
    # Ensure stop invoked
    called = {"stop": False}

    def fake_stop():
        called["stop"] = True

    monkeypatch.setattr(SCANNER, "stop", fake_stop)
    resp = client.post("/scanner/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "stopped"
    assert called["stop"] is True


def test_scanner_stop_idempotent_when_already_stopped(client, monkeypatch):
    # Stopping again should still return "stopped"
    called = {"stop": 0}

    def fake_stop():
        called["stop"] += 1

    monkeypatch.setattr(SCANNER, "stop", fake_stop)
    resp1 = client.post("/scanner/stop")
    resp2 = client.post("/scanner/stop")
    assert resp1.status_code == 200 and resp2.status_code == 200
    assert resp1.json()["status"] == "stopped"
    assert resp2.json()["status"] == "stopped"
    assert called["stop"] == 2
