def test_get_config_returns_current_values(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "config" in body
    cfg = body["config"]
    # Required fields should exist
    assert "external_api_url" in cfg
    assert "webhook_url" in cfg
    assert isinstance(cfg["scan_interval_ms"], int)


def test_patch_config_updates_values(client):
    updated = {
        "external_api_url": "https://example.org/new-api",
        "webhook_url": "https://example.org/new-hook",
        "camera_index": 2,
        "scan_interval_ms": 333,
        "send_unique_only": False,
        "external_api_key": "new-key",
    }
    resp = client.patch("/config", json=updated)
    assert resp.status_code == 200
    cfg = resp.json()["config"]
    for k, v in updated.items():
        assert cfg[k] == v


def test_patch_config_rejects_invalid_url_format(client):
    # invalid URL (not a uri)
    resp = client.patch("/config", json={"external_api_url": "not-a-url"})
    assert resp.status_code == 422


def test_patch_config_rejects_invalid_scan_interval(client):
    # interval too small
    resp = client.patch("/config", json={"scan_interval_ms": 10})
    assert resp.status_code == 422

    # interval too large
    resp = client.patch("/config", json={"scan_interval_ms": 999999})
    assert resp.status_code == 422
