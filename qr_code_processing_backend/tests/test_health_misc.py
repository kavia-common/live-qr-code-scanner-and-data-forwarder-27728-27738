def test_health(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Healthy"


def test_websocket_notes(client):
    resp = client.get("/docs/websocket")
    assert resp.status_code == 200
    assert "note" in resp.json()
