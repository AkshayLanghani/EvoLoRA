"""Pairing a hosted website with the local agent: tokens, CORS, private network access."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evolora.web.remote_access import PairingStore, RemoteAccessMiddleware, valid_origin

SITE = "https://evolora.example"
OWN = "http://testserver"  # TestClient's Host header


@pytest.fixture
def store(tmp_path):
    return PairingStore(tmp_path / "pairings.json")


@pytest.fixture
def client(store):
    app = FastAPI()

    @app.get("/api/agent/hello")
    def hello():
        return {"app": "evolora-agent"}

    @app.post("/api/runs")
    def runs():
        return {"ok": True}

    @app.get("/index.html")
    def page():
        return {"page": True}

    app.add_middleware(RemoteAccessMiddleware, store=store)
    return TestClient(app)


@pytest.mark.parametrize(
    ("origin", "ok"),
    [
        ("https://evolora.vercel.app", True),
        ("http://localhost:3000", True),
        ("http://127.0.0.1:5173", True),
        ("http://evil.example", False),  # plain http only for this machine
        ("https://site.example/path", False),
        ("javascript:alert(1)", False),
        ("", False),
    ],
)
def test_valid_origin(origin, ok):
    assert valid_origin(origin) is ok


def test_pair_check_and_revoke(store):
    token = store.pair(SITE)
    assert store.check(SITE, token)
    assert not store.check(SITE, "wrong")
    assert not store.check("https://other.example", token)  # tokens are per origin
    assert store.origins() == [SITE]

    newer = store.pair(SITE)
    assert not store.check(SITE, token) and store.check(SITE, newer)  # re-pairing rotates

    assert store.revoke(SITE)
    assert not store.check(SITE, newer)


def test_tokens_are_not_stored_in_plain_text(store, tmp_path):
    token = store.pair(SITE)
    assert token not in (tmp_path / "pairings.json").read_text()


def test_same_origin_requests_pass_untouched(client):
    assert client.post("/api/runs").status_code == 200
    assert client.post("/api/runs", headers={"Origin": OWN}).status_code == 200


def test_unpaired_site_is_rejected_but_can_detect_the_agent(client):
    denied = client.post("/api/runs", headers={"Origin": SITE})
    assert denied.status_code == 401
    assert denied.headers["access-control-allow-origin"] == SITE  # readable error

    hello = client.get("/api/agent/hello", headers={"Origin": SITE})
    assert hello.status_code == 200
    assert hello.headers["access-control-allow-origin"] == SITE


def test_paired_site_with_token_is_allowed(client, store):
    token = store.pair(SITE)
    ok = client.post("/api/runs", headers={"Origin": SITE, "Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == SITE

    stolen = client.post(
        "/api/runs", headers={"Origin": "https://evil.example", "Authorization": f"Bearer {token}"}
    )
    assert stolen.status_code == 401


def test_preflight_answers_private_network_access(client):
    resp = client.options(
        "/api/runs",
        headers={
            "Origin": SITE,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert resp.status_code == 204
    assert resp.headers["access-control-allow-private-network"] == "true"
    assert "authorization" in resp.headers["access-control-allow-headers"]


def test_preflight_from_disallowed_origin_is_refused(client):
    resp = client.options("/api/runs", headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403


def test_non_api_paths_are_not_affected(client):
    assert client.get("/index.html", headers={"Origin": SITE}).status_code == 200
