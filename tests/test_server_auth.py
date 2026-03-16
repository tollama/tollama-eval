"""Tests for API authentication and authorization."""

from __future__ import annotations

import pytest

from ts_autopilot.server.auth import (
    APIKeyRecord,
    APIKeyStore,
    AuthConfig,
    AuthUser,
    hash_api_key,
    require_role,
    setup_auth,
    verify_api_key,
)

# Skip if fastapi is not installed
fastapi = pytest.importorskip("fastapi")


# --- Unit tests for key hashing ---


def test_hash_api_key_deterministic() -> None:
    assert hash_api_key("sk-test123") == hash_api_key("sk-test123")


def test_hash_api_key_different_keys() -> None:
    assert hash_api_key("sk-a") != hash_api_key("sk-b")


def test_verify_api_key_correct() -> None:
    key = "sk-mykey"
    h = hash_api_key(key)
    assert verify_api_key(key, h) is True


def test_verify_api_key_wrong() -> None:
    h = hash_api_key("sk-correct")
    assert verify_api_key("sk-wrong", h) is False


# --- Unit tests for APIKeyStore ---


def test_store_add_and_authenticate() -> None:
    store = APIKeyStore()
    store.add_key(
        APIKeyRecord(
            name="test-key",
            key_hash=hash_api_key("sk-test"),
            tenant_id="t1",
            roles=["operator"],
        )
    )
    user = store.authenticate("sk-test")
    assert user is not None
    assert user.user_id == "test-key"
    assert user.tenant_id == "t1"
    assert "operator" in user.roles


def test_store_authenticate_wrong_key() -> None:
    store = APIKeyStore()
    store.add_key(
        APIKeyRecord(name="k", key_hash=hash_api_key("sk-real"))
    )
    assert store.authenticate("sk-fake") is None


def test_store_revoke_key() -> None:
    store = APIKeyStore()
    h = hash_api_key("sk-revoke")
    store.add_key(APIKeyRecord(name="k", key_hash=h))
    assert store.authenticate("sk-revoke") is not None
    store.revoke_key(h)
    assert store.authenticate("sk-revoke") is None


def test_store_list_keys() -> None:
    store = APIKeyStore()
    store.add_key(APIKeyRecord(name="a", key_hash=hash_api_key("sk-a")))
    store.add_key(APIKeyRecord(name="b", key_hash=hash_api_key("sk-b")))
    keys = store.list_keys()
    assert len(keys) == 2
    names = {k["name"] for k in keys}
    assert names == {"a", "b"}
    # Hashes should not appear in listing
    for k in keys:
        assert "key_hash" not in k


def test_store_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLAMA_API_KEYS", "admin:sk-admin:tenant1,reader:sk-read")
    store = APIKeyStore.from_env()
    admin = store.authenticate("sk-admin")
    assert admin is not None
    assert admin.tenant_id == "tenant1"
    reader = store.authenticate("sk-read")
    assert reader is not None
    assert reader.tenant_id == ""


def test_store_from_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOLLAMA_API_KEYS", raising=False)
    store = APIKeyStore.from_env()
    assert store.authenticate("anything") is None


def test_store_from_file(tmp_path: object) -> None:
    import json
    from pathlib import Path

    p = Path(str(tmp_path)) / "keys.json"
    records = [
        {
            "name": "file-key",
            "key_hash": hash_api_key("sk-file"),
            "tenant_id": "t2",
            "roles": ["admin"],
        }
    ]
    p.write_text(json.dumps(records))
    store = APIKeyStore.from_file(p)
    user = store.authenticate("sk-file")
    assert user is not None
    assert user.tenant_id == "t2"
    assert "admin" in user.roles


# --- Integration tests with FastAPI ---


@pytest.fixture()
def auth_client(monkeypatch: pytest.MonkeyPatch):
    """Create a TestClient with API key auth enabled."""
    monkeypatch.setenv("TOLLAMA_API_KEYS", "tester:sk-test123")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    config = AuthConfig(mode="apikey")
    setup_auth(app, config)

    @app.get("/protected")
    async def protected(request: fastapi.Request):
        return {"user": request.state.auth_user.user_id}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return TestClient(app)


def test_health_no_auth_required(auth_client) -> None:
    resp = auth_client.get("/health")
    assert resp.status_code == 200


def test_protected_no_token(auth_client) -> None:
    resp = auth_client.get("/protected")
    assert resp.status_code == 401


def test_protected_invalid_token(auth_client) -> None:
    resp = auth_client.get(
        "/protected", headers={"Authorization": "Bearer sk-wrong"}
    )
    assert resp.status_code == 403


def test_protected_valid_token(auth_client) -> None:
    resp = auth_client.get(
        "/protected", headers={"Authorization": "Bearer sk-test123"}
    )
    assert resp.status_code == 200
    assert resp.json()["user"] == "tester"


def test_protected_bad_scheme(auth_client) -> None:
    resp = auth_client.get(
        "/protected", headers={"Authorization": "Basic dXNlcjpwYXNz"}
    )
    assert resp.status_code == 401


# --- No-auth mode ---


def test_no_auth_mode():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    config = AuthConfig(mode="none")
    setup_auth(app, config)

    @app.get("/anything")
    async def anything():
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/anything")
    assert resp.status_code == 200


# --- require_role ---


def test_require_role_admin_bypasses() -> None:
    user = AuthUser(user_id="a", roles=("admin",))
    require_role(user, "operator")  # Should not raise


def test_require_role_matching() -> None:
    user = AuthUser(user_id="a", roles=("viewer",))
    require_role(user, "viewer")  # Should not raise


def test_require_role_missing() -> None:
    user = AuthUser(user_id="a", roles=("viewer",))
    with pytest.raises(fastapi.HTTPException) as exc_info:
        require_role(user, "admin")
    assert exc_info.value.status_code == 403


def test_require_role_none_user() -> None:
    with pytest.raises(fastapi.HTTPException) as exc_info:
        require_role(None, "operator")
    assert exc_info.value.status_code == 401
