"""Authentication and authorization for the tollama-eval REST API.

Supports two authentication modes:
- API key: Bearer token validated against hashed keys
- JWT/OIDC: Token validated against a configurable JWKS endpoint

Public endpoints (e.g., /health) bypass authentication.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ts_autopilot.logging_config import get_logger

logger = get_logger("server.auth")

# --- Data types ---


@dataclass(frozen=True)
class AuthUser:
    """Represents an authenticated user/service."""

    user_id: str
    tenant_id: str = ""
    roles: tuple[str, ...] = ("operator",)
    api_key_name: str = ""


@dataclass
class APIKeyRecord:
    """A stored API key with metadata."""

    name: str
    key_hash: str
    tenant_id: str = ""
    roles: list[str] = field(default_factory=lambda: ["operator"])
    created_at: float = 0.0
    revoked: bool = False


# --- Key hashing ---


def hash_api_key(key: str) -> str:
    """Hash an API key using SHA-256 for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, key_hash: str) -> bool:
    """Constant-time comparison of an API key against its hash."""
    computed = hashlib.sha256(key.encode()).hexdigest()
    return hmac.compare_digest(computed, key_hash)


# --- Key store ---


class APIKeyStore:
    """In-memory store for API keys, loadable from file or env."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKeyRecord] = {}

    def add_key(self, record: APIKeyRecord) -> None:
        self._keys[record.key_hash] = record

    def revoke_key(self, key_hash: str) -> bool:
        if key_hash in self._keys:
            self._keys[key_hash].revoked = True
            return True
        return False

    def authenticate(self, bearer_token: str) -> AuthUser | None:
        """Validate a bearer token and return the user, or None."""
        token_hash = hash_api_key(bearer_token)
        record = self._keys.get(token_hash)
        if record is None or record.revoked:
            return None
        return AuthUser(
            user_id=record.name,
            tenant_id=record.tenant_id,
            roles=tuple(record.roles),
            api_key_name=record.name,
        )

    def list_keys(self) -> list[dict[str, Any]]:
        """Return metadata (no hashes) for all keys."""
        return [
            {
                "name": r.name,
                "tenant_id": r.tenant_id,
                "roles": r.roles,
                "revoked": r.revoked,
            }
            for r in self._keys.values()
        ]

    @classmethod
    def from_env(cls) -> APIKeyStore:
        """Load keys from TOLLAMA_API_KEYS env var.

        Format: comma-separated ``name:key`` or ``name:key:tenant_id`` pairs.
        Example: ``TOLLAMA_API_KEYS=admin:sk-abc123,reader:sk-xyz789:tenant1``
        """
        store = cls()
        raw = os.environ.get("TOLLAMA_API_KEYS", "")
        if not raw.strip():
            return store
        for entry in raw.split(","):
            parts = entry.strip().split(":")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            key = parts[1].strip()
            tenant_id = parts[2].strip() if len(parts) > 2 else ""
            roles_str = parts[3].strip() if len(parts) > 3 else "operator"
            roles = [r.strip() for r in roles_str.split("+") if r.strip()]
            store.add_key(
                APIKeyRecord(
                    name=name,
                    key_hash=hash_api_key(key),
                    tenant_id=tenant_id,
                    roles=roles,
                    created_at=time.time(),
                )
            )
        logger.info("Loaded %d API key(s) from environment", len(store._keys))
        return store

    @classmethod
    def from_file(cls, path: str | Path) -> APIKeyStore:
        """Load keys from a JSON file.

        Expected format::

            [
                {
                    "name": "admin",
                    "key_hash": "<sha256 hex>",
                    "tenant_id": "",
                    "roles": ["admin", "operator"]
                }
            ]
        """
        store = cls()
        p = Path(path)
        if not p.exists():
            logger.warning("API key file not found: %s", p)
            return store
        records = json.loads(p.read_text())
        for rec in records:
            store.add_key(
                APIKeyRecord(
                    name=rec["name"],
                    key_hash=rec["key_hash"],
                    tenant_id=rec.get("tenant_id", ""),
                    roles=rec.get("roles", ["operator"]),
                    created_at=rec.get("created_at", 0.0),
                    revoked=rec.get("revoked", False),
                )
            )
        logger.info("Loaded %d API key(s) from %s", len(store._keys), p)
        return store


# --- JWT validation ---


class JWTValidator:
    """Validate JWT tokens against an OIDC issuer's JWKS.

    Requires ``python-jose[cryptography]`` to be installed.
    """

    def __init__(
        self,
        issuer_url: str,
        audience: str = "tollama-eval",
        *,
        algorithms: tuple[str, ...] = ("RS256",),
    ) -> None:
        self.issuer_url = issuer_url.rstrip("/")
        self.audience = audience
        self.algorithms = list(algorithms)
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: float = 0.0

    def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch JWKS from the issuer's well-known endpoint."""
        import httpx

        oidc_url = f"{self.issuer_url}/.well-known/openid-configuration"
        resp = httpx.get(oidc_url, timeout=10)
        resp.raise_for_status()
        jwks_uri = resp.json()["jwks_uri"]

        jwks_resp = httpx.get(jwks_uri, timeout=10)
        jwks_resp.raise_for_status()
        return jwks_resp.json()

    def _get_jwks(self) -> dict[str, Any]:
        """Return cached JWKS, refreshing every 5 minutes."""
        now = time.time()
        if self._jwks is None or (now - self._jwks_fetched_at) > 300:
            self._jwks = self._fetch_jwks()
            self._jwks_fetched_at = now
        return self._jwks

    def validate(self, token: str) -> AuthUser | None:
        """Validate a JWT and return the user, or None."""
        try:
            from jose import jwt as jose_jwt

            jwks = self._get_jwks()
            payload = jose_jwt.decode(
                token,
                jwks,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer_url,
            )
            return AuthUser(
                user_id=payload.get("sub", "unknown"),
                tenant_id=payload.get("tenant_id", payload.get("org_id", "")),
                roles=tuple(payload.get("roles", ["operator"])),
            )
        except Exception:
            logger.debug("JWT validation failed", exc_info=True)
            return None


# --- Auth configuration ---


@dataclass
class AuthConfig:
    """Authentication configuration for the server."""

    mode: str = "none"  # "none", "apikey", "jwt", "both"
    api_key_file: str = ""
    oidc_issuer_url: str = ""
    oidc_audience: str = "tollama-eval"
    public_paths: tuple[str, ...] = (
        "/health",
        "/health/live",
        "/health/ready",
        "/health/startup",
        "/docs",
        "/redoc",
        "/openapi.json",
    )


# --- FastAPI integration ---


def setup_auth(app: Any, config: AuthConfig) -> None:
    """Install authentication middleware on a FastAPI app.

    Args:
        app: FastAPI application instance.
        config: Authentication configuration.
    """
    if config.mode == "none":
        logger.info("Auth disabled — all endpoints are public")
        # Store a no-op getter for endpoints that optionally use auth
        app.state.get_auth_user = _no_auth_user
        return

    key_store: APIKeyStore | None = None
    jwt_validator: JWTValidator | None = None

    if config.mode in ("apikey", "both"):
        if config.api_key_file:
            key_store = APIKeyStore.from_file(config.api_key_file)
        else:
            key_store = APIKeyStore.from_env()

    if config.mode in ("jwt", "both"):
        if not config.oidc_issuer_url:
            raise ValueError("oidc_issuer_url is required for JWT auth mode")
        jwt_validator = JWTValidator(
            issuer_url=config.oidc_issuer_url,
            audience=config.oidc_audience,
        )

    app.state.key_store = key_store
    app.state.jwt_validator = jwt_validator
    app.state.auth_config = config

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Any:
            # Skip auth for public paths
            path = request.url.path.rstrip("/")
            if any(path == p.rstrip("/") for p in config.public_paths):
                request.state.auth_user = None
                return await call_next(request)

            # Also skip for /metrics (Prometheus scrape)
            if path == "/metrics":
                request.state.auth_user = None
                return await call_next(request)

            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid Authorization header"},
                )

            token = auth_header[7:]  # Strip "Bearer "
            user: AuthUser | None = None

            # Try API key first
            if key_store is not None:
                user = key_store.authenticate(token)

            # Try JWT if API key didn't match
            if user is None and jwt_validator is not None:
                user = jwt_validator.validate(token)

            if user is None:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid or revoked credentials"},
                )

            request.state.auth_user = user
            return await call_next(request)

    app.add_middleware(AuthMiddleware)
    app.state.get_auth_user = _get_auth_user_from_request
    logger.info("Auth enabled: mode=%s", config.mode)


def _no_auth_user(request: Any) -> AuthUser | None:
    """Return None when auth is disabled."""
    return None


def _get_auth_user_from_request(request: Any) -> AuthUser | None:
    """Extract auth user from request state."""
    return getattr(request.state, "auth_user", None)


def require_role(user: AuthUser | None, role: str) -> None:
    """Raise 403 if the user does not have the required role.

    Args:
        user: Authenticated user or None.
        role: Required role name.

    Raises:
        HTTPException: If user is None or lacks the role.
    """
    if user is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Authentication required")
    if role not in user.roles and "admin" not in user.roles:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' required",
        )
