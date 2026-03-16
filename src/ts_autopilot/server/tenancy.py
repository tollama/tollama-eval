"""Multi-tenancy support for the tollama-eval REST API.

Provides tenant resolution from authenticated users and
per-tenant resource quota enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass

from ts_autopilot.logging_config import get_logger
from ts_autopilot.server.auth import AuthUser
from ts_autopilot.server.job_store import JobStore

logger = get_logger("server.tenancy")


@dataclass
class TenantQuotas:
    """Per-tenant resource limits."""

    max_concurrent_jobs: int = 10
    max_csv_size_mb: int = 500
    max_retention_days: int = 90


class TenantManager:
    """Manages tenant resolution and quota enforcement."""

    def __init__(
        self,
        job_store: JobStore,
        *,
        default_quotas: TenantQuotas | None = None,
        tenant_quotas: dict[str, TenantQuotas] | None = None,
    ) -> None:
        self._store = job_store
        self._default = default_quotas or TenantQuotas()
        self._overrides = tenant_quotas or {}

    def resolve_tenant(self, user: AuthUser | None) -> str:
        """Extract tenant_id from authenticated user."""
        if user is None:
            return ""
        return user.tenant_id or ""

    def get_quotas(self, tenant_id: str) -> TenantQuotas:
        """Get quotas for a tenant, falling back to defaults."""
        return self._overrides.get(tenant_id, self._default)

    def check_concurrent_limit(self, tenant_id: str) -> bool:
        """Return True if tenant is within concurrent job limit."""
        if not tenant_id:
            return True
        quotas = self.get_quotas(tenant_id)
        active = self._store.active_count(tenant_id=tenant_id)
        if active >= quotas.max_concurrent_jobs:
            logger.warning(
                "Tenant %s at concurrent job limit (%d/%d)",
                tenant_id,
                active,
                quotas.max_concurrent_jobs,
            )
            return False
        return True

    def check_csv_size(
        self, tenant_id: str, size_bytes: int
    ) -> bool:
        """Return True if upload is within tenant's size limit."""
        if not tenant_id:
            return True
        quotas = self.get_quotas(tenant_id)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > quotas.max_csv_size_mb:
            logger.warning(
                "Tenant %s CSV size %.1f MB exceeds limit %d MB",
                tenant_id,
                size_mb,
                quotas.max_csv_size_mb,
            )
            return False
        return True
