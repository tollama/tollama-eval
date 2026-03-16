# ADR-0004: API Authentication Model

## Status

Accepted

## Context

The REST API server was initially built without authentication, making it
unsuitable for any network-accessible deployment. Enterprise customers
require authentication, audit trails, and multi-tenancy.

## Decision

Support two authentication modes, selectable via configuration:

1. **API Key** (`mode=apikey`): Bearer token authentication with SHA-256
   hashed keys. Keys loaded from environment variable (`TOLLAMA_API_KEYS`)
   or a JSON file. Simple to set up, suitable for service-to-service auth.

2. **JWT/OIDC** (`mode=jwt`): Token validation against a configurable OIDC
   issuer's JWKS endpoint. Supports enterprise SSO integration (Okta,
   Auth0, Azure AD). Tokens carry `tenant_id` and `roles` claims.

3. **Both** (`mode=both`): Try API key first, fall back to JWT.

Public endpoints (`/health/*`, `/docs`, `/redoc`) bypass authentication.

## Consequences

- API keys are simple but require secure distribution and rotation.
- JWT/OIDC enables SSO but adds network dependency on the identity provider.
- The `both` mode allows gradual migration from API keys to SSO.
- All authenticated requests carry a user identity for audit logging.
