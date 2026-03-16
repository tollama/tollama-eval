# ADR-0001: Frozen results.json Schema

## Status

Accepted

## Context

External consumers (dashboards, CI pipelines, downstream analytics) depend on the
structure of `results.json`. Breaking changes to field names or nesting force all
consumers to update simultaneously, creating coordination overhead and potential outages.

## Decision

The `results.json` output schema is frozen. Fields will never be renamed or removed.
New data is added via:
1. New optional fields in existing objects (additive-only).
2. The separate `details.json` file for detailed/experimental data.

## Consequences

- Consumers can rely on the schema indefinitely without version negotiation.
- Adding data requires careful design to avoid breaking additive-only guarantees.
- Large new features go into `details.json` rather than `results.json`.
