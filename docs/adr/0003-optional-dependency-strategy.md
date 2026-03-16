# ADR-0003: Optional Dependency Strategy

## Status

Accepted

## Context

Enterprise deployments may not allow installing GPU libraries, Prophet, or
other heavy dependencies. The tool must work with only core dependencies
while gracefully supporting additional capabilities when available.

## Decision

- Core models (StatsForecast-based) are always available.
- Optional models (Prophet, LightGBM, neural, etc.) are in separate extras
  (`pip install "tollama-eval[prophet]"`).
- Each optional runner probes its dependency at import time using safe mode
  (subprocess health check for neural models).
- Missing dependencies skip the model with a log warning, never crash.
- The `doctor` command reports which optional dependencies are installed.

## Consequences

- Zero-config installs work immediately with core models.
- Enterprises can precisely control which dependencies are deployed.
- CI tests must cover both with-dependency and without-dependency paths.
