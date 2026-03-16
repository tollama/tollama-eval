# ADR-0002: BaseRunner Abstraction for Model Plugins

## Status

Accepted

## Context

tollama-eval supports 36+ forecasting models from multiple libraries
(StatsForecast, Prophet, MLForecast, NeuralForecast, Tollama TSFM).
Each has different APIs, dependencies, and failure modes.

## Decision

All models implement the `BaseRunner` abstract class with a single
`fit_predict(train, horizon, freq, season_length, n_jobs)` interface.
Third-party models can register via Python entry points
(`ts_autopilot.runners` group).

## Consequences

- Adding a new model requires only a `BaseRunner` subclass.
- The pipeline is model-agnostic; it only interacts with the runner interface.
- Optional dependencies are lazily imported and safely probed.
- Trade-off: some model-specific configuration must be mapped through
  a generic interface.
