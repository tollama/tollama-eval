# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enterprise-grade security hardening (input path validation, SSRF prevention, XSS verification)
- Per-model timeout with configurable `model_timeout_sec` (default 300s)
- Graceful signal handling (SIGINT/SIGTERM) with partial result output
- Memory guard for large datasets with configurable `memory_limit_mb`
- Run ID / correlation ID for log tracing across pipeline execution
- Structured event emission for external monitoring integration
- `tollama-eval doctor` command for environment diagnostics
- Dependency vulnerability scanning in CI (pip-audit)
- SBOM generation in CI pipeline
- Release automation via GitHub Actions (tag-triggered PyPI publish)
- Python 3.13 added to CI test matrix
- Dockerfile for containerized deployment
- CHANGELOG.md, CONTRIBUTING.md, SECURITY.md documentation
- Property-based testing with Hypothesis
- CSV parser fuzz testing

### Changed
- Added LICENSE file (MIT) and CODE_OF_CONDUCT.md for public OSS release
- Removed internal development scripts and generated data files
- Aligned coverage gate to 80% across CI, pyproject.toml, and Makefile
- Tollama URL validation now blocks private/internal IPs by default (SSRF prevention)
- Failed models now recorded with NaN metrics instead of crashing the pipeline
- Coverage gate remains at 80% (to be increased incrementally)
- CI now cancels stale PR runs via concurrency groups

### Security
- Input file paths are resolved and validated against symlink attacks
- Tollama URLs validated for scheme (http/https only) and SSRF safety
- HTML report template verified for autoescape (XSS prevention)

## [0.2.0] - 2025-01-15

### Added
- Commercial-quality HTML report with Plotly visualizations
- Executive summary with auto-generated narrative
- Residual diagnostics (ACF, Ljung-Box, histograms)
- Ensemble model recommendations
- Multi-metric evaluation (MASE, SMAPE, RMSSE, MAE)
- Per-series breakdown in reports
- Tollama TSFM integration (Chronos-2, TimesFM, etc.)
- Data export (CSV leaderboard, per-fold details, per-series MASE)
- Campaign mode for multi-dataset benchmarking
- Configurable retry with exponential backoff
- JSON structured logging (--log-json)
- details.json output for report reproducibility
- PDF export via weasyprint (optional)
- Streamlit dashboard (optional)
- YAML/JSON configuration file support
- mypy strict type checking
- Custom exception hierarchy
- Feature inventory documentation (FEATURES.md)

### Changed
- Tollama rewritten from LLM interpretation to direct TSFM runner
- Expanded model catalog: 5 core + 4 optional + 7 foundation models

## [0.1.0] - 2024-12-01

### Added
- Initial release: automated time series benchmarking
- SeasonalNaive and AutoETS model runners
- Expanding-window cross-validation
- MASE metric computation
- CLI entry point (`tollama-eval run`)
- results.json output with frozen schema
- Basic HTML report generation
- CSV ingestion with long/wide format auto-detection
- Data profiling (frequency, season length, missing ratio)
