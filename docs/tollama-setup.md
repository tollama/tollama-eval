# Tollama Setup Guide

This guide explains how to benchmark Tollama TSFM models with `tollama-eval`.

Tollama is a time series foundation model platform, not an LLM service.

## Supported CLI Flags

- `--tollama-url`: base URL of Tollama server
- `--tollama-models`: comma-separated model IDs
- `--no-tollama`: disable Tollama even if URL is provided

Example:

```bash
tollama-eval run -i data.csv \
  --tollama-url https://your-tollama.example.com \
  --tollama-models chronos2,timesfm
```

## Request Path and Payload

`tollama-eval` calls:

- `POST /v1/forecast`

with payload shape:

```json
{
  "model": "chronos2",
  "series": { "target": [1.0, 2.0, 3.0], "freq": "D" },
  "horizon": 14
}
```

## Model IDs

Common IDs in this project:

- `chronos2`
- `timesfm`
- `moirai`
- `granite-ttm`
- `lag-llama`
- `patchtst`
- `tide`

Use model IDs supported by your Tollama server deployment.

## Local and Private URL Note

`tollama-eval` blocks private/local URLs by default for SSRF safety.

If your Tollama server is local (for example `http://127.0.0.1:8000`), use a config file and set:

```yaml
allow_private_urls: true
```

Minimal config example:

```yaml
input: data.csv
output: out/
horizon: 14
n_folds: 3
tollama_url: http://127.0.0.1:8000
tollama_models:
  - chronos2
  - timesfm
allow_private_urls: true
```

Run:

```bash
tollama-eval run -c local_tollama.yaml
```

## Failure Behavior

Tollama models are zero-shot. There is no model fitting step in `tollama-eval`.

If a per-series Tollama request fails:

- `tollama-eval` logs a warning
- That series prediction falls back to NaN values
- The overall benchmark run can continue

## Quick Connectivity Test

You can sanity-check the endpoint directly:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{"model":"chronos2","series":{"target":[1,2,3,4,5],"freq":"D"},"horizon":2}'
```

Expected response should include a `mean` array with length equal to `horizon`.

## Common Errors and Fixes

| Error | Meaning | Fix |
|---|---|---|
| `Tollama URL must use http:// or https://` | Invalid URL scheme | Use `http://` or `https://` |
| `... resolves to private address ...` | Private/local URL blocked | Set `allow_private_urls: true` in config for trusted local/private hosts |
| `Tollama unavailable at ...` | Network or server unreachable | Verify host, port, firewall, and server status |
| `Invalid tollama response ...` | Response shape mismatch | Confirm server compatibility with `/v1/forecast` response contract |
| `Expected N forecast values, got ...` | `mean` length mismatch | Check Tollama model/service behavior and server logs |
