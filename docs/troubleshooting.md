# Troubleshooting

This guide helps you quickly isolate and fix common `ts-autopilot` failures.

## Quick Triage

1. Run environment diagnostics:

```bash
ts-autopilot doctor
```

2. Re-run with verbose logs and no cache:

```bash
ts-autopilot run -i your_data.csv -v --no-cache
```

3. If you use a config file, validate keys and values:

```bash
ts-autopilot run -c benchmark.yaml
```

## Common Errors

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Error: Input file not found` | Wrong input path | Check `--input` path and file existence |
| `Cannot parse CSV as long or wide format` | Missing required columns in long format, or first column in wide format is not parseable datetime | Ensure long format has `ds` and `y` (plus `unique_id` preferred), or wide format has dates in first column and numeric series columns |
| `Error: One or more training series has zero variation` | At least one series is constant | Remove constant series before running |
| `Error: ... training series has zero variation` or fold failures | Horizon/fold choice too aggressive for data length | Reduce `--horizon` and/or `--n-folds` |
| `Loaded DataFrame uses ... MB, exceeding ...` | Data exceeds memory guard | Reduce dataset size or increase `memory_limit_mb` in config |
| `openpyxl not installed` when using `--excel` | Excel extra not installed | `pip install "ts-autopilot[excel]"` |
| PDF generation import errors when using `--pdf` | PDF extra not installed | `pip install "ts-autopilot[pdf]"` |
| `Tollama URL ... resolves to private address ...` | SSRF protection blocks private/local URLs by default | Use config file with `allow_private_urls: true` for trusted local/private Tollama servers |
| Tollama warnings and NaN predictions | Tollama server/model unavailable or returned invalid response | Verify server URL, model name, and server logs; run can continue with NaN fallback |

## CSV Format Checklist

`ts-autopilot` canonical internal format is:

- `unique_id`: string
- `ds`: datetime (timezone-naive)
- `y`: float

Notes:

- If `unique_id` is missing in long format, it defaults to `series_1`.
- Timezone-aware timestamps are converted to timezone-naive timestamps.
- In wide format, the first column must be parseable as datetime.

## Tollama-Specific Checklist

1. Confirm URL and scheme:
- Must be `http://` or `https://`.

2. Confirm model IDs:
- Example IDs: `chronos2`, `timesfm`, `moirai`, `granite-ttm`.

3. For local/private URLs:
- Put Tollama settings in config and set `allow_private_urls: true`.

4. If benchmark runs but Tollama model outputs are bad:
- Check for warnings in stderr.
- Expect NaN fallback when per-series requests fail.

## CI/Lint/Type Failures (Contributors)

Before pushing:

```bash
ruff check src/ tests/
mypy src/ts_autopilot/
pytest tests/ -v
```

## Reporting an Issue

When opening a bug report, include:

1. Command used
2. Full error output
3. `ts-autopilot doctor` output
4. Minimal sample CSV that reproduces the issue (if possible)
5. OS, Python version, and package version
