# Plan: Commercial-Quality Report Generation

## Goal
Transform the current basic HTML report into a polished, commercial-grade
benchmark report that runs end-to-end without human intervention.

---

## Current State
- Single-page HTML report via Jinja2 + Plotly CDN
- 2 charts (bar chart, heatmap), basic tables, optional LLM section
- Functional but visually plain — looks like a developer tool, not a deliverable
- No PDF export, no executive summary, no forecast visualization

## Target State
A report that a data science team could hand directly to a stakeholder:
professional layout, executive summary, rich visualizations, actionable
insights, and one-click PDF export — all generated automatically.

---

## Phase 1: Data Foundation — Forecast & Residual Capture
> _Prerequisite: the report can only show forecasts if the pipeline produces them._

### 1.1 Extend `ForecastOutput` contract
- Add `y_actual: list[float]` field alongside existing `y_hat`
- Add `y_train_tail: list[float]` (last 2×horizon of training data for context)
- Add `ds_train_tail: list[str]` (corresponding dates)
- Keep backward-compatible (optional fields with defaults)

### 1.2 Capture actuals in `run_benchmark()`
- After `fit_predict()`, attach the held-out test actuals and train tail
  to each fold's `ForecastOutput`
- Store in `ModelResult.folds` so downstream reporting can access them

### 1.3 Compute residual diagnostics
- New module: `src/ts_autopilot/evaluation/diagnostics.py`
  - `residual_stats(y_true, y_hat)` → mean, std, skew, kurtosis
  - `ljung_box_p(residuals, lags)` → p-value (autocorrelation test)
  - `residual_distribution(residuals)` → histogram bin edges & counts
- Attach summary stats to each model's result dict under `"diagnostics"`

---

## Phase 2: Rich Visualizations
> _New charts that tell the story at a glance._

### 2.1 Forecast vs Actual plot (per top-N series)
- Line chart: training tail (gray) → actual (blue) → forecast (orange)
- Shown for the best model, top 3 + bottom 3 series by MASE
- Interactive Plotly with hover showing date, actual, forecast, error

### 2.2 Multi-metric radar chart
- One radar per model showing normalized MASE, SMAPE, RMSSE, MAE
- Quick visual for "which model is balanced vs specialized"

### 2.3 Fold stability line chart
- X = fold number, Y = MASE per model (line per model)
- Shows whether a model degrades or improves with more training data

### 2.4 Residual analysis panel (best model)
- Histogram of residuals with normal overlay
- ACF plot of residuals (bar chart of autocorrelation by lag)
- Residuals vs fitted scatter plot

### 2.5 Error distribution box plot
- Box plot of per-series MASE across all series, one box per model
- Shows spread/outliers — critical for understanding reliability

---

## Phase 3: Report Layout & Design Overhaul
> _From dev tool to boardroom-ready._

### 3.1 Professional CSS theme
- Clean typography (system font stack, proper hierarchy)
- Color palette: muted blues/grays with accent colors for metrics
- Card-based layout for sections (subtle shadows, rounded corners)
- Responsive grid: charts side-by-side on wide screens, stacked on narrow
- Print stylesheet: page breaks, no interactive elements, clean margins

### 3.2 Executive summary section (auto-generated)
- Template-driven natural language summary:
  - "We evaluated {n} models on {n_series} time series over {n_folds} folds."
  - "The best model is **{name}** with MASE={value}, beating naive by {pct}%."
  - "Key risk: {worst_series} has MASE={value}, indicating poor forecastability."
  - Warning callouts if any model MASE > 2.0 or high variance across folds
- No LLM required — pure template logic with conditional branches

### 3.3 Section restructure
```
1. Header (logo placeholder, title, timestamp, version)
2. Executive Summary (auto-generated narrative)
3. Data Profile (enhanced with sparkline-style stats)
4. Warnings & Data Quality (promoted to its own section)
5. Leaderboard (enhanced table + bar chart)
6. Model Comparison (radar chart + fold stability)
7. Forecast Visualization (actual vs predicted)
8. Residual Diagnostics (histogram, ACF, scatter)
9. Per-Series Deep Dive (collapsible, with error distribution)
10. Methodology Notes (auto-filled: CV strategy, metric definitions)
11. LLM Interpretation (if available)
12. Appendix: Raw Metrics Table (all folds × all models × all series)
13. Footer (version, timestamp, reproducibility hash)
```

### 3.4 Table of Contents
- Sticky sidebar TOC on desktop, collapsible hamburger on mobile
- Auto-generated from section headings
- Smooth scroll navigation

---

## Phase 4: PDF Export
> _One-click PDF without external services._

### 4.1 Add `weasyprint` as optional dependency
- Extra: `pip install "ts-autopilot[pdf]"`
- Graceful fallback: if not installed, skip PDF, log info message

### 4.2 PDF generation function
- `reporting/pdf_export.py`
  - `generate_pdf(html_path, output_path)` — converts HTML → PDF
  - Replace Plotly interactive charts with static Plotly images (use `plotly`
    `write_image` via kaleido, or pre-render to SVG/PNG and embed)
  - Inject `@media print` CSS for page breaks, headers, footers

### 4.3 CLI integration
- New flag: `--pdf / --no-pdf` (default: generate if weasyprint available)
- Output: `report.pdf` alongside `report.html` and `results.json`

---

## Phase 5: Pipeline Integration
> _Wire everything together, zero human intervention._

### 5.1 Update `pipeline.py`
- After metrics computation, call diagnostics module
- Pass enriched data (forecasts, actuals, diagnostics) to report generator
- Generate HTML report → optionally generate PDF
- All atomic writes preserved

### 5.2 Update `html_report.py`
- Refactor `generate_report()` to accept enriched data structures
- New `_build_chart_data()` sections for each new chart type
- Keep backward compatibility: if forecast data missing, skip those sections

### 5.3 Update Jinja2 template
- Rewrite `report.html.j2` with new layout and all new sections
- Modular template structure: `{% include %}` partials for each section
- All charts self-contained (no external CDN dependency option)

---

## Phase 6: Testing & Quality
> _Ensure the report is correct and doesn't break existing functionality._

### 6.1 Unit tests
- `tests/test_diagnostics.py` — residual stats, Ljung-Box
- `tests/test_report_enhanced.py` — template rendering with new data
- `tests/test_pdf_export.py` — PDF generation (skip if weasyprint missing)
- `tests/test_executive_summary.py` — narrative generation logic

### 6.2 Integration test
- End-to-end: CSV → pipeline → enriched report.html + report.pdf
- Validate HTML structure (key sections present)
- Validate results.json backward compatibility (schema unchanged)

### 6.3 Visual QA
- Generate report with example dataset, verify rendering
- Test print/PDF layout

---

## Implementation Order

| Step | Scope | Depends On |
|------|-------|------------|
| 1 | Phase 1.1–1.2: Extend contracts & capture forecasts | — |
| 2 | Phase 1.3: Diagnostics module | Step 1 |
| 3 | Phase 3.1: CSS theme overhaul | — |
| 4 | Phase 3.2–3.4: Executive summary, TOC, section restructure | Step 1 |
| 5 | Phase 2.1–2.5: All new visualizations | Steps 1, 2 |
| 6 | Phase 5: Pipeline integration | Steps 1–5 |
| 7 | Phase 4: PDF export | Step 6 |
| 8 | Phase 6: Testing | Steps 1–7 |

Steps 1+3 can run in parallel. Steps 2+4 can run in parallel after Step 1.

---

## Files to Create
- `src/ts_autopilot/evaluation/diagnostics.py`
- `src/ts_autopilot/reporting/pdf_export.py`
- `src/ts_autopilot/reporting/executive_summary.py`
- `src/ts_autopilot/reporting/templates/partials/` (section templates)
- `tests/test_diagnostics.py`
- `tests/test_report_enhanced.py`
- `tests/test_pdf_export.py`

## Files to Modify
- `src/ts_autopilot/contracts.py` — extend ForecastOutput
- `src/ts_autopilot/pipeline.py` — capture actuals, call diagnostics, enrich report
- `src/ts_autopilot/reporting/html_report.py` — new chart builders, enriched data
- `src/ts_autopilot/reporting/templates/report.html.j2` — complete rewrite
- `src/ts_autopilot/cli.py` — add --pdf flag
- `pyproject.toml` — add optional deps (weasyprint, kaleido)

## Key Constraints
- `results.json` schema is **frozen** — new fields only, no renames/removals
- CLI remains zero-logic — all new logic in pipeline/reporting modules
- Backward compatible: old results.json files can still produce (basic) reports
- No mandatory new dependencies for core functionality
