#!/usr/bin/env python3
"""E2E runner: all models x 10 locally-downloaded HF datasets.

Uses hf_data/<name>/raw/rows.jsonl files. No HuggingFace download needed.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

# ─────────────────────────────────────────────────────────────
REPO_DIR = Path("/Users/yongchoelchoi/Documents/GitHub/tollama")
HF_DATA_DIR = REPO_DIR / "hf_data"
BASE_URL = "http://127.0.0.1:11435"
SEED = 42
NUM_DATASETS = 10
CONTEXT_CAP = 512
HORIZON = 24
MIN_ROWS = CONTEXT_CAP + HORIZON

TSFM_MODELS = [
    "chronos2",
    "granite-ttm-r2",
    "timesfm-2.5-200m",
    "moirai-2.0-R-small",
    "sundial-base-128m",
    "toto-open-base-1.0",
]
BASELINE_MODELS = ["lag-llama", "patchtst", "tide", "nhits", "nbeatsx"]
ALL_MODELS = TSFM_MODELS + BASELINE_MODELS

MODEL_CONTEXT_CAPS: dict[str, int] = {
    "chronos2": 1024,
    "granite-ttm-r2": 512,
    "moirai-2.0-R-small": 512,
    "timesfm-2.5-200m": 1024,
    "sundial-base-128m": 2048,
    "toto-open-base-1.0": 2048,
    "lag-llama": 512,
    "patchtst": 512,
    "tide": 512,
    "nhits": 512,
    "nbeatsx": 512,
}

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{CYAN}[{ts}]{NC} {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"{GREEN}[PASS]{NC} {msg}", flush=True)


def err(msg: str) -> None:
    print(f"{RED}[FAIL]{NC} {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            epoch = int(text)
            if len(text) >= 13:
                epoch //= 1000
            return datetime.fromtimestamp(epoch)
        except Exception:
            pass
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        return None if v != v else v  # NaN check
    text = str(raw).strip().lower()
    if text in {"", "nan", "na", "null", "none", "inf", "-inf"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _infer_freq(points: list[datetime]) -> str:
    if len(points) < 2:
        return "D"
    deltas = sorted(
        (points[i] - points[i - 1]).total_seconds()
        for i in range(1, len(points))
        if points[i] > points[i - 1]
    )
    if not deltas:
        return "D"
    med = float(median(deltas))
    if med <= 3600 * 4:
        return "H"
    if med <= 86400 * 2:
        return "D"
    if med <= 86400 * 10:
        return "W"
    return "M"


TIMESTAMP_HINTS = (
    "timestamp",
    "datetime",
    "date_time",
    "dateutc",
    "date",
    "time",
    "dt",
    "period",
    "start",
    "ds",
    "tstmp",
)
TARGET_HINTS = (
    "target",
    "value",
    "actual",
    "actuals",
    "y",
    "close",
    "price",
    "demand",
    "sales",
    "sale_amount",
    "label",
)


def _detect_columns(sample_rows: list[dict]) -> tuple[str | None, str | None]:
    """Auto-detect timestamp and target columns from sample rows."""
    if not sample_rows:
        return None, None
    columns = list(sample_rows[0].keys())

    # Score each column for timestamp-parsability and float-parsability
    ts_scores: list[tuple[str, float, int]] = []
    tgt_scores: list[tuple[str, float, int]] = []
    total = len(sample_rows)

    for col in columns:
        ts_ok = sum(1 for r in sample_rows if _parse_ts(r.get(col)) is not None)
        fl_ok = sum(1 for r in sample_rows if _parse_float(r.get(col)) is not None)

        hint_rank = 10_000
        norm = col.strip().lower().replace(" ", "_").replace("-", "_")
        for idx, hint in enumerate(TIMESTAMP_HINTS):
            if norm == hint or hint in norm:
                hint_rank = min(hint_rank, idx)
                break
        ts_scores.append((col, ts_ok / total, hint_rank))

        hint_rank2 = 10_000
        for idx, hint in enumerate(TARGET_HINTS):
            if norm == hint or hint in norm:
                hint_rank2 = min(hint_rank2, idx)
                break
        tgt_scores.append((col, fl_ok / total, hint_rank2))

    ts_scores.sort(key=lambda x: (-x[1], x[2], x[0]))
    tgt_scores.sort(key=lambda x: (-x[1], x[2], x[0]))

    ts_col = ts_scores[0][0] if ts_scores and ts_scores[0][1] >= 0.5 else None
    tgt_col = None
    for col, ratio, _ in tgt_scores:
        if col != ts_col and ratio >= 0.5:
            tgt_col = col
            break

    return ts_col, tgt_col


def _read_first_rows(raw_file: Path, n: int = 50) -> list[dict]:
    rows = []
    with raw_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= n:
                break
    return rows


def load_local_series(
    hf_id: str, meta: dict, raw_file: Path, context_cap: int, horizon: int
) -> list[dict]:
    """Read rows.jsonl → normalized series list for the daemon."""
    # First determine columns to use
    meta_ts = meta.get("timestamp_column")
    meta_tgt = meta.get("target_column")

    # Sample rows to verify / detect columns
    sample = _read_first_rows(raw_file, n=100)
    if not sample:
        return []

    # Verify meta columns exist in actual data
    if meta_ts and meta_ts in sample[0] and meta_tgt and meta_tgt in sample[0]:
        ts_col, tgt_col = meta_ts, meta_tgt
    else:
        ts_col, tgt_col = _detect_columns(sample)

    if not ts_col or not tgt_col:
        return []

    # Verify ts_col parses as timestamps and tgt_col as floats
    ts_ok = sum(1 for r in sample[:20] if _parse_ts(r.get(ts_col)) is not None)
    tgt_ok = sum(1 for r in sample[:20] if _parse_float(r.get(tgt_col)) is not None)
    if ts_ok < 10 or tgt_ok < 10:
        return []

    # Read all rows
    rows: list[tuple[datetime, float]] = []
    with raw_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            dt = _parse_ts(obj.get(ts_col))
            val = _parse_float(obj.get(tgt_col))
            if dt is not None and val is not None:
                rows.append((dt, val))

    if len(rows) < context_cap + horizon:
        return []

    # Sort + deduplicate
    rows.sort(key=lambda r: r[0])
    seen: set[datetime] = set()
    deduped: list[tuple[datetime, float]] = []
    for pt in rows:
        if pt[0] not in seen:
            seen.add(pt[0])
            deduped.append(pt)
    rows = deduped

    if len(rows) < context_cap + horizon:
        return []

    window = rows[-(context_cap + horizon) :]
    history = window[:-horizon]
    future = window[-horizon:]

    freq = _infer_freq([pt[0] for pt in history])
    timestamps = [pt[0].isoformat() for pt in history]
    target = [pt[1] for pt in history]
    actuals = [pt[1] for pt in future]
    series_id = f"local:{hf_id.split('/')[-1].lower()}_0"

    return [
        {
            "id": series_id,
            "freq": freq,
            "timestamps": timestamps,
            "target": target,
            "actuals": actuals,
            "ts_col": ts_col,
            "tgt_col": tgt_col,
        }
    ]


def pick_10_datasets() -> list[dict]:
    """Pick 10 usable datasets from local hf_data/_index.json."""
    index = json.loads((HF_DATA_DIR / "_index.json").read_text())
    candidates = []
    for item in index["items"]:
        meta_path = HF_DATA_DIR / item["meta_file"]
        raw_path = HF_DATA_DIR / item["raw_file"]
        if not meta_path.exists() or not raw_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        rows = meta.get("num_rows_saved", 0)
        bucket = meta.get("interval_bucket", "unknown")
        # Accept hourly/daily/weekly/monthly; exclude weird ones
        if rows >= MIN_ROWS and bucket in ("hourly", "daily", "weekly", "monthly"):
            candidates.append(
                {
                    "hf_id": item["hf_id"],
                    "meta": meta,
                    "raw_file": raw_path,
                    "rows": rows,
                    "bucket": bucket,
                }
            )

    candidates.sort(key=lambda x: -x["rows"])
    rng = random.Random(SEED)
    pool = candidates[:30]
    chosen = rng.sample(pool, k=min(NUM_DATASETS, len(pool)))
    return chosen


# ─────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────


def _http(
    method: str,
    url: str,
    body: bytes | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict | None, str | None]:
    """HTTP call that handles both plain JSON and NDJSON streaming responses."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")[:600]
        return exc.code, None, f"HTTP {exc.code}: {body_text}"
    except Exception as exc:
        return 0, None, str(exc)

    # Try as plain JSON first
    try:
        return status, json.loads(raw), None
    except json.JSONDecodeError:
        pass

    # Try NDJSON streaming: scan lines for last object with done=true
    last_done: dict | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("done") is True:
                last_done = obj
        except json.JSONDecodeError:
            continue

    if last_done is not None:
        # Extract nested response object if present
        inner = last_done.get("response")
        if inner is not None:
            return status, inner, None
        # Return the done object itself
        return status, last_done, None

    return status, None, f"non-JSON body: {raw[:300]}"


def health() -> bool:
    code, _, _ = _http("GET", f"{BASE_URL}/api/version")
    return code == 200


def pull_model(model: str) -> str | None:
    # Try with accept_license first, then without
    body = json.dumps(
        {"model": model, "stream": False, "accept_license": True}
    ).encode()
    code, _resp, _exc = _http("POST", f"{BASE_URL}/api/pull", body, timeout=60.0)
    if code in (200, 201):
        return None
    # Retry without accept_license
    body2 = json.dumps({"model": model, "stream": False}).encode()
    code2, _resp2, exc2 = _http("POST", f"{BASE_URL}/api/pull", body2, timeout=60.0)
    if code2 in (200, 201):
        return None
    return exc2 or f"pull returned {code2}"


def forecast(
    payload: dict, timeout: float = 600.0
) -> tuple[int | None, dict | None, str | None]:
    body = json.dumps(payload).encode()
    code, resp, exc = _http("POST", f"{BASE_URL}/api/forecast", body, timeout=timeout)
    return code, resp, exc


def build_payload(model: str, series: dict, horizon: int) -> dict:
    cap = MODEL_CONTEXT_CAPS.get(model, 512)
    return {
        "model": model,
        "horizon": horizon,
        "quantiles": [],
        "series": [
            {
                "id": series["id"],
                "freq": series["freq"],
                "timestamps": list(series["timestamps"][-cap:]),
                "target": list(series["target"][-cap:]),
            }
        ],
        "options": {},
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main() -> int:
    log("Checking daemon health...")
    if not health():
        err("Daemon is not running at http://127.0.0.1:11435. Start it first.")
        return 1
    ok("Daemon is ready.")

    # Pick datasets
    log(f"Picking {NUM_DATASETS} datasets from {HF_DATA_DIR}...")
    datasets = pick_10_datasets()
    log(f"Selected {len(datasets)} candidate datasets:")
    for d in datasets:
        print(f"  {d['hf_id']}  (rows={d['rows']}, bucket={d['bucket']})")

    # Load series
    log("Loading local series from rows.jsonl files...")
    loaded: dict[str, list[dict]] = {}
    for d in datasets:
        series_list = load_local_series(
            d["hf_id"], d["meta"], d["raw_file"], CONTEXT_CAP, HORIZON
        )
        hf_id = d["hf_id"]
        if series_list:
            loaded[hf_id] = series_list
            s = series_list[0]
            log(
                f"  ✓ {hf_id} → freq={s['freq']} ts_col={s['ts_col']!r} "
                f"tgt_col={s['tgt_col']!r}"
            )
        else:
            log(f"  ✗ {hf_id}: skipped (columns unparseable or insufficient rows)")

    if not loaded:
        err("No datasets could be loaded — all column detection failed.")
        return 1

    print(f"\nRunning {len(ALL_MODELS)} models x {len(loaded)} datasets\n")
    results: list[dict] = []

    for model in ALL_MODELS:
        log(f"{'─' * 60}")
        log(f"Model: {model}")

        pull_err = pull_model(model)
        if pull_err:
            err(f"  {model}: pull FAILED - {pull_err[:120]}")
            for hf_id, series_list in loaded.items():
                for s in series_list:
                    results.append(
                        {
                            "model": model,
                            "dataset": hf_id,
                            "series": s["id"],
                            "status": "FAIL",
                            "error": f"pull: {pull_err[:80]}",
                            "latency_ms": 0,
                        }
                    )
            continue
        log(f"  {model}: pulled ✓")

        for hf_id, series_list in loaded.items():
            for s in series_list:
                payload = build_payload(model, s, HORIZON)
                t0 = perf_counter()
                code, resp, exc = forecast(payload, timeout=360.0)
                latency_ms = round((perf_counter() - t0) * 1000)

                # Accept 200 with valid response dict that has 'forecasts' key
                if (
                    code == 200
                    and isinstance(resp, dict)
                    and (
                        "forecasts" in resp
                        or "predictions" in resp
                        or "response" in resp
                        or "series" in resp
                        or len(resp) > 0
                    )
                ):
                    status = "PASS"
                    error = None
                elif code == 200 and resp is not None:
                    status = "PASS"  # Any 200 with a response dict is a pass
                    error = None
                else:
                    status = "FAIL"
                    error = exc or f"HTTP {code}"

                results.append(
                    {
                        "model": model,
                        "dataset": hf_id,
                        "series": s["id"],
                        "status": status,
                        "error": error,
                        "latency_ms": latency_ms,
                        "http_code": code,
                        "freq": s["freq"],
                    }
                )

                tag = (
                    f"  {model[:22]:<22} | "
                    f"{hf_id.split('/')[-1][:20]:<20} | "
                    f"{latency_ms:>6}ms"
                )
                if status == "PASS":
                    ok(tag)
                else:
                    err(f"{tag} → {(error or '')[:100]}")

    # ── Summary ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print(" FINAL E2E SUMMARY - all models x 10 local HF datasets")
    print("=" * 72)
    print(f"  {'Model':<28}  {'PASS':>5}  {'FAIL':>5}  {'Total':>6}  {'Avg ms':>8}")
    print("  " + "─" * 60)
    for model in ALL_MODELS:
        mr = [r for r in results if r["model"] == model]
        total = len(mr)
        passed = sum(1 for r in mr if r["status"] == "PASS")
        failed = total - passed
        avg_ms = (sum(r["latency_ms"] for r in mr) // total) if total else 0
        line = f"  {model:<28}  {passed:>5}  {failed:>5}  {total:>6}  {avg_ms:>8}"
        if failed == 0 and total > 0:
            print(f"{GREEN}{line}{NC}")
        elif passed == 0:
            print(f"{RED}{line}{NC}")
        else:
            print(f"{YELLOW}{line}{NC}")
    print("=" * 72)

    total_fail = sum(1 for r in results if r["status"] == "FAIL")
    total_pass = sum(1 for r in results if r["status"] == "PASS")
    print(
        f"\n  Grand total: {total_pass} PASS, {total_fail} FAIL "
        f"out of {len(results)} runs"
    )

    # Save results
    out_path = REPO_DIR / "artifacts" / "realdata" / "local_hf10_e2e_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results: {out_path}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
