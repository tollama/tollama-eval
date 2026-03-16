"""Shared test helpers for artifact-rich benchmark scenarios."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataCharacteristics,
    DataProfile,
    DiagnosticsResult,
    FoldResult,
    ForecastData,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.dashboard import (
    _build_dashboard_artifact_bundle,
    _build_dashboard_filtered_details_json,
    _build_dashboard_filtered_results_json,
    _build_dashboard_snapshot_html,
    _filter_result_for_dashboard,
    _load_result_artifacts,
    _open_saved_results,
    _parse_dashboard_query_artifact_paths,
    _render_saved_artifact_sidebar,
)
from ts_autopilot.runners.optional import OptionalRunnerStatus


def make_rich_result() -> BenchmarkResult:
    """Create a benchmark result with forecasts, diagnostics, and provenance."""
    result = BenchmarkResult(
        profile=DataProfile(
            n_series=2,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=60,
            max_length=60,
            total_rows=120,
        ),
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[
            ModelResult(
                name="AutoETS",
                runtime_sec=0.5,
                folds=[
                    FoldResult(
                        fold=1,
                        cutoff="2020-06-26",
                        mase=0.9,
                        smape=8.2,
                        rmsse=0.88,
                        mae=0.75,
                        series_scores={"s1": 0.81, "s2": 0.99},
                    ),
                    FoldResult(
                        fold=2,
                        cutoff="2020-07-01",
                        mase=0.85,
                        smape=7.9,
                        rmsse=0.84,
                        mae=0.7,
                        series_scores={"s1": 0.79, "s2": 0.96},
                    ),
                ],
                mean_mase=0.875,
                std_mase=0.025,
                mean_smape=8.05,
                mean_rmsse=0.86,
                mean_mae=0.725,
            ),
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[
                    FoldResult(
                        fold=1,
                        cutoff="2020-06-26",
                        mase=1.02,
                        smape=9.6,
                        rmsse=1.01,
                        mae=0.93,
                        series_scores={"s1": 1.08, "s2": 0.96},
                    ),
                    FoldResult(
                        fold=2,
                        cutoff="2020-07-01",
                        mase=0.98,
                        smape=9.1,
                        rmsse=0.97,
                        mae=0.89,
                        series_scores={"s1": 1.02, "s2": 0.94},
                    ),
                ],
                mean_mase=1.0,
                std_mase=0.0,
                mean_smape=9.35,
                mean_rmsse=0.99,
                mean_mae=0.91,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(
                rank=1,
                name="AutoETS",
                mean_mase=0.875,
                mean_smape=8.05,
                mean_rmsse=0.86,
                mean_mae=0.725,
            ),
            LeaderboardEntry(
                rank=2,
                name="SeasonalNaive",
                mean_mase=1.0,
                mean_smape=9.35,
                mean_rmsse=0.99,
                mean_mae=0.91,
            ),
        ],
        forecast_data=[
            ForecastData(
                model_name="AutoETS",
                fold=2,
                unique_id=["s1", "s1", "s2", "s2"],
                ds=["2020-07-02", "2020-07-03", "2020-07-02", "2020-07-03"],
                y_hat=[10.0, 11.0, 20.0, 21.0],
                y_actual=[10.5, 10.8, 20.3, 21.2],
                train_unique_id=["s1", "s1", "s2", "s2"],
                ds_train_tail=[
                    "2020-06-30",
                    "2020-07-01",
                    "2020-06-30",
                    "2020-07-01",
                ],
                y_train_tail=[9.0, 9.5, 19.1, 19.6],
            ),
            ForecastData(
                model_name="SeasonalNaive",
                fold=2,
                unique_id=["s1", "s1", "s2", "s2"],
                ds=["2020-07-02", "2020-07-03", "2020-07-02", "2020-07-03"],
                y_hat=[9.8, 10.1, 19.7, 20.4],
                y_actual=[10.5, 10.8, 20.3, 21.2],
                train_unique_id=["s1", "s1", "s2", "s2"],
                ds_train_tail=[
                    "2020-06-30",
                    "2020-07-01",
                    "2020-06-30",
                    "2020-07-01",
                ],
                y_train_tail=[9.0, 9.5, 19.1, 19.6],
            ),
        ],
        diagnostics=[
            DiagnosticsResult(
                model_name="AutoETS",
                residual_mean=0.1,
                residual_std=0.5,
                residual_skew=0.1,
                residual_kurtosis=-0.2,
                ljung_box_p=0.42,
                histogram_bins=[0.0, 0.5, 1.0],
                histogram_counts=[3, 2],
                acf_lags=[1, 2, 3],
                acf_values=[0.1, -0.05, 0.02],
                residuals=[0.1, -0.2, 0.3, 0.0],
                fitted=[10.0, 11.0, 20.0, 21.0],
            ),
        ],
        data_characteristics=DataCharacteristics(
            y_mean=15.0,
            y_std=5.0,
            y_min=9.0,
            y_max=21.2,
            y_median=15.4,
            trend_strength=0.35,
            seasonality_strength=0.72,
            series_heterogeneity=0.78,
        ),
    )
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="Prophet",
            available=True,
            reason="available",
            runner_names=["Prophet"],
        ),
        OptionalRunnerStatus(
            label="NeuralForecast",
            available=False,
            reason="failed health check",
            runner_names=["NHITS", "NBEATS"],
        ),
    ]
    return result


def write_rich_artifact_dir(
    output_dir: str | Path,
    *,
    result: BenchmarkResult | None = None,
) -> dict[str, Any]:
    """Write a rich benchmark artifact directory for integration tests."""
    from ts_autopilot.pipeline import write_output_artifacts

    output_path = Path(output_dir)
    rich_result = result if result is not None else make_rich_result()
    written_paths = write_output_artifacts(rich_result, output_path)
    return {
        "result": rich_result,
        "output_dir": output_path,
        "written_paths": written_paths,
        "results_path": output_path / "results.json",
        "details_path": output_path / "details.json",
        "report_path": output_path / "report.html",
    }


def open_saved_dashboard_artifact_dir(
    artifact_dir: dict[str, Any],
    *,
    monkeypatch: Any,
    install_streamlit: Any,
) -> Any:
    """Open a written artifact directory through the saved-results dashboard path."""
    fake_st = install_streamlit(monkeypatch)
    _render_saved_artifact_sidebar(
        fake_st,
        artifact_dir["results_path"],
        artifact_dir["details_path"],
    )
    _open_saved_results(
        artifact_dir["results_path"],
        artifact_dir["details_path"],
    )
    return fake_st


def reopen_saved_dashboard_artifact_dir_via_query_params(
    artifact_dir: dict[str, Any],
    *,
    monkeypatch: Any,
    install_streamlit: Any,
    query_updates: dict[str, str] | None = None,
) -> Any:
    """Open saved dashboard artifacts via query-param path resolution."""
    fake_st = install_streamlit(monkeypatch)
    fake_st.query_params.update(
        {
            "results": str(artifact_dir["results_path"]),
            "details": str(artifact_dir["details_path"]),
            **(query_updates or {}),
        }
    )
    parsed_results, parsed_details = _parse_dashboard_query_artifact_paths(fake_st)
    if parsed_results is not None:
        _render_saved_artifact_sidebar(
            fake_st,
            parsed_results,
            parsed_details,
        )
    _open_saved_results(parsed_results, parsed_details)
    return fake_st


def build_filtered_dashboard_exports(
    result: BenchmarkResult,
    *,
    selected_model_names: list[str],
    max_rank: int | None = None,
) -> dict[str, Any]:
    """Build the full filtered dashboard export set from one filtered view."""
    filtered = _filter_result_for_dashboard(
        result,
        selected_model_names=selected_model_names,
        max_rank=max_rank,
    )
    return {
        "filtered_result": filtered,
        "results_json": _build_dashboard_filtered_results_json(filtered),
        "details_json": _build_dashboard_filtered_details_json(filtered),
        "snapshot_html": _build_dashboard_snapshot_html(filtered),
        "bundle": _build_dashboard_artifact_bundle(filtered),
    }


def write_filtered_artifact_dir(
    output_dir: str | Path,
    *,
    result: BenchmarkResult,
    selected_model_names: list[str],
    max_rank: int | None = None,
) -> dict[str, Any]:
    """Write filtered results/detail artifacts for saved-results reopen tests."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exports = build_filtered_dashboard_exports(
        result,
        selected_model_names=selected_model_names,
        max_rank=max_rank,
    )
    results_path = output_path / "results.json"
    details_path = output_path / "details.json"
    results_path.write_text(exports["results_json"], encoding="utf-8")
    details_json = exports["details_json"]
    if details_json is None:
        raise AssertionError("Filtered dashboard exports did not produce details.json")
    details_path.write_text(details_json, encoding="utf-8")

    return {
        "output_dir": output_path,
        "results_path": results_path,
        "details_path": details_path,
        **exports,
    }


def decode_dashboard_bundle(bundle: bytes) -> dict[str, object]:
    """Return decoded files and parsed manifest from a dashboard artifact zip."""
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())
        text_files = {
            name: zf.read(name).decode("utf-8")
            for name in names
            if name.endswith((".json", ".txt", ".html"))
        }
    return {
        "names": names,
        "texts": text_files,
        "manifest": json.loads(text_files["manifest.json"]),
        "readme": text_files["README.txt"],
    }


def assert_saved_dashboard_rich_sections(
    fake_st: Any,
    *,
    leader_name: str,
    expect_per_series: bool,
) -> None:
    """Assert that saved-results dashboard rendering shows rich sections."""
    assert "Benchmark Environment" in fake_st.subheaders
    assert "Data Overview" in fake_st.subheaders
    assert "Forecast vs Actual" in fake_st.subheaders
    assert "Residual Diagnostics" in fake_st.subheaders
    if expect_per_series:
        assert "Per-Series Winners" in fake_st.subheaders
        assert any(
            caption == "Hardest-series drill-downs:" for caption in fake_st.captions
        )
    assert any(
        caption == "Drill-down shortcuts from the current leaderboard winner:"
        for caption in fake_st.captions
    )
    assert any(
        caption == "Diagnostics drill-downs:" for caption in fake_st.captions
    )
    assert any(
        f"Focus forecasts on {leader_name}" in item["text"]
        for item in fake_st.markdowns
    )
    assert any(
        f"View forecasts for {leader_name}" in item["text"]
        for item in fake_st.markdowns
    )


def assert_saved_dashboard_artifact_surface(
    fake_st: Any,
    *,
    snapshot_filename: str,
    overall_status: str = "Complete",
    details_status: str = "Loaded",
) -> None:
    """Assert saved-results dashboard health, manifest, and export surface."""
    assert "Artifact Health" in fake_st.subheaders
    assert "Artifact Manifest" in fake_st.subheaders
    assert "Snapshot Export" in fake_st.subheaders
    assert any(
        f"Overall: {overall_status}" in item["text"]
        for item in fake_st.markdowns
    )
    assert any(
        "results.json: Loaded" in item["text"]
        for item in fake_st.markdowns
    )
    assert any(
        f"details.json: {details_status}" in item["text"]
        for item in fake_st.markdowns
    )
    assert fake_st.downloads
    assert fake_st.downloads[0]["file_name"] == snapshot_filename


def assert_saved_dashboard_sidebar_surface(
    fake_st: Any,
    *,
    results_source: str,
    details_source: str,
    details_status: str = "Loaded",
) -> None:
    """Assert saved-results sidebar badges and source labels."""
    assert fake_st.headers == ["Loaded Artifacts"]
    assert any(
        "results.json: Loaded" in item["text"]
        for item in fake_st.markdowns
    )
    assert any(
        f"details.json: {details_status}" in item["text"]
        for item in fake_st.markdowns
    )
    assert results_source in fake_st.captions
    assert details_source in fake_st.captions


def assert_filtered_export_coherence(
    *,
    results_json: str,
    details_json: str,
    snapshot_html: str,
    bundle: bytes,
    expected_model_names: list[str],
    forbidden_model_names: list[str],
    expect_per_series_section: bool,
    upload_factory: Any,
) -> None:
    """Assert filtered JSON, HTML, and bundle exports stay mutually consistent."""
    leader_slug = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in expected_model_names[0]
    ).strip("-")
    results_payload = json.loads(results_json)
    details_payload = json.loads(details_json)
    bundle_data = decode_dashboard_bundle(bundle)

    assert [model["name"] for model in results_payload["models"]] == (
        expected_model_names
    )
    assert [entry["name"] for entry in results_payload["leaderboard"]] == (
        expected_model_names
    )
    assert [fd["model_name"] for fd in details_payload["forecast_data"]] == (
        expected_model_names
    )
    assert [diag["model_name"] for diag in details_payload["diagnostics"]] == (
        expected_model_names
    )
    assert details_payload["optional_model_environment"][0]["label"] == "Prophet"
    assert details_payload["optional_model_environment"][1]["label"] == (
        "NeuralForecast"
    )
    for forbidden_name in forbidden_model_names:
        assert forbidden_name not in details_json

    assert "Filtered Benchmark Snapshot" in snapshot_html
    assert "Snapshot provenance" in snapshot_html
    assert "Forecast vs Actual" in snapshot_html
    assert "Residual Diagnostics" in snapshot_html
    assert "Optional Model Environment" in snapshot_html
    if expect_per_series_section:
        assert 'id="per-series"' in snapshot_html
    else:
        assert 'id="per-series"' not in snapshot_html
    for expected_name in expected_model_names:
        assert expected_name in snapshot_html
    for forbidden_name in forbidden_model_names:
        assert forbidden_name not in snapshot_html

    results_name = f"dashboard-filtered-results-{leader_slug}.json"
    details_name = f"dashboard-filtered-details-{leader_slug}.json"
    snapshot_name = f"dashboard-snapshot-{leader_slug}.html"
    assert bundle_data["names"] == {
        "manifest.json",
        "README.txt",
        snapshot_name,
        results_name,
        details_name,
    }

    text_files = bundle_data["texts"]
    assert text_files[results_name] == results_json
    assert text_files[details_name] == details_json
    assert text_files[snapshot_name] == snapshot_html

    manifest = bundle_data["manifest"]
    assert manifest["bundle"] == f"dashboard-filtered-bundle-{leader_slug}.zip"
    assert manifest["leader"] == expected_model_names[0]
    assert manifest["model_count"] == len(expected_model_names)
    assert manifest["artifacts"][2]["features"] == [
        "forecast_data",
        "diagnostics",
        "data_characteristics",
        "optional_model_environment",
    ]

    readme = bundle_data["readme"]
    assert snapshot_name in readme
    assert results_name in readme
    assert details_name in readme

    loaded = _load_result_artifacts(
        upload_factory(
            json.loads(text_files[results_name]),
            name="dashboard-filtered-results.json",
        ),
        upload_factory(
            json.loads(text_files[details_name]),
            name="dashboard-filtered-details.json",
        ),
    )
    assert [model.name for model in loaded.models] == expected_model_names
    assert [fd.model_name for fd in loaded.forecast_data] == expected_model_names
    assert [diag.model_name for diag in loaded.diagnostics] == expected_model_names
