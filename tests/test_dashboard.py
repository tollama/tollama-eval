"""Tests for dashboard benchmark environment provenance."""

import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

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
    _artifact_manifest_summary,
    _artifact_source_label,
    _build_dashboard_artifact_bundle,
    _build_dashboard_bundle_manifest,
    _build_dashboard_bundle_readme,
    _build_dashboard_filtered_details_json,
    _build_dashboard_filtered_results_json,
    _build_dashboard_query_link,
    _build_dashboard_shareable_link,
    _build_dashboard_snapshot_html,
    _dashboard_bundle_filename,
    _dashboard_filtered_details_filename,
    _dashboard_filtered_results_filename,
    _dashboard_snapshot_filename,
    _filter_forecast_chart_data,
    _filter_per_series_chart_data,
    _filter_result_for_dashboard,
    _load_result_artifacts,
    _open_saved_results,
    _optional_model_environment_summary,
    _parse_dashboard_args,
    _parse_dashboard_query_artifact_paths,
    _render_artifact_health,
    _render_artifact_manifest,
    _render_data_overview_panel,
    _render_diagnostics_panel,
    _render_display_filters,
    _render_drilldown_links,
    _render_forecast_panels,
    _render_optional_model_environment,
    _render_per_series_panel,
    _render_saved_artifact_sidebar,
    _render_shareable_link,
    _render_snapshot_export,
    _resolve_saved_result_sources,
    _saved_artifact_sidebar_rows,
)
from ts_autopilot.runners.optional import OptionalRunnerStatus


def _make_result() -> BenchmarkResult:
    return BenchmarkResult(
        profile=DataProfile(
            n_series=1,
            frequency="D",
            missing_ratio=0.0,
            season_length_guess=7,
            min_length=60,
            max_length=60,
            total_rows=60,
        ),
        config=BenchmarkConfig(horizon=7, n_folds=2),
        models=[
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[],
                mean_mase=1.0,
                std_mase=0.0,
            )
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="SeasonalNaive", mean_mase=1.0),
        ],
    )


def _make_rich_result() -> BenchmarkResult:
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


class _FakeColumn:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, object]] = []

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))


class _FakeExpander:
    def __enter__(self) -> "_FakeExpander":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeStreamlit:
    def __init__(self) -> None:
        self.headers: list[str] = []
        self.subheaders: list[str] = []
        self.markdowns: list[dict[str, object]] = []
        self.captions: list[str] = []
        self.dataframes: list[object] = []
        self.columns_created: list[_FakeColumn] = []
        self.expanders: list[str] = []
        self.plotly_calls: int = 0
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.info_messages: list[object] = []
        self.downloads: list[dict[str, object]] = []
        self.query_params: dict[str, str] = {}
        self.text_inputs: list[dict[str, object]] = []

    def header(self, text: str) -> None:
        self.headers.append(text)

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def markdown(self, text: str, **kwargs: object) -> None:
        self.markdowns.append(
            {
                "text": text,
                "kwargs": kwargs,
            }
        )

    def columns(self, count: int) -> list[_FakeColumn]:
        self.columns_created = [_FakeColumn() for _ in range(count)]
        return self.columns_created

    def dataframe(self, data: object, **_: object) -> None:
        self.dataframes.append(data)

    def expander(self, label: str, **_: object) -> _FakeExpander:
        self.expanders.append(label)
        return _FakeExpander()

    def plotly_chart(self, *_: object, **__: object) -> None:
        self.plotly_calls += 1

    def slider(self, _label: str, **kwargs: object) -> object:
        return kwargs["value"]

    def multiselect(
        self,
        _label: str,
        _options: list[str],
        *,
        default: list[str],
    ) -> list[str]:
        return default

    def warning(self, text: str) -> None:
        self.warnings.append(text)

    def error(self, text: str) -> None:
        self.errors.append(text)

    def info(self, text: object) -> None:
        self.info_messages.append(text)

    def divider(self) -> None:
        return None

    def download_button(
        self,
        label: str,
        *,
        data: object,
        file_name: str,
        mime: str,
    ) -> None:
        self.downloads.append(
            {
                "label": label,
                "data": data,
                "file_name": file_name,
                "mime": mime,
            }
        )

    def text_input(self, label: str, *, value: str) -> str:
        self.text_inputs.append(
            {
                "label": label,
                "value": value,
            }
        )
        return value

    def experimental_get_query_params(self) -> dict[str, list[str]]:
        return {
            key: [value]
            for key, value in self.query_params.items()
        }

    def experimental_set_query_params(self, **kwargs: object) -> None:
        self.query_params = {
            key: str(value)
            for key, value in kwargs.items()
        }


class _FakeUpload:
    def __init__(self, payload: dict, name: str = "artifact.json") -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.name = name

    def getvalue(self) -> bytes:
        return self._payload


def _install_fake_streamlit_module(monkeypatch) -> _FakeStreamlit:
    fake_st = _FakeStreamlit()
    fake_module = types.ModuleType("streamlit")
    for attr in (
        "header",
        "subheader",
        "caption",
        "markdown",
        "columns",
        "dataframe",
        "expander",
        "plotly_chart",
        "slider",
        "multiselect",
        "warning",
        "error",
        "info",
        "divider",
        "download_button",
        "text_input",
    ):
        setattr(fake_module, attr, getattr(fake_st, attr))
    fake_module.query_params = fake_st.query_params
    monkeypatch.setitem(sys.modules, "streamlit", fake_module)
    return fake_st


def test_optional_model_environment_summary_counts_models_and_groups() -> None:
    result = _make_result()
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

    summary = _optional_model_environment_summary(result)

    assert summary["enabled_models"] == 1
    assert summary["enabled_groups"] == 1
    assert summary["skipped_groups"] == 1
    assert summary["rows"][1]["Detail"] == "failed health check"


def test_render_optional_model_environment_uses_streamlit_block() -> None:
    result = _make_result()
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="LightGBM",
            available=False,
            reason="missing dependency: lightgbm",
            runner_names=["LightGBM"],
        )
    ]
    fake_st = _FakeStreamlit()

    _render_optional_model_environment(fake_st, result)

    assert fake_st.subheaders == ["Benchmark Environment"]
    assert fake_st.columns_created[2].metrics == [("Skipped Groups", 1)]
    assert fake_st.dataframes


def test_render_optional_model_environment_skips_empty_context() -> None:
    fake_st = _FakeStreamlit()

    _render_optional_model_environment(fake_st, _make_result())

    assert fake_st.subheaders == []
    assert fake_st.dataframes == []


def test_load_result_artifacts_merges_details_json() -> None:
    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="Prophet",
            available=True,
            reason="available",
            runner_names=["Prophet"],
        )
    ]

    loaded = _load_result_artifacts(
        _FakeUpload(result.to_dict()),
        _FakeUpload(result.to_details_dict()),
    )

    assert loaded.forecast_data[0].model_name == "SeasonalNaive"
    assert loaded._optional_runner_statuses[0]["label"] == "Prophet"


def test_load_result_artifacts_supports_results_only() -> None:
    result = _make_result()

    loaded = _load_result_artifacts(_FakeUpload(result.to_dict()))

    assert loaded.leaderboard[0].name == "SeasonalNaive"
    assert loaded.forecast_data == []


def test_load_result_artifacts_supports_filesystem_paths(tmp_path) -> None:
    result = _make_result()
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="LightGBM",
            available=False,
            reason="missing dependency: lightgbm",
            runner_names=["LightGBM"],
        )
    ]
    results_path = tmp_path / "results.json"
    details_path = tmp_path / "details.json"
    results_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    details_path.write_text(json.dumps(result.to_details_dict()), encoding="utf-8")

    loaded = _load_result_artifacts(results_path, details_path)

    assert loaded._optional_runner_statuses[0]["label"] == "LightGBM"


def test_written_artifacts_roundtrip_into_dashboard_loader(tmp_path) -> None:
    from ts_autopilot.pipeline import write_output_artifacts

    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="Prophet",
            available=True,
            reason="available",
            runner_names=["Prophet"],
        )
    ]

    out_dir = tmp_path / "out"
    write_output_artifacts(result, out_dir)
    loaded = _load_result_artifacts(
        out_dir / "results.json",
        out_dir / "details.json",
    )

    assert loaded.leaderboard[0].name == "SeasonalNaive"
    assert loaded.forecast_data[0].model_name == "SeasonalNaive"
    assert loaded._optional_runner_statuses[0]["label"] == "Prophet"


def test_open_saved_results_via_artifact_dir_launch_renders_dashboard(
    tmp_path,
    monkeypatch,
) -> None:
    from ts_autopilot.pipeline import write_output_artifacts

    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
            train_unique_id=["s1"],
            ds_train_tail=["2020-01-01"],
            y_train_tail=[0.9],
        )
    ]
    result.diagnostics = [
        DiagnosticsResult(
            model_name="SeasonalNaive",
            residual_mean=0.0,
            residual_std=0.6,
            residual_skew=0.0,
            residual_kurtosis=-0.1,
            ljung_box_p=0.35,
        )
    ]
    out_dir = tmp_path / "out"
    write_output_artifacts(result, out_dir)

    parsed_results, parsed_details = _parse_dashboard_args(
        ["--artifact-dir", str(out_dir)]
    )
    fake_st = _install_fake_streamlit_module(monkeypatch)

    _open_saved_results(parsed_results, parsed_details)

    assert "Artifact Health" in fake_st.subheaders
    assert "Artifact Manifest" in fake_st.subheaders
    assert "Snapshot Export" in fake_st.subheaders
    assert fake_st.downloads


def test_open_saved_results_via_query_param_reopen_renders_dashboard(
    tmp_path,
    monkeypatch,
) -> None:
    from ts_autopilot.pipeline import write_output_artifacts

    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]
    out_dir = tmp_path / "out"
    write_output_artifacts(result, out_dir)

    fake_st = _install_fake_streamlit_module(monkeypatch)
    fake_st.query_params.update(
        {
            "results": str(out_dir / "results.json"),
            "details": str(out_dir / "details.json"),
            "forecast_series": '["s1"]',
        }
    )
    parsed_results, parsed_details = _parse_dashboard_query_artifact_paths(fake_st)

    _open_saved_results(parsed_results, parsed_details)

    assert "Artifact Health" in fake_st.subheaders
    assert "Forecast vs Actual" in fake_st.subheaders
    assert fake_st.downloads[0]["file_name"] == "dashboard-snapshot-seasonalnaive.html"


def test_filtered_export_roundtrip_renders_filtered_saved_results(
    tmp_path,
    monkeypatch,
) -> None:
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
                unique_id=["s1"],
                ds=["2020-07-02"],
                y_hat=[10.0],
                y_actual=[10.5],
            ),
            ForecastData(
                model_name="SeasonalNaive",
                fold=2,
                unique_id=["s1"],
                ds=["2020-07-02"],
                y_hat=[9.8],
                y_actual=[10.5],
            ),
        ],
    )
    filtered = _filter_result_for_dashboard(
        result,
        selected_model_names=["SeasonalNaive"],
        max_rank=2,
    )
    results_path = tmp_path / "filtered-results.json"
    details_path = tmp_path / "filtered-details.json"
    results_path.write_text(
        _build_dashboard_filtered_results_json(filtered),
        encoding="utf-8",
    )
    details_payload = _build_dashboard_filtered_details_json(filtered)
    assert details_payload is not None
    details_path.write_text(details_payload, encoding="utf-8")

    fake_st = _install_fake_streamlit_module(monkeypatch)
    _open_saved_results(results_path, details_path)

    assert fake_st.downloads[0]["file_name"] == "dashboard-snapshot-seasonalnaive.html"
    assert "SeasonalNaive" in fake_st.downloads[0]["data"]
    assert "AutoETS" not in fake_st.downloads[0]["data"]


def test_write_output_artifacts_roundtrip_preserves_rich_dashboard_views(
    tmp_path,
    monkeypatch,
) -> None:
    from ts_autopilot.pipeline import write_output_artifacts

    result = _make_rich_result()

    out_dir = tmp_path / "rich-out"
    write_output_artifacts(result, out_dir)
    fake_st = _install_fake_streamlit_module(monkeypatch)
    _open_saved_results(out_dir / "results.json", out_dir / "details.json")

    assert "Artifact Health" in fake_st.subheaders
    assert "Benchmark Environment" in fake_st.subheaders
    assert "Data Overview" in fake_st.subheaders
    assert "Forecast vs Actual" in fake_st.subheaders
    assert "Residual Diagnostics" in fake_st.subheaders
    assert "Per-Series Winners" in fake_st.subheaders
    assert any(
        caption == "Drill-down shortcuts from the current leaderboard winner:"
        for caption in fake_st.captions
    )
    assert any(caption == "Diagnostics drill-downs:" for caption in fake_st.captions)
    assert any(
        caption == "Hardest-series drill-downs:" for caption in fake_st.captions
    )
    assert any(
        "Focus forecasts on AutoETS" in item["text"]
        for item in fake_st.markdowns
    )
    assert any(
        "View forecasts for AutoETS" in item["text"]
        for item in fake_st.markdowns
    )


def test_cross_artifact_outputs_stay_consistent_for_rich_result(
    tmp_path,
    monkeypatch,
) -> None:
    from ts_autopilot.pipeline import write_output_artifacts
    from ts_autopilot.reporting.export import export_excel

    openpyxl = pytest.importorskip("openpyxl")

    result = _make_rich_result()
    out_dir = tmp_path / "artifact-consistency"
    write_output_artifacts(result, out_dir)
    excel_path = export_excel(result, out_dir / "report.xlsx")

    results_payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    details_payload = json.loads((out_dir / "details.json").read_text(encoding="utf-8"))
    report_html = (out_dir / "report.html").read_text(encoding="utf-8")

    assert results_payload["leaderboard"][0]["name"] == "AutoETS"
    assert len(details_payload["forecast_data"]) == 2
    assert details_payload["diagnostics"][0]["model_name"] == "AutoETS"
    assert details_payload["data_characteristics"]["seasonality_strength"] == 0.72
    assert details_payload["optional_model_environment"][1]["reason"] == (
        "failed health check"
    )

    for expected in (
        "Data Overview",
        "Forecast vs Actual",
        "Residual Diagnostics",
        "Per-Series Winners",
        "Optional Model Environment",
        "AutoETS",
    ):
        assert expected in report_html

    workbook = openpyxl.load_workbook(excel_path)
    assert {"Leaderboard", "Per-Series Winners", "Optional Models"}.issubset(
        set(workbook.sheetnames)
    )
    assert workbook["Leaderboard"]["B2"].value == "AutoETS"
    assert workbook["Optional Models"]["A1"].value == "Optional Model Environment"
    assert workbook["Optional Models"]["D9"].value == "failed health check"

    winner_sheet = workbook["Per-Series Winners"]
    winner_rows = list(
        winner_sheet.iter_rows(min_row=2, max_col=4, values_only=True)
    )
    assert ("s1", "AutoETS", 0.8, "SeasonalNaive") in winner_rows
    assert ("s2", "SeasonalNaive", 0.95, "AutoETS") in winner_rows

    fake_st = _install_fake_streamlit_module(monkeypatch)
    _open_saved_results(out_dir / "results.json", out_dir / "details.json")

    assert "Benchmark Environment" in fake_st.subheaders
    assert "Data Overview" in fake_st.subheaders
    assert "Forecast vs Actual" in fake_st.subheaders
    assert "Residual Diagnostics" in fake_st.subheaders
    assert "Per-Series Winners" in fake_st.subheaders
    assert any(
        "Focus forecasts on AutoETS" in item["text"]
        for item in fake_st.markdowns
    )
    assert any(
        "View forecasts for AutoETS" in item["text"]
        for item in fake_st.markdowns
    )


def test_filter_result_for_dashboard_trims_models_forecasts_and_diagnostics() -> None:
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
                folds=[],
                mean_mase=0.875,
                std_mase=0.025,
            ),
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[],
                mean_mase=1.0,
                std_mase=0.0,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.875),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.0),
        ],
        forecast_data=[
            ForecastData(
                model_name="AutoETS",
                fold=2,
                unique_id=["s1"],
                ds=["2020-07-02"],
                y_hat=[10.0],
                y_actual=[10.5],
            ),
            ForecastData(
                model_name="SeasonalNaive",
                fold=2,
                unique_id=["s1"],
                ds=["2020-07-02"],
                y_hat=[9.8],
                y_actual=[10.5],
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
            ),
            DiagnosticsResult(
                model_name="SeasonalNaive",
                residual_mean=0.0,
                residual_std=0.6,
                residual_skew=0.0,
                residual_kurtosis=-0.1,
                ljung_box_p=0.35,
            ),
        ],
    )

    filtered = _filter_result_for_dashboard(
        result,
        selected_model_names=["SeasonalNaive"],
        max_rank=2,
    )

    assert [model.name for model in filtered.models] == ["SeasonalNaive"]
    assert [entry.name for entry in filtered.leaderboard] == ["SeasonalNaive"]
    assert filtered.leaderboard[0].rank == 1
    assert [fd.model_name for fd in filtered.forecast_data] == ["SeasonalNaive"]
    assert [diag.model_name for diag in filtered.diagnostics] == ["SeasonalNaive"]


def test_parse_dashboard_args_uses_artifact_dir_defaults(tmp_path) -> None:
    results_path = tmp_path / "results.json"
    details_path = tmp_path / "details.json"
    results_path.write_text("{}", encoding="utf-8")
    details_path.write_text("{}", encoding="utf-8")

    parsed_results, parsed_details = _parse_dashboard_args(
        ["--artifact-dir", str(tmp_path)]
    )

    assert parsed_results == results_path
    assert parsed_details == details_path


def test_parse_dashboard_args_prefers_explicit_results_path(tmp_path) -> None:
    explicit_results = tmp_path / "custom-results.json"

    parsed_results, parsed_details = _parse_dashboard_args(
        [
            "--artifact-dir",
            str(tmp_path),
            "--results",
            str(explicit_results),
        ]
    )

    assert parsed_results == explicit_results
    assert parsed_details is None


def test_parse_dashboard_query_artifact_paths_reads_local_paths() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "results": "/tmp/out/results.json",
        "details": "/tmp/out/details.json",
    }

    parsed_results, parsed_details = _parse_dashboard_query_artifact_paths(fake_st)

    assert parsed_results == Path("/tmp/out/results.json")
    assert parsed_details == Path("/tmp/out/details.json")


def test_saved_artifact_sidebar_rows_marks_details_not_provided(tmp_path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text("{}", encoding="utf-8")

    rows = _saved_artifact_sidebar_rows(results_path, None)

    assert rows[0]["Status"] == "Loaded"
    assert rows[1]["Status"] == "Not Provided"
    assert rows[1]["Source"] == "not provided"


def test_saved_artifact_sidebar_rows_marks_missing_details_on_disk(tmp_path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text("{}", encoding="utf-8")
    details_path = tmp_path / "details.json"

    rows = _saved_artifact_sidebar_rows(results_path, details_path)

    assert rows[0]["Status"] == "Loaded"
    assert rows[1]["Status"] == "Missing on Disk"
    assert rows[1]["Source"] == str(details_path)


def test_render_saved_artifact_sidebar_uses_badges_and_sources(tmp_path) -> None:
    fake_st = _FakeStreamlit()
    results_path = tmp_path / "results.json"
    results_path.write_text("{}", encoding="utf-8")

    _render_saved_artifact_sidebar(fake_st, results_path, tmp_path / "details.json")

    assert fake_st.headers == ["Loaded Artifacts"]
    assert "results.json: Loaded" in fake_st.markdowns[0]["text"]
    assert "details.json: Missing on Disk" in fake_st.markdowns[1]["text"]
    assert fake_st.captions[-1] == str(tmp_path / "details.json")


def test_resolve_saved_result_sources_stops_when_results_path_is_missing() -> None:
    fake_st = _FakeStreamlit()

    resolved = _resolve_saved_result_sources(
        fake_st,
        Path("/tmp/out/missing-results.json"),
        Path("/tmp/out/missing-details.json"),
    )

    assert resolved is None
    assert "Saved results artifact was not found" in fake_st.errors[0]
    assert "local artifact path that no longer exists" in fake_st.info_messages[0]


def test_resolve_saved_result_sources_drops_missing_details_path(tmp_path) -> None:
    fake_st = _FakeStreamlit()
    results_path = tmp_path / "results.json"
    results_path.write_text("{}", encoding="utf-8")

    resolved = _resolve_saved_result_sources(
        fake_st,
        results_path,
        tmp_path / "missing-details.json",
    )

    assert resolved == (results_path, None, True)
    assert "Optional details artifact was not found" in fake_st.warnings[0]
    assert "Continuing with `results.json` only" in fake_st.info_messages[0]


def test_artifact_source_label_uses_uploaded_name() -> None:
    label = _artifact_source_label(_FakeUpload({}, name="results.json"))

    assert label == "results.json"


def test_dashboard_snapshot_filename_uses_leader_name() -> None:
    result = _make_result()

    filename = _dashboard_snapshot_filename(result)

    assert filename == "dashboard-snapshot-seasonalnaive.html"


def test_dashboard_filtered_json_filenames_use_leader_name() -> None:
    result = _make_result()

    assert (
        _dashboard_filtered_results_filename(result)
        == "dashboard-filtered-results-seasonalnaive.json"
    )
    assert (
        _dashboard_filtered_details_filename(result)
        == "dashboard-filtered-details-seasonalnaive.json"
    )
    assert (
        _dashboard_bundle_filename(result)
        == "dashboard-filtered-bundle-seasonalnaive.zip"
    )


def test_build_dashboard_snapshot_html_respects_filtered_models() -> None:
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
                folds=[],
                mean_mase=0.875,
                std_mase=0.025,
            ),
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[],
                mean_mase=1.0,
                std_mase=0.0,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.875),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.0),
        ],
    )
    filtered = _filter_result_for_dashboard(
        result,
        selected_model_names=["SeasonalNaive"],
        max_rank=2,
    )

    html = _build_dashboard_snapshot_html(filtered)

    assert "Filtered Benchmark Snapshot" in html
    assert "SeasonalNaive" in html
    assert "AutoETS" not in html
    assert "Snapshot provenance" in html
    assert "exported from the dashboard filtered view with 1 visible model(s)" in html
    assert "leader at export time: SeasonalNaive (MASE 1.0000)" in html


def test_render_display_filters_restores_query_param_state() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "display_rank": "2",
        "display_models": '["AutoETS"]',
    }
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
                folds=[],
                mean_mase=0.875,
                std_mase=0.025,
            ),
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[],
                mean_mase=1.0,
                std_mase=0.0,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.875),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.0),
        ],
    )

    filtered = _render_display_filters(fake_st, result)

    assert [model.name for model in filtered.models] == ["AutoETS"]
    assert fake_st.query_params["display_rank"] == "2"
    assert fake_st.query_params["display_models"] == '["AutoETS"]'


def test_build_dashboard_shareable_link_encodes_current_filters() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "results": "/tmp/out/results.json",
        "details": "/tmp/out/details.json",
        "display_rank": "2",
        "display_models": '["AutoETS"]',
        "forecast_models": '["AutoETS"]',
        "forecast_series": '["s1"]',
    }

    link = _build_dashboard_shareable_link(fake_st)

    assert link is not None
    assert link.startswith("?results=%2Ftmp%2Fout%2Fresults.json")
    assert "details=%2Ftmp%2Fout%2Fdetails.json" in link
    assert "display_models=%5B%22AutoETS%22%5D" in link
    assert "forecast_series=%5B%22s1%22%5D" in link


def test_build_dashboard_query_link_applies_overrides() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "display_models": '["AutoETS","SeasonalNaive"]',
        "forecast_series": '["s1"]',
    }

    link = _build_dashboard_query_link(
        fake_st,
        {
            "forecast_models": '["AutoETS"]',
            "forecast_series": '["s2"]',
        },
    )

    assert link is not None
    assert "display_models=%5B%22AutoETS%22%2C%22SeasonalNaive%22%5D" in link
    assert "forecast_models=%5B%22AutoETS%22%5D" in link
    assert "forecast_series=%5B%22s2%22%5D" in link


def test_render_shareable_link_outputs_copyable_url() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "display_rank": "2",
        "display_models": '["AutoETS"]',
    }

    _render_shareable_link(fake_st)

    assert fake_st.subheaders == ["Shareable Link"]
    assert fake_st.text_inputs[0]["label"] == "Shareable URL"
    assert fake_st.text_inputs[0]["value"].startswith("?display_rank=2")


def test_render_drilldown_links_outputs_markdown_links() -> None:
    fake_st = _FakeStreamlit()

    _render_drilldown_links(
        fake_st,
        "Drill-down shortcuts:",
        [("Focus winner", "?display_models=%5B%22AutoETS%22%5D")],
    )

    assert fake_st.captions == ["Drill-down shortcuts:"]
    assert "[Focus winner](?display_models=%5B%22AutoETS%22%5D)" in fake_st.markdowns[
        0
    ]["text"]


def test_build_dashboard_filtered_results_json_respects_filtered_models() -> None:
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
                folds=[],
                mean_mase=0.875,
                std_mase=0.025,
            ),
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[],
                mean_mase=1.0,
                std_mase=0.0,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.875),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.0),
        ],
    )
    filtered = _filter_result_for_dashboard(
        result,
        selected_model_names=["SeasonalNaive"],
        max_rank=2,
    )

    payload = _build_dashboard_filtered_results_json(filtered)

    assert '"SeasonalNaive"' in payload
    assert '"AutoETS"' not in payload


def test_build_dashboard_filtered_details_json_returns_none_without_details() -> None:
    assert _build_dashboard_filtered_details_json(_make_result()) is None


def test_build_dashboard_filtered_details_json_includes_filtered_details() -> None:
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
                folds=[],
                mean_mase=0.875,
                std_mase=0.025,
            ),
            ModelResult(
                name="SeasonalNaive",
                runtime_sec=0.1,
                folds=[],
                mean_mase=1.0,
                std_mase=0.0,
            ),
        ],
        leaderboard=[
            LeaderboardEntry(rank=1, name="AutoETS", mean_mase=0.875),
            LeaderboardEntry(rank=2, name="SeasonalNaive", mean_mase=1.0),
        ],
        forecast_data=[
            ForecastData(
                model_name="AutoETS",
                fold=2,
                unique_id=["s1"],
                ds=["2020-07-02"],
                y_hat=[10.0],
                y_actual=[10.5],
            ),
            ForecastData(
                model_name="SeasonalNaive",
                fold=2,
                unique_id=["s1"],
                ds=["2020-07-02"],
                y_hat=[9.8],
                y_actual=[10.5],
            ),
        ],
    )
    filtered = _filter_result_for_dashboard(
        result,
        selected_model_names=["SeasonalNaive"],
        max_rank=2,
    )

    payload = _build_dashboard_filtered_details_json(filtered)

    assert payload is not None
    assert '"SeasonalNaive"' in payload
    assert '"AutoETS"' not in payload


def test_build_dashboard_artifact_bundle_contains_expected_files() -> None:
    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]

    bundle = _build_dashboard_artifact_bundle(result)

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "README.txt" in names
        assert "dashboard-snapshot-seasonalnaive.html" in names
        assert "dashboard-filtered-results-seasonalnaive.json" in names
        assert "dashboard-filtered-details-seasonalnaive.json" in names


def test_build_dashboard_artifact_bundle_skips_details_when_absent() -> None:
    bundle = _build_dashboard_artifact_bundle(_make_result())

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "README.txt" in names
        assert "dashboard-snapshot-seasonalnaive.html" in names
        assert "dashboard-filtered-results-seasonalnaive.json" in names
        assert "dashboard-filtered-details-seasonalnaive.json" not in names


def test_build_dashboard_bundle_manifest_describes_artifacts() -> None:
    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]
    manifest = json.loads(_build_dashboard_bundle_manifest(result))

    assert manifest["bundle"] == "dashboard-filtered-bundle-seasonalnaive.zip"
    assert manifest["leader"] == "SeasonalNaive"
    assert manifest["model_count"] == 1
    artifact_names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert "dashboard-snapshot-seasonalnaive.html" in artifact_names
    assert "dashboard-filtered-results-seasonalnaive.json" in artifact_names
    assert "dashboard-filtered-details-seasonalnaive.json" in artifact_names


def test_build_dashboard_bundle_readme_describes_included_files() -> None:
    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]

    readme = _build_dashboard_bundle_readme(result)

    assert "Filtered Dashboard Artifact Bundle" in readme
    assert "dashboard-snapshot-seasonalnaive.html" in readme
    assert "dashboard-filtered-results-seasonalnaive.json" in readme
    assert "dashboard-filtered-details-seasonalnaive.json" in readme


def test_build_dashboard_bundle_readme_notes_missing_details() -> None:
    readme = _build_dashboard_bundle_readme(_make_result())

    assert (
        "not included because the filtered view has no detail-only payloads"
        in readme
    )


def test_artifact_manifest_summary_marks_missing_details() -> None:
    result = _make_result()

    manifest = _artifact_manifest_summary(
        result,
        results_source="results.json",
        details_source=None,
        details_requested=False,
    )

    assert manifest["loaded_count"] == 1
    assert manifest["missing_count"] == 1
    assert manifest["rows"][1]["Artifact"] == "details.json"
    assert manifest["rows"][1]["Status"] == "Not Provided"
    assert manifest["rows"][1]["Source"] == "not provided"


def test_artifact_manifest_summary_lists_detail_features() -> None:
    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]
    result._optional_runner_statuses = [
        OptionalRunnerStatus(
            label="Prophet",
            available=True,
            reason="available",
            runner_names=["Prophet"],
        )
    ]

    manifest = _artifact_manifest_summary(
        result,
        results_source="results.json",
        details_source="details.json",
        details_requested=True,
    )

    assert manifest["rows"][1]["Status"] == "Loaded"
    assert "forecast data" in manifest["rows"][1]["Contents"]
    assert "environment provenance" in manifest["rows"][1]["Contents"]


def test_artifact_manifest_summary_marks_details_missing_on_disk() -> None:
    result = _make_result()

    manifest = _artifact_manifest_summary(
        result,
        results_source="results.json",
        details_source="/tmp/out/details.json",
        details_requested=False,
        details_missing_on_disk=True,
    )

    assert manifest["rows"][1]["Status"] == "Missing on Disk"
    assert manifest["rows"][1]["Source"] == "/tmp/out/details.json"
    assert "requested, but local file was not available" in manifest["rows"][1][
        "Contents"
    ]


def test_render_artifact_health_uses_manifest_states() -> None:
    fake_st = _FakeStreamlit()

    _render_artifact_health(
        fake_st,
        {
            "loaded_count": 1,
            "missing_count": 1,
            "rows": [
                {
                    "Artifact": "results.json",
                    "Status": "Loaded",
                    "Source": "results.json",
                    "Contents": "profile, config, models, leaderboard",
                },
                {
                    "Artifact": "details.json",
                    "Status": "Missing on Disk",
                    "Source": "/tmp/out/details.json",
                    "Contents": "requested, but local file was not available",
                },
            ],
        },
    )

    assert fake_st.subheaders == ["Artifact Health"]
    assert fake_st.columns_created[0].metrics == [("Overall", "Degraded")]
    assert fake_st.columns_created[1].metrics == [("results.json", "Loaded")]
    assert fake_st.columns_created[2].metrics == [("details.json", "Missing on Disk")]
    assert "Overall: Degraded" in fake_st.markdowns[0]["text"]
    assert "background:#fee2e2" in fake_st.markdowns[0]["text"]
    assert fake_st.markdowns[0]["kwargs"] == {"unsafe_allow_html": True}


def test_render_artifact_manifest_uses_streamlit_block() -> None:
    fake_st = _FakeStreamlit()

    _render_artifact_manifest(
        fake_st,
        {
            "loaded_count": 1,
            "missing_count": 1,
            "rows": [
                {
                    "Artifact": "results.json",
                    "Status": "Loaded",
                    "Source": "results.json",
                    "Contents": "profile, config, models, leaderboard",
                },
                {
                    "Artifact": "details.json",
                    "Status": "Not Provided",
                    "Source": "not provided",
                    "Contents": "-",
                },
            ],
        },
    )

    assert fake_st.subheaders == ["Artifact Manifest"]
    assert fake_st.columns_created[0].metrics == [("Artifacts Loaded", 1)]
    assert fake_st.columns_created[1].metrics == [("Artifacts Missing", 1)]
    assert fake_st.dataframes


def test_render_snapshot_export_adds_download_button() -> None:
    fake_st = _FakeStreamlit()

    _render_snapshot_export(fake_st, _make_result())

    assert fake_st.subheaders == ["Snapshot Export"]
    assert fake_st.downloads[0]["file_name"] == "dashboard-snapshot-seasonalnaive.html"
    assert fake_st.downloads[0]["mime"] == "text/html"
    assert (
        fake_st.downloads[1]["file_name"]
        == "dashboard-filtered-results-seasonalnaive.json"
    )
    assert fake_st.downloads[1]["mime"] == "application/json"
    assert (
        fake_st.downloads[2]["file_name"]
        == "dashboard-filtered-bundle-seasonalnaive.zip"
    )
    assert fake_st.downloads[2]["mime"] == "application/zip"
    assert len(fake_st.downloads) == 3


def test_render_snapshot_export_adds_details_download_when_available() -> None:
    fake_st = _FakeStreamlit()
    result = _make_result()
    result.forecast_data = [
        ForecastData(
            model_name="SeasonalNaive",
            fold=1,
            unique_id=["s1"],
            ds=["2020-01-02"],
            y_hat=[1.0],
            y_actual=[1.1],
        )
    ]

    _render_snapshot_export(fake_st, result)

    assert len(fake_st.downloads) == 4
    assert (
        fake_st.downloads[2]["file_name"]
        == "dashboard-filtered-details-seasonalnaive.json"
    )
    assert fake_st.downloads[2]["mime"] == "application/json"
    assert (
        fake_st.downloads[3]["file_name"]
        == "dashboard-filtered-bundle-seasonalnaive.zip"
    )
    assert fake_st.downloads[3]["mime"] == "application/zip"


def test_render_forecast_panels_outputs_chart_blocks() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "forecast_models": '["AutoETS"]',
        "forecast_series": '["s1"]',
    }

    _render_forecast_panels(
        fake_st,
        {
            "fold": 2,
            "models": [
                {
                    "name": "AutoETS",
                    "rank": 1,
                    "mean_mase": 0.875,
                    "summary": "Mean MASE is 0.8750.",
                    "series": [
                        {
                            "name": "s1",
                            "note": "MASE 0.8000; this model beats the naive baseline.",
                            "ds_history": ["2020-07-01"],
                            "y_history": [9.5],
                            "ds_actual": ["2020-07-02"],
                            "y_actual": [10.5],
                            "ds_forecast": ["2020-07-02"],
                            "y_hat": [10.0],
                        }
                    ],
                }
            ],
        },
    )

    assert fake_st.subheaders == ["Forecast vs Actual"]
    assert fake_st.expanders == ["AutoETS (MASE=0.8750)"]
    assert fake_st.plotly_calls == 1
    assert fake_st.query_params["forecast_models"] == '["AutoETS"]'
    assert fake_st.query_params["forecast_series"] == '["s1"]'


def test_filter_forecast_chart_data_limits_models_and_series() -> None:
    filtered = _filter_forecast_chart_data(
        {
            "fold": 2,
            "selected_series": ["s1", "s2"],
            "models": [
                {
                    "name": "AutoETS",
                    "rank": 1,
                    "mean_mase": 0.875,
                    "summary": "Mean MASE is 0.8750.",
                    "series": [
                        {"name": "s1"},
                        {"name": "s2"},
                    ],
                },
                {
                    "name": "SeasonalNaive",
                    "rank": 2,
                    "mean_mase": 1.0,
                    "summary": "Mean MASE is 1.0000.",
                    "series": [
                        {"name": "s1"},
                    ],
                },
            ],
        },
        selected_model_names=["SeasonalNaive"],
        selected_series=["s1"],
    )

    assert filtered["selected_series"] == ["s1"]
    assert [model["name"] for model in filtered["models"]] == ["SeasonalNaive"]
    assert filtered["models"][0]["series"] == [{"name": "s1"}]


def test_render_diagnostics_panel_outputs_metrics_and_charts() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "display_models": '["AutoETS","SeasonalNaive"]',
    }

    _render_diagnostics_panel(
        fake_st,
        {
            "model_name": "AutoETS",
            "residual_mean": 0.1,
            "residual_std": 0.5,
            "ljung_box_p": 0.42,
            "histogram_bins": [0.0, 0.5, 1.0],
            "histogram_counts": [3, 2],
            "acf_lags": [1, 2],
            "acf_values": [0.1, -0.05],
        },
    )

    assert fake_st.subheaders == ["Residual Diagnostics"]
    assert fake_st.columns_created[2].metrics == [("Ljung-Box p", "0.4200")]
    assert fake_st.plotly_calls == 2
    assert "Diagnostics drill-downs:" in fake_st.captions
    assert "View forecasts for AutoETS" in fake_st.markdowns[0]["text"]


def test_render_data_overview_panel_outputs_chart_blocks() -> None:
    fake_st = _FakeStreamlit()

    _render_data_overview_panel(
        fake_st,
        {
            "fold": 2,
            "insights": [
                "No missing values were detected in the input data.",
            ],
            "series": [
                {
                    "name": "s1",
                    "summary": (
                        "s1 shows 2 recent training points and 2 holdout "
                        "observations."
                    ),
                    "ds_history": ["2020-06-30", "2020-07-01"],
                    "y_history": [9.0, 9.5],
                    "ds_actual": ["2020-07-02", "2020-07-03"],
                    "y_actual": [10.5, 10.8],
                }
            ],
        },
    )

    assert fake_st.subheaders == ["Data Overview"]
    assert "No missing values were detected in the input data." in fake_st.captions
    assert fake_st.plotly_calls == 1


def test_render_per_series_panel_outputs_charts_and_table() -> None:
    fake_st = _FakeStreamlit()
    fake_st.query_params = {
        "per_series_models": '["AutoETS"]',
        "per_series_series": '["s2"]',
    }

    _render_per_series_panel(
        fake_st,
        {
            "series_total": 2,
            "models": ["AutoETS", "SeasonalNaive"],
            "winner_summary": [
                {"name": "AutoETS", "count": 2},
            ],
            "heatmap_series": ["s2", "s1"],
            "heatmap_z": [[0.975, 1.06], [0.775, 0.94]],
            "heatmap_text": [["0.9750", "1.0600"], ["0.7750", "0.9400"]],
            "table_rows": [
                {
                    "series_id": "s2",
                    "winner": "AutoETS",
                    "winner_mase": 0.975,
                    "runner_up": "SeasonalNaive",
                    "runner_up_mase": 1.06,
                    "margin": 0.085,
                    "spread": 0.085,
                }
            ],
            "insights": [
                "AutoETS wins 2 of 2 comparable series.",
            ],
        },
    )

    assert fake_st.subheaders == ["Per-Series Winners"]
    assert fake_st.plotly_calls == 2
    assert fake_st.dataframes
    assert "Hardest-series drill-downs:" in fake_st.captions
    assert "Open forecasts for hardest series s2" in fake_st.markdowns[0]["text"]
    assert fake_st.query_params["per_series_models"] == '["AutoETS"]'
    assert fake_st.query_params["per_series_series"] == '["s2"]'


def test_filter_per_series_chart_data_limits_models_and_series() -> None:
    filtered = _filter_per_series_chart_data(
        {
            "series_total": 2,
            "models": ["AutoETS", "SeasonalNaive"],
            "winner_summary": [
                {"name": "AutoETS", "count": 2},
                {"name": "SeasonalNaive", "count": 0},
            ],
            "heatmap_series": ["s2", "s1"],
            "heatmap_z": [[0.975, 1.06], [0.775, 0.94]],
            "heatmap_text": [["0.9750", "1.0600"], ["0.7750", "0.9400"]],
            "table_rows": [
                {
                    "series_id": "s2",
                    "winner": "AutoETS",
                    "winner_mase": 0.975,
                    "runner_up": "SeasonalNaive",
                    "runner_up_mase": 1.06,
                    "margin": 0.085,
                    "spread": 0.085,
                    "scores": {"AutoETS": 0.975, "SeasonalNaive": 1.06},
                }
            ],
            "insights": [],
        },
        selected_model_names=["AutoETS"],
        selected_series=["s2"],
    )

    assert filtered["models"] == ["AutoETS"]
    assert filtered["heatmap_series"] == ["s2"]
    assert filtered["heatmap_z"] == [[0.975]]
    assert filtered["table_rows"][0]["scores"] == {"AutoETS": 0.975}
