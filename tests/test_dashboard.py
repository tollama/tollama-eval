"""Tests for dashboard benchmark environment provenance."""

import json

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    DiagnosticsResult,
    ForecastData,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.dashboard import (
    _artifact_manifest_summary,
    _artifact_source_label,
    _filter_forecast_chart_data,
    _filter_per_series_chart_data,
    _filter_result_for_dashboard,
    _load_result_artifacts,
    _optional_model_environment_summary,
    _parse_dashboard_args,
    _render_artifact_manifest,
    _render_diagnostics_panel,
    _render_forecast_panels,
    _render_optional_model_environment,
    _render_per_series_panel,
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
        self.subheaders: list[str] = []
        self.captions: list[str] = []
        self.dataframes: list[object] = []
        self.columns_created: list[_FakeColumn] = []
        self.expanders: list[str] = []
        self.plotly_calls: int = 0
        self.warnings: list[str] = []
        self.info_messages: list[object] = []

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def caption(self, text: str) -> None:
        self.captions.append(text)

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

    def info(self, text: object) -> None:
        self.info_messages.append(text)

    def divider(self) -> None:
        return None


class _FakeUpload:
    def __init__(self, payload: dict, name: str = "artifact.json") -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.name = name

    def getvalue(self) -> bytes:
        return self._payload


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


def test_artifact_source_label_uses_uploaded_name() -> None:
    label = _artifact_source_label(_FakeUpload({}, name="results.json"))

    assert label == "results.json"


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
    assert manifest["rows"][1]["Status"] == "Missing"


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
                    "Status": "Missing",
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


def test_render_forecast_panels_outputs_chart_blocks() -> None:
    fake_st = _FakeStreamlit()

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


def test_render_per_series_panel_outputs_charts_and_table() -> None:
    fake_st = _FakeStreamlit()

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
