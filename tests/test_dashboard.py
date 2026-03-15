"""Tests for dashboard benchmark environment provenance."""

import json

from ts_autopilot.contracts import (
    BenchmarkConfig,
    BenchmarkResult,
    DataProfile,
    ForecastData,
    LeaderboardEntry,
    ModelResult,
)
from ts_autopilot.reporting.dashboard import (
    _load_result_artifacts,
    _optional_model_environment_summary,
    _render_optional_model_environment,
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
        self.metrics: list[tuple[str, int]] = []

    def metric(self, label: str, value: int) -> None:
        self.metrics.append((label, value))


class _FakeStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.captions: list[str] = []
        self.dataframes: list[object] = []
        self.columns_created: list[_FakeColumn] = []

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def columns(self, count: int) -> list[_FakeColumn]:
        self.columns_created = [_FakeColumn() for _ in range(count)]
        return self.columns_created

    def dataframe(self, data: object, **_: object) -> None:
        self.dataframes.append(data)


class _FakeUpload:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

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
