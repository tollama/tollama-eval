"""Tests for statistical model runners."""

from ts_autopilot.runners.statistical import (
    AutoARIMARunner,
    AutoETSRunner,
    AutoThetaRunner,
    SeasonalNaiveRunner,
)


def test_seasonal_naive_row_count(tiny_long_df):
    runner = SeasonalNaiveRunner()
    result = runner.fit_predict(tiny_long_df, horizon=14, freq="D", season_length=7)
    n_series = tiny_long_df["unique_id"].nunique()
    assert len(result.y_hat) == n_series * 14


def test_auto_ets_row_count(tiny_long_df):
    runner = AutoETSRunner()
    result = runner.fit_predict(tiny_long_df, horizon=14, freq="D", season_length=7)
    n_series = tiny_long_df["unique_id"].nunique()
    assert len(result.y_hat) == n_series * 14


def test_forecast_output_has_runtime(tiny_long_df):
    runner = SeasonalNaiveRunner()
    result = runner.fit_predict(tiny_long_df, horizon=7, freq="D", season_length=7)
    assert result.runtime_sec > 0


def test_forecast_output_unique_ids_match_input(tiny_long_df):
    runner = SeasonalNaiveRunner()
    result = runner.fit_predict(tiny_long_df, horizon=7, freq="D", season_length=7)
    input_uids = set(tiny_long_df["unique_id"].unique())
    output_uids = set(result.unique_id)
    assert input_uids == output_uids


def test_forecast_output_model_name(tiny_long_df):
    runner = AutoETSRunner()
    result = runner.fit_predict(tiny_long_df, horizon=7, freq="D", season_length=7)
    assert result.model_name == "AutoETS"


def test_runner_inherits_stats_forecast_runner():
    from ts_autopilot.runners.base import StatsForecastRunner

    assert isinstance(SeasonalNaiveRunner(), StatsForecastRunner)
    assert isinstance(AutoETSRunner(), StatsForecastRunner)


def test_make_model_returns_correct_type(tiny_long_df):
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

    sn = SeasonalNaiveRunner()
    assert isinstance(sn._make_model(7), SeasonalNaive)
    ets = AutoETSRunner()
    assert isinstance(ets._make_model(7), AutoETS)
    arima = AutoARIMARunner()
    assert isinstance(arima._make_model(7), AutoARIMA)
    theta = AutoThetaRunner()
    assert isinstance(theta._make_model(7), AutoTheta)


def test_auto_arima_row_count(tiny_long_df):
    runner = AutoARIMARunner()
    result = runner.fit_predict(tiny_long_df, horizon=14, freq="D", season_length=7)
    n_series = tiny_long_df["unique_id"].nunique()
    assert len(result.y_hat) == n_series * 14


def test_auto_arima_model_name(tiny_long_df):
    runner = AutoARIMARunner()
    result = runner.fit_predict(tiny_long_df, horizon=7, freq="D", season_length=7)
    assert result.model_name == "AutoARIMA"


def test_auto_theta_row_count(tiny_long_df):
    runner = AutoThetaRunner()
    result = runner.fit_predict(tiny_long_df, horizon=14, freq="D", season_length=7)
    n_series = tiny_long_df["unique_id"].nunique()
    assert len(result.y_hat) == n_series * 14


def test_auto_theta_model_name(tiny_long_df):
    runner = AutoThetaRunner()
    result = runner.fit_predict(tiny_long_df, horizon=7, freq="D", season_length=7)
    assert result.model_name == "AutoTheta"


def test_runner_inherits_stats_forecast_runner_all():
    from ts_autopilot.runners.base import StatsForecastRunner

    assert isinstance(AutoARIMARunner(), StatsForecastRunner)
    assert isinstance(AutoThetaRunner(), StatsForecastRunner)
