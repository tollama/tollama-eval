"""Tests for statistical model runners."""

from ts_autopilot.runners.statistical import AutoETSRunner, SeasonalNaiveRunner


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
