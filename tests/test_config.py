"""Tests for YAML/JSON config file support."""

from __future__ import annotations

import json

import pytest

from ts_autopilot.config import FileConfig, load_config


def test_load_yaml_config(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("input: data.csv\nhorizon: 7\nn_folds: 2\n")
    cfg = load_config(cfg_path)
    assert cfg.input == "data.csv"
    assert cfg.horizon == 7
    assert cfg.n_folds == 2


def test_load_json_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"input": "data.csv", "horizon": 7}))
    cfg = load_config(cfg_path)
    assert cfg.input == "data.csv"
    assert cfg.horizon == 7


def test_load_yaml_with_models_list(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("models:\n  - SeasonalNaive\n  - AutoETS\n")
    cfg = load_config(cfg_path)
    assert cfg.models == ["SeasonalNaive", "AutoETS"]


def test_load_yaml_with_models_string(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("models: SeasonalNaive,AutoETS\n")
    cfg = load_config(cfg_path)
    assert cfg.models == ["SeasonalNaive", "AutoETS"]


def test_unknown_key_raises(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("input: data.csv\nbogus_key: true\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(cfg_path)


def test_invalid_horizon_raises(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("horizon: -5\n")
    with pytest.raises(ValueError, match="positive integer"):
        load_config(cfg_path)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yml")


def test_unsupported_format(tmp_path):
    cfg_path = tmp_path / "config.txt"
    cfg_path.write_text("input: data.csv\n")
    with pytest.raises(ValueError, match="Unsupported config format"):
        load_config(cfg_path)


def test_empty_yaml_returns_defaults(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("")
    cfg = load_config(cfg_path)
    assert cfg == FileConfig()


def test_tollama_url_in_config(tmp_path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("tollama_url: http://localhost:8000\n")
    cfg = load_config(cfg_path)
    assert cfg.tollama_url == "http://localhost:8000"


def test_cli_config_flag(tmp_path):
    """Integration: --config loads settings and runs benchmark."""
    import pandas as pd
    from typer.testing import CliRunner

    from ts_autopilot.cli import app

    # Create CSV
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": str(d.date()), "y": float(i)})
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Create config
    cfg_path = tmp_path / "config.yml"
    out_dir = tmp_path / "results"
    cfg_path.write_text(
        f"input: {csv_path}\n"
        f"output: {out_dir}\n"
        "horizon: 7\n"
        "n_folds: 2\n"
        "models:\n"
        "  - SeasonalNaive\n"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert (out_dir / "results.json").exists()


def test_cli_flags_override_config(tmp_path):
    """CLI flags take precedence over config file values."""
    import pandas as pd
    from typer.testing import CliRunner

    from ts_autopilot.cli import app

    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for uid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": str(d.date()), "y": float(i)})
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    cfg_path = tmp_path / "config.yml"
    out_dir = tmp_path / "results"
    cfg_path.write_text(
        f"input: {csv_path}\n"
        f"output: {out_dir}\n"
        "horizon: 14\n"
        "n_folds: 3\n"
    )

    # Override horizon and n_folds via CLI
    runner = CliRunner()
    cli_out_dir = tmp_path / "cli_out"
    result = runner.invoke(
        app,
        [
            "run",
            "--config", str(cfg_path),
            "-H", "7",
            "-k", "2",
            "-o", str(cli_out_dir),
            "-m", "SeasonalNaive",
        ],
    )
    assert result.exit_code == 0
    assert (cli_out_dir / "results.json").exists()


def test_cli_config_bad_file(tmp_path):
    """Bad config file produces friendly error."""
    from typer.testing import CliRunner

    from ts_autopilot.cli import app

    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("bad_key: true\n")

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--config", str(cfg_path)])
    assert result.exit_code != 0
    assert "Config error" in result.output
