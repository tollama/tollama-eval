"""Tests for PDF export module."""

from __future__ import annotations

import sys
from types import ModuleType

from ts_autopilot.reporting.pdf_export import _inject_static_charts, is_available


def test_is_available_returns_bool():
    result = is_available()
    assert isinstance(result, bool)


def test_inject_static_charts_covers_new_report_sections(monkeypatch):
    class FakeFigure:
        def __init__(self, data=None):
            self.data = []
            if data is not None:
                self.data.append(data)
            self.layout = {}

        def update_layout(self, **kwargs):
            self.layout.update(kwargs)

        def add_vline(self, **kwargs):
            self.layout.setdefault("vlines", []).append(kwargs)

        def add_trace(self, trace):
            self.data.append(trace)

    def _trace(kind, **kwargs):
        return {"kind": kind, **kwargs}

    fake_go = ModuleType("plotly.graph_objects")
    fake_go.Figure = FakeFigure
    fake_go.Bar = lambda **kwargs: _trace("bar", **kwargs)
    fake_go.Heatmap = lambda **kwargs: _trace("heatmap", **kwargs)
    fake_go.Scatter = lambda **kwargs: _trace("scatter", **kwargs)

    fake_io = ModuleType("plotly.io")
    fake_io.to_image = lambda fig, format, scale: b"png-bytes"

    fake_plotly = ModuleType("plotly")
    fake_plotly.graph_objects = fake_go
    fake_plotly.io = fake_io

    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)
    monkeypatch.setitem(sys.modules, "plotly.io", fake_io)
    monkeypatch.setattr(
        "ts_autopilot.reporting.pdf_export._kaleido_available",
        lambda: True,
    )

    html = """
    <html><body>
    <div id="chart-mase"></div>
    <div id="chart-heatmap"></div>
    <div id="chart-per-series-wins"></div>
    <div id="chart-per-series-heatmap"></div>
    <div id="chart-data-overview-1"></div>
    <div id="chart-forecast-1-1"></div>
    <script>
    var chartData = {"bar_names":["A"],"bar_mase":[0.8],"bar_colors":["#2563eb"],
    "heatmap_z":[[0.8]],"heatmap_folds":["Fold 1"],"heatmap_models":["A"],
    "per_series":{"winner_summary":[{"name":"A","count":2}],"models":["A","B"],
    "heatmap_series":["s2","s1"],"heatmap_z":[[0.9,1.1],[0.7,0.8]]},
    "data_overview":{"series":[{"name":"s1","chart_id":"chart-data-overview-1",
    "ds_history":["2020-01-01"],"y_history":[1.0],"ds_actual":["2020-01-02"],
    "y_actual":[1.1],"mase":0.8}]},
    "forecast":{"models":[{"name":"A","series":[{"name":"s1","chart_id":"chart-forecast-1-1",
    "ds_history":["2020-01-01"],"y_history":[1.0],"ds_actual":["2020-01-02"],
    "y_actual":[1.1],"ds_forecast":["2020-01-02"],"y_hat":[1.05],"mase":0.8}]}]}};
    var fontFamily = 'sans-serif';
    </script>
    </body></html>
    """

    rendered = _inject_static_charts(html)

    assert rendered.count("static-chart-img") >= 6
    assert '<div id="chart-per-series-wins"></div>' in rendered
    assert "Per-series winner counts" in rendered
    assert "Data overview for s1" in rendered
    assert "Forecast chart for A on s1" in rendered
