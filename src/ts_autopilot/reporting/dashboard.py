"""Interactive Streamlit dashboard for ts-autopilot.

Launch with:
    streamlit run -m ts_autopilot.reporting.dashboard -- --input data.csv

Or from code:
    python -m ts_autopilot.reporting.dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path


def _check_streamlit() -> bool:
    try:
        import streamlit  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    if not _check_streamlit():
        print(
            "Streamlit is required for the dashboard. "
            'Install with: pip install "ts-autopilot[dashboard]"',
            file=sys.stderr,
        )
        sys.exit(1)

    import streamlit as st

    st.set_page_config(
        page_title="ts-autopilot Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("ts-autopilot Interactive Dashboard")
    st.markdown("Upload a CSV to run benchmarks interactively.")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"], help="Long or wide format time series CSV"
        )
        horizon = st.number_input("Forecast Horizon", min_value=1, value=14)
        n_folds = st.number_input("CV Folds", min_value=1, value=3)

        available_models = [
            "SeasonalNaive",
            "AutoETS",
            "AutoARIMA",
            "AutoTheta",
            "CES",
        ]
        selected_models = st.multiselect(
            "Models",
            available_models,
            default=["SeasonalNaive", "AutoETS"],
        )

        run_button = st.button(
            "Run Benchmark", type="primary", use_container_width=True
        )

    if uploaded_file is None:
        st.info("Upload a CSV file to get started.")
        _show_sample_format()
        return

    if not run_button:
        # Show data preview
        import pandas as pd

        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"{len(df):,} rows, {len(df.columns)} columns")
        return

    # Run benchmark
    _run_benchmark(uploaded_file, int(horizon), int(n_folds), selected_models)


def _show_sample_format() -> None:
    """Show expected CSV format."""
    import streamlit as st

    st.subheader("Expected CSV Format")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Long format** (recommended)")
        st.code(
            "unique_id,ds,y\n"
            "series_1,2020-01-01,10.5\n"
            "series_1,2020-01-02,11.2\n"
            "series_2,2020-01-01,20.0",
            language="csv",
        )
    with col2:
        st.markdown("**Wide format**")
        st.code(
            "date,series_1,series_2\n2020-01-01,10.5,20.0\n2020-01-02,11.2,21.5",
            language="csv",
        )


def _run_benchmark(
    uploaded_file: object,
    horizon: int,
    n_folds: int,
    model_names: list[str],
) -> None:
    """Execute benchmark and display results."""
    import tempfile

    import pandas as pd
    import streamlit as st

    from ts_autopilot.ingestion.loader import load_csv
    from ts_autopilot.ingestion.profiler import profile_dataframe
    from ts_autopilot.pipeline import run_benchmark
    from ts_autopilot.reporting.executive_summary import generate_executive_summary

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    progress_bar = st.progress(0, text="Loading data...")

    try:
        df = load_csv(tmp_path)
        profile = profile_dataframe(df)
        progress_bar.progress(10, text="Data loaded. Running benchmark...")

        # Progress tracking

        def progress_cb(step: str, current: int, total: int) -> None:
            if step == "model":
                pct = min(10 + int(80 * current / total), 90)
                progress_bar.progress(pct, text=f"Running model {current}/{total}...")

        result = run_benchmark(
            df,
            horizon=horizon,
            n_folds=n_folds,
            model_names=model_names if model_names else None,
            progress_callback=progress_cb,
        )
        progress_bar.progress(100, text="Benchmark complete!")

    except Exception as exc:
        st.error(f"Benchmark failed: {exc}")
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Display results
    st.divider()

    # Executive summary
    summary = generate_executive_summary(result)
    st.info(summary)

    # Data profile
    st.subheader("Dataset Profile")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Series", profile.n_series)
    col2.metric("Total Rows", f"{profile.total_rows:,}")
    col3.metric("Frequency", profile.frequency)
    col4.metric("Season Length", profile.season_length_guess)

    # Warnings
    if result.warnings:
        with st.expander(f"Warnings ({len(result.warnings)})", expanded=False):
            for w in result.warnings:
                st.warning(w)

    # Leaderboard
    st.subheader("Leaderboard")
    lb_data = []
    for entry in result.leaderboard:
        lb_data.append(
            {
                "Rank": entry.rank,
                "Model": entry.name,
                "MASE": round(entry.mean_mase, 4),
                "SMAPE": f"{entry.mean_smape:.2f}%",
                "RMSSE": round(entry.mean_rmsse, 4),
                "MAE": round(entry.mean_mae, 4),
            }
        )
    if lb_data:
        st.dataframe(pd.DataFrame(lb_data), use_container_width=True, hide_index=True)

    # Charts
    if result.leaderboard:
        st.subheader("Model Comparison")

        import plotly.graph_objects as go

        # MASE bar chart
        names = [e.name for e in result.leaderboard]
        mase_vals = [e.mean_mase for e in result.leaderboard]
        bar_colors = ["#2563eb" if v < 1.0 else "#dc2626" for v in mase_vals]

        fig = go.Figure(
            go.Bar(
                y=names[::-1],
                x=mase_vals[::-1],
                orientation="h",
                marker_color=bar_colors[::-1],
                text=[f"{v:.4f}" for v in mase_vals[::-1]],
                textposition="outside",
            )
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="#9ca3af")
        fig.update_layout(
            title="MASE Comparison",
            xaxis_title="MASE (lower is better)",
            height=max(250, len(names) * 45 + 100),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Fold stability
        if result.models and len(result.models[0].folds) > 1:
            fig2 = go.Figure()
            colors = [
                "#2563eb",
                "#dc2626",
                "#059669",
                "#d97706",
                "#7c3aed",
                "#db2777",
                "#0891b2",
                "#65a30d",
            ]
            for i, model in enumerate(result.models):
                fig2.add_trace(
                    go.Scatter(
                        x=[f"Fold {f.fold}" for f in model.folds],
                        y=[f.mase for f in model.folds],
                        name=model.name,
                        mode="lines+markers",
                        line={"color": colors[i % len(colors)]},
                    )
                )
            fig2.add_hline(y=1.0, line_dash="dash", line_color="#9ca3af")
            fig2.update_layout(title="MASE Stability Across Folds", height=350)
            st.plotly_chart(fig2, use_container_width=True)

    # Per-model details
    if result.models:
        st.subheader("Model Details")
        for model in result.models:
            with st.expander(f"{model.name} (MASE={model.mean_mase:.4f})"):
                fold_data = []
                for f in model.folds:
                    fold_data.append(
                        {
                            "Fold": f.fold,
                            "Cutoff": f.cutoff,
                            "MASE": round(f.mase, 4),
                            "SMAPE": f"{f.smape:.2f}%",
                            "RMSSE": round(f.rmsse, 4),
                            "MAE": round(f.mae, 4),
                        }
                    )
                if fold_data:
                    st.dataframe(
                        pd.DataFrame(fold_data),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.caption(f"Runtime: {model.runtime_sec:.2f}s")


if __name__ == "__main__":
    main()
