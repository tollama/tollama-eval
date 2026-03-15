"""Shared chart and interpretation builders for report and dashboard views."""

from __future__ import annotations

from ts_autopilot.contracts import BenchmarkResult


def build_series_score_lookup(result: BenchmarkResult) -> dict[str, dict[str, float]]:
    """Average per-series MASE across folds for each model."""
    lookup: dict[str, dict[str, float]] = {}
    for model in result.models:
        totals: dict[str, list[float]] = {}
        for fold in model.folds:
            for sid, score in fold.series_scores.items():
                totals.setdefault(sid, []).append(score)
        if totals:
            lookup[model.name] = {
                sid: round(sum(scores) / len(scores), 4)
                for sid, scores in totals.items()
            }
    return lookup


def build_per_series_competition_data(result: BenchmarkResult) -> dict:
    """Build per-series winner and difficulty analysis."""
    scores_by_model = build_series_score_lookup(result)
    if len(scores_by_model) < 2:
        return {}

    ordered_models = [
        entry.name for entry in result.leaderboard if entry.name in scores_by_model
    ]
    for model in result.models:
        if model.name in scores_by_model and model.name not in ordered_models:
            ordered_models.append(model.name)

    all_series = sorted(
        sid for model_scores in scores_by_model.values() for sid in model_scores
    )

    rows = []
    winner_counts = {model_name: 0 for model_name in ordered_models}

    for sid in all_series:
        comparisons = [
            (model_name, model_scores[sid])
            for model_name, model_scores in scores_by_model.items()
            if sid in model_scores
        ]
        if len(comparisons) < 2:
            continue

        comparisons.sort(key=lambda item: item[1])
        winner_name, winner_score = comparisons[0]
        runner_name, runner_score = comparisons[1]
        margin = round(runner_score - winner_score, 4)
        spread = round(comparisons[-1][1] - winner_score, 4)
        winner_counts[winner_name] = winner_counts.get(winner_name, 0) + 1
        row_scores = {
            model_name: scores_by_model.get(model_name, {}).get(sid)
            for model_name in ordered_models
        }
        rows.append(
            {
                "series_id": sid,
                "winner": winner_name,
                "winner_mase": round(winner_score, 4),
                "runner_up": runner_name,
                "runner_up_mase": round(runner_score, 4),
                "margin": margin,
                "spread": spread,
                "scores": row_scores,
            }
        )

    if not rows:
        return {}

    hardest_rows = sorted(
        rows,
        key=lambda item: (-item["winner_mase"], item["margin"], item["series_id"]),
    )
    closest_rows = sorted(
        rows,
        key=lambda item: (item["margin"], item["winner_mase"], item["series_id"]),
    )

    display_rows = hardest_rows[: min(15, len(hardest_rows))]
    heatmap_rows = hardest_rows[: min(25, len(hardest_rows))]
    heatmap_z = []
    heatmap_text = []
    heatmap_labels = []
    for row in heatmap_rows:
        heatmap_labels.append(row["series_id"])
        z_row: list[float | None] = []
        text_row: list[str] = []
        for model_name in ordered_models:
            score = row["scores"].get(model_name)
            z_row.append(score)
            text_row.append("" if score is None else f"{score:.4f}")
        heatmap_z.append(z_row)
        heatmap_text.append(text_row)

    winner_summary = [
        {"name": model_name, "count": count}
        for model_name, count in sorted(
            winner_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count > 0
    ]

    insights = build_per_series_insights(
        rows=rows,
        winner_summary=winner_summary,
        closest_rows=closest_rows,
        hardest_rows=hardest_rows,
    )

    return {
        "models": ordered_models,
        "winner_summary": winner_summary,
        "series_total": len(rows),
        "displayed_total": len(heatmap_rows),
        "heatmap_series": heatmap_labels,
        "heatmap_z": heatmap_z,
        "heatmap_text": heatmap_text,
        "table_rows": display_rows,
        "insights": insights,
    }


def build_per_series_insights(
    rows: list[dict],
    winner_summary: list[dict],
    closest_rows: list[dict],
    hardest_rows: list[dict],
) -> list[str]:
    """Generate short interpretation notes for per-series winners."""
    insights: list[str] = []

    if winner_summary:
        top = winner_summary[0]
        insights.append(
            f"{top['name']} wins {top['count']} of {len(rows)} comparable series."
        )

    if closest_rows:
        closest = closest_rows[0]
        insights.append(
            f"The closest race is {closest['series_id']}: "
            f"{closest['winner']} beats {closest['runner_up']} by only "
            f"{closest['margin']:.4f} MASE."
        )

    if hardest_rows:
        hardest = hardest_rows[0]
        if hardest["winner_mase"] < 1.0:
            insights.append(
                f"The hardest highlighted series is {hardest['series_id']} "
                f"(best MASE {hardest['winner_mase']:.4f}); "
                "it is still beatable, but model spread is large."
            )
        else:
            insights.append(
                f"{hardest['series_id']} remains difficult for every model: "
                f"the best available MASE is {hardest['winner_mase']:.4f}."
            )

    return insights[:3]


def get_reference_forecast_data(result: BenchmarkResult):
    """Pick one forecast payload to represent the evaluation window."""
    if not result.forecast_data:
        return None

    last_fold = max(fd.fold for fd in result.forecast_data)
    best_model = result.leaderboard[0].name if result.leaderboard else None
    if best_model is not None:
        for fd in result.forecast_data:
            if fd.model_name == best_model and fd.fold == last_fold:
                return fd

    for fd in result.forecast_data:
        if fd.fold == last_fold:
            return fd
    return result.forecast_data[0]


def group_forecast_series(fd) -> dict[str, dict]:
    """Group train/evaluation points by unique_id for chart rendering."""
    series: dict[str, dict] = {}

    for i, uid in enumerate(fd.unique_id):
        if uid not in series:
            series[uid] = {
                "ds_history": [],
                "y_history": [],
                "ds_actual": [],
                "y_actual": [],
                "ds_forecast": [],
                "y_hat": [],
            }
        if i < len(fd.ds):
            series[uid]["ds_actual"].append(fd.ds[i])
            series[uid]["ds_forecast"].append(fd.ds[i])
        if i < len(fd.y_actual):
            series[uid]["y_actual"].append(fd.y_actual[i])
        if i < len(fd.y_hat):
            series[uid]["y_hat"].append(fd.y_hat[i])

    train_ids = fd.train_unique_id
    if not train_ids and len(set(fd.unique_id)) == 1:
        train_ids = [fd.unique_id[0]] * len(fd.ds_train_tail)

    for i, uid in enumerate(train_ids):
        if uid not in series:
            series[uid] = {
                "ds_history": [],
                "y_history": [],
                "ds_actual": [],
                "y_actual": [],
                "ds_forecast": [],
                "y_hat": [],
            }
        if i < len(fd.ds_train_tail):
            series[uid]["ds_history"].append(fd.ds_train_tail[i])
        if i < len(fd.y_train_tail):
            series[uid]["y_history"].append(fd.y_train_tail[i])

    return series


def select_representative_series(
    result: BenchmarkResult,
    available_series: list[str],
    scores_by_model: dict[str, dict[str, float]],
    max_series: int = 3,
) -> list[str]:
    """Pick best, median, and worst series to keep charts focused."""
    if not available_series:
        return []

    best_model = result.leaderboard[0].name if result.leaderboard else None
    score_map = scores_by_model.get(best_model or "", {})
    ranked = [(sid, score_map[sid]) for sid in available_series if sid in score_map]
    ranked.sort(key=lambda item: item[1])

    selected: list[str] = []
    if ranked:
        candidate_positions = [0, len(ranked) // 2, len(ranked) - 1]
        for pos in candidate_positions:
            sid = ranked[pos][0]
            if sid not in selected:
                selected.append(sid)
        for sid, _ in ranked:
            if len(selected) >= min(max_series, len(available_series)):
                break
            if sid not in selected:
                selected.append(sid)

    if len(selected) < min(max_series, len(available_series)):
        for sid in sorted(available_series):
            if sid not in selected:
                selected.append(sid)
            if len(selected) >= min(max_series, len(available_series)):
                break

    return selected


def build_data_overview_insights(result: BenchmarkResult) -> list[str]:
    """Generate short chart-adjacent data interpretation notes."""
    insights: list[str] = []
    profile = result.profile
    data_chars = result.data_characteristics

    if profile.missing_ratio == 0:
        insights.append("No missing values were detected in the input data.")
    elif profile.missing_ratio < 0.05:
        insights.append(
            f"Missingness is low at {profile.missing_ratio:.1%}, "
            "so ranking noise from gaps should be limited."
        )
    else:
        insights.append(
            f"Missingness is {profile.missing_ratio:.1%}; "
            "treat model differences cautiously on sparse regions."
        )

    if data_chars is not None:
        if data_chars.seasonality_strength >= 0.5:
            insights.append(
                f"Seasonality is strong "
                f"({data_chars.seasonality_strength:.2f}), "
                "so seasonal models should have an advantage."
            )
        elif data_chars.trend_strength >= 0.6:
            insights.append(
                f"Trend strength is elevated ({data_chars.trend_strength:.2f}), "
                "so trend-tracking behavior matters more than seasonal fit."
            )

        if data_chars.series_heterogeneity >= 0.7:
            insights.append(
                f"Series heterogeneity is high "
                f"({data_chars.series_heterogeneity:.2f}), "
                "which increases the chance that one global winner "
                "hides per-series losers."
            )

    if profile.min_length < profile.season_length_guess * 2:
        insights.append(
            "The shortest series are relatively short for the "
            "inferred seasonality, which can reduce forecast stability."
        )

    return insights[:3]


def build_data_overview_chart_data(result: BenchmarkResult) -> dict:
    """Build recent-history charts to explain the benchmarked data window."""
    reference_fd = get_reference_forecast_data(result)
    if reference_fd is None:
        return {"series": [], "insights": []}

    scores_by_model = build_series_score_lookup(result)
    grouped = group_forecast_series(reference_fd)
    selected = select_representative_series(
        result=result,
        available_series=list(grouped.keys()),
        scores_by_model=scores_by_model,
    )

    best_model = result.leaderboard[0].name if result.leaderboard else ""
    best_scores = scores_by_model.get(best_model, {})
    series_payload = []
    for idx, sid in enumerate(selected, start=1):
        payload = grouped[sid]
        score = best_scores.get(sid)
        series_payload.append(
            {
                "name": sid,
                "chart_id": f"chart-data-overview-{idx}",
                "ds_history": payload["ds_history"],
                "y_history": payload["y_history"],
                "ds_actual": payload["ds_actual"],
                "y_actual": payload["y_actual"],
                "mase": score,
                "summary": describe_series_snapshot(
                    sid=sid,
                    mase=score,
                    history_points=len(payload["y_history"]),
                    horizon_points=len(payload["y_actual"]),
                ),
            }
        )

    return {
        "series": series_payload,
        "fold": reference_fd.fold,
        "reference_model": reference_fd.model_name,
        "insights": build_data_overview_insights(result),
    }


def describe_series_snapshot(
    sid: str,
    mase: float | None,
    history_points: int,
    horizon_points: int,
) -> str:
    """Describe the data window shown for one representative series."""
    parts = [
        f"{sid} shows {history_points} recent training points "
        f"and {horizon_points} holdout observations."
    ]
    if mase is not None:
        if mase < 1.0:
            parts.append(
                f"The winning model beats naive on this series "
                f"(MASE {mase:.4f})."
            )
        else:
            parts.append(
                "This series remains difficult for the winning model "
                f"(MASE {mase:.4f})."
            )
    return " ".join(parts)


def describe_model_forecast(
    model_name: str,
    model_score: float,
    leader_name: str | None,
    leader_score: float | None,
    selected_scores: list[float],
) -> str:
    """Summarize what a model's forecast charts mean."""
    parts = [f"Mean MASE is {model_score:.4f}."]
    if model_score < 1.0:
        parts.append(
            f"That is {(1.0 - model_score) * 100:.1f}% better "
            "than the naive baseline."
        )
    elif model_score > 1.0:
        parts.append(
            f"That is {(model_score - 1.0) * 100:.1f}% worse "
            "than the naive baseline."
        )
    else:
        parts.append("That matches the naive baseline.")

    if leader_name and leader_score is not None and model_name != leader_name:
        parts.append(
            f"It trails {leader_name} "
            f"by {model_score - leader_score:.4f} MASE."
        )

    if selected_scores:
        beats_naive = sum(score < 1.0 for score in selected_scores)
        parts.append(
            f"Among the highlighted series, "
            f"{beats_naive}/{len(selected_scores)} beat the naive baseline."
        )

    return " ".join(parts)


def build_forecast_chart_data(result: BenchmarkResult) -> dict:
    """Build forecast vs actual chart data for every model on the last fold."""
    if not result.forecast_data:
        return {"models": [], "selected_series": []}

    last_fold = max(fd.fold for fd in result.forecast_data)
    scores_by_model = build_series_score_lookup(result)

    reference_fd = get_reference_forecast_data(result)
    if reference_fd is None:
        return {"models": [], "selected_series": []}

    selected_series = select_representative_series(
        result=result,
        available_series=list(group_forecast_series(reference_fd).keys()),
        scores_by_model=scores_by_model,
    )

    leaderboard_map = {entry.name: entry.mean_mase for entry in result.leaderboard}
    leader_name = result.leaderboard[0].name if result.leaderboard else None
    leader_score = result.leaderboard[0].mean_mase if result.leaderboard else None

    models_payload = []
    for model_idx, model in enumerate(result.models, start=1):
        fd = next(
            (
                item
                for item in result.forecast_data
                if item.model_name == model.name and item.fold == last_fold
            ),
            None,
        )
        if fd is None:
            continue

        grouped = group_forecast_series(fd)
        per_series_scores = scores_by_model.get(model.name, {})
        series_payload = []
        selected_scores: list[float] = []
        for series_idx, sid in enumerate(selected_series, start=1):
            if sid not in grouped:
                continue
            payload = grouped[sid]
            score = per_series_scores.get(sid)
            if score is not None:
                selected_scores.append(score)
            series_payload.append(
                {
                    "name": sid,
                    "chart_id": f"chart-forecast-{model_idx}-{series_idx}",
                    "mase": score,
                    "note": describe_series_forecast(score),
                    "ds_history": payload["ds_history"],
                    "y_history": payload["y_history"],
                    "ds_actual": payload["ds_actual"],
                    "y_actual": payload["y_actual"],
                    "ds_forecast": payload["ds_forecast"],
                    "y_hat": payload["y_hat"],
                }
            )

        if not series_payload:
            continue

        models_payload.append(
            {
                "name": model.name,
                "rank": next(
                    (
                        entry.rank
                        for entry in result.leaderboard
                        if entry.name == model.name
                    ),
                    None,
                ),
                "mean_mase": leaderboard_map.get(model.name, model.mean_mase),
                "summary": describe_model_forecast(
                    model_name=model.name,
                    model_score=leaderboard_map.get(model.name, model.mean_mase),
                    leader_name=leader_name,
                    leader_score=leader_score,
                    selected_scores=selected_scores,
                ),
                "series": series_payload,
            }
        )

    return {
        "models": models_payload,
        "selected_series": selected_series,
        "fold": last_fold,
    }


def describe_series_forecast(mase: float | None) -> str:
    """Explain one per-series forecast panel in plain language."""
    if mase is None:
        return "Per-series MASE is unavailable for this panel."
    if mase < 1.0:
        return (
            f"MASE {mase:.4f}; this model beats the naive baseline "
            "on the highlighted series."
        )
    if mase > 1.0:
        return (
            f"MASE {mase:.4f}; this model underperforms the naive baseline "
            "on the highlighted series."
        )
    return (
        "MASE 1.0000; this model matches the naive baseline "
        "on the highlighted series."
    )


def build_diagnostics_chart_data(result: BenchmarkResult) -> dict:
    """Build diagnostics chart data for the best model."""
    if not result.diagnostics or not result.leaderboard:
        return {}

    best_model = result.leaderboard[0].name
    diag = next(
        (d for d in result.diagnostics if d.model_name == best_model),
        None,
    )

    if not diag:
        return {}

    return {
        "model_name": diag.model_name,
        "residual_mean": diag.residual_mean,
        "residual_std": diag.residual_std,
        "residual_skew": diag.residual_skew,
        "residual_kurtosis": diag.residual_kurtosis,
        "ljung_box_p": diag.ljung_box_p,
        "histogram_bins": diag.histogram_bins,
        "histogram_counts": diag.histogram_counts,
        "acf_lags": diag.acf_lags,
        "acf_values": diag.acf_values,
        "residuals": diag.residuals,
        "fitted": diag.fitted,
    }
