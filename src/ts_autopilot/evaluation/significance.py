"""Statistical significance testing for model comparison.

Implements the Friedman test with Nemenyi post-hoc analysis for
multi-model benchmarking, following Demsar (2006) methodology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from ts_autopilot.logging_config import get_logger

logger = get_logger("significance")

# Nemenyi critical values (q_alpha) for alpha=0.05
# Indexed by number of groups k (2..20).
# From: Demsar (2006), Table in Appendix.
_NEMENYI_Q_005: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.314,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}


@dataclass
class PairwiseComparison:
    """Result of a pairwise post-hoc comparison."""

    model_a: str
    model_b: str
    rank_diff: float
    critical_difference: float
    significant: bool  # |rank_diff| > CD at alpha=0.05


@dataclass
class SignificanceReport:
    """Result of the Friedman + Nemenyi significance analysis."""

    friedman_statistic: float
    friedman_p_value: float
    n_models: int
    n_series: int  # number of observation units (series)
    mean_ranks: dict[str, float]  # model_name -> mean rank (1 = best)
    pairwise: list[PairwiseComparison] = field(default_factory=list)
    critical_difference: float = 0.0


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    Uses scipy if available, otherwise Wilson-Hilferty approximation.
    """
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, df))
    except ImportError:
        pass

    if df <= 0 or x <= 0:
        return 1.0 if x <= 0 else 0.0
    # Wilson-Hilferty approximation
    z = ((x / df) ** (1.0 / 3) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
    # Standard normal CDF via error function
    p_cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return max(0.0, min(1.0, 1.0 - p_cdf))


def friedman_test(
    per_series_scores: dict[str, dict[str, float]],
    alpha: float = 0.05,
) -> SignificanceReport | None:
    """Friedman test + Nemenyi post-hoc for model comparison.

    Args:
        per_series_scores: {model_name: {series_id: mase_score}}.
            Each series is treated as an independent observation unit.
            Models are ranked per series (lower MASE = rank 1).
        alpha: Significance level (default 0.05).

    Returns:
        SignificanceReport or None if insufficient data (< 2 models or < 3 series).
    """
    model_names = sorted(per_series_scores.keys())
    k = len(model_names)
    if k < 2:
        return None

    # Find series that are present in ALL models
    all_series: set[str] = set()
    for scores in per_series_scores.values():
        if not all_series:
            all_series = set(scores.keys())
        else:
            all_series &= set(scores.keys())

    series_ids = sorted(all_series)
    n = len(series_ids)
    if n < 3:
        logger.info(
            "Need at least 3 common series for Friedman test, got %d", n
        )
        return None

    # Build score matrix: (n_series, n_models)
    score_matrix = np.zeros((n, k))
    for j, model in enumerate(model_names):
        for i, sid in enumerate(series_ids):
            score_matrix[i, j] = per_series_scores[model][sid]

    # Rank models within each series (lower score = rank 1)
    rank_matrix = np.zeros_like(score_matrix)
    for i in range(n):
        order = np.argsort(score_matrix[i, :])
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, k + 1, dtype=float)
        # Handle ties: average ranks
        for j in range(k):
            tied = score_matrix[i, :] == score_matrix[i, j]
            if np.sum(tied) > 1:
                ranks[tied] = np.mean(ranks[tied])
        rank_matrix[i, :] = ranks

    mean_ranks = rank_matrix.mean(axis=0)
    mean_rank_dict = {model_names[j]: float(mean_ranks[j]) for j in range(k)}

    # Friedman chi-squared statistic
    chi2_f = (12 * n / (k * (k + 1))) * np.sum(
        (mean_ranks - (k + 1) / 2) ** 2
    )
    p_value = _chi2_sf(float(chi2_f), k - 1)

    report = SignificanceReport(
        friedman_statistic=float(chi2_f),
        friedman_p_value=p_value,
        n_models=k,
        n_series=n,
        mean_ranks=mean_rank_dict,
    )

    # Nemenyi post-hoc (only if Friedman is significant)
    if p_value < alpha and k in _NEMENYI_Q_005:
        q_alpha = _NEMENYI_Q_005[k]
        cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
        report.critical_difference = cd

        pairwise: list[PairwiseComparison] = []
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(mean_ranks[i] - mean_ranks[j])
                pairwise.append(
                    PairwiseComparison(
                        model_a=model_names[i],
                        model_b=model_names[j],
                        rank_diff=float(diff),
                        critical_difference=cd,
                        significant=diff > cd,
                    )
                )
        report.pairwise = pairwise
    elif p_value < alpha and k > 20:
        logger.info(
            "Nemenyi post-hoc not available for k=%d models (max 20)", k
        )

    return report


def render_critical_difference_svg(
    mean_ranks: dict[str, float],
    cd: float,
    width: int = 600,
    height: int | None = None,
) -> str:
    """Render a Demsar (2006) critical difference diagram as inline SVG.

    Models whose rank difference <= CD are connected by a horizontal bar,
    indicating no statistically significant difference.

    Args:
        mean_ranks: {model_name: mean_rank} (rank 1 = best).
        cd: Critical difference value from Nemenyi test.
        width: SVG width in pixels.
        height: SVG height (auto-calculated if None).

    Returns:
        SVG string ready for inline HTML embedding.
    """
    if not mean_ranks or cd <= 0:
        return ""

    sorted_models = sorted(mean_ranks.items(), key=lambda x: x[1])
    k = len(sorted_models)

    # Layout constants
    margin_left = 20
    margin_right = 20
    margin_top = 40
    tick_height = 20
    model_spacing = 22
    clique_spacing = 12
    axis_y = margin_top + tick_height

    min_rank = 1.0
    max_rank = float(k)
    scale = (width - margin_left - margin_right) / max(max_rank - min_rank, 1)

    def rank_to_x(rank: float) -> float:
        return margin_left + (rank - min_rank) * scale

    # Find cliques (groups of models not significantly different)
    cliques: list[list[str]] = []
    for i in range(k):
        for j in range(i + 1, k):
            name_i, rank_i = sorted_models[i]
            name_j, rank_j = sorted_models[j]
            if abs(rank_j - rank_i) <= cd:
                # Check if this pair extends an existing clique
                merged = False
                for clique in cliques:
                    if name_i in clique or name_j in clique:
                        if name_i not in clique:
                            clique.append(name_i)
                        if name_j not in clique:
                            clique.append(name_j)
                        merged = True
                        break
                if not merged:
                    cliques.append([name_i, name_j])

    # Deduplicate cliques (merge overlapping)
    merged_cliques: list[set[str]] = []
    for clique_names in cliques:
        clique_set = set(clique_names)
        merged = False
        for mc in merged_cliques:
            if mc & clique_set:
                mc |= clique_set
                merged = True
                break
        if not merged:
            merged_cliques.append(clique_set)

    # Verify clique validity: all pairs within a clique must be within CD
    valid_cliques: list[set[str]] = []
    for clique_set in merged_cliques:
        clique_rank_pairs = [(name, mean_ranks[name]) for name in clique_set]
        clique_rank_pairs.sort(key=lambda x: x[1])
        if len(clique_rank_pairs) >= 2:
            min_r = clique_rank_pairs[0][1]
            max_r = clique_rank_pairs[-1][1]
            if max_r - min_r <= cd:
                valid_cliques.append(clique_set)
            else:
                # Split into valid sub-cliques
                for ci in range(len(clique_rank_pairs)):
                    for cj in range(ci + 1, len(clique_rank_pairs)):
                        sub = [clique_rank_pairs[x][0] for x in range(ci, cj + 1)]
                        sub_min = clique_rank_pairs[ci][1]
                        sub_max = clique_rank_pairs[cj][1]
                        if sub_max - sub_min <= cd and len(sub) >= 2:
                            sub_set = set(sub)
                            # Only add if not a subset of an existing clique
                            if not any(sub_set <= vc for vc in valid_cliques):
                                valid_cliques.append(sub_set)
        else:
            valid_cliques.append(clique_set)

    # Remove subset cliques
    final_cliques: list[set[str]] = []
    for vc in sorted(valid_cliques, key=len, reverse=True):
        if not any(vc < fc for fc in final_cliques):
            final_cliques.append(vc)

    n_cliques = len(final_cliques)

    # Split models into top half and bottom half for label placement
    top_models = sorted_models[: (k + 1) // 2]
    bottom_models = sorted_models[(k + 1) // 2 :]

    top_label_y = axis_y - 8
    bottom_label_start_y = axis_y + tick_height + n_cliques * clique_spacing + 10

    if height is None:
        height = int(
            bottom_label_start_y
            + len(bottom_models) * model_spacing
            + 20
        )

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; '
        f'font-size: 11px;" role="img" '
        f'aria-label="Critical difference diagram">'
    )

    # Background
    lines.append(f'<rect width="{width}" height="{height}" fill="transparent"/>')

    # CD indicator bar
    cd_px = cd * scale
    cd_x = margin_left
    cd_y = margin_top - 28
    lines.append(
        f'<line x1="{cd_x}" y1="{cd_y}" x2="{cd_x + cd_px}" y2="{cd_y}" '
        f'stroke="currentColor" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{cd_x}" y1="{cd_y - 4}" x2="{cd_x}" y2="{cd_y + 4}" '
        f'stroke="currentColor" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{cd_x + cd_px}" y1="{cd_y - 4}" x2="{cd_x + cd_px}" '
        f'y2="{cd_y + 4}" stroke="currentColor" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{cd_x + cd_px / 2}" y="{cd_y - 6}" '
        f'text-anchor="middle" fill="currentColor" font-size="10">'
        f'CD = {cd:.2f}</text>'
    )

    # Axis line
    lines.append(
        f'<line x1="{margin_left}" y1="{axis_y}" '
        f'x2="{width - margin_right}" y2="{axis_y}" '
        f'stroke="currentColor" stroke-width="1.5"/>'
    )

    # Tick marks and labels
    for tick_rank in range(1, k + 1):
        x = rank_to_x(float(tick_rank))
        lines.append(
            f'<line x1="{x}" y1="{axis_y - 4}" x2="{x}" y2="{axis_y + 4}" '
            f'stroke="currentColor" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{x}" y="{axis_y - 8}" text-anchor="middle" '
            f'fill="currentColor" font-size="10">{tick_rank}</text>'
        )

    # Model labels and connecting lines — top models
    for idx, (name, model_rank) in enumerate(top_models):
        x = rank_to_x(model_rank)
        label_y = top_label_y - (len(top_models) - idx) * model_spacing
        esc_name = xml_escape(name)
        lines.append(
            f'<line x1="{x}" y1="{axis_y - 4}" x2="{x}" y2="{label_y + 4}" '
            f'stroke="currentColor" stroke-width="0.8" stroke-dasharray="2,2"/>'
        )
        lines.append(
            f'<circle cx="{x}" cy="{axis_y}" r="3" fill="currentColor"/>'
        )
        lines.append(
            f'<text x="{x + 4}" y="{label_y}" fill="currentColor" '
            f'font-size="11" font-weight="{600 if idx == 0 else 400}">'
            f'{esc_name}</text>'
        )

    # Model labels — bottom models
    for idx, (name, model_rank) in enumerate(bottom_models):
        x = rank_to_x(model_rank)
        label_y = bottom_label_start_y + idx * model_spacing
        esc_name = xml_escape(name)
        lines.append(
            f'<line x1="{x}" y1="{axis_y + 4}" x2="{x}" y2="{label_y - 8}" '
            f'stroke="currentColor" stroke-width="0.8" stroke-dasharray="2,2"/>'
        )
        lines.append(
            f'<circle cx="{x}" cy="{axis_y}" r="3" fill="currentColor"/>'
        )
        lines.append(
            f'<text x="{x + 4}" y="{label_y}" fill="currentColor" '
            f'font-size="11">{esc_name}</text>'
        )

    # Clique bars (horizontal bars connecting non-significantly-different models)
    bar_colors = [
        "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
        "#db2777", "#0891b2", "#65a30d",
    ]
    for ci, clique_set in enumerate(final_cliques):
        clique_rank_values = sorted(mean_ranks[name] for name in clique_set)
        x1 = rank_to_x(clique_rank_values[0])
        x2 = rank_to_x(clique_rank_values[-1])
        bar_y = axis_y + tick_height + ci * clique_spacing
        color = bar_colors[ci % len(bar_colors)]
        lines.append(
            f'<line x1="{x1}" y1="{bar_y}" x2="{x2}" y2="{bar_y}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round" '
            f'opacity="0.7"/>'
        )

    lines.append("</svg>")
    return "\n".join(lines)
