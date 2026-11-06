#!/usr/bin/env python3
"""Generate the three figures used in the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "results/tables"

INK = "#1D2731"
MUTED = "#5C6770"
LIGHT = "#EDF2F4"
IRRIGATED = "#2C7FB8"
RAINFED = "#F2B134"
TEAL = "#2A8C99"
VERMILLION = "#D95F59"
REGIME = {
    "V0_random_day": "#5B3A8A",
    "V1_within_year": "#456A9C",
    "V2_same_year_cross": "#2A8C99",
    "V3_strict_cross": "#73A942",
}
REGIME_SHORT = {
    "V0_random_day": "V0\nrandom",
    "V1_within_year": "V1\nyear",
    "V2_same_year_cross": "V2\nsame-year",
    "V3_strict_cross": "V3\nstrict",
}
REGIMES = list(REGIME)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, fc="white", ec=INK, lw=1.0, size=7.2, bold=False, radius=0.018):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=INK, weight="bold" if bold else "normal", linespacing=1.15)
    return patch


def arrow(ax, p1, p2, color=MUTED, connection="arc3,rad=0", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
            color=color, connectionstyle=connection, shrinkA=2, shrinkB=2,
        )
    )


def figure1_design() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Section headers separate the four parts of the analysis workflow.
    headers = [
        (0.02, 0.18, "PAIRED DATA"),
        (0.225, 0.19, "PREDICTORS"),
        (0.445, 0.14, "MODELS"),
        (0.61, 0.36, "VALIDATION AUDIT"),
    ]
    for x, w, label in headers:
        ax.add_patch(Rectangle((x, 0.91), w, 0.07, color=TEAL, ec="none"))
        ax.text(x + w / 2, 0.945, label, color="white", weight="bold", ha="center", va="center", fontsize=7.2)

    box(ax, (0.03, 0.58), (0.16, 0.23), "US-Ne2\nIRRIGATED\n1,491 site-days", fc="#DCEEF8", ec=IRRIGATED, lw=1.2, bold=True)
    box(ax, (0.03, 0.25), (0.16, 0.23), "US-Ne3\nRAINFED\n1,507 site-days", fc="#FFF0CC", ec=RAINFED, lw=1.2, bold=True)
    ax.text(0.11, 0.525, "paired at 1.6 km • 2001–2013", ha="center", va="center", fontsize=6.5, color=MUTED)

    features = ["TA", "RH", "WS", "PA", "SW$_{in}$", "LW$_{in}$", "NDVI", "VPD", "sin/cos DOY"]
    for i, label in enumerate(features):
        row, col = divmod(i, 3)
        x = 0.235 + col * 0.061
        y = 0.72 - row * 0.16
        box(ax, (x, y), (0.052, 0.10), label, fc=LIGHT, ec="#B8C2CC", lw=0.7, size=6.2, radius=0.012)

    model_y = [0.70, 0.50, 0.30]
    model_names = ["15-day\nclimatology", "Ridge", "Random\nForest"]
    for y, label in zip(model_y, model_names):
        box(ax, (0.455, y), (0.12, 0.115), label, fc="white", ec=TEAL, lw=1.0, size=6.8, bold=label == "Random\nForest")

    arrow(ax, (0.19, 0.67), (0.228, 0.67))
    arrow(ax, (0.415, 0.53), (0.45, 0.53))
    arrow(ax, (0.575, 0.53), (0.61, 0.53))

    # Validation regimes arranged by whether the site and year were observed.
    x0, y0, cw, ch = 0.675, 0.30, 0.135, 0.22
    ax.text(x0 + cw, 0.84, "YEAR", ha="center", va="center", fontsize=7.2, weight="bold", color=INK)
    ax.text(x0 + cw / 2, 0.79, "seen", ha="center", fontsize=6.5, color=MUTED)
    ax.text(x0 + 1.5 * cw, 0.79, "unseen", ha="center", fontsize=6.5, color=MUTED)
    ax.text(0.63, y0 + ch, "SITE", rotation=90, ha="center", va="center", fontsize=7.2, weight="bold", color=INK)
    ax.text(0.655, y0 + 1.5 * ch, "seen", rotation=90, ha="center", va="center", fontsize=6.4, color=MUTED)
    ax.text(0.655, y0 + 0.5 * ch, "unseen", rotation=90, ha="center", va="center", fontsize=6.4, color=MUTED)
    cells = [
        (x0, y0 + ch, "V0\nrandom-day", REGIME["V0_random_day"]),
        (x0 + cw, y0 + ch, "V1\nheld-out year", REGIME["V1_within_year"]),
        (x0, y0, "V2\nsame-year", REGIME["V2_same_year_cross"]),
        (x0 + cw, y0, "V3\nstrict", REGIME["V3_strict_cross"]),
    ]
    for x, y, label, color in cells:
        ax.add_patch(Rectangle((x, y), cw, ch, facecolor=color, alpha=0.92, edgecolor="white", linewidth=1.5))
        ax.text(x + cw / 2, y + ch / 2, label, color="white", ha="center", va="center", fontsize=6.8, weight="bold")

    # Paired contrasts reported in the results.
    outputs = [
        (0.61, "V1 − V0", "random-split\noptimism"),
        (0.745, "V3 − V1", "strict-transfer\npenalty"),
        (0.88, "V3 − V2", "same-year\noptimism"),
    ]
    for x, top, bottom in outputs:
        box(ax, (x, 0.05), (0.11, 0.14), f"{top}\n{bottom}", fc="#F7F9FA", ec="#AAB4BE", lw=0.8, size=6.2, bold=True)
        arrow(ax, (0.81, 0.29), (x + 0.055, 0.20), color="#9AA4AD", connection="arc3,rad=0.08", lw=0.8)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.01)
    save(fig, "fig1_study_design")


def half_violin(ax, arrays, positions, colors, width=0.72):
    parts = ax.violinplot(arrays, positions=positions, widths=width, showmeans=False, showmedians=False, showextrema=False)
    for body, pos, color in zip(parts["bodies"], positions, colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.82)
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.maximum(verts[:, 0], pos)
        q1, med, q3 = np.percentile(arrays[positions.index(pos)], [25, 50, 75])
        ax.add_patch(Rectangle((pos - 0.12, q1), 0.22, q3 - q1, facecolor=color, edgecolor=INK, lw=0.65, zorder=4))
        ax.plot([pos - 0.12, pos + 0.10], [med, med], color="white", lw=1.2, zorder=5)
        rng = np.random.default_rng(2400 + int(pos * 11))
        jitter = rng.uniform(-0.28, -0.15, len(arrays[positions.index(pos)]))
        ax.scatter(pos + jitter, arrays[positions.index(pos)], s=9, color=color, edgecolor="white", linewidth=0.25, zorder=5, alpha=0.92)


def forest_panel(ax, data, effect, title):
    models = ["climatology", "ridge", "random_forest"]
    labels = ["Climatology", "Ridge", "Random Forest"]
    y = np.arange(len(models))[::-1]
    for offset, site, color, marker in [(-0.10, "US-Ne2", IRRIGATED, "o"), (0.10, "US-Ne3", RAINFED, "s")]:
        sub = data[(data.effect == effect) & (data.target_site == site)].set_index("model").loc[models]
        x = sub.median_rmse_difference.to_numpy()
        lo = sub.ci_low.to_numpy()
        hi = sub.ci_high.to_numpy()
        ax.errorbar(x, y + offset, xerr=np.vstack([x - lo, hi - x]), fmt=marker, color=color, ecolor=color,
                    ms=4.5, capsize=2, lw=1.1, label="Ne2 irrigated" if site == "US-Ne2" else "Ne3 rainfed")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("paired RMSE difference (W m$^{-2}$)")
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="x", color="#D8DEE3", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def figure2_main() -> None:
    annual = pd.read_csv(TABLES / "yearly_validation_metrics.csv")
    annual = annual[annual.model == "random_forest"]
    penalties = pd.read_csv(TABLES / "penalty_summary.csv")
    fig = plt.figure(figsize=(7.16, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.43, wspace=0.40)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    all_vals = []
    for panel, site, title in [(axes[0], "US-Ne2", "(a) Target: irrigated US-Ne2"), (axes[1], "US-Ne3", "(b) Target: rainfed US-Ne3")]:
        arrays = [annual[(annual.target_site == site) & (annual.regime == r)].sort_values("outer_year").rmse.to_numpy() for r in REGIMES]
        all_vals.extend(arrays)
        positions = list(range(1, 5))
        half_violin(panel, arrays, positions, [REGIME[r] for r in REGIMES])
        for pos, arr in zip(positions, arrays):
            panel.text(pos + 0.04, max(arr) + 0.65, f"{np.median(arr):.1f}", color=INK, fontsize=6.4, ha="center")
        panel.set_xticks(positions, [REGIME_SHORT[r] for r in REGIMES])
        panel.set_ylabel("yearly RMSE (W m$^{-2}$)")
        panel.set_title(title, loc="left", weight="bold")
        panel.grid(axis="y", color="#D8DEE3", lw=0.55)
        panel.spines[["top", "right"]].set_visible(False)
    ymin = min(map(np.min, all_vals)) - 2
    ymax = max(map(np.max, all_vals)) + 3
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin, ymax)

    forest_panel(axes[2], penalties, "same_year_transfer_optimism", "(c) V3 − V2: same-year optimism")
    forest_panel(axes[3], penalties, "strict_transfer_penalty", "(d) V3 − V1: transfer penalty")
    handles = [
        plt.Line2D([0], [0], marker="o", color=IRRIGATED, lw=0, label="Ne2 irrigated"),
        plt.Line2D([0], [0], marker="s", color=RAINFED, lw=0, label="Ne3 rainfed"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.52, 0.005))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.975, bottom=0.13)
    save(fig, "fig2_validation_audit")


def classify_v3() -> pd.DataFrame:
    pred = pd.read_csv(ROOT / "results/predictions_main.csv")
    pred = pred[(pred.regime == "V3_strict_cross") & (pred.model == "random_forest")].copy()
    features = pd.read_csv(ROOT / "data/processed/site_daily_features.csv")
    main = features[features.growing_season & (features.QC >= 0.75)]
    thresholds = []
    for source in pred.source_site.unique():
        for year in pred.outer_year.unique():
            train = main[(main.site == source) & (main.year != year)]
            thresholds.append((source, year, train.VPD.quantile(2 / 3)))
    th = pd.DataFrame(thresholds, columns=["source_site", "outer_year", "vpd_threshold"])
    pred = pred.merge(th, on=["source_site", "outer_year"], validate="many_to_one")
    pred["high_active"] = (pred.VPD >= pred.vpd_threshold) & (pred.NDVI >= 0.40)
    return pred


def scatter_panel(ax, frame, title):
    other = frame[~frame.high_active]
    high = frame[frame.high_active]
    ax.scatter(other.observed_LE, other.predicted_LE, s=6, color="#7E9AA8", alpha=0.36, edgecolor="none", label="other days")
    ax.scatter(high.observed_LE, high.predicted_LE, s=13, marker="^", color=RAINFED, alpha=0.82,
               edgecolor="white", linewidth=0.25, label="high VPD + active canopy")
    lim = 240
    ax.plot([0, lim], [0, lim], color=INK, lw=0.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed LE (W m$^{-2}$)")
    ax.set_ylabel("predicted LE (W m$^{-2}$)")
    rmse = np.sqrt(np.mean((frame.predicted_LE - frame.observed_LE) ** 2))
    bias = np.mean(frame.predicted_LE - frame.observed_LE)
    ax.text(0.05, 0.91, f"RMSE = {rmse:.1f}\nBias = {bias:+.1f}", transform=ax.transAxes, va="top", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CDD5DB", alpha=0.90))
    ax.set_title(title, loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def figure3_failure() -> None:
    pred = classify_v3()
    contrast = pd.read_csv(TABLES / "directional_diagnostic_summary.csv")
    fig = plt.figure(figsize=(7.16, 4.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.78], hspace=0.48, wspace=0.33)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    scatter_panel(ax1, pred[(pred.source_site == "US-Ne3") & (pred.target_site == "US-Ne2")], "(a) Ne3 rainfed → Ne2 irrigated")
    scatter_panel(ax2, pred[(pred.source_site == "US-Ne2") & (pred.target_site == "US-Ne3")], "(b) Ne2 irrigated → Ne3 rainfed")
    ax2.legend(frameon=False, loc="lower right", fontsize=6.4)

    models = ["climatology", "ridge", "random_forest"]
    labels = ["Climatology", "Ridge", "Random Forest"]
    y = np.arange(3)[::-1]
    for offset, source, target, color, marker, label in [
        (-0.10, "US-Ne3", "US-Ne2", IRRIGATED, "o", "Ne3 → Ne2"),
        (0.10, "US-Ne2", "US-Ne3", RAINFED, "s", "Ne2 → Ne3"),
    ]:
        sub = contrast[(contrast.source_site == source) & (contrast.target_site == target)].set_index("model").loc[models]
        x = sub.median_bias_contrast.to_numpy()
        lo = sub.ci_low.to_numpy()
        hi = sub.ci_high.to_numpy()
        ax3.errorbar(x, y + offset, xerr=np.vstack([x - lo, hi - x]), fmt=marker, color=color, ecolor=color,
                     ms=4.5, capsize=2, lw=1.15, label=label)
    ax3.axvline(0, color=INK, lw=0.85)
    ax3.set_yticks(y, labels)
    ax3.set_xlabel("bias contrast: high-VPD active canopy − other days (W m$^{-2}$)")
    ax3.set_title("(c) Direction-specific high-demand bias", loc="left", weight="bold")
    ax3.grid(axis="x", color="#D8DEE3", lw=0.55)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(frameon=False, ncol=2, loc="upper right")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.095)
    save(fig, "fig3_failure_conditions")


def main() -> None:
    style()
    figure1_design()
    figure2_main()
    figure3_failure()
    print(f"Created figures in {OUT}")


if __name__ == "__main__":
    main()
