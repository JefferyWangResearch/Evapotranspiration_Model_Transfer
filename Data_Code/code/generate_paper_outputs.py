#!/usr/bin/env python3
"""Generate compact, paper-ready figures and tables from completed analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "V0_random_day": "#4C78A8",
    "V1_within_year": "#72B7B2",
    "V2_same_year_cross": "#F2CF5B",
    "V3_strict_cross": "#E45756",
}
REGIME_LABELS = {
    "V0_random_day": "V0 random-day",
    "V1_within_year": "V1 held-out year",
    "V2_same_year_cross": "V2 same-year transfer",
    "V3_strict_cross": "V3 strict transfer",
}
TARGET_LABELS = {"US-Ne2": "US-Ne2 (irrigated)", "US-Ne3": "US-Ne3 (rainfed)"}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1_design() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.8), sharey=True)
    panels = [
        ("V0", "Random-day", "target site", "mixed years", "seen site\nseen seasons"),
        ("V1", "Held-out year", "target site", "other years", "seen site\nunseen year"),
        ("V2", "Same-year transfer", "other site", "includes year y", "unseen site\nseen year"),
        ("V3", "Strict transfer", "other site", "excludes year y", "unseen site\nunseen year"),
    ]
    for ax, (code, title, site, years, test) in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"{code}: {title}", fontsize=10, weight="bold")
        train_box = FancyBboxPatch(
            (0.08, 0.58), 0.84, 0.22, boxstyle="round,pad=0.025", fc="#DCEAF7", ec="#4C78A8"
        )
        test_box = FancyBboxPatch(
            (0.08, 0.17), 0.84, 0.22, boxstyle="round,pad=0.025", fc="#FCE0DD", ec="#E45756"
        )
        ax.add_patch(train_box)
        ax.add_patch(test_box)
        ax.text(0.5, 0.69, f"TRAIN\n{site}; {years}", ha="center", va="center", fontsize=8.5)
        ax.annotate("", xy=(0.5, 0.43), xytext=(0.5, 0.56), arrowprops={"arrowstyle": "->", "lw": 1.3})
        ax.text(0.5, 0.28, f"TEST\n{test}", ha="center", va="center", fontsize=8.5)
    fig.suptitle("Four validation regimes isolate temporal and management-domain shifts", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "figure1_validation_design")


def figure2_results() -> None:
    summary = pd.read_csv(TABLES / "validation_summary.csv")
    rmse = summary[(summary["metric"] == "rmse") & (summary["model"] == "random_forest")].copy()
    penalty = pd.read_csv(TABLES / "penalty_summary.csv")
    penalty = penalty[penalty["model"] == "random_forest"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7))
    ax = axes[0]
    regimes = list(REGIME_LABELS)
    x = np.arange(len(regimes))
    for offset, site, marker in [(-0.08, "US-Ne2", "o"), (0.08, "US-Ne3", "s")]:
        sub = rmse[rmse["target_site"] == site].set_index("regime").loc[regimes]
        y = sub["median"].to_numpy()
        yerr = np.vstack([y - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - y])
        ax.errorbar(x + offset, y, yerr=yerr, marker=marker, capsize=3, lw=1.5, label=TARGET_LABELS[site])
    ax.set_xticks(x, ["V0", "V1", "V2", "V3"])
    ax.set_ylabel("Median yearly RMSE (W m$^{-2}$)")
    ax.set_title("A. Random Forest validation ladder", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    effects = ["random_split_optimism", "strict_transfer_penalty", "same_year_transfer_optimism"]
    labels = ["V1 − V0\nrandom-split optimism", "V3 − V1\nstrict transfer penalty", "V3 − V2\nsame-year optimism"]
    x = np.arange(len(effects))
    for offset, site, marker in [(-0.08, "US-Ne2", "o"), (0.08, "US-Ne3", "s")]:
        sub = penalty[penalty["target_site"] == site].set_index("effect").loc[effects]
        y = sub["median_rmse_difference"].to_numpy()
        yerr = np.vstack([y - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - y])
        ax.errorbar(x + offset, y, yerr=yerr, marker=marker, capsize=3, lw=0, elinewidth=1.5, label=TARGET_LABELS[site])
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("Paired RMSE difference (W m$^{-2}$)")
    ax.set_title("B. Year-paired validation effects", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "figure2_main_results")


def figure3_diagnostic() -> None:
    data = pd.read_csv(TABLES / "directional_diagnostic_summary.csv")
    model_order = ["climatology", "ridge", "random_forest"]
    directions = [("US-Ne3", "US-Ne2"), ("US-Ne2", "US-Ne3")]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(len(model_order))
    for offset, (source, target), marker in zip([-0.09, 0.09], directions, ["o", "s"]):
        sub = data[(data["source_site"] == source) & (data["target_site"] == target)].set_index("model").loc[model_order]
        y = sub["median_bias_contrast"].to_numpy()
        yerr = np.vstack([y - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - y])
        label = f"{source} → {target}"
        ax.errorbar(x + offset, y, yerr=yerr, marker=marker, capsize=3, lw=0, elinewidth=1.5, label=label)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x, ["Climatology", "Ridge", "Random Forest"])
    ax.set_ylabel("Bias contrast: high-VPD active canopy\nminus other days (W m$^{-2}$)")
    ax.set_title("Strict-transfer failure is direction-dependent", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "figure3_directional_diagnostic")

    # Single-column version for the compact symposium manuscript.
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    x = np.arange(len(model_order))
    for offset, (source, target), marker in zip([-0.07, 0.07], directions, ["o", "s"]):
        sub = data[(data["source_site"] == source) & (data["target_site"] == target)].set_index("model").loc[model_order]
        y = sub["median_bias_contrast"].to_numpy()
        yerr = np.vstack([y - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - y])
        label = "Ne3→Ne2" if source == "US-Ne3" else "Ne2→Ne3"
        ax.errorbar(x + offset, y, yerr=yerr, marker=marker, capsize=2, lw=0, elinewidth=1.1, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, ["Clim.", "Ridge", "RF"])
    ax.set_ylabel("Bias contrast (W m$^{-2}$)")
    ax.set_title("High-VPD active canopy vs. other days", fontsize=9, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout(pad=0.5)
    save(fig, "figure3_directional_diagnostic_column")


def paper_tables() -> None:
    audit = pd.read_csv(ROOT / "data/processed/data_audit_by_site.csv")
    table1 = audit[["site", "management", "start_date", "end_date", "rows", "growing_QC_ge_075"]].copy()
    table1.to_csv(TABLES / "paper_table1_dataset.csv", index=False)

    validation = pd.read_csv(TABLES / "validation_summary.csv")
    validation = validation[validation["metric"] == "rmse"]
    med = validation.pivot_table(index=["target_site", "model"], columns="regime", values="median").reset_index()
    effects = pd.read_csv(TABLES / "penalty_summary.csv")
    keep_effects = effects[effects["effect"].isin(["strict_transfer_penalty", "same_year_transfer_optimism"])].copy()
    keep_effects["display"] = keep_effects.apply(
        lambda r: f"{r.median_rmse_difference:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1
    )
    eff = keep_effects.pivot_table(index=["target_site", "model"], columns="effect", values="display", aggfunc="first").reset_index()
    table2 = med.merge(eff, on=["target_site", "model"])
    table2 = table2.rename(columns={
        "V0_random_day": "V0_RMSE",
        "V1_within_year": "V1_RMSE",
        "V2_same_year_cross": "V2_RMSE",
        "V3_strict_cross": "V3_RMSE",
        "strict_transfer_penalty": "V3_minus_V1_95CI",
        "same_year_transfer_optimism": "V3_minus_V2_95CI",
    })
    for col in ["V0_RMSE", "V1_RMSE", "V2_RMSE", "V3_RMSE"]:
        table2[col] = table2[col].round(2)
    table2.to_csv(TABLES / "paper_table2_main_results.csv", index=False)

    def markdown_table(frame: pd.DataFrame) -> str:
        labels = [str(x) for x in frame.columns]
        rows = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(labels)) + " |"]
        rows.extend("| " + " | ".join(str(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))
        return "\n".join(rows)

    lines = ["# Paper-ready tables", "", "## Table 1 — Dataset", "", markdown_table(table1), "", "## Table 2 — Main RMSE results", "", markdown_table(table2)]
    (TABLES / "paper_tables.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    figure1_design()
    figure2_results()
    figure3_diagnostic()
    paper_tables()
    print("Paper figures and tables created.")


if __name__ == "__main__":
    main()
