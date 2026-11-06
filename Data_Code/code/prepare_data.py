#!/usr/bin/env python3
"""Prepare and audit the US-Ne2/US-Ne3 daily ET analysis table."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "aggregated_daily"
PROCESSED_DIR = ROOT / "data" / "processed"
EXPECTED_COLUMNS = [
    "TIMESTAMP", "LE", "QC", "TA", "WS", "RH", "SW_IN", "LW_IN", "PA", "NDVI"
]
SITES = {"US-Ne2": "irrigated", "US-Ne3": "rainfed"}
EXPECTED_DATES = pd.date_range("2001-01-01", "2013-12-31", freq="D")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_site(site: str, management: str) -> pd.DataFrame:
    path = RAW_DIR / f"{site}_daily.csv"
    frame = pd.read_csv(path)
    assert frame.columns.tolist() == EXPECTED_COLUMNS, (site, frame.columns.tolist())
    frame["TIMESTAMP"] = pd.to_datetime(frame["TIMESTAMP"], errors="raise")
    assert not frame.isna().any().any(), f"Unexpected missing data at {site}"
    assert np.isfinite(frame.drop(columns="TIMESTAMP").to_numpy()).all(), site
    assert frame["TIMESTAMP"].is_unique, site
    assert frame["TIMESTAMP"].sort_values().reset_index(drop=True).equals(pd.Series(EXPECTED_DATES)), site

    frame.insert(0, "site", site)
    frame.insert(1, "management", management)
    frame["year"] = frame["TIMESTAMP"].dt.year
    frame["month"] = frame["TIMESTAMP"].dt.month
    frame["doy"] = frame["TIMESTAMP"].dt.dayofyear
    frame["growing_season"] = frame["month"].between(5, 9)
    frame["rh_clipped_for_vpd"] = frame["RH"].clip(0, 100)
    es = 0.6108 * np.exp(17.27 * frame["TA"] / (frame["TA"] + 237.3))
    frame["VPD"] = es * (1 - frame["rh_clipped_for_vpd"] / 100)
    frame["sin_DOY"] = np.sin(2 * np.pi * frame["doy"] / 365.25)
    frame["cos_DOY"] = np.cos(2 * np.pi * frame["doy"] / 365.25)
    frame["ET_mm_day_display"] = frame["LE"] * 86400 / 2.45e6
    return frame


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = [load_site(site, management) for site, management in SITES.items()]
    data = pd.concat(frames, ignore_index=True)

    metadata = pd.read_csv(ROOT / "data" / "site_year_metadata.csv")
    crop_long = pd.concat(
        [
            metadata[["year", f"{site}_crop", "crop_matched"]]
            .rename(columns={f"{site}_crop": "crop"})
            .assign(site=site)
            for site in SITES
        ],
        ignore_index=True,
    )
    data = data.merge(crop_long, on=["site", "year"], how="left", validate="many_to_one")
    assert data["crop"].notna().all()

    data.to_csv(PROCESSED_DIR / "site_daily_features.csv", index=False)

    audit_rows = []
    for site, group in data.groupby("site", sort=True):
        audit_rows.append(
            {
                "site": site,
                "management": group["management"].iloc[0],
                "rows": len(group),
                "start_date": group["TIMESTAMP"].min().date().isoformat(),
                "end_date": group["TIMESTAMP"].max().date().isoformat(),
                "unique_dates": group["TIMESTAMP"].nunique(),
                "missing_cells": int(group.isna().sum().sum()),
                "negative_LE_days": int((group["LE"] < 0).sum()),
                "RH_above_100_days": int((group["RH"] > 100).sum()),
                "QC_ge_050": int((group["QC"] >= 0.50).sum()),
                "QC_ge_075": int((group["QC"] >= 0.75).sum()),
                "QC_ge_090": int((group["QC"] >= 0.90).sum()),
                "QC_eq_100": int((group["QC"] == 1.00).sum()),
                "growing_QC_ge_075": int((group["growing_season"] & (group["QC"] >= 0.75)).sum()),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(PROCESSED_DIR / "data_audit_by_site.csv", index=False)

    cohorts = {
        "main_growing_qc075": data["growing_season"] & (data["QC"] >= 0.75),
        "growing_qc050": data["growing_season"] & (data["QC"] >= 0.50),
        "growing_qc090": data["growing_season"] & (data["QC"] >= 0.90),
        "growing_qc100": data["growing_season"] & (data["QC"] == 1.00),
        "all_year_qc075": data["QC"] >= 0.75,
        "growing_qc075_crop_matched": data["growing_season"] & (data["QC"] >= 0.75) & data["crop_matched"],
    }
    count_frames = []
    for cohort, mask in cohorts.items():
        count_frames.append(
            data.loc[mask].groupby(["site", "year"], as_index=False).size().assign(cohort=cohort)
        )
    counts = pd.concat(count_frames, ignore_index=True)
    counts.to_csv(PROCESSED_DIR / "cohort_counts_by_site_year.csv", index=False)

    paired = data.pivot(index="TIMESTAMP", columns="site", values=["QC", "growing_season"])
    paired_summary = {
        "paired_all_dates": int(len(paired)),
        "paired_growing_qc050": int(
            (paired[("growing_season", "US-Ne2")].astype(bool)
             & (paired[("QC", "US-Ne2")] >= 0.50)
             & (paired[("QC", "US-Ne3")] >= 0.50)).sum()
        ),
        "paired_growing_qc075": int(
            (paired[("growing_season", "US-Ne2")].astype(bool)
             & (paired[("QC", "US-Ne2")] >= 0.75)
             & (paired[("QC", "US-Ne3")] >= 0.75)).sum()
        ),
        "archive_sha256": sha256(ROOT / "data/raw/aggregated_daily.zip"),
    }
    (PROCESSED_DIR / "preparation_manifest.json").write_text(
        json.dumps(paired_summary, indent=2) + "\n", encoding="utf-8"
    )

    dictionary = pd.DataFrame(
        [
            ("LE", "daily mean latent heat flux", "W m^-2", "target"),
            ("QC", "fraction of hourly LE inputs observed", "0-1", "filter only"),
            ("TA", "air temperature", "deg C", "predictor"),
            ("WS", "wind speed", "m s^-1", "predictor"),
            ("RH", "relative humidity", "%", "predictor"),
            ("SW_IN", "incoming shortwave radiation", "W m^-2", "predictor"),
            ("LW_IN", "incoming longwave radiation", "W m^-2", "predictor"),
            ("PA", "air pressure", "kPa", "predictor"),
            ("NDVI", "normalized difference vegetation index", "unitless", "predictor"),
            ("VPD", "vapor pressure deficit derived from TA/RH", "kPa", "predictor/diagnostic"),
            ("sin_DOY", "cyclic day-of-year sine", "unitless", "predictor"),
            ("cos_DOY", "cyclic day-of-year cosine", "unitless", "predictor"),
        ],
        columns=["field", "description", "unit", "analysis_role"],
    )
    dictionary.to_csv(PROCESSED_DIR / "data_dictionary.csv", index=False)

    print(audit.to_string(index=False))
    print("\nCohort per-year minimum/median/maximum:")
    print(counts.groupby(["cohort", "site"])["size"].agg(["min", "median", "max"]).to_string())
    print("\nManifest:", paired_summary)


if __name__ == "__main__":
    main()
