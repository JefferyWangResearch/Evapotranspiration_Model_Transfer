#!/usr/bin/env python3
"""Run the prespecified V0-V3 validation audit on the main cohort."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/site_daily_features.csv"
RESULTS = ROOT / "results"
FEATURES = ["TA", "RH", "WS", "PA", "SW_IN", "LW_IN", "NDVI", "VPD", "sin_DOY", "cos_DOY"]
SITES = ["US-Ne2", "US-Ne3"]
YEARS = list(range(2001, 2014))
RANDOM_SEEDS = [1103, 2213, 3323, 4433, 5543, 6653, 7763, 8873, 9983, 10993]

MODEL_GRIDS = {
    "ridge": [{"alpha": x} for x in [0.1, 1.0, 10.0, 100.0]],
    "random_forest": list(
        ParameterGrid(
            {
                "max_depth": [None, 10],
                "max_features": [0.7],
                "min_samples_leaf": [1, 3, 7],
            }
        )
    ),
}


@dataclass
class ClimatologyModel:
    means: pd.Series | None = None
    global_mean: float = np.nan

    def fit(self, frame: pd.DataFrame) -> "ClimatologyModel":
        bins = ((frame["doy"] - 1) // 15).astype(int)
        self.means = frame.assign(_bin=bins).groupby("_bin")["LE"].mean()
        self.global_mean = float(frame["LE"].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.means is not None
        bins = ((frame["doy"] - 1) // 15).astype(int)
        return bins.map(self.means).fillna(self.global_mean).to_numpy()


def make_model(name: str, params: dict, seed: int):
    if name == "climatology":
        return ClimatologyModel()
    if name == "ridge":
        return Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=params["alpha"]))]
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=150,
            random_state=seed,
            n_jobs=-1,
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
        )
    raise ValueError(name)


def fit_predict(name: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    model = make_model(name, params, seed)
    if name == "climatology":
        model.fit(train)
        return model.predict(test)
    model.fit(train[FEATURES], train["LE"])
    return model.predict(test[FEATURES])


def tune(name: str, train: pd.DataFrame, outer_year: int) -> tuple[dict, float]:
    if name == "climatology":
        return {}, np.nan
    groups = train["year"].to_numpy()
    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    best_params = None
    best_rmse = np.inf
    for params in MODEL_GRIDS[name]:
        fold_rmse = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, groups=groups)):
            tr = train.iloc[tr_idx]
            va = train.iloc[va_idx]
            pred = fit_predict(name, params, tr, va, seed=outer_year * 100 + fold)
            fold_rmse.append(mean_squared_error(va["LE"], pred) ** 0.5)
        score = float(np.mean(fold_rmse))
        if score < best_rmse:
            best_rmse = score
            best_params = params
    assert best_params is not None
    return dict(best_params), best_rmse


def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "n_test": int(len(observed)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "mae": float(mean_absolute_error(observed, predicted)),
        "bias": float(np.mean(predicted - observed)),
        "r2": float(r2_score(observed, predicted)) if len(observed) >= 2 else np.nan,
    }


def prediction_frame(test: pd.DataFrame, pred: np.ndarray, **labels) -> pd.DataFrame:
    result = test[
        ["TIMESTAMP", "year", "site", "management", "crop", "crop_matched", "LE", "QC", "VPD", "NDVI"]
    ].copy()
    result = result.rename(columns={"site": "target_site", "LE": "observed_LE"})
    result["predicted_LE"] = pred
    for key, value in labels.items():
        result[key] = value
    return result


def main() -> None:
    start = time.time()
    RESULTS.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP"])
    data = data[data["growing_season"] & (data["QC"] >= 0.75)].copy()
    assert set(data["year"]) == set(YEARS)

    predictions: list[pd.DataFrame] = []
    metrics: list[dict] = []
    tuning: list[dict] = []
    best: dict[tuple[str, int, str], dict] = {}

    # Tune once for each site/year/model on the strict 12-year training set.
    for site in SITES:
        for year in YEARS:
            train = data[(data["site"] == site) & (data["year"] != year)].copy()
            for model in ["ridge", "random_forest"]:
                params, inner_rmse = tune(model, train, year)
                best[(site, year, model)] = params
                tuning.append(
                    {
                        "site": site,
                        "outer_year": year,
                        "model": model,
                        "n_train": len(train),
                        "best_params": json.dumps(params, sort_keys=True),
                        "inner_group_rmse": inner_rmse,
                    }
                )
        print(f"Tuning complete: {site}", flush=True)

    # V1 and V3 share a strict site/year model; score on same-site and other-site targets.
    for source in SITES:
        other = SITES[1] if source == SITES[0] else SITES[0]
        for year in YEARS:
            train = data[(data["site"] == source) & (data["year"] != year)].copy()
            for model in ["climatology", "ridge", "random_forest"]:
                params = {} if model == "climatology" else best[(source, year, model)]
                for target, regime in [(source, "V1_within_year"), (other, "V3_strict_cross")]:
                    test = data[(data["site"] == target) & (data["year"] == year)].copy()
                    pred = fit_predict(model, params, train, test, seed=year + 17)
                    labels = {
                        "regime": regime,
                        "source_site": source,
                        "target_site": target,
                        "model": model,
                        "outer_year": year,
                        "repeat": 0,
                    }
                    predictions.append(prediction_frame(test, pred, **labels))
                    metrics.append({**labels, **metric_row(test["LE"].to_numpy(), pred), "n_train": len(train)})
        print(f"V1/V3 complete: {source}", flush=True)

    # V2: include source year y; exclude each other year once; aggregate 12 equal-year-count fits.
    for source in SITES:
        target = SITES[1] if source == SITES[0] else SITES[0]
        for year in YEARS:
            test = data[(data["site"] == target) & (data["year"] == year)].copy()
            for model in ["climatology", "ridge", "random_forest"]:
                params = {} if model == "climatology" else best[(source, year, model)]
                variant_predictions = []
                train_sizes = []
                for excluded_year in [x for x in YEARS if x != year]:
                    train = data[(data["site"] == source) & (data["year"] != excluded_year)].copy()
                    variant_predictions.append(
                        fit_predict(model, params, train, test, seed=year * 100 + excluded_year)
                    )
                    train_sizes.append(len(train))
                pred = np.mean(np.vstack(variant_predictions), axis=0)
                labels = {
                    "regime": "V2_same_year_cross",
                    "source_site": source,
                    "target_site": target,
                    "model": model,
                    "outer_year": year,
                    "repeat": 0,
                }
                predictions.append(prediction_frame(test, pred, **labels))
                metrics.append(
                    {
                        **labels,
                        **metric_row(test["LE"].to_numpy(), pred),
                        "n_train": float(np.mean(train_sizes)),
                    }
                )
        print(f"V2 complete: {source}", flush=True)

    # V0: random holdout with the same target-year test size, ten prespecified seeds.
    for site in SITES:
        pool = data[data["site"] == site].copy().reset_index(drop=True)
        for year in YEARS:
            n_test = int((pool["year"] == year).sum())
            for repeat, seed in enumerate(RANDOM_SEEDS, start=1):
                rng = np.random.default_rng(seed + year)
                test_idx = rng.choice(pool.index.to_numpy(), size=n_test, replace=False)
                test = pool.loc[test_idx].copy()
                train = pool.drop(index=test_idx).copy()
                for model in ["climatology", "ridge", "random_forest"]:
                    params = {} if model == "climatology" else best[(site, year, model)]
                    pred = fit_predict(model, params, train, test, seed=seed)
                    labels = {
                        "regime": "V0_random_day",
                        "source_site": site,
                        "target_site": site,
                        "model": model,
                        "outer_year": year,
                        "repeat": repeat,
                    }
                    predictions.append(prediction_frame(test, pred, **labels))
                    metrics.append({**labels, **metric_row(test["LE"].to_numpy(), pred), "n_train": len(train)})
        print(f"V0 complete: {site}", flush=True)

    pred_all = pd.concat(predictions, ignore_index=True)
    metrics_all = pd.DataFrame(metrics)
    tuning_all = pd.DataFrame(tuning)
    pred_all.to_csv(RESULTS / "predictions_main.csv", index=False)
    metrics_all.to_csv(RESULTS / "metrics_raw_main.csv", index=False)
    tuning_all.to_csv(RESULTS / "tuning_main.csv", index=False)

    elapsed = time.time() - start
    manifest = {
        "cohort": "May-September and site-specific QC >= 0.75",
        "features": FEATURES,
        "years": YEARS,
        "random_seeds": RANDOM_SEEDS,
        "prediction_rows": len(pred_all),
        "metric_rows": len(metrics_all),
        "elapsed_seconds": elapsed,
    }
    (RESULTS / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
