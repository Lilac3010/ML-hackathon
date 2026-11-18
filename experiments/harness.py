"""Shared experiment harness: every model is trained / cross-validated / scored identically.

A model lives in experiments/zoo/<name>.py and defines (see zoo/template.py):
    NAME, FAMILY, PARAMS (initial config), build(params) -> sklearn-style estimator
    optional: MAX_TRAIN_ROWS, fit(est, X, y, Xval, yval), history(est), search_space(trial)

run_experiment(name) does:
  1. 5-fold GROUP CV on the 80 train trips (folds from pathsnap.evaluate.assign_folds). Only
     ref_valid rows are used for fitting. Per-fold: candidate macro P/R/F1, log loss, Top-1.
  2. refit on all train, predict student_test: same metrics + confusion matrix +
     classifier-only route continuity + training / inference time.
  3. writes   results/experiments/<tag>.json
              data/processed/probs_<tag>_{train_oof,student_test}.parquet   (decoder tuning input)
              models/<tag>.joblib   ({"model", "feature_cols"})
"""
import importlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from pathsnap.features import FEATURE_COLS
from pathsnap.evaluate import (assign_folds, N_FOLDS, candidate_metrics, observation_top1,
                               route_continuity)
from pathsnap.io import trip_dir
from pathsnap.pipeline import PreparedTrip

EXP_DIR = config.RESULTS_DIR / "experiments"


def load_zoo(name):
    mod = importlib.import_module(f"experiments.zoo.{name}")
    assert mod.NAME == name, f"NAME in {name}.py must equal the file name"
    return mod


def load_tables():
    tr = pd.read_parquet(config.PROCESSED_DIR / "train_features.parquet")
    te = pd.read_parquet(config.PROCESSED_DIR / "student_test_features.parquet")
    return tr, te


def _subsample_points(df, max_rows, seed):
    """Optionally shrink the training set for slow models (keeps whole candidate sets of
    randomly chosen GPS points). Must be disclosed in the report if used."""
    if not max_rows or len(df) <= max_rows:
        return df
    key = df["trip_id"].astype(str) + "_" + df["obs_idx"].astype(str)
    uniq = key.unique()
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(uniq, size=int(len(uniq) * max_rows / len(df)), replace=False))
    return df[key.isin(keep)]


def _fit(mod, est, X, y, Xva=None, yva=None):
    if hasattr(mod, "fit"):
        return mod.fit(est, X, y, Xva, yva)
    return est.fit(X, y)


def _proba(est, X):
    return est.predict_proba(X)[:, 1]


def cross_validate(mod, params, tr, seed=config.SEED, folds=None):
    """Group CV. Returns (oof_prob array aligned with tr, per-fold metric list, total fit seconds)."""
    fold = tr["trip_id"].map(assign_folds(tr["trip_id"].unique(), seed=seed)).to_numpy()
    oof = np.zeros(len(tr))
    rows, fit_s = [], 0.0
    for k in (folds if folds is not None else range(N_FOLDS)):
        fit_df = tr[(fold != k) & tr["ref_valid"].to_numpy()]
        fit_df = _subsample_points(fit_df, getattr(mod, "MAX_TRAIN_ROWS", None), seed)
        est = mod.build(params)
        t0 = time.perf_counter()
        _fit(mod, est, fit_df[FEATURE_COLS], fit_df["label"])
        fit_s += time.perf_counter() - t0
        va = tr[fold == k]
        p = _proba(est, va[FEATURE_COLS])
        oof[fold == k] = p
        m = candidate_metrics(va, p)
        m.update(observation_top1(va, p))
        m["fold"] = int(k)
        rows.append(m)
    return oof, rows, fit_s


def _clf_continuity(te, prob):
    """Classifier-only route continuity on student_test (top-1 candidate per GPS point)."""
    vals, wts = [], []
    for tid, g in te.assign(_p=prob).groupby("trip_id", sort=False):
        prep = PreparedTrip(trip_dir("student_test", tid), with_labels=False)
        top = g.loc[g.groupby("obs_idx")["_p"].idxmax()]
        pred = np.full(len(prep.P), -1, dtype=np.int64)
        pred[top["obs_idx"].to_numpy()] = top["arc_id"].to_numpy()
        vals.append(route_continuity(pred, prep.net))
        wts.append(len(prep.P))
    return float(np.average(vals, weights=wts))


def run_experiment(name, params=None, tag=None, seed=config.SEED, tables=None, verbose=True):
    mod = load_zoo(name)
    params = {**mod.PARAMS, **(params or {})}
    tag = tag or name
    tr, te = tables or load_tables()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    oof, fold_rows, cv_fit_s = cross_validate(mod, params, tr, seed)
    cvdf = pd.DataFrame(fold_rows)
    if verbose:
        print(f"[{tag}] CV  macroF1 {cvdf.f1_macro.mean():.4f}+-{cvdf.f1_macro.std():.4f}  "
              f"top1 {cvdf.top1_accuracy.mean():.4f}  logloss {cvdf.log_loss.mean():.4f}  ({cv_fit_s:.0f}s fit)")

    # final refit on all valid train rows
    fit_df = _subsample_points(tr[tr["ref_valid"]], getattr(mod, "MAX_TRAIN_ROWS", None), seed)
    est = mod.build(params)
    tev = te[te["ref_valid"]]
    t0 = time.perf_counter()
    _fit(mod, est, fit_df[FEATURE_COLS], fit_df["label"], tev[FEATURE_COLS], tev["label"])
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pte = _proba(est, te[FEATURE_COLS])
    infer_s = time.perf_counter() - t0
    m = candidate_metrics(te, pte)
    m.update(observation_top1(te, pte))
    yv = te.loc[te["ref_valid"], "label"].to_numpy()
    ph = (np.clip(pte[te["ref_valid"].to_numpy()], 0, 1) >= 0.5).astype(int)
    cm = [[int(((yv == 0) & (ph == 0)).sum()), int(((yv == 0) & (ph == 1)).sum())],
          [int(((yv == 1) & (ph == 0)).sum()), int(((yv == 1) & (ph == 1)).sum())]]
    m.update(confusion_matrix=cm, route_continuity_clf_only=_clf_continuity(te, pte),
             train_time_s=train_s, inference_time_s=infer_s,
             inference_ms_per_1k_rows=1000 * infer_s / (len(te) / 1000), n_train_rows=int(len(fit_df)))
    if verbose:
        print(f"[{tag}] TEST macroF1 {m['f1_macro']:.4f}  top1 {m['top1_accuracy']:.4f}  logloss {m['log_loss']:.4f}  "
              f"continuity {m['route_continuity_clf_only']:.4f}  train {train_s:.0f}s  infer {infer_s:.2f}s")

    hist = mod.history(est) if hasattr(mod, "history") else None
    result = dict(name=tag, base_model=name, family=mod.FAMILY, params=params, seed=seed,
                  cv_folds=fold_rows,
                  cv=dict(f1_macro_mean=cvdf.f1_macro.mean(), f1_macro_std=cvdf.f1_macro.std(),
                          top1_mean=cvdf.top1_accuracy.mean(), top1_std=cvdf.top1_accuracy.std(),
                          log_loss_mean=cvdf.log_loss.mean(), precision_macro_mean=cvdf.precision_macro.mean(),
                          recall_macro_mean=cvdf.recall_macro.mean(), cv_fit_time_s=cv_fit_s),
                  student_test=m, history=hist,
                  max_train_rows=getattr(mod, "MAX_TRAIN_ROWS", None))
    with open(EXP_DIR / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    for part, df, p in (("train_oof", tr, oof), ("student_test", te, pte)):
        o = df[["trip_id", "obs_idx", "arc_id"]].copy()
        o["prob"] = np.asarray(p, dtype=np.float32)
        o.to_parquet(config.PROCESSED_DIR / f"probs_{tag}_{part}.parquet", index=False)
    (config.ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": est, "feature_cols": FEATURE_COLS}, config.ROOT / "models" / f"{tag}.joblib")
    return result
