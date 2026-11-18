"""DEV STAND-IN ONLY (not one of the compared models): quick LightGBM to produce candidate
probabilities so the HMM decoder can be developed/tuned before the real model exists.

Writes data/processed/probs_<name>_{train_oof,student_test}.parquet with columns
trip_id, obs_idx, arc_id, prob  (same row order as the features parquet).
Out-of-fold for train (5 group folds from pathsnap.evaluate.assign_folds) so decoder
parameters are never tuned on probabilities from a model that saw the same trip.

The real model should write the SAME file format (see docs/HANDOFF.md), and the decoder
tuning script then re-runs on it:  python scripts/tune_hmm.py --probs <name>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from pathsnap.features import FEATURE_COLS
from pathsnap.evaluate import assign_folds, N_FOLDS, candidate_metrics, observation_top1

NAME = "standin_lgbm"
PARAMS = dict(n_estimators=300, learning_rate=0.08, num_leaves=31, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=config.SEED,
              n_jobs=-1, verbose=-1)


def main():
    tr = pd.read_parquet(config.PROCESSED_DIR / "train_features.parquet")
    te = pd.read_parquet(config.PROCESSED_DIR / "student_test_features.parquet")
    fold = tr.trip_id.map(assign_folds(tr.trip_id.unique())).to_numpy()
    oof = np.zeros(len(tr))
    for k in range(N_FOLDS):
        fit = (fold != k) & tr.ref_valid.to_numpy()
        m = lgb.LGBMClassifier(**PARAMS).fit(tr.loc[fit, FEATURE_COLS], tr.loc[fit, "label"])
        oof[fold == k] = m.predict_proba(tr.loc[fold == k, FEATURE_COLS])[:, 1]
    fit = tr.ref_valid.to_numpy()
    m = lgb.LGBMClassifier(**PARAMS).fit(tr.loc[fit, FEATURE_COLS], tr.loc[fit, "label"])
    pte = m.predict_proba(te[FEATURE_COLS])[:, 1]
    import joblib
    (config.ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": m, "feature_cols": FEATURE_COLS}, config.ROOT / "models" / f"{NAME}.joblib")
    for tag, df, p in (("train_oof", tr, oof), ("student_test", te, pte)):
        out = df[["trip_id", "obs_idx", "arc_id"]].copy()
        out["prob"] = p.astype(np.float32)
        out.to_parquet(config.PROCESSED_DIR / f"probs_{NAME}_{tag}.parquet", index=False)
        print(tag, "candidate-level:", {k: round(v, 4) for k, v in candidate_metrics(df, p).items()},
              "| top1:", observation_top1(df, p))


if __name__ == "__main__":
    main()
