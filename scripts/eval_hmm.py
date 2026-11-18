"""Compare classifier-only vs classifier + HMM/Viterbi on a partition.

  python scripts/eval_hmm.py --probs standin_lgbm --partition student_test
  python scripts/eval_hmm.py --probs standin_lgbm --partition train        (uses OOF probs)

Writes results/hmm_eval_<probs>_<partition>.csv (one row per trip + an ALL row).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from pathsnap.io import list_trips, trip_dir
from pathsnap.pipeline import PreparedTrip
from pathsnap.hmm import HMMParams, TransitionGeometry, decode
from pathsnap.evaluate import continuity_counts, continuity_from_counts, sequence_scores


def load_params(path):
    return HMMParams(**json.load(open(path))) if path and Path(path).exists() else HMMParams()


def eval_trip(partition, tid, prob, params):
    prep = PreparedTrip(trip_dir(partition, tid), with_labels=True)
    pt = prep.point_table()
    valid = pt.ref_valid.to_numpy()
    N = len(pt)
    # classifier-only: highest-probability candidate per point
    c = prep.cand
    order = np.lexsort((-prob, c["obs"]))
    first = np.r_[0, np.flatnonzero(np.diff(c["obs"][order])) + 1]
    clf = np.full(N, -1, dtype=np.int64)
    clf[c["obs"][order][first]] = c["arc"][order][first]
    tg = TransitionGeometry(prep)
    hmm = decode(prep, prob, params, tg)["pred_arc"]
    ref = pt.ref_arc.to_numpy()
    row = dict(trip_id=tid, n_valid=int(valid.sum()))
    rc = continuity_counts(ref, prep.net)
    row["ref_changes"] = rc["n_changes"]
    for name, pred in (("clf", clf), ("hmm", hmm)):
        s = sequence_scores(pred, ref, valid)
        cc = continuity_counts(pred, prep.net)
        row[f"{name}_top1"] = s["pointwise_accuracy"]
        row[f"{name}_route_f1"] = s["route_arc_f1"]
        row[f"{name}_continuity"] = continuity_from_counts(cc)
        row[f"{name}_cont_good"] = cc["good"]
        row[f"{name}_changes"] = cc["n_changes"]
        row[f"{name}_uturns"] = cc["n_uturn"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="standin_lgbm")
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--params", default=str(config.RESULTS_DIR / "hmm_best_params.json"))
    a = ap.parse_args()
    tag = "train_oof" if a.partition == "train" else a.partition
    pr = pd.read_parquet(config.PROCESSED_DIR / f"probs_{a.probs}_{tag}.parquet")
    params = load_params(a.params)
    ids = list_trips(a.partition)
    by = {t: g["prob"].to_numpy() for t, g in pr.groupby("trip_id", sort=False)}
    rows = Parallel(n_jobs=-1)(delayed(eval_trip)(a.partition, t, by[t], params) for t in ids)
    df = pd.DataFrame(rows)
    w = df.n_valid
    allrow = {"trip_id": "ALL(weighted)", "n_valid": int(w.sum())}
    for c in df.columns[2:]:
        if c.endswith(("_cont_good", "_changes", "_uturns")):
            allrow[c] = int(df[c].sum())
        elif not c.endswith("_continuity"):
            allrow[c] = float(np.average(df[c].fillna(0), weights=w))
    for name in ("clf", "hmm"):     # continuity is pooled over arc changes, not averaged over trips
        allrow[f"{name}_continuity"] = allrow[f"{name}_cont_good"] / max(allrow[f"{name}_changes"], 1)
        allrow[f"{name}_change_ratio"] = allrow[f"{name}_changes"] / max(allrow["ref_changes"], 1)
    df = pd.concat([df, pd.DataFrame([allrow])], ignore_index=True)
    out = config.RESULTS_DIR / f"hmm_eval_{a.probs}_{a.partition}.csv"
    df.to_csv(out, index=False)
    print("params:", params.to_dict())
    print(df.tail(1).round(4).T.to_string())
    print("saved", out)


if __name__ == "__main__":
    main()
