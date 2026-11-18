"""Grid-tune the HMM decoder parameters on the TRAINING trips (out-of-fold probabilities),
then report the winner on student_test. (Evaluator data is never touched.)

  python scripts/tune_hmm.py --probs standin_lgbm
Outputs: results/hmm_tuning_<probs>.csv (every setting) and results/hmm_best_params.json
Objective: pointwise Top-1 accuracy on valid points (ties broken by route continuity).
"""
import argparse
import itertools
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
from pathsnap.evaluate import continuity_counts

GRID = dict(beta=[8.0, 15.0, 30.0, 60.0], lam_break=[6.0, 12.0, 24.0],
            lam_uturn=[0.0, 6.0, 12.0], eps=[1e-3, 1e-4], back_pen=[1.0], back_free_m=[10.0], w_head=[0.0])


def settings():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def trip_counts(partition, tid, prob, sets):
    prep = PreparedTrip(trip_dir(partition, tid), with_labels=True)
    pt = prep.point_table()
    valid = pt.ref_valid.to_numpy()
    ref = pt.ref_arc.to_numpy()
    tg = TransitionGeometry(prep)
    out = []
    for s in sets:
        pred = decode(prep, prob, HMMParams(**s), tg)["pred_arc"]
        cc = continuity_counts(pred, prep.net)
        out.append((int((pred[valid] == ref[valid]).sum()), int(valid.sum()), cc["good"], cc["n_changes"]))
    return out


def run(partition, tag, probs, sets):
    pr = pd.read_parquet(config.PROCESSED_DIR / f"probs_{probs}_{tag}.parquet")
    by = {t: g["prob"].to_numpy() for t, g in pr.groupby("trip_id", sort=False)}
    res = Parallel(n_jobs=-1)(delayed(trip_counts)(partition, t, by[t], sets) for t in list_trips(partition))
    arr = np.array(res)                                   # trips x sets x 4
    tot = arr.sum(0)
    return pd.DataFrame(sets).assign(top1=tot[:, 0] / tot[:, 1], continuity=tot[:, 2] / tot[:, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="standin_lgbm")
    ap.add_argument("--grid", default=None, help='JSON overriding GRID entries, e.g. \'{"beta":[8],"back_pen":[1,4]}\'')
    ap.add_argument("--no-save-best", action="store_true", help="do not overwrite results/hmm_best_params.json")
    a = ap.parse_args()
    if a.grid:
        GRID.update(json.loads(a.grid))
    sets = settings()
    print(len(sets), "settings")
    tr = run("train", "train_oof", a.probs, sets)
    tr.to_csv(config.RESULTS_DIR / f"hmm_tuning_{a.probs}.csv", index=False)
    best = tr.sort_values(["top1", "continuity"], ascending=False).iloc[0]
    print(tr.sort_values(["top1", "continuity"], ascending=False).head(8).round(4).to_string())
    bp = {k: float(best[k]) for k in GRID}
    if not a.no_save_best:
        json.dump(bp, open(config.RESULTS_DIR / "hmm_best_params.json", "w"), indent=2)
    te = run("student_test", "student_test", a.probs, [bp, HMMParams().to_dict()])
    print("\nstudent_test  [tuned, default]:\n", te.round(4).to_string())
    print("best:", bp)


if __name__ == "__main__":
    main()
