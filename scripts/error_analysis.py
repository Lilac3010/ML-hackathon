"""Where does the decoder go wrong on a trip, and why?  (uses out-of-fold probabilities: honest)

   python scripts/error_analysis.py 00000072 [--probs lightgbm_gbdt]

Each wrong GPS point (valid reference only) is classed as:
  reverse_twin     predicted arc is the opposite direction of the reference arc
  route_neighbour  predicted arc is another arc of the reference route within +-3 positions (boundary shift)
  route_far        predicted arc is on the reference route but far along it (loop / revisit / jump)
  off_route        predicted arc is not on the reference route at all
  ref_not_in_cand  the reference arc was not even a candidate (candidate miss)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from pathsnap.io import trip_dir
from pathsnap.pipeline import PreparedTrip
from pathsnap.hmm import TransitionGeometry, decode
from pathsnap.inference import load_hmm_params
from pathsnap.evaluate import route_continuity


def runs(mask):
    """lengths of consecutive True runs"""
    m = np.r_[False, mask, False]
    d = np.diff(m.astype(int))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def analyse(tid, probs="lightgbm_gbdt", partition="train", verbose=True):
    prep = PreparedTrip(trip_dir(partition, tid), with_labels=True)
    pr = pd.read_parquet(config.PROCESSED_DIR / f"probs_{probs}_{'train_oof' if partition == 'train' else partition}.parquet")
    pr = pr[pr.trip_id == tid]
    assert len(pr) == len(prep.cand["obs"]), "probability rows must align with candidate rows"
    prob = pr["prob"].to_numpy()
    res = decode(prep, prob, load_hmm_params(), TransitionGeometry(prep))
    pred, ref, valid = res["pred_arc"], prep.ref["ref_arc"], prep.ref["ref_valid"]
    ridx = prep.ref["ref_idx"]
    net, route = prep.net, prep.route
    pos = {}
    for i, a in enumerate(route):
        pos.setdefault(int(a), []).append(i)

    bad = np.flatnonzero(valid & (pred != ref))
    cand_hit = np.zeros(len(pred), bool)
    m = prep.cand["arc"] == ref[prep.cand["obs"]]
    cand_hit[prep.cand["obs"][m]] = True
    kinds = []
    for i in bad:
        p, r = int(pred[i]), int(ref[i])
        if not cand_hit[i]:
            kinds.append("ref_not_in_cand")
        elif net.reverse_arc[r] == p:
            kinds.append("reverse_twin")
        elif p in pos and min(abs(j - ridx[i]) for j in pos[p]) <= 3:
            kinds.append("route_neighbour")
        elif p in pos:
            kinds.append("route_far")
        else:
            kinds.append("off_route")
    k = pd.Series(kinds).value_counts()
    n_valid = int(valid.sum())
    err_runs = runs(valid & (pred != ref))
    out = dict(trip=tid, n_obs=len(pred), n_valid=n_valid, hmm_top1=1 - len(bad) / n_valid,
               cont=route_continuity(pred, net), n_error_points=len(bad),
               longest_error_run=int(err_runs.max()) if len(err_runs) else 0,
               n_error_runs=len(err_runs), **{f"err_{x}": int(k.get(x, 0)) for x in
                                              ["reverse_twin", "route_neighbour", "route_far", "off_route", "ref_not_in_cand"]})
    if verbose:
        print(f"\n=== trip {tid} ({partition}) ===")
        for a, b in out.items():
            print(f"  {a:20s} {b if not isinstance(b, float) else round(b, 4)}")
        two_way = float((net.reverse_arc[route] >= 0).mean())
        stp = np.hypot(*np.diff(prep.P, axis=0).T)
        print(f"  route arcs two-way share {two_way:.2f} | median arc length {np.median(net.length[route]):.0f} m | "
              f"median GPS step {np.median(stp):.1f} m | repeated route arcs {len(route) - len(set(route.tolist()))}")
        print(f"  ref_dist: median {np.median(prep.ref['ref_dist']):.1f} m, p95 {np.percentile(prep.ref['ref_dist'], 95):.1f} m"
              f" | route-arc connectivity {np.mean(net.v[route[:-1]] == net.u[route[1:]]):.3f}")
    return out, prep, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_id")
    ap.add_argument("--probs", default="lightgbm_gbdt")
    ap.add_argument("--partition", default="train")
    a = ap.parse_args()
    analyse(a.trip_id, a.probs, a.partition)
