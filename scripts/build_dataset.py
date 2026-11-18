"""Build the candidate feature tables for every trip in train/ and student_test/.

Outputs (data/processed/):
  <partition>_features.parquet   one row per (GPS point, candidate arc)   <- for modelling
  <partition>_points.parquet     one row per GPS point (reference arc, miss flag, ...)
  <partition>_trips.csv          one row per trip (sizes, label coverage, miss rate, timing)

Usage:  python scripts/build_dataset.py            (all partitions)
        python scripts/build_dataset.py train
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from pathsnap.io import list_trips, trip_dir
from pathsnap.pipeline import PreparedTrip


def one_trip(partition, tid):
    t0 = time.time()
    pt = PreparedTrip(trip_dir(partition, tid), with_labels=True)
    feats, pts = pt.table(), pt.point_table()
    v = pts[pts.ref_valid]
    row = dict(partition=partition, trip_id=tid, n_obs=len(pts), n_nodes=pt.net.n_nodes,
               n_arcs=pt.net.n_arcs, n_route_arcs=len(pt.route), n_cand_rows=len(feats),
               mean_cand_per_obs=len(feats) / len(pts), pct_ref_valid=pts.ref_valid.mean(),
               ref_dist_median_m=pts.ref_dist_m.median(), ref_dist_p95_m=pts.ref_dist_m.quantile(.95),
               miss_rate_valid=1 - v.ref_in_cand.mean() if len(v) else np.nan,
               miss_rate_all=1 - pts.ref_in_cand.mean(), pts_without_candidates=int((pts.n_cand == 0).sum()),
               seconds=time.time() - t0)
    return feats, pts, row


def main(parts):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for part in parts:
        ids = list_trips(part)
        t0 = time.time()
        out = Parallel(n_jobs=-1, verbose=0)(delayed(one_trip)(part, t) for t in ids)
        feats = pd.concat([o[0] for o in out], ignore_index=True)
        pts = pd.concat([o[1] for o in out], ignore_index=True)
        trips = pd.DataFrame([o[2] for o in out])
        feats.to_parquet(config.PROCESSED_DIR / f"{part}_features.parquet", index=False)
        pts.to_parquet(config.PROCESSED_DIR / f"{part}_points.parquet", index=False)
        trips.to_csv(config.PROCESSED_DIR / f"{part}_trips.csv", index=False)
        print(f"[{part}] {len(ids)} trips, {len(pts):,} GPS points, {len(feats):,} candidate rows "
              f"in {time.time() - t0:.0f}s | valid-ref {pts.ref_valid.mean():.3f} | "
              f"miss(valid) {1 - pts[pts.ref_valid].ref_in_cand.mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(config.PARTITIONS))
