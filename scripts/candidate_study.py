"""Choose the candidate search radius R and max-K from the training trips.

For each (R, K) we measure the candidate MISS RATE: the share of GPS points (with a valid
reference arc) whose reference arc is NOT among the K nearest arcs within R metres,
and the average candidate-set size (cost for the models / HMM).
Needs: python scripts/build_dataset.py train   (for the reference arcs)
Output: results/candidate_study.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from pathsnap.io import list_trips, trip_dir, load_trip
from pathsnap.geo import Projector
from pathsnap.network import RoadNetwork
from pathsnap.candidates import generate_candidates

RADII = [15, 20, 30, 40, 50, 75, 100]
KS = [3, 5, 8, 10, 15, 20, 30]


def study_trip(tid, pts):
    tr = load_trip(trip_dir("train", tid))
    pr = Projector(tr.track[:, 0].mean(), tr.track[:, 1].mean())
    P = pr.to_xy(tr.track[:, 0], tr.track[:, 1])
    net = RoadNetwork(pr.to_xy(tr.nodes[:, 0], tr.nodes[:, 1]), tr.arcs)
    c = generate_candidates(P, net, radius=max(RADII), radius_max=max(RADII), max_k=max(KS))
    p = pts[pts.trip_id == tid].set_index("obs_idx")
    ref = p["ref_arc"].to_numpy()[c["obs"]]
    valid = p["ref_valid"].to_numpy()[c["obs"]]
    is_ref = (c["arc"] == ref) & valid
    n_valid = int(p["ref_valid"].sum())
    rows = []
    for R in RADII:
        for K in KS:
            m = (c["dist"] <= R) & (c["rank"] < K)
            rows.append((R, K, n_valid, int(is_ref[m].sum()), int(m.sum()), len(P)))
    return rows


def main():
    pts = pd.read_parquet(config.PROCESSED_DIR / "train_points.parquet", columns=["trip_id", "obs_idx", "ref_arc", "ref_valid"])
    out = Parallel(n_jobs=-1)(delayed(study_trip)(t, pts[pts.trip_id == t]) for t in list_trips("train"))
    df = pd.DataFrame([r for o in out for r in o], columns=["radius_m", "max_k", "n_valid", "n_hit", "n_rows", "n_obs"])
    g = df.groupby(["radius_m", "max_k"]).sum()
    res = pd.DataFrame({"miss_rate": 1 - g.n_hit / g.n_valid, "cands_per_point": g.n_rows / g.n_obs}).reset_index()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(config.RESULTS_DIR / "candidate_study.csv", index=False)
    print(res.pivot(index="radius_m", columns="max_k", values="miss_rate").round(4).to_string())
    print()
    print(res.pivot(index="radius_m", columns="max_k", values="cands_per_point").round(1).to_string())


if __name__ == "__main__":
    main()
