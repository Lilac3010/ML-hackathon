"""Raw-data quality scan over every trip (missing values, validity, ordering, network sanity).
   python scripts/data_quality_scan.py      -> results/data_quality_<partition>.csv + printed summary
Only reads raw files; nothing is modified."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from pathsnap.io import list_trips, trip_dir


def scan(partition, tid):
    d = trip_dir(partition, tid)
    r = dict(partition=partition, trip_id=tid)
    tr = np.loadtxt(d / f"{tid}.track", ndmin=2)
    nd = np.loadtxt(d / f"{tid}.nodes", ndmin=2)
    ar = np.loadtxt(d / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    rt = np.loadtxt(d / f"{tid}.route", dtype=np.int64, ndmin=1)
    r.update(track_cols=tr.shape[1], n_obs=len(tr), n_nodes=len(nd), n_arcs=len(ar), n_route=len(rt))
    r["nan_track"] = int(np.isnan(tr).sum())
    r["nan_nodes"] = int(np.isnan(nd).sum())
    lon, lat, t = tr[:, 0], tr[:, 1], tr[:, 2]
    r["bad_lonlat_track"] = int(((np.abs(lon) > 180) | (np.abs(lat) > 90) | ((lon == 0) & (lat == 0))).sum())
    r["bad_lonlat_nodes"] = int(((np.abs(nd[:, 0]) > 180) | (np.abs(nd[:, 1]) > 90)).sum())
    dt = np.diff(t)
    r["dt_nonpositive"] = int((dt <= 0).sum())          # out-of-order / duplicate timestamps
    r["pct_dt_1s"] = float((dt == 1).mean())
    r["n_gaps_gt5s"] = int((dt > 5).sum())
    r["max_gap_s"] = float(dt.max())
    r["dup_consecutive_points"] = int(((np.diff(lon) == 0) & (np.diff(lat) == 0)).sum())
    # network / route consistency
    r["arc_node_out_of_range"] = int(((ar < 0) | (ar >= len(nd))).any(axis=1).sum())
    r["self_loops"] = int((ar[:, 0] == ar[:, 1]).sum())
    r["duplicate_arcs"] = int(len(ar) - len({(a, b) for a, b in ar.tolist()}))
    r["isolated_nodes"] = int(len(nd) - len(np.unique(ar)))
    r["route_arc_out_of_range"] = int(((rt < 0) | (rt >= len(ar))).sum())
    ok = rt[(rt >= 0) & (rt < len(ar))]
    r["route_disconnected_steps"] = int((ar[ok[:-1], 1] != ar[ok[1:], 0]).sum()) if len(ok) > 1 else 0
    # track inside the network's bounding box?
    r["track_outside_network_bbox"] = int(((lon < nd[:, 0].min()) | (lon > nd[:, 0].max()) |
                                           (lat < nd[:, 1].min()) | (lat > nd[:, 1].max())).sum())
    return r


def main():
    rows = [scan(p, t) for p in config.PARTITIONS for t in list_trips(p)]
    df = pd.DataFrame(rows)
    for p, g in df.groupby("partition"):
        g.to_csv(config.RESULTS_DIR / f"data_quality_{p}.csv", index=False)
    pd.set_option("display.width", 200)
    for p, g in df.groupby("partition"):
        print(f"\n=== {p}: {len(g)} trips, {int(g.n_obs.sum()):,} GPS points ===")
        for c in ["nan_track", "nan_nodes", "bad_lonlat_track", "bad_lonlat_nodes", "dt_nonpositive",
                  "dup_consecutive_points", "arc_node_out_of_range", "self_loops", "duplicate_arcs",
                  "route_arc_out_of_range", "route_disconnected_steps", "track_outside_network_bbox"]:
            print(f"  {c:28s} total {int(g[c].sum()):>8,}   trips affected {int((g[c] > 0).sum())}")
        print(f"  trips with a time gap >5 s: {int((g.n_gaps_gt5s > 0).sum())}  (max gap {g.max_gap_s.max():.0f} s, "
              f"{int(g.n_gaps_gt5s.sum())} gaps total) | share of steps exactly 1 s: "
              f"{np.average(g.pct_dt_1s, weights=g.n_obs):.3f}")
        print(f"  isolated nodes: {int(g.isolated_nodes.sum()):,} across {int((g.isolated_nodes > 0).sum())} trips")
        print(f"  columns in .track: {sorted(g.track_cols.unique())}")


if __name__ == "__main__":
    main()
