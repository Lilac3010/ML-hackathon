"""Robustness / latency check of the API entry point (run_trip) on EVERY trip of a partition.

   python scripts/stress_test.py                 # train partition, lightgbm_gbdt model
   python scripts/stress_test.py student_test

Records per trip: crash (with traceback), seconds per stage, peak memory, route breaks, imputed points,
classifier-only vs HMM Top-1 and continuity. NOTE: for the train partition the model has seen these trips,
so the accuracy columns only show the decoder's behaviour, not generalisation - the point of this script is
"does it run on everything, and how fast". Output: results/stress_test_<partition>.csv
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

import config
from pathsnap.io import list_trips, trip_dir
from pathsnap.inference import run_trip, load_hmm_params

try:
    import psutil
    _proc = psutil.Process()
    rss_mb = lambda: _proc.memory_info().rss / 1e6
except ImportError:  # memory column just stays empty
    rss_mb = lambda: float("nan")


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "train"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "lightgbm_gbdt"
    art = joblib.load(config.ROOT / "models" / f"{model_name}.joblib")
    predict = lambda X: art["model"].predict_proba(X)[:, 1]
    params = load_hmm_params()
    rows = []
    for tid in list_trips(part):
        row = dict(trip_id=tid)
        t0 = time.perf_counter()
        try:
            o = run_trip(trip_dir(part, tid), predict, params, evaluate=True)
            m = o["metrics"]
            row.update(ok=True, n_obs=o["n_obs"], seconds=o["timing_s"]["total"],
                       t_prepare=o["timing_s"]["prepare"], t_model=o["timing_s"]["model"],
                       t_decode=o["timing_s"]["decode"], rss_mb=rss_mb(), breaks=o["route_breaks"],
                       imputed=o["n_imputed"], n_valid=m["n_valid_points"],
                       clf_top1=m["classifier_only"]["top1_accuracy"], hmm_top1=m["hmm"]["top1_accuracy"],
                       cont_clf=o["continuity_clf"], cont_hmm=o["continuity_hmm"],
                       payload_kb=len(str(o)) / 1024)
        except Exception as e:  # keep going, report at the end
            row.update(ok=False, seconds=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        rows.append(row)
        print(row["trip_id"], "OK " if row["ok"] else "FAIL", f"{row['seconds']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / f"stress_test_{part}.csv"
    df.to_csv(out, index=False)
    print(f"\n{int(df.ok.sum())}/{len(df)} trips ran without error -> {out}")
    ok = df[df.ok]
    if len(ok):
        print(ok[["seconds", "rss_mb", "breaks", "imputed", "clf_top1", "hmm_top1", "payload_kb"]].describe().round(2).to_string())
        print("\nslowest:\n", ok.nlargest(5, "seconds")[["trip_id", "n_obs", "seconds", "t_prepare", "t_decode", "rss_mb"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
