"""Generate an interactive Folium map of one trip (report / EDA; includes the reference route).

   python scripts/make_map.py student_test 00000007
   python scripts/make_map.py train 00000054 --model lightgbm_gbdt --no-network

Writes results/maps/<partition>_<trip>.html. Open it in a browser; take a screenshot for the PDF.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np

import config
from pathsnap.io import trip_dir
from pathsnap.pipeline import PreparedTrip
from pathsnap.features import FEATURE_COLS
from pathsnap.hmm import TransitionGeometry, decode, connect_route
from pathsnap.inference import load_hmm_params
from pathsnap.viz import build_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("partition")
    ap.add_argument("trip_id")
    ap.add_argument("--model", default="lightgbm_gbdt")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    art = joblib.load(config.ROOT / "models" / f"{a.model}.joblib")
    prep = PreparedTrip(trip_dir(a.partition, a.trip_id), with_labels=True)
    prob = art["model"].predict_proba(prep.table()[FEATURE_COLS])[:, 1]
    res = decode(prep, prob, load_hmm_params(), TransitionGeometry(prep))
    c = prep.cand
    order = np.lexsort((-prob, c["obs"]))
    first = np.r_[0, np.flatnonzero(np.diff(c["obs"][order])) + 1]
    clf = np.full(len(prep.P), -1, dtype=np.int64)
    clf[c["obs"][order][first]] = c["arc"][order][first]
    route, _ = connect_route(prep, res["pred_arc"])
    m = build_map(prep, pred_arc=res["pred_arc"], route_arcs=route, clf_arc=clf, reference=True,
                  show_network=not a.no_network)
    out = Path(a.out) if a.out else config.RESULTS_DIR / "maps" / f"{a.partition}_{a.trip_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    main()
