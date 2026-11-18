"""The single entry point the API calls:  run_trip(trip_dir, predict_proba, ...)

predict_proba : callable taking a DataFrame with exactly features.FEATURE_COLS (in this order)
                and returning a 1-D array of P(candidate is the true arc). Person 2's final
                model is wrapped as e.g.  lambda X: model.predict_proba(X)[:, 1]
evaluate      : server-side only. When True the reference route is loaded INSIDE this function
                to score the prediction; reference arcs are never returned or fed to the model
                (features/candidates/decoder never read them).
"""
import json
import time
import numpy as np

import config
from pathsnap.features import FEATURE_COLS
from pathsnap.pipeline import PreparedTrip
from pathsnap.hmm import HMMParams, TransitionGeometry, decode, connect_route
from pathsnap.evaluate import route_continuity, sequence_scores


def load_hmm_params(path=None) -> HMMParams:
    path = path or (config.RESULTS_DIR / "hmm_best_params.json")
    try:
        return HMMParams(**json.load(open(path)))
    except FileNotFoundError:
        return HMMParams()


def arcs_to_polyline(prep, arcs):
    """[[lat, lon], ...] following the arcs in order (Folium / Leaflet order)."""
    net = prep.net
    arcs = np.asarray(arcs)
    if len(arcs) == 0:
        return []
    xy = np.vstack([net.a[arcs[0]][None, :], net.b[arcs]])
    ll = prep.proj.to_lonlat(xy)
    return ll[:, ::-1].tolist()


def run_trip(tdir, predict_proba, params: HMMParams | None = None, evaluate=False):
    t0 = time.perf_counter()
    prep = PreparedTrip(tdir, with_labels=evaluate)
    t_prep = time.perf_counter()
    X = prep.table()[FEATURE_COLS]
    prob = np.asarray(predict_proba(X), float)
    t_model = time.perf_counter()
    res = decode(prep, prob, params or load_hmm_params())
    pred = res["pred_arc"]
    route_arcs, breaks = connect_route(prep, pred)
    t_dec = time.perf_counter()

    # classifier-only prediction (for the clf vs clf+HMM comparison shown in the UI/report)
    c = prep.cand
    order = np.lexsort((-prob, c["obs"]))
    first = np.r_[0, np.flatnonzero(np.diff(c["obs"][order])) + 1]
    clf = np.full(len(prep.P), -1, dtype=np.int64)
    clf[c["obs"][order][first]] = c["arc"][order][first]

    out = dict(
        trip_id=prep.trip_id,
        n_obs=int(len(prep.P)),
        gps=prep.track[:, [1, 0]].tolist(),                      # [[lat, lon], ...]
        pred_arc_per_obs=pred.tolist(),
        pred_route_arcs=route_arcs,
        pred_route_polyline=arcs_to_polyline(prep, route_arcs),
        route_breaks=int(breaks),
        n_imputed=int(res["imputed"].sum()),
        continuity_hmm=route_continuity(pred, prep.net),      # share of arc changes that are drivable
        continuity_clf=route_continuity(clf, prep.net),
        arc_changes_hmm=int(np.sum((pred[:-1] != pred[1:]) & (pred[:-1] >= 0) & (pred[1:] >= 0))),
        arc_changes_clf=int(np.sum((clf[:-1] != clf[1:]) & (clf[:-1] >= 0) & (clf[1:] >= 0))),
        timing_s=dict(prepare=t_prep - t0, model=t_model - t_prep, decode=t_dec - t_model,
                      total=t_dec - t0),
    )
    if evaluate:
        pt = prep.point_table()
        valid = pt.ref_valid.to_numpy()
        ref = pt.ref_arc.to_numpy()
        h = sequence_scores(pred, ref, valid)
        k = sequence_scores(clf, ref, valid)
        out["metrics"] = dict(
            n_valid_points=int(valid.sum()),
            hmm=dict(top1_accuracy=h["pointwise_accuracy"], route_arc_f1=h["route_arc_f1"]),
            classifier_only=dict(top1_accuracy=k["pointwise_accuracy"], route_arc_f1=k["route_arc_f1"]),
        )
    return out
