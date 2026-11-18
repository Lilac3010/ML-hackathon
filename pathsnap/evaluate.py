"""Shared metric + CV-fold definitions. EVERYONE (modelling, HMM, API) uses these so the
numbers in the report are comparable.

Conventions
-----------
* Only rows/points with ref_valid == True are evaluated (reference arc within
  config.REF_VALID_M of the GPS point; otherwise the reference label is not trustworthy).
* Candidate-level metrics (precision / recall / F1 / log loss) treat every candidate row
  as a binary example (label 1 = it is the reference arc).
* Observation-level Top-1: for each GPS point pick the candidate with the highest score;
  correct if it is the reference arc. If the reference arc is not in the candidate set the
  point counts as WRONG (conservative; the miss rate is reported separately).
* Route continuity: share of predicted ARC CHANGES that are drivable (<= 3 forward arcs, no
  U-turn onto the twin), pooled over all changes of all trips. Always report the number of arc
  changes with it (ratio to the reference route's changes; ideal 1.0).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score

N_FOLDS = 5


def assign_folds(trip_ids, n_folds=N_FOLDS, seed=42) -> dict:
    """Deterministic trip -> fold map (group-aware CV: a trip is never split across folds)."""
    ids = np.array(sorted(set(trip_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {t: i % n_folds for i, t in enumerate(ids)}


def top1_pick(table: pd.DataFrame, score) -> pd.DataFrame:
    """One row per GPS point: the highest-scoring candidate (columns kept from `table`)."""
    t = table.assign(_s=np.asarray(score))
    idx = t.groupby(["trip_id", "obs_idx"], sort=False)["_s"].idxmax()
    return t.loc[idx.to_numpy()]


def observation_top1(table: pd.DataFrame, score) -> dict:
    """Observation-level Top-1 accuracy over valid points (misses count as wrong)."""
    v = table[table["ref_valid"]]
    s = np.asarray(score)[table["ref_valid"].to_numpy()]
    pick = top1_pick(v, s)
    n_points = v.groupby(["trip_id", "obs_idx"]).ngroups
    return dict(top1_accuracy=float(pick["label"].sum() / n_points), n_points=int(n_points))


def candidate_metrics(table: pd.DataFrame, prob, threshold=0.5) -> dict:
    """Candidate-level binary metrics on valid rows. Macro = mean over the two classes."""
    m = table["ref_valid"].to_numpy()
    y = table["label"].to_numpy()[m]
    p = np.clip(np.asarray(prob, float)[m], 1e-7, 1 - 1e-7)
    yhat = (p >= threshold).astype(int)
    return dict(accuracy=accuracy_score(y, yhat),
                precision_macro=precision_score(y, yhat, average="macro", zero_division=0),
                recall_macro=recall_score(y, yhat, average="macro", zero_division=0),
                f1_macro=f1_score(y, yhat, average="macro", zero_division=0),
                log_loss=log_loss(y, p, labels=[0, 1]))


def miss_rate(points: pd.DataFrame) -> float:
    """Share of valid GPS points whose reference arc is absent from the candidate set."""
    v = points[points["ref_valid"]]
    return float(1.0 - v["ref_in_cand"].mean())


CONT_MAX_HOPS = 3   # reference route scores 0.908 / 0.993 / 0.996 with 1 / 3 / 5 hops (student_test)


def _reachable(net, x, y, max_hops):
    """Can a vehicle on arc x get onto arc y within max_hops forward arcs, never turning back
    onto the twin of the arc it is on?"""
    frontier, seen = [x], {x}
    for _ in range(max_hops):
        nxt = []
        for f in frontier:
            for s in net.successors(f):
                if s == net.reverse_arc[f] or s in seen:
                    continue
                if s == y:
                    return True
                seen.add(s)
                nxt.append(s)
        frontier = nxt
    return False


def continuity_counts(pred_arcs, net, max_hops=CONT_MAX_HOPS) -> dict:
    """Raw counts behind route_continuity (sum these over trips, then divide = pooled score).

    Only ARC CHANGES are judged (consecutive observations on the same arc say nothing about
    continuity, and at ~1 Hz they are 75-85% of all pairs). A change x -> y is good if y is
    reachable from x within `max_hops` forward arcs without a U-turn; a flip onto the reverse
    twin of the same road (x -> reverse(x)) is always bad. -1 (no prediction) breaks the
    sequence: pairs across it are not judged.
    Returns good, n_changes, n_uturn."""
    a = np.asarray(pred_arcs, dtype=np.int64)
    good = n_ch = n_ut = 0
    for x, y in zip(a[:-1], a[1:]):
        if x < 0 or y < 0 or x == y:
            continue
        n_ch += 1
        if net.reverse_arc[x] == y:
            n_ut += 1
        elif _reachable(net, x, y, max_hops):
            good += 1
    return dict(good=good, n_changes=n_ch, n_uturn=n_ut)


def continuity_from_counts(counts) -> float:
    """Pooled continuity from one counts dict or an iterable of them (NaN if no arc changes)."""
    if isinstance(counts, dict):
        counts = [counts]
    good = sum(c["good"] for c in counts)
    n = sum(c["n_changes"] for c in counts)
    return good / n if n else float("nan")


def route_continuity(pred_arcs, net, max_hops=CONT_MAX_HOPS) -> float:
    """Share of predicted ARC CHANGES that are drivable (see continuity_counts).
    NaN when the prediction never changes arc - always report it next to the number of arc
    changes (or their ratio to the reference route's), otherwise a constant prediction looks
    perfect. This is a plausibility check, not accuracy: an HMM decoder enforces connectivity
    by construction, so judge it on Top-1 / route F1 against the reference."""
    return continuity_from_counts(continuity_counts(pred_arcs, net, max_hops))


def step_consistency(pred_arcs, net) -> float:
    """LEGACY metric (was called route_continuity before 2026-11-18). Share of ALL consecutive
    pairs that are the same arc or a direct hand-over (v[i] == u[j]). Inflated: same-arc pairs
    dominate (a constant prediction scores 1.0), U-turns onto the twin count as fine, and
    passing through a short arc between two fixes counts as broken. Do NOT report as accuracy."""
    a = np.asarray(pred_arcs)
    ok = (a[:-1] >= 0) & (a[1:] >= 0)
    a0, a1 = a[:-1][ok], a[1:][ok]
    cont = (a0 == a1) | (net.v[a0] == net.u[a1])
    return float(cont.mean()) if len(cont) else float("nan")


def sequence_scores(pred_arcs, ref_arcs, valid) -> dict:
    """Sequence-level agreement on valid points: collapse repeats to a route (arc sequence)
    and compare as arc sets (precision/recall/F1) plus pointwise accuracy."""
    pred = np.asarray(pred_arcs)[valid]
    ref = np.asarray(ref_arcs)[valid]

    def collapse(x):
        return x[np.r_[True, x[1:] != x[:-1]]] if len(x) else x

    ps, rs = set(collapse(pred).tolist()), set(collapse(ref).tolist())
    tp = len(ps & rs)
    prec = tp / max(len(ps), 1)
    rec = tp / max(len(rs), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(pointwise_accuracy=float((pred == ref).mean()) if len(pred) else float("nan"),
                route_arc_precision=prec, route_arc_recall=rec, route_arc_f1=f1)
