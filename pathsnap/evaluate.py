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


def route_continuity(pred_arcs, net) -> float:
    """Share of consecutive observation pairs whose predicted arcs are physically
    consistent: same arc, or arc j starts where arc i ends (v[i] == u[j]).
    pred_arcs: (N,) predicted arc id per observation (-1 = no prediction, skipped)."""
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
