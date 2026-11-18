"""Candidate generation: for every GPS point, the nearest directed arcs.

Both directions of a two-way road are separate candidates (the reference route uses
directed arc ids), so heading is what later tells them apart."""
import numpy as np
import shapely
from shapely.strtree import STRtree

import config
from pathsnap.geo import seg_distance


def generate_candidates(P, net, radius=None, radius_max=None, max_k=None):
    """P: (N,2) projected GPS points (m). Returns dict of flat arrays, one row per
    (point, candidate), sorted by (point, distance):
    obs, arc, dist, t (fraction along arc), lat (signed lateral offset), rank.
    Points with no arc within `radius` get the radius doubled up to `radius_max`."""
    radius = radius or config.CAND_RADIUS_M
    radius_max = radius_max or config.CAND_RADIUS_MAX_M
    max_k = max_k or config.CAND_MAX_K

    # only index arcs near the track (huge networks otherwise)
    pad = radius_max
    lo, hi = P.min(0) - pad, P.max(0) + pad
    amin = np.minimum(net.a, net.b); amax = np.maximum(net.a, net.b)
    keep = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                    (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
    segs = shapely.linestrings(np.stack([net.a[keep], net.b[keep]], axis=1))
    tree = STRtree(segs)
    pts = shapely.points(P)

    obs_l, arc_l = [], []
    todo = np.arange(len(P))
    r = radius
    while len(todo) and r <= radius_max:
        pi, si = tree.query(pts[todo], predicate="dwithin", distance=r)
        obs_l.append(todo[pi]); arc_l.append(keep[si])
        found = np.zeros(len(P), bool); found[todo[pi]] = True
        todo = todo[~found[todo]]
        r *= 2.0
    if not obs_l:
        return _empty()
    obs = np.concatenate(obs_l); arc = np.concatenate(arc_l)
    d, t, lat = seg_distance(P[obs], net.a[arc], net.b[arc])
    order = np.lexsort((d, obs))
    obs, arc, d, t, lat = obs[order], arc[order], d[order], t[order], lat[order]
    # rank within point, keep top-K
    first = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    counts = np.diff(np.r_[first, len(obs)])
    rank = np.arange(len(obs)) - np.repeat(first, counts)
    m = rank < max_k
    return dict(obs=obs[m], arc=arc[m], dist=d[m], t=t[m], lat=lat[m], rank=rank[m])


def _empty():
    z = np.zeros(0)
    return dict(obs=z.astype(int), arc=z.astype(int), dist=z, t=z, lat=z, rank=z.astype(int))
