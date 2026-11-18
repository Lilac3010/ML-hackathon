"""Reference labelling: turn the .route (a short list of arcs) into one reference arc
per GPS point.

The route is an ordered list of arcs, the track is an ordered list of points, so we do a
monotone alignment (a Viterbi over route positions): each point is assigned a route index
that never decreases, may stay or jump forward a few arcs (vehicle speed), and the cost is
the (capped) distance from the point to that arc. This is *only* used to build training /
evaluation labels - never at inference."""
import numpy as np
from scipy.ndimage import minimum_filter1d

import config
from pathsnap.geo import seg_distance

SKIP_PEN = 0.2   # metres-equivalent penalty per route arc skipped between consecutive points


def _wmin_simple(a, K):
    # reliable O(M*log K) doubling construction of the sliding minimum over [j-K, j]
    out = a.copy()
    span = 1
    while span < K + 1:
        step = min(span, K + 1 - span)
        sh = np.concatenate([np.full(step, np.inf), out[:-step]]) if step < len(out) else np.full_like(out, np.inf)
        out = np.minimum(out, sh)
        span += step
    return out


def align_route(P, t, net, route, cap=None, kbase=15):
    """P (N,2) metres, t (N,) seconds, route (M,) arc ids.
    Returns ref_idx (N,) index into route, ref_arc (N,), ref_dist (N,) metres."""
    cap = cap or config.REF_CAP_M
    N, M = len(P), len(route)
    a, b = net.a[route], net.b[route]
    dt = np.diff(t, prepend=t[0])
    Kmax = np.clip(kbase * np.maximum(dt, 1.0), kbase, M).astype(int)

    dp = np.empty((N, M), dtype=np.float32)

    def cost_row(i):
        d, _, _ = seg_distance(np.broadcast_to(P[i], (M, 2)), a, b)
        return d

    d0 = cost_row(0)
    dp[0] = np.minimum(d0, cap)
    j = np.arange(M, dtype=np.float64)
    for i in range(1, N):
        K = int(Kmax[i])
        prev = dp[i - 1].astype(np.float64)
        # min_k prev[j-k] + SKIP_PEN*k  =  SKIP_PEN*j + min_{m in [j-K, j]} (prev[m] - SKIP_PEN*m)
        best = SKIP_PEN * j + _wmin_simple(prev - SKIP_PEN * j, K)
        d = cost_row(i)
        dp[i] = np.minimum(d, cap) + best
    # backtrack
    ref_idx = np.empty(N, dtype=np.int64)
    ref_idx[-1] = int(np.argmin(dp[-1]))
    for i in range(N - 1, 0, -1):
        jj = ref_idx[i]
        K = int(Kmax[i])
        lo = max(0, jj - K)
        cand = dp[i - 1, lo:jj + 1].astype(np.float64) + SKIP_PEN * (jj - np.arange(lo, jj + 1))
        ref_idx[i - 1] = lo + int(np.argmin(cand))
    ref_dist, _, _ = seg_distance(P, a[ref_idx], b[ref_idx])
    return ref_idx, route[ref_idx], ref_dist
