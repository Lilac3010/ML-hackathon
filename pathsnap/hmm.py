"""HMM / Viterbi route reconstruction on top of classifier probabilities.

Hidden states   : at GPS observation i the state is *which candidate arc* the vehicle is on
                  (the candidate set built in candidates.py, <= CAND_MAX_K arcs).
Emission        : log of the classifier probability that the candidate is the reference arc
                  (floored at `eps` so one very unlikely candidate cannot veto a path).
Transition      : Newson & Krumm style. For previous candidate a (at fraction t_a of its arc)
                  and current candidate b (fraction t_b) the *network* travel distance is
                      d_net = (1-t_a)*len(a) + shortest_path(end(a) -> start(b)) + t_b*len(b)
                  (same arc: (t_b-t_a)*len).  It is compared with the straight-line distance
                  d_gc between the two GPS points:
                      log P(a->b) = - |d_net - d_gc| / beta_i
                  beta_i grows with the time gap, so long GPS gaps tolerate bigger mismatch.
                  Unreachable / absurd pairs get the soft floor -lambda_break (NOT -inf), so
                  the chain can "restart" when the road graph is disconnected or a gap is huge.
                  U-turns (b == reverse of a) and backward motion on an arc are extra penalised.
Decoding        : Viterbi  delta_i(b) = logE_i(b) + max_a [ delta_{i-1}(a) + logT(a->b) ].
Missing cands   : observations with no candidate are skipped by the chain (time gap is measured
                  between the observations actually used) and get the last decoded arc afterwards.

The expensive part (graph search) only depends on the trip, so it is computed once
(`TransitionGeometry`) and decoder parameters can then be tuned cheaply.
"""
from dataclasses import dataclass, asdict
import heapq
import numpy as np

from pathsnap.geo import bearing, ang_diff

SEARCH_BASE_M = 350.0   # graph-search radius; larger for big GPS jumps (rounded up to multiples)
HEAD_WINDOW = 7         # points each side for the long-window direction of travel
HEAD_MIN_MOVE_M = 20.0  # below this displacement over the window, direction is treated as unknown


@dataclass
class HMMParams:
    beta: float = 15.0          # metres: scale of |d_net - d_gc| mismatch penalty
    lam_break: float = 12.0     # log-penalty floor for unreachable / inconsistent transitions
    lam_uturn: float = 6.0      # extra log-penalty for a U-turn (b == reverse(a))
    back_pen: float = 1.0       # extra penalty per (backward metres beyond back_free_m)/beta on one arc
    back_free_m: float = 10.0   # backward jitter tolerated for free (GPS noise); lower = stricter direction
    eps: float = 1e-4           # probability floor used in the emission log
    w_head: float = 0.0         # weight of the long-window direction term (0 = off); see TransitionGeometry
    gap_scale_max: float = 8.0  # beta_i = beta * min(max(dt,1), gap_scale_max)

    def to_dict(self):
        return asdict(self)


class _Graph:
    """Bounded Dijkstra with memoisation (python lists: much faster than numpy scalars)."""
    def __init__(self, net):
        indptr, order = net.out_adj
        self.indptr = indptr.tolist()
        self.order = order.tolist()
        self.v = net.v.tolist()
        self.len = net.length.tolist()
        self.memo = {}

    def ball(self, node, cutoff):
        key = (node, cutoff)
        r = self.memo.get(key)
        if r is not None:
            return r
        dist = {node: 0.0}
        heap = [(0.0, node)]
        indptr, order, v, ln = self.indptr, self.order, self.v, self.len
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, 1e18):
                continue
            for k in range(indptr[n], indptr[n + 1]):
                a = order[k]
                nd = d + ln[a]
                if nd <= cutoff:
                    m = v[a]
                    if nd < dist.get(m, 1e18):
                        dist[m] = nd
                        heapq.heappush(heap, (nd, m))
        self.memo[key] = dist
        return dist

    def path_arcs(self, src, dst, cutoff):
        """Arc-id list of the shortest path src node -> dst node (None if > cutoff)."""
        if src == dst:
            return []
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        indptr, order, v, ln = self.indptr, self.order, self.v, self.len
        while heap:
            d, n = heapq.heappop(heap)
            if n == dst:
                break
            if d > dist.get(n, 1e18):
                continue
            for k in range(indptr[n], indptr[n + 1]):
                a = order[k]
                nd = d + ln[a]
                if nd <= cutoff:
                    m = v[a]
                    if nd < dist.get(m, 1e18):
                        dist[m] = nd
                        pred[m] = a
                        heapq.heappush(heap, (nd, m))
        if dst not in pred:
            return None
        out, n = [], dst
        while n != src:
            a = pred[n]
            out.append(a)
            n = self.u_of(a)
        return out[::-1]

    def u_of(self, a):
        return self._u[a]


class TransitionGeometry:
    """Parameter-independent part of the transitions for one trip."""
    def __init__(self, prep):
        net, cand, P, t = prep.net, prep.cand, prep.P, prep.t
        g = _Graph(net)
        g._u = net.u.tolist()
        self.graph = g
        N = len(P)
        obs = cand["obs"]
        counts = np.bincount(obs, minlength=N)
        offs = np.r_[0, np.cumsum(counts)]
        self.offs, self.counts = offs, counts
        self.used = np.flatnonzero(counts > 0)           # observations that enter the chain
        arc, tt = cand["arc"], cand["t"]
        # long-window direction of travel (+-HEAD_WINDOW points): averages out GPS jitter that makes the
        # short-window heading feature useless at walking speed. Angle (0..180) between it and each
        # candidate arc's direction; 0 when the vehicle barely moved (no direction information).
        idx = np.arange(N)
        dd = P[np.clip(idx + HEAD_WINDOW, 0, N - 1)] - P[np.clip(idx - HEAD_WINDOW, 0, N - 1)]
        moved = np.hypot(dd[:, 0], dd[:, 1]) > HEAD_MIN_MOVE_M
        hl = np.where(moved, bearing(dd[:, 0], dd[:, 1]), np.nan)[obs]
        self.hd_long = np.nan_to_num(ang_diff(hl, net.bearing[arc]), nan=0.0)
        L = net.length
        u, v, rev = net.u, net.v, net.reverse_arc
        self.drt, self.uturn, self.back, self.dgc, self.dt = [None], [None], [None], [0.0], [0.0]
        for s in range(1, len(self.used)):
            i0, i1 = self.used[s - 1], self.used[s]
            ra, rb = slice(offs[i0], offs[i0 + 1]), slice(offs[i1], offs[i1 + 1])
            a, b = arc[ra], arc[rb]
            ta, tb = tt[ra], tt[rb]
            dgc = float(np.hypot(*(P[i1] - P[i0])))
            dt = float(t[i1] - t[i0])
            m = SEARCH_BASE_M * np.ceil(max(dgc * 1.6 + 60.0, SEARCH_BASE_M) / SEARCH_BASE_M)
            ka, kb = len(a), len(b)
            drt = np.full((ka, kb), np.inf)
            back = np.zeros((ka, kb))
            ub = u[b].tolist()
            for x in range(ka):
                ball = g.ball(int(v[a[x]]), m)
                tail = (1.0 - ta[x]) * L[a[x]]
                for y in range(kb):
                    if a[x] == b[y]:
                        fwd = (tb[y] - ta[x]) * L[b[y]]
                        if fwd >= 0:
                            drt[x, y] = fwd
                        else:
                            drt[x, y] = 0.0
                            back[x, y] = -fwd
                    else:
                        sp = ball.get(ub[y])
                        if sp is not None:
                            drt[x, y] = tail + sp + tb[y] * L[b[y]]
            uturn = (rev[a][:, None] == b[None, :]) & (rev[a][:, None] >= 0)
            self.drt.append(drt); self.uturn.append(uturn); self.back.append(back)
            self.dgc.append(dgc); self.dt.append(dt)


def decode(prep, prob, params: HMMParams | None = None, tg: TransitionGeometry | None = None):
    """Viterbi decode one trip.
    prep : PreparedTrip (needs .cand, .P, .t, .net)
    prob : (n_candidate_rows,) classifier probability aligned with prep.cand rows
    Returns dict: pred_arc (N,) arc id per GPS observation, pred_row (N,) candidate row used
    (-1 if imputed), imputed (N,) bool, route_logp."""
    p = params or HMMParams()
    tg = tg or TransitionGeometry(prep)
    N = len(prep.P)
    pred_arc = np.full(N, -1, dtype=np.int64)
    pred_row = np.full(N, -1, dtype=np.int64)
    used = tg.used
    if len(used) == 0:
        return dict(pred_arc=pred_arc, pred_row=pred_row, imputed=np.ones(N, bool), route_logp=float("nan"))
    logE = np.log(np.maximum(np.asarray(prob, float), p.eps)) - p.w_head * tg.hd_long / 180.0
    offs = tg.offs
    delta = logE[offs[used[0]]:offs[used[0] + 1]].copy()
    back_ptr = [None]
    for s in range(1, len(used)):
        i = used[s]
        beta = p.beta * min(max(tg.dt[s], 1.0), p.gap_scale_max)
        drt = tg.drt[s]
        lt = -np.abs(drt - tg.dgc[s]) / beta
        lt = np.where(np.isfinite(drt), lt, -p.lam_break)
        lt = np.maximum(lt, -p.lam_break)
        lt = lt - p.lam_uturn * tg.uturn[s] - p.back_pen * np.maximum(tg.back[s] - p.back_free_m, 0.0) / beta
        tot = delta[:, None] + lt
        bp = np.argmax(tot, axis=0)
        delta = tot[bp, np.arange(tot.shape[1])] + logE[offs[i]:offs[i + 1]]
        back_ptr.append(bp)
    k = int(np.argmax(delta))
    best = float(delta[k])
    for s in range(len(used) - 1, -1, -1):
        i = used[s]
        row = offs[i] + k
        pred_row[i] = row
        pred_arc[i] = prep.cand["arc"][row]
        if s > 0:
            k = int(back_ptr[s][k])
    imputed = pred_row < 0
    last = -1                                       # forward-fill observations without candidates
    first_valid = pred_arc[used[0]]
    for i in range(N):
        if pred_row[i] >= 0:
            last = pred_arc[i]
        else:
            pred_arc[i] = last if last >= 0 else first_valid
    return dict(pred_arc=pred_arc, pred_row=pred_row, imputed=imputed, route_logp=best)


def connect_route(prep, pred_arc, max_search_m=2500.0):
    """Turn per-point arcs into one connected arc sequence (for drawing / API output):
    repeats are collapsed and consecutive arcs that are not adjacent are bridged with the
    shortest network path when one exists within `max_search_m` (else left as a break)."""
    g = _Graph(prep.net)
    g._u = prep.net.u.tolist()
    net = prep.net
    seq = pred_arc[np.r_[True, pred_arc[1:] != pred_arc[:-1]]]
    out, breaks = [int(seq[0])], 0
    for a in seq[1:]:
        prev = out[-1]
        if net.v[prev] != net.u[a]:
            path = g.path_arcs(int(net.v[prev]), int(net.u[a]), max_search_m)
            if path is None:
                breaks += 1
            else:
                out.extend(path)
        out.append(int(a))
    return out, breaks
