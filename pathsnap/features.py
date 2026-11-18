"""Per-(GPS point, candidate arc) feature engineering.

Deliberately NOT used as features: absolute coordinates (trips are from different cities,
that would just memorise places) and anything from the reference route."""
import numpy as np

from pathsnap.geo import seg_distance, bearing, ang_diff

# order matters: this is the column order the model is trained on (saved with the artifact)
FEATURE_COLS = [
    # candidate <-> point geometry
    "dist_m", "lat_off_m", "proj_frac", "arc_len_m", "along_m", "to_end_m",
    # relative position inside the candidate set
    "rank", "dist_gap_to_best", "dist_ratio_to_best", "n_cand", "has_reverse",
    # heading agreement (direction of travel vs arc direction)
    "heading_diff", "cos_heading", "heading_valid",
    # kinematics of the GPS trace
    "speed_mps", "accel", "turn_rate", "dt_prev", "dt_next",
    # temporal support: do neighbouring GPS points also lie near this arc?
    "d_prev1", "d_next1", "d_prev3", "d_next3", "support_mean",
    # topology of the arc
    "start_out_deg", "end_out_deg", "start_in_deg", "end_in_deg",
]
ID_COLS = ["trip_id", "obs_idx", "arc_id"]
LABEL_COLS = ["label", "ref_valid"]   # present only when the route is available


def point_kinematics(P, t):
    """Heading (deg, NaN if standing still), speed (m/s), accel, turn rate from the trace."""
    N = len(P)
    idx = np.arange(N)
    i0 = np.clip(idx - 2, 0, N - 1)
    i1 = np.clip(idx + 2, 0, N - 1)
    d = P[i1] - P[i0]
    span_t = np.maximum(t[i1] - t[i0], 1e-3)
    dist = np.hypot(d[:, 0], d[:, 1])
    speed = dist / span_t
    head = bearing(d[:, 0], d[:, 1])
    head_valid = dist > 3.0                    # <3 m over the window = effectively stationary
    head = np.where(head_valid, head, np.nan)
    prev_i = np.clip(idx - 1, 0, N - 1)
    next_i = np.clip(idx + 1, 0, N - 1)
    dt_prev = np.maximum(t - t[prev_i], 0.0)
    dt_next = np.maximum(t[next_i] - t, 0.0)
    sp_prev = np.r_[speed[0], speed[:-1]]
    accel = (speed - sp_prev) / np.maximum(dt_prev, 1.0)
    hp = np.r_[head[0], head[:-1]]
    turn = np.where(np.isnan(head) | np.isnan(hp), 0.0, ang_diff(head, hp)) / np.maximum(dt_prev, 1.0)
    return dict(head=head, head_valid=head_valid, speed=speed, accel=accel, turn=turn,
                dt_prev=dt_prev, dt_next=dt_next)


def build_features(P, t, net, cand):
    """Return dict of column arrays (len = number of candidate rows) for FEATURE_COLS."""
    obs, arc = cand["obs"], cand["arc"]
    N = len(P)
    kin = point_kinematics(P, t)
    n_cand = np.bincount(obs, minlength=N)

    first = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    counts = np.diff(np.r_[first, len(obs)])
    best = np.repeat(cand["dist"][first], counts)           # rows sorted by (obs, dist)

    L = net.length[arc]
    head = kin["head"][obs]
    hd = ang_diff(head, net.bearing[arc])
    hvalid = ~np.isnan(hd)
    hd = np.where(hvalid, hd, 90.0)                          # neutral value when heading unknown
    a, b = net.a[arc], net.b[arc]

    def neighbour_dist(off):
        j = np.clip(obs + off, 0, N - 1)
        d, _, _ = seg_distance(P[j], a, b)
        return d

    dp1, dn1, dp3, dn3 = (neighbour_dist(o) for o in (-1, 1, -3, 3))
    support = (cand["dist"] + dp1 + dn1 + dp3 + dn3) / 5.0

    f = {
        "dist_m": cand["dist"], "lat_off_m": cand["lat"], "proj_frac": cand["t"],
        "arc_len_m": L, "along_m": cand["t"] * L, "to_end_m": (1 - cand["t"]) * L,
        "rank": cand["rank"].astype(float),
        "dist_gap_to_best": cand["dist"] - best,
        "dist_ratio_to_best": cand["dist"] / np.maximum(best, 0.5),
        "n_cand": n_cand[obs].astype(float),
        "has_reverse": (net.reverse_arc[arc] >= 0).astype(float),
        "heading_diff": hd, "cos_heading": np.cos(np.radians(hd)) * hvalid,
        "heading_valid": hvalid.astype(float),
        "speed_mps": kin["speed"][obs], "accel": kin["accel"][obs], "turn_rate": kin["turn"][obs],
        "dt_prev": kin["dt_prev"][obs], "dt_next": kin["dt_next"][obs],
        "d_prev1": dp1, "d_next1": dn1, "d_prev3": dp3, "d_next3": dn3, "support_mean": support,
        "start_out_deg": net.out_deg[net.u[arc]].astype(float),
        "end_out_deg": net.out_deg[net.v[arc]].astype(float),
        "start_in_deg": net.in_deg[net.u[arc]].astype(float),
        "end_in_deg": net.in_deg[net.v[arc]].astype(float),
    }
    assert list(f) == FEATURE_COLS
    return f
