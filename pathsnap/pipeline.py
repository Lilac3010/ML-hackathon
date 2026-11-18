"""End-to-end trip -> candidate feature table. The same class is used for training
tables (with_labels=True) and for the API at inference time (with_labels=False)."""
import numpy as np
import pandas as pd

import config
from pathsnap.io import load_trip
from pathsnap.geo import Projector
from pathsnap.network import RoadNetwork
from pathsnap.candidates import generate_candidates
from pathsnap.features import build_features, FEATURE_COLS
from pathsnap.labels import align_route


class PreparedTrip:
    """Everything the decoder / API needs for one trip (no labels unless requested)."""

    def __init__(self, tdir, with_labels=False, radius=None, max_k=None):
        trip = load_trip(tdir, with_route=with_labels)
        self.trip_id = trip.trip_id
        self.track = trip.track
        self.t = trip.track[:, 2]
        self.proj = Projector(trip.track[:, 0].mean(), trip.track[:, 1].mean())
        self.P = self.proj.to_xy(trip.track[:, 0], trip.track[:, 1])
        self.net = RoadNetwork(self.proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1]), trip.arcs)
        self.cand = generate_candidates(self.P, self.net, radius=radius, max_k=max_k)
        self.feat = build_features(self.P, self.t, self.net, self.cand)
        self.route = trip.route
        self.ref = None
        if with_labels:
            idx, arc, dist = align_route(self.P, self.t, self.net, trip.route)
            self.ref = dict(ref_idx=idx, ref_arc=arc, ref_dist=dist,
                            ref_valid=dist <= config.REF_VALID_M)

    def table(self) -> pd.DataFrame:
        """Candidate feature table (one row per point-candidate)."""
        c = self.cand
        df = pd.DataFrame({"trip_id": self.trip_id, "obs_idx": c["obs"].astype(np.int32),
                           "arc_id": c["arc"].astype(np.int64)})
        for k in FEATURE_COLS:
            df[k] = self.feat[k].astype(np.float32)
        if self.ref is not None:
            ra = self.ref["ref_arc"][c["obs"]]
            df["label"] = (c["arc"] == ra).astype(np.int8)
            df["ref_valid"] = self.ref["ref_valid"][c["obs"]]
        return df

    def point_table(self) -> pd.DataFrame:
        """One row per GPS observation (+ reference info, whether it is in the candidate set)."""
        N = len(self.P)
        df = pd.DataFrame({"trip_id": self.trip_id, "obs_idx": np.arange(N, dtype=np.int32),
                           "lon": self.track[:, 0], "lat": self.track[:, 1], "t": self.t,
                           "n_cand": np.bincount(self.cand["obs"], minlength=N)})
        if self.ref is not None:
            c = self.cand
            hit = np.zeros(N, bool)
            m = c["arc"] == self.ref["ref_arc"][c["obs"]]
            hit[c["obs"][m]] = True
            df["ref_arc"] = self.ref["ref_arc"]
            df["ref_route_idx"] = self.ref["ref_idx"]
            df["ref_dist_m"] = self.ref["ref_dist"]
            df["ref_valid"] = self.ref["ref_valid"]
            df["ref_in_cand"] = hit
        return df
