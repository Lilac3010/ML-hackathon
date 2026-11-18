"""Road network = directed arcs between nodes; each arc is a straight segment."""
import numpy as np
from scipy.sparse import csr_matrix

from pathsnap.geo import bearing


class RoadNetwork:
    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray):
        self.xy = nodes_xy
        self.u = arcs[:, 0]
        self.v = arcs[:, 1]
        self.n_nodes = len(nodes_xy)
        self.n_arcs = len(arcs)
        self.a = nodes_xy[self.u]           # arc start coords
        self.b = nodes_xy[self.v]           # arc end coords
        d = self.b - self.a
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing(d[:, 0], d[:, 1])
        self.out_deg = np.bincount(self.u, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.v, minlength=self.n_nodes)
        # reverse-arc lookup: arc index of (v,u) or -1
        key = self.u.astype(np.int64) * self.n_nodes + self.v
        rkey = self.v.astype(np.int64) * self.n_nodes + self.u
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[order][pos] == rkey
        self.reverse_arc = np.where(hit, order[pos], -1)
        self._out = None

    @property
    def out_adj(self):
        """CSR: row = node, cols = outgoing arc ids (built lazily, used by the HMM)."""
        if self._out is None:
            order = np.argsort(self.u, kind="stable")
            indptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
            np.cumsum(self.out_deg, out=indptr[1:])
            self._out = (indptr, order)
        return self._out

    def successors(self, arc_id: int) -> np.ndarray:
        """Arc ids that can follow `arc_id` (arcs leaving its end node)."""
        indptr, order = self.out_adj
        n = self.v[arc_id]
        return order[indptr[n]:indptr[n + 1]]

    def components(self):
        """Weakly connected components over nodes (for EDA)."""
        from scipy.sparse.csgraph import connected_components
        g = csr_matrix((np.ones(self.n_arcs), (self.u, self.v)), shape=(self.n_nodes,) * 2)
        return connected_components(g, directed=False)
