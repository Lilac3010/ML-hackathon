"""Raw file loading. One trip = one folder holding <id>.track/.nodes/.arcs/(.route)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import config


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t(sec)
    nodes: np.ndarray            # (V,2) lon, lat  (row index = node_id)
    arcs: np.ndarray             # (A,2) from_node, to_node (row index = arc_id)
    route: np.ndarray | None     # (M,) reference arc_ids, or None (never loaded at inference)


def list_trips(partition: str, root: Path | None = None) -> list[str]:
    root = Path(root) if root else config.DATA_DIR
    return sorted(p.name for p in (root / partition).iterdir() if p.is_dir())


def trip_dir(partition: str, trip_id: str, root: Path | None = None) -> Path:
    root = Path(root) if root else config.DATA_DIR
    return root / partition / trip_id


def load_trip(tdir: str | Path, with_route: bool = False) -> Trip:
    """Load a trip folder. `.route` is only read when with_route=True so that the
    inference path can never touch the reference labels."""
    tdir = Path(tdir)
    tid = tdir.name
    track = np.loadtxt(tdir / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(tdir / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(tdir / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    route = None
    if with_route:
        route = np.loadtxt(tdir / f"{tid}.route", dtype=np.int64, ndmin=1)
    return Trip(tid, track, nodes, arcs, route)
