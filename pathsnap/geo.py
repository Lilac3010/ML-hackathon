"""Geographic helpers. Trips come from different cities (lon ~8.6, ~11.0, ~37.4 ...)
so we pick the UTM zone per trip from the track centroid, giving metres locally."""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.column_stack([x, y])

    def to_lonlat(self, xy):
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])


def seg_distance(p, a, b):
    """Distance from points p to segments a->b (all (M,2)).
    Returns dist, t (fraction along segment 0..1), signed lateral offset (+ = left of a->b)."""
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    ap = p - a
    t = np.clip((ap * ab).sum(1) / L2, 0.0, 1.0)
    closest = a + t[:, None] * ab
    d = np.hypot(*(p - closest).T)
    cross = ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]
    lat = cross / np.sqrt(L2)
    return d, t, lat


def bearing(dx, dy):
    """Compass-style bearing in degrees [0,360) from east/north deltas."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def ang_diff(a, b):
    """Smallest absolute angular difference in degrees, 0..180."""
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)
