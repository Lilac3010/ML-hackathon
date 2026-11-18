"""Shared Folium map for one trip (report figures, EDA, and the UI).

Layers (toggle with the layer control, legend bottom-left):
  road network   thin grey   - arcs near the track only (two-way roads drawn once)
  raw GPS        blue dots   - thinned if the trip is very long; popup shows index/time
  reference      green line  - OFFLINE/REPORT ONLY. Pass reference=True explicitly. The API/UI must
                               never send or draw it (ground truth stays server-side).
  predicted      red line    - classifier + HMM route (connected arcs)
  classifier-only violet dots - where the per-point classifier choice disagrees with the HMM
  errors         orange dots - GPS points (valid reference) whose HMM arc differs from the reference
"""
import folium
import numpy as np
from scipy.spatial import cKDTree

from pathsnap.inference import arcs_to_polyline

COLORS = dict(network="#9aa0a6", gps="#1a73e8", reference="#188038", predicted="#d93025",
              clf="#9334e6", error="#f29900")


def _network_segments(prep, pad_m=120.0, max_segments=25000):
    net = prep.net
    tree = cKDTree(prep.P)
    mid = (net.a + net.b) / 2
    keep = np.where(tree.query(mid, k=1)[0] <= pad_m)[0]
    # draw two-way roads once
    rev = net.reverse_arc[keep]
    keep = keep[(rev < 0) | (keep < rev)]
    if len(keep) > max_segments:
        keep = keep[np.linspace(0, len(keep) - 1, max_segments).astype(int)]
    ends = prep.proj.to_lonlat(np.vstack([net.a[keep], net.b[keep]]))
    n = len(keep)
    return [[[ends[i, 1], ends[i, 0]], [ends[n + i, 1], ends[n + i, 0]]] for i in range(n)]


def _legend(items):
    rows = "".join(f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{t}</div>' for t, c in items)
    html = ('<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;'
            'padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px sans-serif">'
            f'<b>Legend</b>{rows}</div>')
    return folium.Element(html)


def build_map(prep, pred_arc=None, route_arcs=None, clf_arc=None, reference=False,
              max_gps=3000, show_network=True):
    """prep: PreparedTrip. pred_arc: (N,) HMM arc per observation. route_arcs: connected arc list to draw
    (defaults to the collapsed pred_arc). clf_arc: (N,) classifier-only arc. reference=True needs a
    PreparedTrip built with_labels=True."""
    ll = prep.track[:, [1, 0]]
    m = folium.Map(location=ll.mean(0).tolist(), zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    legend = []

    if show_network:
        fg = folium.FeatureGroup(name="Road network (near track)", show=True)
        folium.PolyLine(_network_segments(prep), color=COLORS["network"], weight=1.5, opacity=0.7).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))

    fg = folium.FeatureGroup(name="Raw GPS observations", show=True)
    step = max(1, len(ll) // max_gps)
    for i in range(0, len(ll), step):
        folium.CircleMarker(ll[i].tolist(), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, popup=f"obs {i}, t={prep.t[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    legend.append(("Raw GPS", COLORS["gps"]))

    if reference:
        if prep.route is None:
            raise ValueError("reference=True needs PreparedTrip(with_labels=True)")
        fg = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
        folium.PolyLine(arcs_to_polyline(prep, prep.route), color=COLORS["reference"], weight=6, opacity=0.6,
                        tooltip="reference route").add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    if pred_arc is not None or route_arcs is not None:
        if route_arcs is None:
            seq = np.asarray(pred_arc)
            route_arcs = seq[np.r_[True, seq[1:] != seq[:-1]]].tolist()
        fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)", show=True)
        folium.PolyLine(arcs_to_polyline(prep, route_arcs), color=COLORS["predicted"], weight=3, opacity=0.95,
                        tooltip="predicted route").add_to(fg)
        fg.add_to(m)
        legend.append(("Predicted route (HMM)", COLORS["predicted"]))

    if pred_arc is not None and clf_arc is not None:
        diff = np.flatnonzero(np.asarray(clf_arc) != np.asarray(pred_arc))
        fg = folium.FeatureGroup(name="Classifier-only disagrees with HMM", show=False)
        for i in diff[::max(1, len(diff) // 1500)]:
            folium.CircleMarker(ll[i].tolist(), radius=3, color=COLORS["clf"], fill=True, weight=0,
                                popup=f"obs {i}: classifier arc {clf_arc[i]} vs HMM arc {pred_arc[i]}").add_to(fg)
        fg.add_to(m)
        legend.append(("Classifier-only differs", COLORS["clf"]))

    if reference and pred_arc is not None and prep.ref is not None:
        bad = np.flatnonzero(prep.ref["ref_valid"] & (np.asarray(pred_arc) != prep.ref["ref_arc"]))
        fg = folium.FeatureGroup(name="HMM errors vs reference", show=True)
        for i in bad[::max(1, len(bad) // 1500)]:
            folium.CircleMarker(ll[i].tolist(), radius=3, color=COLORS["error"], fill=True, weight=0,
                                popup=f"obs {i}: predicted {pred_arc[i]}, reference {prep.ref['ref_arc'][i]}").add_to(fg)
        fg.add_to(m)
        legend.append((f"HMM errors ({len(bad)})", COLORS["error"]))

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    lo, hi = ll.min(0), ll.max(0)
    m.fit_bounds([lo.tolist(), hi.tolist()])
    return m
