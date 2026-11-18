"""Reproduce P4's dataset exploration from the student data and trip summaries."""

import argparse
import sys
from pathlib import Path

import folium
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from pathsnap.geo import Projector
from pathsnap.io import list_trips, trip_dir
from pathsnap.network import RoadNetwork


SCHEMA = {
    "track": (3, float, "longitude latitude timestamp_seconds"),
    "nodes": (2, float, "longitude latitude"),
    "arcs": (2, np.int64, "from_node_id to_node_id"),
    "route": (1, np.int64, "arc_id"),
}
COLORS = {"train": "#145f79", "student_test": "#ab4824"}


def scan_files(folder, partition):
    rows, arrays = [], {}
    for suffix, (width, dtype, columns) in SCHEMA.items():
        filename = folder / f"{folder.name}.{suffix}"
        row = dict(partition=partition, trip_id=folder.name, file_type=suffix,
                   source_fields=columns, expected_columns=width, parsed_dtype=np.dtype(dtype).name,
                   file_bytes=0, rows=None, status="missing")
        if filename.is_file():
            row["file_bytes"] = filename.stat().st_size
            if row["file_bytes"] == 0:
                row["status"] = "empty"
            else:
                try:
                    data = np.loadtxt(filename, dtype=dtype, ndmin=2)
                    row.update(rows=len(data), actual_columns=data.shape[1])
                    if len(data) and data.shape[1] == width:
                        row["status"] = "ok"
                        arrays[suffix] = data
                    else:
                        row["status"] = "unexpected_shape"
                except ValueError:
                    row["status"] = "parse_error"
        rows.append(row)
    return rows, arrays


def coordinate_mask(coords):
    return ((np.abs(coords) <= [180, 90]).all(axis=1)
            & (coords != 0).any(axis=1))


def track_stats(track):
    gps, t = track[:, :2], track[:, 2]
    plausible = coordinate_mask(gps)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    positive = dt[dt > 0]
    row = dict(n_obs=len(track), gps_rows_with_nan=int(np.isnan(track).any(axis=1).sum()),
               gps_rows_with_nonfinite=int((~np.isfinite(track).all(axis=1)).sum()),
               invalid_coordinates=int((np.isfinite(gps).all(axis=1) & ~plausible).sum()),
               repeated_timestamps=int((dt == 0).sum()), backward_timestamps=int((dt < 0).sum()),
               repeated_gps=int((gps[1:] == gps[:-1]).all(axis=1).sum()),
               finite_intervals=len(dt), one_second_intervals=int((dt == 1).sum()),
               gaps_gt_10s=int((dt > 10).sum()), gaps_gt_1000s=int((dt > 1000).sum()),
               median_positive_gap_s=np.median(positive) if len(positive) else np.nan,
               max_gap_s=positive.max() if len(positive) else np.nan,
               elapsed_s=t[-1] - t[0])
    valid_gps = gps[plausible]
    if len(valid_gps):
        lo, hi = valid_gps.min(axis=0), valid_gps.max(axis=0)
        proj = Projector(*valid_gps.mean(axis=0))
        xy = proj.to_xy(*valid_gps.T)
        adjacent = np.diff(np.flatnonzero(plausible)) == 1
        deltas = np.diff(xy, axis=0)[adjacent]
        row.update(lon_min=lo[0], lon_max=hi[0], lat_min=lo[1], lat_max=hi[1],
                   extent_east_km=np.ptp(xy[:, 0]) / 1000,
                   extent_north_km=np.ptp(xy[:, 1]) / 1000,
                   gps_path_km=np.linalg.norm(deltas, axis=1).sum() / 1000)
    return row, valid_gps


def graph_stats(arrays):
    row = {}
    nodes, arcs, route = (arrays.get(kind) for kind in ("nodes", "arcs", "route"))
    if nodes is not None:
        row.update(n_nodes=len(nodes), invalid_node_coordinates=int((~coordinate_mask(nodes)).sum()))
    if arcs is not None:
        row["n_arcs"] = len(arcs)
    if nodes is not None and arcs is not None:
        valid = ((arcs >= 0) & (arcs < len(nodes))).all(axis=1)
        row["invalid_arc_endpoints"] = int((~valid).sum())
        kept = arcs[valid]
        if len(kept):
            # The component and degree calculations depend on IDs, so coordinates are irrelevant here.
            network = RoadNetwork(nodes, kept)
            count, labels = network.components()
            largest = np.bincount(labels).max()
            row.update(weak_components=count, largest_component_nodes=largest,
                       largest_component_fraction=largest / len(nodes),
                       isolated_nodes=int(((network.in_deg + network.out_deg) == 0).sum()),
                       self_loop_arcs=int((network.u == network.v).sum()),
                       duplicate_arcs=len(kept) - len(np.unique(kept, axis=0)))
        else:
            row.update(weak_components=len(nodes), largest_component_nodes=1,
                       largest_component_fraction=1 / len(nodes), isolated_nodes=len(nodes))
    if route is not None:
        ids = route[:, 0]
        row.update(n_route_arcs=len(ids), unique_route_arcs=len(np.unique(ids)),
                   repeated_route_steps=int((ids[1:] == ids[:-1]).sum()))
        if arcs is not None:
            valid = (ids >= 0) & (ids < len(arcs))
            pairs = valid[1:] & valid[:-1]
            row.update(invalid_route_arc_ids=int((~valid).sum()),
                       disconnected_route_steps=int((arcs[ids[:-1][pairs], 1]
                                                     != arcs[ids[1:][pairs], 0]).sum()))
    return row


def partition_summary(trips):
    rows = []
    for part, group in trips.groupby("partition", sort=False):
        row = dict(partition=part, trips=len(group))
        for column in ("n_obs", "gps_rows_with_nan", "invalid_coordinates", "gaps_gt_10s",
                       "gaps_gt_1000s", "isolated_nodes", "invalid_route_arc_ids",
                       "disconnected_route_steps"):
            row[column + "_total"] = group[column].sum(min_count=1) if column in group else np.nan
        for column in ("n_obs", "gps_path_km", "elapsed_s", "n_nodes", "n_arcs", "weak_components", "n_route_arcs"):
            if column in group:
                for statistic in ("min", "median", "max"):
                    row[f"{column}_{statistic}"] = getattr(group[column], statistic)()
        intervals = group["finite_intervals"].sum()
        row.update(pct_intervals_1s=100 * group["one_second_intervals"].sum() / intervals if intervals else np.nan,
                   max_gap_s=group["max_gap_s"].max(),
                   disconnected_networks=int((group["weak_components"] > 1).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_coordinates(coordinates, output):
    samples = {part: np.concatenate(values) for part, values in coordinates.items() if values}
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for i, (axis, label) in enumerate(zip(axes, ("Longitude", "Latitude"))):
        bins = np.histogram_bin_edges(np.concatenate(list(samples.values()))[:, i], bins=55)
        for part, data in samples.items():
            counts, _ = np.histogram(data[:, i], bins=bins)
            axis.stairs(100 * counts / len(data), bins, label=part, color=COLORS[part], linewidth=1.7)
        axis.set(xlabel=f"{label} (°)", ylabel="GPS observations per bin (%)")
        axis.legend(frameon=False)
    figure.suptitle("GPS coordinate distributions by partition")
    figure.savefig(output / "gps_distributions.png", dpi=170)
    plt.close(figure)


def examples(trips, raw_root, output, maps):
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained", squeeze=False)
    for row_index, (part, group) in enumerate(trips.groupby("partition", sort=False)):
        group = group.dropna(subset=["n_obs", "max_gap_s"])
        typical = group.loc[(group["n_obs"] - group["n_obs"].median()).abs().idxmin()]
        extreme = group.loc[group["max_gap_s"].idxmax()]
        for column, (label, selected) in enumerate((("Near-median point count", typical), ("Largest timestamp gap", extreme))):
            _, arrays = scan_files(trip_dir(part, selected.trip_id, raw_root), part)
            gps = arrays["track"][:, :2]
            gps = gps[coordinate_mask(gps)]
            projection = Projector(*gps.mean(axis=0))
            xy = projection.to_xy(*gps.T)
            relative = (xy - xy[0]) / 1000
            axis = axes[row_index, column]
            axis.plot(*relative.T, color=COLORS[part], alpha=0.65, lw=1)
            axis.scatter(*relative[[0, -1]].T, c=["#247653", "#ab4824"], s=30, zorder=3)
            axis.set(title=f"{part} / {selected.trip_id}\n{label}", xlabel="East of start (km)", ylabel="North of start (km)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=.2)
            if maps and column == 0:
                save_map(part, selected.trip_id, arrays, gps, xy, projection, maps)
    figure.suptitle("Example GPS trajectories · green start, orange end")
    figure.savefig(output / "trajectory_examples.png", dpi=170)
    plt.close(figure)


def save_map(part, tid, arrays, gps, xy, projection, maps):
    arcs = arrays["arcs"]
    coords = arrays["nodes"]
    node_xy = projection.to_xy(*coords.T)
    net = RoadNetwork(node_xy, arcs)
    midpoint = (net.a + net.b) / 2
    dist = cKDTree(xy).query(midpoint)[0]
    ids = np.arange(len(arcs))
    visible = (dist <= 120) & ((net.reverse_arc < 0) | (ids <= net.reverse_arc))
    view = folium.Map(location=gps.mean(axis=0)[::-1].tolist(), control_scale=True)
    roads = folium.FeatureGroup(name="Nearby road network")
    folium.PolyLine(coords[arcs[visible]][:, :, ::-1].tolist(), color="#596772", weight=2).add_to(roads)
    roads.add_to(view)
    points = folium.FeatureGroup(name="GPS observations")
    for lon, lat in gps:
        folium.CircleMarker((lat, lon), radius=2, stroke=False, fill=True, fill_color="#146ab8",
                            fill_opacity=.85).add_to(points)
    points.add_to(view)
    folium.LayerControl(collapsed=False).add_to(view)
    view.fit_bounds([gps.min(axis=0)[::-1].tolist(), gps.max(axis=0)[::-1].tolist()])
    maps.mkdir(parents=True, exist_ok=True)
    view.save(maps / f"{part}_{tid}_eda.html")


def run(raw_root, output, maps=None):
    files, trips = [], []
    coordinates = {part: [] for part in config.PARTITIONS}
    for part in config.PARTITIONS:
        for tid in list_trips(part, raw_root):
            inventory, arrays = scan_files(trip_dir(part, tid, raw_root), part)
            files.extend(inventory)
            row = {"partition": part, "trip_id": tid, **graph_stats(arrays)}
            if "track" in arrays:
                metrics, gps = track_stats(arrays["track"])
                row.update(metrics)
                if len(gps):
                    coordinates[part].append(gps)
            trips.append(row)

    trips = pd.DataFrame(trips)
    inventory = pd.DataFrame(files)
    availability = inventory.assign(available=inventory.status.ne("missing"), parsed=inventory.status.eq("ok"))
    availability = availability.groupby(["partition", "file_type"], sort=False).agg(
        expected=("trip_id", "size"), available=("available", "sum"), parsed=("parsed", "sum"),
        bytes_total=("file_bytes", "sum"), bytes_min=("file_bytes", "min"),
        bytes_median=("file_bytes", "median"), bytes_max=("file_bytes", "max")).reset_index()

    summaries = []
    for part in config.PARTITIONS:
        path = config.PROCESSED_DIR / f"{part}_trips.csv"
        if path.is_file():
            processed = pd.read_csv(path, dtype={"trip_id": str})
            summaries.append(processed[["partition", "trip_id", "pct_ref_valid", "ref_dist_median_m", "ref_dist_p95_m"]])
    if summaries:
        trips = trips.merge(pd.concat(summaries), on=["partition", "trip_id"], how="left", validate="1:1")

    output.mkdir(parents=True, exist_ok=True)
    summary = partition_summary(trips)
    for filename, table in (("files_by_trip", inventory), ("file_summary", availability),
                            ("trips", trips), ("partition_summary", summary)):
        table.to_csv(output / f"{filename}.csv", index=False)
    plot_coordinates(coordinates, output)
    examples(trips, raw_root, output, maps)
    print(summary[["partition", "trips", "n_obs_total", "pct_intervals_1s", "max_gap_s",
                   "disconnected_networks"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_DIR)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "eda")
    parser.add_argument("--maps-dir", type=Path, help="Local folder for Folium HTML with GPS geometry")
    args = parser.parse_args()
    run(args.data, args.out, args.maps_dir)
