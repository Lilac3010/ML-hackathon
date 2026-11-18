"""Data-understanding figures (EDA) built only from the real processed tables / result CSVs.
   python scripts/eda_plots.py   -> results/figures/eda_*.png
Needs data/processed/{train,student_test}_{features,points}.parquet (run scripts/build_dataset.py first)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

FIG = config.RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
P = config.PROCESSED_DIR
C_TR, C_TE, C_POS, C_NEG = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"eda_{name}.png", dpi=150)
    plt.close()
    print("wrote", FIG / f"eda_{name}.png")


def trip_sizes(trips):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col, title in zip(ax, ["n_obs", "n_route_arcs", "n_arcs"],
                             ["GPS points per trip", "Route length (arcs) per trip", "Road-network size (arcs) per trip"]):
        for part, c in (("train", C_TR), ("student_test", C_TE)):
            a.hist(trips[trips.partition == part][col], bins=25, alpha=0.65, color=c, label=part)
        a.set_title(title); a.set_xlabel(col); a.set_ylabel("trips")
        if col == "n_arcs":
            a.set_xscale("log")
    ax[0].legend()
    save("trip_sizes")


def sampling_and_speed(pts, feats):
    pts = pts.sort_values(["trip_id", "obs_idx"])
    dt = pts.groupby("trip_id")["t"].diff().dropna()
    f1 = feats.drop_duplicates(["trip_id", "obs_idx"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(dt.clip(upper=10), bins=np.arange(0, 10.5, 0.5), color=C_TR)
    ax[0].set_title(f"GPS sampling interval (median {dt.median():.1f}s, {100 * (dt > 5).mean():.2f}% gaps > 5s)")
    ax[0].set_xlabel("seconds between consecutive points (clipped at 10)"); ax[0].set_ylabel("points")
    ax[0].set_yscale("log")
    sp = f1["speed_mps"] * 3.6
    ax[1].hist(sp.clip(upper=130), bins=60, color=C_TR)
    ax[1].set_title("Speed estimated from the trace"); ax[1].set_xlabel("km/h (clipped at 130)"); ax[1].set_ylabel("points")
    save("sampling_speed")


def ref_distance(pts):
    d = pts["ref_dist_m"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(d[d <= 60], bins=60, color=C_TR)
    ax[0].axvline(config.REF_VALID_M, color="red", ls="--", label=f"valid threshold {config.REF_VALID_M:.0f} m")
    ax[0].set_title("GPS point -> reference-route distance"); ax[0].set_xlabel("metres (<= 60 shown)")
    ax[0].set_ylabel("points"); ax[0].legend()
    pv = pts.groupby("trip_id")["ref_valid"].mean().sort_values()
    ax[1].bar(range(len(pv)), pv.values, color=C_TR, width=1.0)
    ax[1].set_title(f"Per-trip share of points with a valid reference ({100 * pts.ref_valid.mean():.1f}% overall)")
    ax[1].set_xlabel("trips (sorted)"); ax[1].set_ylabel("fraction ref_valid"); ax[1].set_ylim(0, 1.02)
    save("reference_distance")


def class_balance(feats):
    v = feats[feats.ref_valid]
    pos = int(v.label.sum()); neg = len(v) - pos
    per_pt = feats.groupby(["trip_id", "obs_idx"]).size()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(["label = 0\n(wrong arc)", "label = 1\n(true arc)"], [neg, pos], color=[C_NEG, C_POS])
    for i, n in enumerate([neg, pos]):
        ax[0].text(i, n, f"{n:,}\n({100 * n / len(v):.1f}%)", ha="center", va="bottom")
    ax[0].set_title("Class balance of candidate rows (ref_valid only)"); ax[0].set_ylabel("rows")
    ax[0].set_ylim(0, max(neg, pos) * 1.2)
    ax[1].hist(per_pt, bins=np.arange(0.5, config.CAND_MAX_K + 1.5, 1), color=C_TR)
    ax[1].set_title(f"Candidates per GPS point (mean {per_pt.mean():.1f}, max K={config.CAND_MAX_K})")
    ax[1].set_xlabel("candidate arcs"); ax[1].set_ylabel("GPS points")
    save("class_balance")


def candidate_study():
    cs = pd.read_csv(config.RESULTS_DIR / "candidate_study.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, g in cs.groupby("max_k"):
        g = g.sort_values("radius_m")
        ax[0].plot(g.radius_m, 100 * g.miss_rate, marker="o", label=f"K={k}")
        ax[1].plot(g.radius_m, g.cands_per_point, marker="o", label=f"K={k}")
    ax[0].set_title("Candidate miss rate (true arc not in set)"); ax[0].set_ylabel("miss rate %")
    ax[1].set_title("Cost: candidates per point"); ax[1].set_ylabel("mean candidates")
    for a in ax:
        a.set_xlabel("search radius (m)")
    ax[0].axvline(config.CAND_RADIUS_M, color="red", ls="--", lw=1)
    ax[1].axvline(config.CAND_RADIUS_M, color="red", ls="--", lw=1)
    ax[0].legend(title=f"red = chosen r={config.CAND_RADIUS_M:.0f} m, K={config.CAND_MAX_K}", fontsize=8)
    save("candidate_study")


def feature_by_class(feats):
    v = feats[feats.ref_valid]
    cols = ["dist_m", "support_mean", "heading_diff", "dist_gap_to_best"]
    lims = {"dist_m": 60, "support_mean": 80, "heading_diff": 180, "dist_gap_to_best": 50}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, c in zip(ax, cols):
        bins = np.linspace(0, lims[c], 50)
        for lab, col, name in ((0, C_NEG, "wrong arc"), (1, C_POS, "true arc")):
            a.hist(v.loc[v.label == lab, c].clip(upper=lims[c]), bins=bins, density=True, alpha=0.6,
                   color=col, label=name)
        a.set_title(c); a.set_xlabel(f"{c} (clipped at {lims[c]})"); a.set_ylabel("density")
    ax[0].legend()
    fig.suptitle("Do the features separate true arcs from wrong arcs?")
    save("feature_by_class")


def feature_importance():
    fi = pd.read_csv(config.RESULTS_DIR / "feature_importance_lightgbm.csv").sort_values("gain_pct")
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(fi.feature, fi.gain_pct, color=C_TR)
    ax.set_xlabel("gain share (%)"); ax.set_title("LightGBM feature importance (gain)")
    save("feature_importance")


def feature_corr(feats):
    from pathsnap.features import FEATURE_COLS
    s = feats[feats.ref_valid].sample(200_000, random_state=config.SEED)[FEATURE_COLS]
    cm = s.corr(method="spearman").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=7)
    ax.grid(False); fig.colorbar(im, shrink=0.8)
    ax.set_title("Feature correlation (Spearman, 200k sampled rows)")
    save("feature_correlation")


def main():
    trips = pd.concat([pd.read_csv(P / "train_trips.csv"), pd.read_csv(P / "student_test_trips.csv")])
    trip_sizes(trips)
    pts = pd.read_parquet(P / "train_points.parquet")
    feats = pd.read_parquet(P / "train_features.parquet")
    sampling_and_speed(pts, feats)
    ref_distance(pts)
    class_balance(feats)
    candidate_study()
    feature_by_class(feats)
    feature_importance()
    feature_corr(feats)


if __name__ == "__main__":
    main()
