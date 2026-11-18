"""Model evaluation: comparison metrics, confusion-matrix heatmaps, CV & test plots,
per-fold stability, and loss curves.

Usage:
    python scripts/model_evaluation.py

Outputs (all in results/figures/):
    evaluation_confusion_matrices.png   -- 2x3 grid of per-model confusion matrices
    evaluation_cv_metrics.png           -- CV Top-1, macro-F1, log-loss with error bars
    evaluation_test_metrics.png         -- student-test bar charts (Top-1, F1, log-loss, continuity)
    evaluation_inference_speed.png      -- inference ms/1k-rows + train time comparison
    evaluation_fold_stability.png       -- per-fold Top-1 and macro-F1 for every model
    evaluation_summary_table.csv        -- machine-readable summary of all metrics

All numbers come directly from results/experiments/*.json (never from live models).
No new training is performed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import config

# ── paths ──────────────────────────────────────────────────────────────────────
EXP = config.RESULTS_DIR / "experiments"
FIG = config.RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    """Return a list of raw result dicts (skips template placeholders)."""
    results = []
    for f in sorted(EXP.glob("*.json")):
        r = json.load(open(f))
        if r.get("base_model") == "template":
            continue
        results.append(r)
    return results


def build_summary(results):
    """Flat DataFrame, one row per model, with CV and student-test metrics."""
    rows = []
    for r in results:
        cv = r["cv"]
        st = r["student_test"]
        rows.append(dict(
            model=r["name"],
            family=r["family"],
            # CV metrics
            cv_top1_mean=cv["top1_mean"],
            cv_top1_std=cv["top1_std"],
            cv_f1_macro_mean=cv["f1_macro_mean"],
            cv_f1_macro_std=cv["f1_macro_std"],
            cv_log_loss_mean=cv["log_loss_mean"],
            cv_precision_mean=cv.get("precision_macro_mean", np.nan),
            cv_recall_mean=cv.get("recall_macro_mean", np.nan),
            # student-test metrics
            test_top1=st["top1_accuracy"],
            test_f1_macro=st["f1_macro"],
            test_precision_macro=st["precision_macro"],
            test_recall_macro=st["recall_macro"],
            test_log_loss=st["log_loss"],
            route_continuity=st.get("route_continuity_clf_only", np.nan),
            route_change_ratio=st.get("route_change_ratio_clf", np.nan),
            uturn_flips=st.get("uturn_flips_clf", np.nan),
            # speed
            train_time_s=st.get("train_time_s", np.nan),
            inference_ms_per_1k=st.get("inference_ms_per_1k_rows", np.nan),
            n_train_rows=st.get("n_train_rows", np.nan),
        ))
    df = pd.DataFrame(rows).sort_values("cv_top1_mean", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Confusion matrices
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(results):
    n = len(results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, r in zip(axes, results):
        cm = r["student_test"].get("confusion_matrix")
        if cm is None:
            ax.set_visible(False)
            continue
        cm = np.array(cm, dtype=float)
        # Row-normalise to show rates
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.where(row_sums == 0, 1, row_sums)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nNeg", "Predicted\nPos"])
        ax.set_yticklabels(["Actual\nNeg", "Actual\nPos"])

        for i in range(2):
            for j in range(2):
                raw = int(cm[i, j])
                rate = cm_norm[i, j]
                colour = "white" if rate > 0.6 else "black"
                ax.text(j, i, f"{raw:,}\n({rate:.1%})",
                        ha="center", va="center", fontsize=10, color=colour, fontweight="bold")

        ax.set_title(r["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted label (threshold = 0.5)")
        ax.set_ylabel("True label")

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices – student_test partition (candidate-level, row-normalised)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG / "evaluation_confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 – CV metrics with error bars
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_metrics(df):
    models = df["model"].tolist()
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = PALETTE[:len(models)]

    # Top-1
    axes[0].bar(x, df["cv_top1_mean"], yerr=df["cv_top1_std"], color=colors,
                capsize=5, error_kw={"linewidth": 1.5})
    axes[0].set_title("CV Observation-level Top-1 Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Top-1 Accuracy")
    axes[0].set_ylim(0, 1)
    for i, (v, e) in enumerate(zip(df["cv_top1_mean"], df["cv_top1_std"])):
        axes[0].text(i, v + e + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Macro F1
    axes[1].bar(x, df["cv_f1_macro_mean"], yerr=df["cv_f1_macro_std"], color=colors,
                capsize=5, error_kw={"linewidth": 1.5})
    axes[1].set_title("CV Candidate-level Macro F1", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1)
    for i, (v, e) in enumerate(zip(df["cv_f1_macro_mean"], df["cv_f1_macro_std"])):
        axes[1].text(i, v + e + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Log loss
    axes[2].bar(x, df["cv_log_loss_mean"], color=colors)
    axes[2].set_title("CV Log Loss (lower is better)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Log Loss")
    for i, v in enumerate(df["cv_log_loss_mean"]):
        axes[2].text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("5-Fold Group Cross-Validation Metrics (80 train trips, grouped by trip, seed=42)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG / "evaluation_cv_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 – Student-test bar charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_test_metrics(df):
    models = df["model"].tolist()
    x = np.arange(len(models))
    colors = PALETTE[:len(models)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("test_top1",         "Student-test Observation Top-1 Accuracy",   "Top-1 Accuracy",   False),
        ("test_f1_macro",     "Student-test Candidate Macro F1",            "Macro F1",          False),
        ("test_log_loss",     "Student-test Log Loss (lower is better)",    "Log Loss",          True),
        ("route_continuity",  "Classifier-only Route Continuity",           "Continuity Score",  False),
    ]

    for ax, (col, title, ylabel, lower_better) in zip(axes.flatten(), metrics):
        vals = df[col].fillna(0).tolist()
        bars = ax.bar(x, vals, color=colors)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not lower_better:
            ax.set_ylim(0, 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        # highlight the best
        best_i = int(np.nanargmin(vals) if lower_better else np.nanargmax(vals))
        bars[best_i].set_edgecolor("gold")
        bars[best_i].set_linewidth(2.5)

    fig.suptitle("Student-test Partition Evaluation (10 trips, honest held-out set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG / "evaluation_test_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 – Inference speed + training time
# ══════════════════════════════════════════════════════════════════════════════

def plot_inference_speed(df):
    models = df["model"].tolist()
    x = np.arange(len(models))
    colors = PALETTE[:len(models)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Inference speed
    infer = df["inference_ms_per_1k"].fillna(0).tolist()
    bars = ax1.bar(x, infer, color=colors)
    ax1.set_title("Inference Speed (ms per 1,000 candidate rows)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("ms / 1k rows")
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for bar, v in zip(bars, infer):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(infer) * 0.01,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # Train time
    train = df["train_time_s"].fillna(0).tolist()
    bars2 = ax2.bar(x, train, color=colors)
    ax2.set_title("Training Time on Full Train Set (seconds)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Seconds")
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    for bar, v in zip(bars2, train):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(train) * 0.01,
                 f"{v:.1f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Efficiency Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG / "evaluation_inference_speed.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 – Per-fold stability
# ══════════════════════════════════════════════════════════════════════════════

def plot_fold_stability(results):
    n = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fold_range = None

    for idx, (r, color) in enumerate(zip(results, PALETTE)):
        folds = r.get("cv_folds", [])
        if not folds:
            continue
        folds_sorted = sorted(folds, key=lambda f: f["fold"])
        xs = [f["fold"] for f in folds_sorted]
        if fold_range is None:
            fold_range = xs

        top1 = [f["top1_accuracy"] for f in folds_sorted]
        f1   = [f["f1_macro"]      for f in folds_sorted]

        axes[0].plot(xs, top1, marker="o", label=r["name"], color=color, linewidth=1.8)
        axes[1].plot(xs, f1,   marker="o", label=r["name"], color=color, linewidth=1.8)

    for ax, metric in zip(axes, ["Observation Top-1 Accuracy", "Candidate Macro F1"]):
        ax.set_title(f"Per-Fold {metric}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Fold index")
        ax.set_ylabel(metric)
        ax.set_xticks(fold_range or [])
        ax.set_xticklabels([f"Fold {k}" for k in (fold_range or [])])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("CV Fold-by-Fold Stability (all models, same 5-fold group split)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG / "evaluation_fold_stability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 – Loss curves (MLP + LightGBM only)
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(results):
    any_saved = False
    for r in results:
        h = r.get("history")
        if not (h and h.get("train_loss")):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["train_loss"], label="Train log-loss", color=PALETTE[0], linewidth=1.8)
        if h.get("val_loss"):
            ax.plot(h["val_loss"], label="Student-test log-loss", color=PALETTE[1],
                    linewidth=1.8, linestyle="--")
        ax.set_xlabel("Iteration / epoch")
        ax.set_ylabel("Log Loss")
        ax.set_title(f"Loss Curve – {r['name']}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = FIG / f"loss_curve_{r['name']}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  saved {out}")
        any_saved = True
    if not any_saved:
        print("  no loss curves found (only MLP/LightGBM have them)")


# ══════════════════════════════════════════════════════════════════════════════
# Summary table with pretty printing
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(df):
    out = config.RESULTS_DIR / "evaluation_summary_table.csv"
    df.to_csv(out, index=False)
    print(f"\n  saved {out}")

    # Pretty print to terminal
    display_cols = [
        "model", "cv_top1_mean", "cv_top1_std", "cv_f1_macro_mean",
        "test_top1", "test_f1_macro", "test_log_loss",
        "route_continuity", "inference_ms_per_1k", "train_time_s"
    ]
    display_names = [
        "Model", "CV Top-1 (mean)", "CV Top-1 (std)", "CV Macro F1",
        "Test Top-1", "Test Macro F1", "Test Log Loss",
        "Route Continuity", "Infer ms/1k", "Train (s)"
    ]
    subset = df[display_cols].copy()
    subset.columns = display_names
    print("\n" + "=" * 100)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 100)
    print(subset.round(4).to_string(index=False))
    print("=" * 100)
    print("  Gold border in bar charts = best-performing model for that metric.")
    print("  Confusion matrices use threshold=0.5 on candidate-level probabilities.")
    print("  CV = 5-fold group CV on 80 train trips (seed 42); Test = 10 student_test trips.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading experiment results …")
    results = load_all()
    if not results:
        print("ERROR: No result JSONs found in", EXP, "— run scripts/run_experiment.py for each model first.")
        sys.exit(1)

    print(f"  Found {len(results)} models: {[r['name'] for r in results]}")

    # Sort results to match the summary df order (best CV Top-1 first)
    df = build_summary(results)
    name_order = df["model"].tolist()
    results_sorted = sorted(results, key=lambda r: name_order.index(r["name"]))

    print("\nGenerating plots …")
    plot_confusion_matrices(results_sorted)
    plot_cv_metrics(df)
    plot_test_metrics(df)
    plot_inference_speed(df)
    plot_fold_stability(results_sorted)
    plot_loss_curves(results_sorted)
    save_summary(df)

    print(f"\nAll outputs in {FIG}")


if __name__ == "__main__":
    main()
