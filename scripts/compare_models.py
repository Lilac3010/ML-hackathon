"""Collect every results/experiments/*.json into the report's comparison table + plots.
   python scripts/compare_models.py
Outputs: results/model_comparison.csv and results/figures/*.png (real results only).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

EXP = config.RESULTS_DIR / "experiments"
FIG = config.RESULTS_DIR / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(EXP.glob("*.json")):
        r = json.load(open(f))
        if r["base_model"] == "template":
            continue
        s, c = r["student_test"], r["cv"]
        rows.append(dict(model=r["name"], family=r["family"], cv_f1_macro=c["f1_macro_mean"], cv_f1_std=c["f1_macro_std"],
                         cv_top1=c["top1_mean"], cv_top1_std=c["top1_std"], cv_log_loss=c["log_loss_mean"],
                         test_top1=s["top1_accuracy"], test_precision_macro=s["precision_macro"],
                         test_recall_macro=s["recall_macro"], test_f1_macro=s["f1_macro"], test_log_loss=s["log_loss"],
                         route_continuity=s["route_continuity_clf_only"],
                         route_change_ratio=s.get("route_change_ratio_clf", np.nan),   # ideal 1.0
                         train_time_s=s["train_time_s"],
                         inference_s=s["inference_time_s"], inference_ms_per_1k_rows=s["inference_ms_per_1k_rows"],
                         n_train_rows=s["n_train_rows"]))
    df = pd.DataFrame(rows).sort_values("cv_top1", ascending=False)
    df.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    print(df.round(4).to_string(index=False))

    x = np.arange(len(df)); names = df.model.tolist()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x, df.cv_f1_macro, yerr=df.cv_f1_std, capsize=3); ax[0].set_title("Group-CV macro F1 (mean +- std over folds)")
    ax[1].bar(x, df.cv_top1, yerr=df.cv_top1_std, capsize=3); ax[1].set_title("Group-CV observation Top-1")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(FIG / "cv_comparison.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    w = 0.38
    ax[0].bar(x - w / 2, df.test_top1, w, label="Top-1"); ax[0].bar(x + w / 2, df.test_f1_macro, w, label="macro F1")
    ax[0].set_title("Student-test Top-1 & macro F1"); ax[0].legend()
    ax[1].bar(x, df.test_log_loss); ax[1].set_title("Student-test log loss (lower is better)")
    ax[2].bar(x, df.inference_ms_per_1k_rows); ax[2].set_title("Inference ms per 1000 candidate rows")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(FIG / "student_test_comparison.png", dpi=150); plt.close()

    # epoch-wise loss curves only for models that really have a history
    for f in sorted(EXP.glob("*.json")):
        r = json.load(open(f))
        h = r.get("history")
        if h and h.get("train_loss"):
            plt.figure(figsize=(5, 3.5)); plt.plot(h["train_loss"], label="train")
            if h.get("val_loss"):
                plt.plot(h["val_loss"], label="student-test")
            plt.xlabel("iteration / epoch"); plt.ylabel("log loss"); plt.title(r["name"]); plt.legend()
            plt.tight_layout(); plt.savefig(FIG / f"loss_curve_{r['name']}.png", dpi=150); plt.close()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
