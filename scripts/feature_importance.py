"""Gain-based feature importance of the trained LightGBM model (evidence for the feature-engineering section).
   python scripts/feature_importance.py  -> results/feature_importance_lightgbm.csv
Gain = how much each feature reduced the loss across all the tree splits that used it (share of total, %).
NOTE: importance shows what the model *used*, not a causal proof; correlated features share credit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

import config

art = joblib.load(config.ROOT / "models" / "lightgbm_gbdt.joblib")
b = art["model"].booster_
df = pd.DataFrame({"feature": b.feature_name(), "gain": b.feature_importance("gain")})
df["gain_pct"] = 100 * df.gain / df.gain.sum()
df = df.sort_values("gain_pct", ascending=False).reset_index(drop=True)
df[["feature", "gain_pct"]].round(2).to_csv(config.RESULTS_DIR / "feature_importance_lightgbm.csv", index=False)
print(df[["feature", "gain_pct"]].round(1).to_string())
