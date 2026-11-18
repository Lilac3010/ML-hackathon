# PathSnap — Map-Matching System

**PathSnap** matches noisy 1 Hz GPS tracks to directed road-network arcs using a
per-candidate classifier (5 models compared) + Hidden Markov Model (Viterbi) decoder,
served through a FastAPI backend and a Leaflet.js evaluator UI.

> PES University UE24CS352A ML Hackathon — Team of 4

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup](#3-setup)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Model Training & Evaluation](#5-model-training--evaluation)
6. [Hyperparameter Optimisation](#6-hyperparameter-optimisation)
7. [HMM Decoder](#7-hmm-decoder)
8. [API Server (Laptop 1)](#8-api-server-laptop-1)
9. [UI (Laptop 2)](#9-ui-laptop-2)
10. [Two-Laptop Deployment](#10-two-laptop-deployment)
11. [Visualisations (Folium)](#11-visualisations-folium)
12. [Results Summary](#12-results-summary)

---

## 1. Project Overview

```
GPS track (.track)
      │
      ▼
Candidate generation    ← pathsnap/candidates.py   (≤20 arcs per point, R=50 m)
      │
      ▼
Feature extraction      ← pathsnap/features.py     (28 features per candidate row)
      │
      ▼
Classifier (one of 5)   ← experiments/zoo/         (P(candidate = reference arc))
      │
      ▼
HMM / Viterbi decoder   ← pathsnap/hmm.py          (emission + transition → route)
      │
      ▼
Connected route         ← [[lat, lon], …]           (served via FastAPI)
```

**Pipeline:** candidate generation → per-candidate binary classifier → HMM/Viterbi decoding → FastAPI (Laptop 1) + Leaflet UI (Laptop 2).

**Datasets:** 80 train trips, 10 student-test trips, 10 evaluator trips (server-side only).

---

## 2. Repository Structure

```
ML-hackathon/
├── config.py                   # Central config — all paths & constants
├── requirements.txt
├── pathsnap/                   # Core library
│   ├── io.py                   # Trip loader
│   ├── geo.py                  # Per-trip UTM projector, distances, bearings
│   ├── network.py              # RoadNetwork (arc geometry, successors, reverse arc)
│   ├── candidates.py           # Candidate arc generation (≤20 arcs, R=50 m)
│   ├── labels.py               # Reference arc alignment (monotone DP)
│   ├── features.py             # 28-feature extraction (FEATURE_COLS)
│   ├── pipeline.py             # PreparedTrip — one call to get features table
│   ├── evaluate.py             # Shared CV folds, Top-1, macro-F1, continuity
│   ├── hmm.py                  # HMM emission/transition + Viterbi decoder
│   ├── inference.py            # run_trip() — entry point called by the API
│   └── viz.py                  # Folium map builder (road network, GPS, route)
├── experiments/
│   ├── harness.py              # Training harness: group CV, refit, scoring, logging
│   └── zoo/
│       ├── template.py         # Model template
│       ├── lightgbm_gbdt.py    # Boosting (LightGBM)
│       ├── logreg.py           # Logistic Regression (Linear)
│       ├── knn.py              # K-Nearest Neighbours (Instance-based)
│       ├── random_forest.py    # Random Forest (Tree ensemble)
│       └── mlp.py              # MLP Neural Network
├── scripts/
│   ├── build_dataset.py        # Build processed parquet feature tables
│   ├── run_experiment.py       # Train + CV + score ONE model
│   ├── run_bayes_opt.py        # Optuna TPE Bayesian optimisation
│   ├── compare_models.py       # Merge all results → CSV + comparison figures
│   ├── model_evaluation.py     # Confusion matrices, stability, speed plots
│   ├── eval_hmm.py             # Classifier-only vs classifier+HMM comparison
│   ├── tune_hmm.py             # Grid search HMM parameters on OOF probs
│   ├── eda_plots.py            # EDA figures (class balance, correlations, …)
│   ├── candidate_study.py      # Candidate coverage study (miss rate, K, R)
│   ├── make_map.py             # Generate Folium HTML maps for the report
│   ├── error_analysis.py       # Per-trip error classification
│   ├── feature_importance.py   # Feature importance (LightGBM SHAP)
│   ├── stress_test.py          # Stress test run_trip on all 80 train trips
│   └── data_quality_scan.py    # Raw-data quality audit across all 90 trips
├── api/
│   └── main.py                 # FastAPI server (GET /trips, POST /predict)
├── ui/
│   ├── index.html              # Evaluator interface
│   ├── app.js                  # Trip selection, map rendering, metrics display
│   └── styles.css
├── models/                     # Saved .joblib model artefacts (git-ignored if >50 MB)
├── results/
│   ├── experiments/            # Per-model JSON logs (CV folds + student-test metrics)
│   ├── figures/                # All plots (CV, test, confusion matrices, loss curves)
│   ├── bayes/                  # Bayesian optimisation trial logs + best params
│   ├── model_comparison.csv
│   ├── evaluation_summary_table.csv
│   ├── hmm_best_params.json
│   └── candidate_study.csv
└── docs/
    ├── HANDOFF.md              # Full context handoff document
    ├── TEAM_SPLIT.md           # Role assignments
    └── PERSON1_EXPLAINER.md    # Data / feature / HMM explainer
```

---

## 3. Setup

### 3.1 Prerequisites

- Python 3.10+
- Windows or Linux (tested on Windows 11 / Python 3.12)
- The **Student folder** dataset (provided by organisers): a folder containing `train/` and `student_test/`, each with one sub-folder per trip named by trip ID.

### 3.2 Install dependencies

```bash
git clone https://github.com/Lilac3010/ML-hackathon
cd ML-hackathon
pip install -r requirements.txt
```

### 3.3 Point to the dataset

`config.py` auto-detects the data in these locations (checked in order):

| Priority | Location |
|---|---|
| 1 | `PATHSNAP_DATA` environment variable |
| 2 | `<repo>/Student folder/Student folder/` |
| 3 | `<repo>/Student folder/` |
| 4 | `<repo parent>/Student folder/Student folder/` |
| 5 | `D:\downloads\Student folder\Student folder\` |

If your data is elsewhere, set the env var:

```bash
# Windows
set PATHSNAP_DATA=C:\path\to\Student folder\Student folder

# Linux / macOS
export PATHSNAP_DATA=/path/to/Student folder/Student folder
```

Verify detection:

```bash
python -c "from config import DATA_DIR; print(DATA_DIR, DATA_DIR.exists())"
```

### 3.4 Build processed feature tables (~40 s)

```bash
python scripts/build_dataset.py
```

Writes `data/processed/train_features.parquet`, `student_test_features.parquet`, and the corresponding `_points.parquet` and `_trips.csv` files. These are **git-ignored** (100 MB+).

---

## 4. Running the Pipeline

### Quick sanity check

```python
from config import DATA_DIR
from pathsnap.pipeline import PreparedTrip
from pathsnap.io import trip_dir

prep = PreparedTrip(trip_dir("student_test", "00000000"), with_labels=True)
print(prep.table().shape)   # (n_candidate_rows, 28 features + metadata)
```

### Full pipeline in code

```python
import joblib
from pathsnap.inference import run_trip, load_hmm_params

art = joblib.load("models/lightgbm_gdbt.joblib")
params = load_hmm_params()                         # loads results/hmm_best_params.json

out = run_trip(
    trip_dir("student_test", "00000000"),
    predict_proba=lambda X: art["model"].predict_proba(X)[:, 1],
    params=params,
    evaluate=True,                                 # scores vs ground truth (server-side only)
)
print(out["metrics"])
```

---

## 5. Model Training & Evaluation

### 5.1 Train one model

```bash
python scripts/run_experiment.py lightgbm_gbdt
python scripts/run_experiment.py logreg
python scripts/run_experiment.py knn
python scripts/run_experiment.py random_forest
python scripts/run_experiment.py mlp
```

Each run does:
- 5-fold **group CV** on the 80 train trips (trip-grouped, seed 42)
- Refit on all valid train rows
- Score on student_test: Top-1 accuracy, macro Precision/Recall/F1, log loss, confusion matrix, route continuity
- Saves `results/experiments/<name>.json`, `models/<name>.joblib`, and OOF probabilities

### 5.2 Generate comparison table + standard plots

```bash
python scripts/compare_models.py
```

Output: `results/model_comparison.csv` + `results/figures/cv_comparison.png`, `student_test_comparison.png`, loss curves.

### 5.3 Full evaluation suite (confusion matrices, fold stability, speed)

```bash
python scripts/model_evaluation.py
```

Output: `results/figures/evaluation_*.png`, `results/evaluation_summary_table.csv`

### 5.4 Results summary

| Model | CV Top-1 | Test Top-1 | Test Macro F1 | Test Log Loss | Infer ms/1k rows |
|---|---|---|---|---|---|
| Random Forest | **0.8838** | 0.8894 | 0.9354 | 0.0581 | 2.4 |
| LightGBM | 0.8826 | 0.8916 | **0.9364** | 0.0650 | 1.8 |
| MLP | 0.8808 | **0.8929** | 0.9356 | 0.0601 | 0.4 |
| Logistic Reg | 0.8715 | 0.8694 | 0.9107 | 0.0920 | 0.3 |
| KNN | 0.8547 | 0.8582 | 0.9096 | 0.1199 | 376.7 ⚠️ |

> KNN uses `MAX_TRAIN_ROWS = 300,000` (disclosed); full dataset would be prohibitively slow.
> Primary selection metric: CV observation-level Top-1. Secondary: CV macro-F1, log loss, inference time.

---

## 6. Hyperparameter Optimisation

Bayesian optimisation (Optuna TPE) is run on **one chosen model** only. Objective = mean group-CV observation Top-1 on the 80 train trips. Student-test and evaluator data are **never used**.

```bash
python scripts/run_bayes_opt.py lightgbm_gbdt --trials 25
```

- Budget: ~1–1.5 hours (LightGBM, 25 trials × 5 folds)
- Logs every trial to `results/bayes/<name>_trials.csv`
- Saves best params to `results/bayes/<name>_best_params.json`
- Automatically refits and scores the tuned model as `<name>_tuned`

---

## 7. HMM Decoder

### Tune HMM parameters on OOF probabilities

```bash
python scripts/tune_hmm.py --probs lightgbm_gbdt
```

Saves `results/hmm_best_params.json` (emission beta, transition lambda).

### Evaluate classifier-only vs classifier + HMM

```bash
python scripts/eval_hmm.py --probs lightgbm_gbdt
```

| Metric | Classifier only | With HMM |
|---|---|---|
| Top-1 accuracy | 89.2% | **94.3%** |
| Route continuity | 92.9% | **98.1%** |

HMM helped on every one of the 10 student-test trips.

---

## 8. API Server (Laptop 1)

### Start the server

```bash
# Windows — open firewall port 8000 first (see Section 10)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The server loads the model and HMM params once at startup.

### Configure the trip folder

```bash
# Default: student_test/ (development)
# On evaluation day, point to the evaluator folder:
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator\trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/trips` | Returns list of trip IDs in the configured folder |
| `POST` | `/predict` | Body: `{"trip_id": "00000000"}` → returns route, GPS, metrics |
| `GET` | `/health` | Returns `{"status": "ok"}` |

### Example request

```bash
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"trip_id\": \"00000000\"}"
```

### API response schema

```json
{
  "trip_id": "00000000",
  "n_obs": 1234,
  "gps": [[lat, lon], ...],
  "pred_route_polyline": [[lat, lon], ...],
  "pred_arc_per_obs": [...],
  "pred_route_arcs": [...],
  "route_breaks": 0,
  "continuity_hmm": 0.981,
  "continuity_clf": 0.929,
  "timing_s": {"prepare": 0.4, "model": 0.1, "decode": 0.2, "total": 0.7},
  "metrics": {
    "n_valid_points": 1180,
    "hmm":             {"top1_accuracy": 0.943, "route_arc_f1": 0.91},
    "classifier_only": {"top1_accuracy": 0.892, "route_arc_f1": 0.87}
  }
}
```

> **Ground truth is never returned.** The `.route` reference file is read server-side only for scoring; it never appears in any API response.

---

## 9. UI (Laptop 2)

The UI is a plain HTML + JavaScript application in `ui/`. No build step required.

**Open `ui/index.html` in a browser**, then:

1. Enter the prediction server URL (e.g. `http://192.168.1.100:8000`)
2. Click **Connect** — the trip dropdown populates
3. Select a trip, click **Match route**
4. The map shows GPS points (blue) and the matched route (red)
5. The metrics panel shows Top-1, Route arc F1, and Continuity for classifier and HMM

**Layer controls** (top-right of map): toggle GPS observations and matched route independently.

---

## 10. Two-Laptop Deployment

### Prerequisites

- Both laptops on the **same Wi-Fi / hotspot**
- Python + dependencies installed on Laptop 1

### Step 1 — Find Laptop 1's IP address

```bash
# Windows
ipconfig
# Look for IPv4 Address under your Wi-Fi adapter, e.g. 192.168.137.1
```

### Step 2 — Open port 8000 on Laptop 1 (Windows Firewall)

```
Control Panel → Windows Defender Firewall → Advanced Settings
  → Inbound Rules → New Rule → Port → TCP → 8000 → Allow
```

Or via PowerShell (run as Administrator):

```powershell
netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000
```

### Step 3 — Start API on Laptop 1

```bash
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Watch the terminal — one log line is printed per request (trip ID, timings, accuracy).

### Step 4 — Open UI on Laptop 2

Open `ui/index.html` in a browser and enter `http://<Laptop1-IP>:8000` as the server URL.

### Step 5 — Test all 10 trips

Click through each trip and verify the map renders and metrics appear. Time each request.

---

## 11. Visualisations (Folium)

Generate an interactive HTML map for any trip:

```bash
python scripts/make_map.py 00000000 --partition student_test --probs lightgbm_gbdt
# Output: results/maps/trip_00000000.html
```

The map has **layer controls** and a **legend** (bottom-left):

| Layer | Colour | Toggle |
|---|---|---|
| Road network (near track) | Grey | ✅ |
| Raw GPS observations | Blue | ✅ |
| Predicted route (HMM) | Red | ✅ |
| Classifier-only disagrees | Violet | ❌ (hidden by default) |
| HMM errors vs reference | Orange | ✅ (report maps only) |

> Reference route is shown **in report/offline maps only** (`reference=True`). It is never sent to the UI.

---

## 12. Results Summary

### Model comparison

See [`results/model_comparison.csv`](results/model_comparison.csv) and [`results/evaluation_summary_table.csv`](results/evaluation_summary_table.csv).

### Key figures in `results/figures/`

| File | Description |
|---|---|
| `evaluation_confusion_matrices.png` | Per-model confusion matrices (all 5 models) |
| `evaluation_cv_metrics.png` | CV Top-1, macro-F1, log loss with ±std error bars |
| `evaluation_test_metrics.png` | Student-test bar charts |
| `evaluation_fold_stability.png` | Per-fold metric lines (variance analysis) |
| `evaluation_inference_speed.png` | Inference speed + training time |
| `loss_curve_lightgbm_gbdt.png` | LightGBM boosting iteration loss curve |
| `loss_curve_mlp.png` | MLP epoch loss curve |
| `eda_*.png` | EDA: class balance, feature correlation, candidate study, etc. |
| `cv_comparison.png` | Standard harness CV comparison |
| `student_test_comparison.png` | Standard harness student-test comparison |

### HMM improvement on student_test

| | Classifier only | Classifier + HMM |
|---|---|---|
| Top-1 accuracy | 89.2% | **94.3%** (+5.1 pp) |
| Route continuity | 92.9% | **98.1%** (+5.2 pp) |

---

## Notes

- **Never commit** the raw dataset (`Student folder/`) or `data/processed/` — both are git-ignored.
- **Never commit** API keys or passwords.
- All numbers in this README come from real runs saved in `results/`. Reproduce any result by re-running the corresponding script.
- AI assistance was used in this project; all team members can explain their own code and results as required by the assessment policy.
