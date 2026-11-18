# PathSnap - Context Hand-off (paste this into your LLM before starting)

Team of 4, PES University UE24CS352A ML hackathon. **PathSnap = map-matching:** noisy 1 Hz GPS
points -> the road segment (arc) the vehicle was really on -> a connected route.
Pipeline: candidate generation -> per-candidate classifier -> HMM/Viterbi decoding -> FastAPI
(Laptop 1) + UI (Laptop 2). Evaluator trips (10, unseen) are only available on the day.

Roles (everyone trains at least one model so the 5-model comparison is done in parallel):
**P1** data/candidates/features (DONE) + HMM decoder + LightGBM | **P2** Logistic Regression + KNN
(+ coordinates the comparison) | **P3** FastAPI + UI + 2-laptop setup + README, and Random Forest |
**P4** EDA, Folium maps, PDF report, and MLP.

### Model split (5 models, 5 different algorithm families)
| # | Model (file in `experiments/zoo/`) | Family | Owner | Notes |
|---|---|---|---|---|
| 1 | `lightgbm_gbdt` | Boosting | **P1** | DONE (reference implementation, use it as an example) |
| 2 | `logreg` | Linear / probabilistic | **P2** | needs scaling (inside a Pipeline) |
| 3 | `knn` | Instance-based | **P2** | slow: set `MAX_TRAIN_ROWS` (e.g. 300k) and disclose it in the report |
| 4 | `random_forest` | Tree ensemble | **P3** | use `n_jobs=-1`; Extra Trees is fine if RF is too slow |
| 5 | `mlp` | Neural network | **P4** | scaling inside a Pipeline; expose `loss_curve_` via `history()` for the loss plot |
Each owner works ONLY in their own `experiments/zoo/<name>.py` + their own results files, so merges never conflict.
Your own model code must be your own (the organisers compare code/configs/weights across teams) and you must be able
to explain it in the viva: why this family suits PathSnap, strengths/limits, initial hyper-parameters, loss/objective,
over/underfitting evidence (CV vs student-test gap). These are the per-model questions the report requires (Deliverables.pdf 3.2).

### The workflow for every model (30-60 min each)
1. `git pull`, run `python scripts/build_dataset.py` once, read section 4.
2. `copy experiments\zoo\template.py experiments\zoo\<name>.py` and edit NAME/FAMILY/PARAMS/build (+ optional hooks).
3. `python scripts/run_experiment.py <name>` -> 5-fold group CV on train + refit + student-test scoring + timing.
   Writes `results/experiments/<name>.json`, `data/processed/probs_<name>_*.parquet`, `models/<name>.joblib`.
   **Commit only** your zoo file, `results/experiments/<name>.json` and `models/<name>.joblib` (if small, <50 MB).
4. When all five exist: `python scripts/compare_models.py` -> `results/model_comparison.csv` + figures for the report.
5. **Model selection (done together, after step 4):** primary metric = CV observation-level Top-1 accuracy (it
   is what the route needs), ties/secondary = CV macro-F1 and log loss (probability quality for the HMM), plus inference
   time and stability across folds (std). Write your reasoning down - the viva asks for it.
6. **Bayesian optimisation** (one model only: the winner): its owner runs
   `python scripts/run_bayes_opt.py <name> --trials 25` (Optuna TPE; objective = mean group-CV Top-1 on the 80 train trips;
   trials logged in `results/bayes/`; refits and scores as `<name>_tuned`). Needs a `search_space(trial)` function in the zoo
   file (see `lightgbm_gbdt.py`). **Budget time:** one trial = 5 fits (LightGBM: 1-4 min per trial -> 25 trials is ~1-1.5 h;
   slow models may need fewer trials or a smaller `MAX_TRAIN_ROWS`). Start it as soon as the winner is clear, leave it running.
7. P1 then re-tunes the HMM on the final model's probabilities and the final model + decoder go to the API (P3).

## 0. Rules that affect everyone (from the organisers' PDFs)
* >= 5 models from different families, compared on CV + student-test; Bayesian hyper-parameter
  search (Optuna TPE) with **group-aware CV on the 80 train trips** for ONE chosen model.
* Never touch evaluator data; never hard-code outputs; ground truth stays server-side
  (API computes accuracy internally; the UI must never receive reference routes).
* Every number in the report must come from a real run saved in `results/`. Everyone must be able
  to explain their code in the viva. Disclose AI assistance per event policy.
* Do not commit passwords/keys or the raw dataset.

## 1. Setup (5 minutes)
```
git clone https://github.com/Lilac3010/ML-hackathon && cd ML-hackathon
pip install -r requirements.txt
set PATHSNAP_DATA=C:\path\to\"Student folder\Student folder"     (folder containing train\ and student_test\)
python scripts/build_dataset.py          # ~40 s -> data/processed/*.parquet  (NOT in git: 100 MB+)
```
`config.py` holds all paths/constants. Always `from config import ...`; never hard-code paths.

## 2. Raw data (verified facts)
Per trip folder `<id>/<id>.{track,nodes,arcs,route}`, no headers, whitespace separated.
* `.track` lon lat t(s) ~1 Hz (97% of steps exactly 1 s, gaps up to 160 s). Trips are from
  **different cities** (lon ~8.6, ~11.0, ~37.4 ...) -> projection is chosen per trip (UTM).
* `.nodes` lon lat (row = node_id). `.arcs` from_node,to_node (row = arc_id). **Each trip has its
  own road network** (299 - 155,003 nodes). Arcs are **directed**; ~82% have a reverse twin.
* `.route` = reference arc ids in travel order (the label source; much shorter than the track).
* 80 train trips, 10 student_test trips (evaluator: 10 more, not provided).

## 3. What P1 built (all in `pathsnap/`)
| module | role |
|---|---|
| `io.py` | load a trip; `.route` only read when `with_route=True` |
| `geo.py` | per-trip UTM projector, point-to-segment distance, bearings |
| `network.py` | `RoadNetwork` (arc geometry, lengths, bearings, degrees, reverse-arc map, successors) |
| `candidates.py` | for every GPS point the <= 20 nearest directed arcs within 50 m (radius doubles to 400 m if none) |
| `labels.py` | aligns `.route` to the track (monotone DP) -> reference arc per GPS point (**training/eval only**) |
| `features.py` | 28 features per (point, candidate) - `FEATURE_COLS` (order matters) |
| `pipeline.py` | `PreparedTrip(trip_dir, with_labels)` -> `.table()` features, `.point_table()` |
| `evaluate.py` | **shared** CV folds + metrics (use these, so numbers are comparable) |
| `hmm.py` | HMM/Viterbi decoder + route connection |
| `inference.py` | `run_trip(trip_dir, predict_proba, evaluate)` = what the API calls |

Scripts: `build_dataset.py`, `candidate_study.py`, `run_experiment.py`, `run_bayes_opt.py`, `compare_models.py`,
`eval_hmm.py`, `tune_hmm.py`, `dev_standin_probs.py` (decoder-development only; superseded by `lightgbm_gbdt`).
`experiments/` = harness + one file per model in `experiments/zoo/`.

## 4. THE TABLES (everyone training a model starts here)
`data/processed/{train,student_test}_features.parquet` - one row per (GPS point, candidate arc).
* ids: `trip_id, obs_idx, arc_id` (**not features**)
* **features: `pathsnap.features.FEATURE_COLS` (28)**: dist/lateral offset/projection fraction/arc length,
  candidate rank & gaps to best, heading difference vs arc direction, speed/accel/turn-rate/time-gaps,
  neighbouring-point support (d_prev1, d_next1, d_prev3, d_next3, support_mean), node degrees.
  No NaNs. Tree models need no scaling; scale for LR/KNN/SVM/MLP **fitting the scaler on train folds only**.
  No absolute coordinates on purpose (different cities -> would memorise places).
* `label` (1 = candidate is the reference arc, ~11.7% positives), `ref_valid` (bool).
* **Use only `ref_valid == True` rows for training and evaluation.** ref_valid = the reference arc is
  within 40 m of the GPS point; otherwise the reference label is untrustworthy (see section 6).
* `{train,student_test}_points.parquet`: one row per GPS point (+ `ref_arc`, `ref_dist_m`, `ref_valid`,
  `ref_in_cand`). `*_trips.csv`: per-trip stats.

Sizes: train 208,417 points / 1,737,911 rows; student_test 19,160 points / 183,413 rows.

### CV + metrics - use `pathsnap/evaluate.py`
```python
from pathsnap.evaluate import assign_folds, N_FOLDS, candidate_metrics, observation_top1
fold = tr.trip_id.map(assign_folds(tr.trip_id.unique()))   # 5 folds, grouped by trip, seed 42
candidate_metrics(table, prob)   # accuracy, macro precision/recall/F1, log loss (candidate rows, valid only)
observation_top1(table, score)   # per GPS point: best-scoring candidate == reference arc (miss = wrong)
```
Primary selection metric suggestion: observation-level Top-1 + candidate macro-F1 (CV, grouped by trip);
log loss for probability quality (HMM consumes probabilities).

### Training harness (`experiments/harness.py`) - you do NOT write CV/metrics code
`scripts/run_experiment.py <name>` does the group CV (same 5 folds for everyone), only trains on `ref_valid` rows, and logs
per fold + student-test: accuracy, macro precision/recall/F1, log loss, observation Top-1, confusion matrix, classifier-only
route continuity, train time, inference time, config and seed - exactly the columns of the report's comparison table.
`results/model_comparison.csv` is built from those JSONs. It also saves out-of-fold probabilities
(`data/processed/probs_<model>_{train_oof,student_test}.parquet`: `trip_id, obs_idx, arc_id, prob`) which P1 uses to
tune the HMM decoder (`python scripts/tune_hmm.py --probs <model>`), and `models/<name>.joblib` =
`{"model", "feature_cols"}` (estimator exposes `predict_proba(X)[:, 1]`; this is what the API loads).
Loss/accuracy curves: only models with a real epoch history (MLP, LightGBM's boosting iterations) get a plot - do not fake
curves for the others (Deliverables.pdf says so explicitly).

## 5. Interfaces for P3 (API) - works NOW with the stand-in model
```python
import joblib; from pathsnap.inference import run_trip
art = joblib.load("models/standin_lgbm.joblib")           # swap for the final model later
out = run_trip(trip_dir, lambda X: art["model"].predict_proba(X)[:, 1], evaluate=True)
```
`run_trip` returns JSON-serialisable dict: `trip_id, n_obs, gps [[lat,lon]..], pred_route_polyline
[[lat,lon]..], pred_arc_per_obs, pred_route_arcs, route_breaks, continuity_hmm/clf, timing_s`, and with
`evaluate=True` also `metrics{n_valid_points, hmm{top1_accuracy,route_arc_f1}, classifier_only{...}}`.
* With `evaluate=True` the reference route is read **server side inside the function**; it is never
  returned. Never add `.route` content to a response.
* Load the model + HMM params (`results/hmm_best_params.json`, `load_hmm_params()`) once at API startup.
  Typical request: 0.2 - 4 s.
* Evaluator trips will sit in a folder on Laptop 1 laid out like `train/<id>/<id>.*` (incl. `.route` for
  scoring). Make the folder configurable (e.g. env var), default to `student_test` for development.
* Suggested endpoints: `GET /trips` (ids), `POST /predict {"trip_id": ...}`. Print one log line per
  request (evaluators ask to see API logs). `uvicorn ... --host 0.0.0.0 --port 8000`, open the
  firewall port, enable CORS; UI reads API base URL from config.

## 6. Results so far (all real runs; reproduce with the scripts)
* Candidate study (`results/candidate_study.csv`): R=50 m, K=20 -> **miss rate 0.06%**, ~8 candidates/point.
* Heuristic floors (valid points): nearest arc only = 56% top-1; distance+heading rule = 84.6% (train),
  84.3% (student_test). Beat these!
* **Model 1, LightGBM (`results/experiments/lightgbm_gbdt.json`, initial config):** CV macro-F1 0.931 +- 0.015,
  CV Top-1 0.883; student-test Top-1 0.892, macro-F1 0.936, log loss 0.065, classifier-only route continuity 0.929.
  Train time ~15 s, inference ~0.3 s for all 10 test trips.
* Harness sanity check with Gaussian NB (template, not one of the five): student-test Top-1 0.817 - below the 0.843
  heuristic, so a model must clear that bar to be worth keeping.
* **HMM on top of LightGBM (student_test): top-1 89.2% -> 94.3%, route continuity 92.9% -> 98.1%**
  (decoder tuned on the earlier stand-in copy of this same model; re-tuned on the final model later);
  HMM helped on every one of the 10 student-test trips.
* Data caveat: train trips `00000054` (10.7% valid) and `00000033` (33%) have very noisy GPS (median 120-140 m
  from the reference route; 169 km of track for a 31 km route) and `00000002` is 87% valid. Their invalid points
  are excluded by `ref_valid`. Decoding noisy trips like these is the hard case to expect in evaluation.

## 7. Open items / caveats
* The selected model's probabilities replace the stand-in; decoder params must be re-tuned on them (P1).
* Candidate-level macro-F1 uses a 0.5 threshold on imbalanced labels (~12% positives); Top-1 is the primary metric.
* HMM best params sit on the edge of the search grid (beta=8, lam_break=24) - small gain, re-check later.
* Labels come from aligning the supplied reference route to the track: they are reference labels, not
  independently verified truth. Say so in the report.
* The raw dataset and `data/processed/` are git-ignored on purpose.
