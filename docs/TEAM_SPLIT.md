# PathSnap - Who does what (read your own section, then skim the others)

Shared context for everything: `docs/HANDOFF.md`. Rule for all: commit only the files you own (listed below),
`git pull --rebase` before every push, never commit the raw dataset or `data/processed/`.

## At a glance
| | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| **Pipeline role** | data, candidates, features, HMM | modelling lead | API + UI + deployment | EDA, maps, report |
| **Model(s) to train** | LightGBM (done) | Logistic Regression + KNN | Random Forest | MLP |
| **Report sections written** | 2.2 preprocessing/features, 5 (HMM), own model | own 2 models, 3.1 table, 3.3 plots | 7 (API/UI/deployment), own model | 2.2 EDA, 6 (Folium), own model, assembly |
| **Starts from** | nothing (done) | `docs/HANDOFF.md` | `run_trip()` + stand-in model | processed tables + `results/` |

Order of work: **P1 DONE -> everyone's models in parallel -> pick winner together -> Bayesian opt (winner's owner)
-> P1 retunes HMM -> P3 plugs final model into API -> full two-laptop test -> report.**

---------------------------------------------------------------------------------------------------------
## P1 - Data, candidates, features, HMM  (+ LightGBM)
**Done:** `pathsnap/` (io, geo, network, candidates, labels, features, pipeline, evaluate, hmm, inference),
`scripts/build_dataset.py`, `candidate_study.py`, `eval_hmm.py`, `tune_hmm.py`, the model harness
(`experiments/`), `lightgbm_gbdt`, `docs/`.
**Still to do**
1. Stress-test `run_trip` on all 80 train trips (noisy trips 54/33, biggest trip 19k points, edge cases); fix what breaks.
2. HMM error analysis on the weakest trips; widen the decoder grid (best values were at the grid edge).
3. `pathsnap/viz.py`: one shared Folium prediction-map function (used by P3's UI and P4's report).
4. After the winner is chosen: `python scripts/tune_hmm.py --probs <winner>` -> final `results/hmm_best_params.json`;
   tell P3 the final model name. Re-run `scripts/eval_hmm.py` for the classifier-only vs classifier+HMM table.
5. Review teammates' `experiments/zoo/*.py` for leakage (scaler outside a Pipeline, student-test used for selection).
**Owns (files):** `pathsnap/`, `scripts/{build_dataset,candidate_study,eval_hmm,tune_hmm,dev_standin_probs}.py`,
`experiments/harness.py`, `experiments/zoo/lightgbm_gbdt.py`, `results/hmm_*`, `docs/PERSON1_EXPLAINER.md`.
**Report:** 2.2 (preprocessing, features and why, candidate study, label limitations), 5 (HMM: the 6 questions + clf vs
HMM table, where HMM did not help), 3.2 for LightGBM. Material: `docs/PERSON1_EXPLAINER.md`.
**Done when:** final HMM params saved, clf-vs-HMM numbers on student_test, no crash on any train trip.

---------------------------------------------------------------------------------------------------------
## P2 - Modelling lead: Logistic Regression + KNN
1. `copy experiments\zoo\template.py experiments\zoo\logreg.py` -> `StandardScaler` + `LogisticRegression` inside a Pipeline.
   Initial params you choose and justify (C, penalty, class_weight, max_iter ...). Run `python scripts/run_experiment.py logreg`.
2. Same for `experiments/zoo/knn.py` (scaler + `KNeighborsClassifier`; distance weighting; **set `MAX_TRAIN_ROWS`**
   ~300k because KNN is slow, and disclose it in the report). Run `python scripts/run_experiment.py knn`.
3. When all 5 results exist: `python scripts/compare_models.py`, check the table/figures, lead the model-selection
   discussion (primary metric = CV observation Top-1; secondary macro-F1, log loss, inference time, fold stability).
4. Write the consolidated comparison table (Deliverables 3.1) and the comparison plots (3.3) from `results/model_comparison.csv`.
**Owns:** `experiments/zoo/{logreg,knn}.py`, `results/experiments/{logreg,knn}.json`, `models/{logreg,knn}.joblib`,
`results/model_comparison.csv`, `results/figures/`.
**Report:** 3.1 overall table, 3.2 A/B/C answers for LogReg and KNN (strengths/limits, initial hyper-parameters, loss
function, results, overfitting/underfitting evidence from the CV-vs-student-test gap), 3.3 plots, model-selection reasoning.
**Done when:** 2 result JSONs committed + comparison table generated.

---------------------------------------------------------------------------------------------------------
## P3 - API, UI, deployment  (+ Random Forest)
**Priority = the live demo**, so do the API first with the stand-in model, the model second.
1. `api/main.py` (FastAPI): load model (`models/<name>.joblib`, start with `lightgbm_gbdt`) and `load_hmm_params()` ONCE at
   startup. `GET /trips` (ids of the trips in the configured folder), `POST /predict {"trip_id": ...}` -> `run_trip(..., evaluate=True)`.
   Reference route is read server-side only and must never appear in a response. One log line per request (trip id,
   timings, accuracy) - the evaluators ask to see the API logs. CORS on; run with `--host 0.0.0.0 --port 8000`.
2. Trip folder configurable by env var (default `student_test` for dev, evaluator folder on the day).
3. `ui/` (Streamlit or plain HTML+JS) on Laptop 2: lists the 10 trips, sends the selected id, shows the map (GPS, predicted
   route; use P1's `viz.py`) and metrics (HMM vs classifier-only). API base URL is a setting.
4. Two-laptop test (same Wi-Fi/hotspot, Windows firewall rule for port 8000) - do it early, not the night before.
   It must work for all 10 trips; time each.
5. `README.md` (install, run, deploy), deployment config, screenshots of the UI and of the API logs on the receiving laptop.
6. Random Forest: `experiments/zoo/random_forest.py` (`n_jobs=-1`; Extra Trees acceptable if RF is too slow), run
   `python scripts/run_experiment.py random_forest`.
**Owns:** `api/`, `ui/`, `README.md`, deployment files, `experiments/zoo/random_forest.py`, its results JSON and model file.
**Report:** 7 (API, UI, deployment, screenshots), 3.2 for Random Forest.
**Done when:** UI on Laptop 2 shows a predicted route + metrics for each of the 10 student-test trips via the API on Laptop 1.

---------------------------------------------------------------------------------------------------------
## P4 - EDA, maps, report  (+ MLP)
1. EDA from `Dataset Exploration.pdf` section 4, saved to `results/eda/` + `notebooks/`: trips per partition, points per trip,
   file sizes/availability, missing/invalid values, timestamp gaps, geographic extent, network size and connectivity
   (components, isolated nodes: `RoadNetwork.components()`), anomalous trips, reference-route structure. Material already in
   `data/processed/*_trips.csv`. State which partition each analysis used.
2. Plots: GPS coordinate distributions, trajectory patterns, road-network size/connectivity tables.
3. Folium maps: GPS points, road network, reference route, predicted route, legend + layer controls (Deliverables section 6).
   Use P1's `viz.py` once it lands. Commit the HTML or the code that makes it; screenshot one for the PDF.
4. MLP: `experiments/zoo/mlp.py` (sklearn `MLPClassifier` in a Pipeline with a scaler; expose `loss_curve_` via `history()` so the
   loss plot is real). Run `python scripts/run_experiment.py mlp`.
5. Assemble the PDF in the required structure (Deliverables 2.2-7), collect everyone's sections, check every number against
   `results/`. Keep the data-quality/assumptions log.
**Owns:** `notebooks/`, `results/eda/`, `viz/` or map scripts, `experiments/zoo/mlp.py`, its results JSON and model file, `report/`.
**Report:** 2.2 EDA part (missing values, trajectory plots, network visualisations, Folium), 6 (Folium), 3.2 for MLP, final assembly.
**Done when:** EDA outputs + maps committed, MLP result committed, report skeleton with all sections filled from real results.

---------------------------------------------------------------------------------------------------------
## Joint tasks
| Task | Who | When |
|---|---|---|
| Model selection (look at the comparison table together, agree the winner and why) | all | when all 5 results are in |
| Bayesian optimisation of the winner (`python scripts/run_bayes_opt.py <name> --trials 25`, 1-1.5 h, leave it running) | winner's owner | right after selection |
| Tuned vs untuned comparison for the report (section 4: 6 questions) | winner's owner | after BO |
| HMM retune on the final model | P1 | after BO |
| Plug final model + HMM params into the API, full test over two laptops | P3 (+P1 help) | after HMM retune |
| Viva rehearsal: each person explains their part to the others, and the others' parts at a high level | all | before the event |

## Rules that bite (from the organisers' PDFs)
* Evaluator data is never used for training, features, selection or tuning. No hard-coded outputs.
* Every number/plot in the report comes from a real run; the PDF must be verifiable against the repo.
* AI use must not replace your own analysis; everyone must be able to explain the code, experiments and conclusions.
* Never commit keys/passwords or the raw dataset.
