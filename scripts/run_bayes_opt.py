"""Bayesian hyper-parameter optimisation (Optuna TPE) for ONE chosen model.

   python scripts/run_bayes_opt.py lightgbm_gbdt --trials 30

* Search space: `search_space(trial)` in the model's zoo file.
* Objective: mean group-CV observation-level Top-1 accuracy on the 80 train trips (5 folds,
  trip-grouped, the same folds as every other experiment). Student-test and evaluator data are NOT used.
* Every trial is logged to results/bayes/<name>_trials.csv; the best params to
  results/bayes/<name>_best_params.json; then the model is refitted with them and scored by the
  same harness under the tag <name>_tuned (so tuned vs untuned are directly comparable).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
import pandas as pd

import config
from experiments.harness import load_zoo, load_tables, cross_validate, run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    mod = load_zoo(a.name)
    tr, te = load_tables()
    out = config.RESULTS_DIR / "bayes"
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        params = {**mod.PARAMS, **mod.search_space(trial)}
        t0 = time.perf_counter()
        _, rows, _ = cross_validate(mod, params, tr)
        df = pd.DataFrame(rows)
        trial.set_user_attr("cv_f1_macro", float(df.f1_macro.mean()))
        trial.set_user_attr("cv_log_loss", float(df.log_loss.mean()))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(df.top1_accuracy.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=a.trials)
    total = time.perf_counter() - t0
    tdf = study.trials_dataframe()
    tdf.to_csv(out / f"{a.name}_trials.csv", index=False)
    best = {**mod.PARAMS, **study.best_params}
    json.dump(dict(best_params=best, best_cv_top1=study.best_value, n_trials=a.trials, seconds=total),
              open(out / f"{a.name}_best_params.json", "w"), indent=2)
    print(f"best CV top-1 {study.best_value:.4f} after {a.trials} trials in {total / 60:.1f} min\n{best}")
    run_experiment(a.name, best, tag=f"{a.name}_tuned", tables=(tr, te))


if __name__ == "__main__":
    main()
