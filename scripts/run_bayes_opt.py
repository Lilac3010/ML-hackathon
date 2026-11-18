"""Bayesian hyper-parameter optimisation (Optuna TPE) for ONE chosen model.

   python scripts/run_bayes_opt.py lightgbm_gbdt --trials 30

* Search space: `search_space(trial)` in the model's zoo file.
* Objective: mean group-CV observation-level Top-1 accuracy on the 80 train trips (5 folds,
  trip-grouped, the same folds as every other experiment). Student-test and evaluator data are NOT used.
* Trial 0 is the untuned config; trials are pruned fold-by-fold (median rule); the final pick is the
  best CV Top-1 whose CV log loss is no worse than the baseline's (the HMM needs good probabilities).
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
from pathsnap.evaluate import N_FOLDS


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
        rows = []
        for k in range(N_FOLDS):             # fold by fold so hopeless trials can be pruned early
            _, r, _ = cross_validate(mod, params, tr, folds=[k])
            rows += r
            trial.report(float(pd.DataFrame(rows).top1_accuracy.mean()), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        df = pd.DataFrame(rows)
        trial.set_user_attr("cv_f1_macro", float(df.f1_macro.mean()))
        trial.set_user_attr("cv_log_loss", float(df.log_loss.mean()))
        trial.set_user_attr("cv_top1_std", float(df.top1_accuracy.std()))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(df.top1_accuracy.mean())

    # multivariate TPE: models interactions such as n_estimators x learning_rate
    sampler = optuna.samplers.TPESampler(seed=config.SEED, multivariate=True)
    # prune a trial whose running CV Top-1 is below the median of earlier trials at the same fold
    # (same fold -> same difficulty, so the comparison is fair); never before 2 folds / 5 trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    # trial 0 = the untuned initial config, so "tuned" can never be worse than the baseline
    # (a 0.0 on a log-scale range is impossible; 1e-3 L2 is numerically the same as none here)
    base = {k: (1e-3 if k == "reg_lambda" and v == 0 else v) for k, v in mod.PARAMS.items()}
    study.enqueue_trial(base)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=a.trials)
    total = time.perf_counter() - t0
    tdf = study.trials_dataframe()
    tdf.to_csv(out / f"{a.name}_trials.csv", index=False)

    # Selection: highest CV Top-1, but the HMM consumes the probabilities, so the pick may not have a
    # worse CV log loss than the untuned baseline (trial 0). Falls back to plain best if none qualify.
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    base_ll = done[0].user_attrs["cv_log_loss"] if done and done[0].number == 0 else float("inf")
    ok = [t for t in done if t.user_attrs["cv_log_loss"] <= base_ll] or done
    pick = max(ok, key=lambda t: t.value)
    best = {**mod.PARAMS, **pick.params}
    summary = dict(best_params=best, best_trial=pick.number, best_cv_top1=pick.value,
                   best_cv_log_loss=pick.user_attrs["cv_log_loss"], best_cv_f1_macro=pick.user_attrs["cv_f1_macro"],
                   best_cv_top1_std=pick.user_attrs["cv_top1_std"],
                   baseline_cv_top1=done[0].value if done and done[0].number == 0 else None,
                   baseline_cv_log_loss=None if base_ll == float("inf") else base_ll,
                   unconstrained_best_trial=study.best_trial.number, unconstrained_best_cv_top1=study.best_value,
                   n_trials=a.trials, n_complete=len(done),
                   n_pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials), seconds=total)
    json.dump(summary, open(out / f"{a.name}_best_params.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    run_experiment(a.name, best, tag=f"{a.name}_tuned", tables=(tr, te))


if __name__ == "__main__":
    main()
