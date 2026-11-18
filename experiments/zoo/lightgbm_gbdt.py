"""Model 1 (Person 1): LightGBM gradient-boosted decision trees - boosting family."""
import lightgbm as lgb

import config

NAME = "lightgbm_gbdt"
FAMILY = "Boosting (gradient-boosted trees, LightGBM)"
PARAMS = dict(n_estimators=300, learning_rate=0.08, num_leaves=31, min_child_samples=40,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0)


def build(params):
    return lgb.LGBMClassifier(objective="binary", subsample_freq=1, random_state=config.SEED,
                              n_jobs=-1, verbose=-1, **params)


def fit(est, X, y, Xval=None, yval=None):
    if Xval is None:
        return est.fit(X, y)
    return est.fit(X, y, eval_set=[(X, y), (Xval, yval)], eval_metric="binary_logloss")


def history(est):
    r = getattr(est, "evals_result_", None)
    if not r:
        return None
    k = list(r)
    return {"train_loss": r[k[0]]["binary_logloss"], "val_loss": r[k[1]]["binary_logloss"]}


def search_space(trial):
    """Bayesian-optimisation space (Optuna TPE). Ranges and reasons:
    * n_estimators 100-800 x learning_rate 0.02-0.2 (log): the pair sets the total boosting budget.
      Searched jointly (multivariate TPE) because they trade off; 800 trees keeps lr=0.02 from
      underfitting, 100 allows a short, heavily-shrunk model. No early stopping: the harness forbids
      using any held-out data for fitting decisions.
    * num_leaves 15-127 (log): tree complexity. 28 features with mostly smooth effects (distance,
      heading) rarely need >127 leaves; log scale because the effect is multiplicative.
    * min_child_samples 20-1000 (log): main regulariser. Consecutive 1 Hz GPS points give
      near-duplicate rows, so 1.58M rows are far fewer independent samples; leaves must be allowed
      to be large to stop the trees memorising single trips.
    * subsample 0.6-1.0 (row bagging, subsample_freq=1) and colsample_bytree 0.5-1.0 (feature
      bagging over 28 correlated features, e.g. dist_m / d_prev1 / support_mean): variance reduction.
    * reg_lambda 1e-3-10 (log): L2 on leaf values; shrinks extreme leaf scores -> better calibrated
      probabilities, which the HMM consumes.
    Deliberately NOT tuned: class weighting (scale_pos_weight / is_unbalance). Top-1 only ranks
    candidates within a point, so reweighting cannot help it, and it would distort the
    probabilities the HMM needs (11.7% positives is a mild imbalance anyway)."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 1000, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
