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
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
