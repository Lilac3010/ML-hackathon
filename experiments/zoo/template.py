"""TEMPLATE - copy this file to experiments/zoo/<your_model>.py and edit. (Gaussian NB here is
only an example and is NOT one of the five compared models.)

Contract (the harness only needs these names):
  NAME             must equal the file name (no .py)
  FAMILY           algorithm family, goes in the report table
  PARAMS           your INITIAL hyper-parameters (documented in the report: "important hyperparameters
                   and their initial values")
  build(params)    returns an sklearn-style estimator with fit / predict_proba. Put scalers INSIDE a
                   sklearn Pipeline so they are fitted on the training folds only (no leakage).
  --- optional ---
  MAX_TRAIN_ROWS   cap on training rows for slow models (random GPS points are kept whole). If you use
                   it, say so in the report.
  fit(est,X,y,Xval,yval)   custom fit (e.g. to record an epoch-wise history). Xval/yval are only given
                   on the final refit and are for plotting curves, never for early stopping/selection.
  history(est)     -> {"train_loss": [...], "val_loss": [...]} for models with epochs, else omit.
  search_space(trial)      Optuna search space -> params dict (needed only if your model gets tuned).

Run:   python scripts/run_experiment.py <name>
Output: results/experiments/<name>.json, data/processed/probs_<name>_*.parquet, models/<name>.joblib
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "template"
FAMILY = "Probabilistic (Gaussian Naive Bayes) - template only"
PARAMS = dict(var_smoothing=1e-9)


def build(params):
    return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=params["var_smoothing"]))
