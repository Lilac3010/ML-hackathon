"""Model 4 (Person 3): Random Forest - tree-ensemble family (bagging of decision trees)."""
from sklearn.ensemble import RandomForestClassifier

import config

NAME = "random_forest"
FAMILY = "Tree ensemble (bagging, Random Forest)"
PARAMS = dict(n_estimators=100, max_depth=18, min_samples_leaf=20, max_features="sqrt")


def build(params):
    return RandomForestClassifier(n_jobs=-1, random_state=config.SEED, **params)


def search_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
        max_depth=trial.suggest_int("max_depth", 8, 24),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100, log=True),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    )
