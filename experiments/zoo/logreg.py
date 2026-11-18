"""Model 2 (Person 2): Logistic Regression."""

import config

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

NAME = "logreg"
FAMILY = "Linear / probabilistic (Logistic Regression)"

PARAMS = dict(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight=None,
)


def build(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=config.SEED,
            **params
        )),
    ])
