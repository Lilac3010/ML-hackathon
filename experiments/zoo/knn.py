"""Model 3 (Person 2): K-Nearest Neighbours."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

NAME = "knn"
FAMILY = "Instance-based (K-Nearest Neighbours)"

# Limit training size because KNN is computationally expensive.
MAX_TRAIN_ROWS = 300_000

PARAMS = dict(
    n_neighbors=15,
    weights="distance",
    metric="minkowski",
    p=2,
    n_jobs=-1,
)


def build(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(**params)),
    ])
