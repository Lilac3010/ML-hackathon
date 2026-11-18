from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

NAME = "mlp"
FAMILY = "Neural network (multilayer perceptron)"
PARAMS = dict(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=1024,
    learning_rate_init=0.001,
    max_iter=30,
    early_stopping=False,
)


def build(params):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(random_state=config.SEED, **params),
    )


def history(est):
    return {"train_loss": est.named_steps["mlpclassifier"].loss_curve_}
