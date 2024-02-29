from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel.pca import PCATransformer
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import Arsenal, RocketClassifier, TimeSeriesSVC
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.feature_based import Catch22Classifier, TSFreshClassifier, RandomIntervalClassifier
from sktime.classification.deep_learning import (
    CNNClassifier,
    FCNClassifier,
    LSTMFCNClassifier,
    MACNNClassifier,
    MCDCNNClassifier,
    MLPClassifier,
    ResNetClassifier,
    SimpleRNNClassifier,
    TapNetClassifier,
)


def init_estimators(config):
    seed = config.time_series.seed

    trafo_dict = {
        'None': IdentityTransformer(),
        'PCA': PCATransformer(),
    }
    trafos = {trafo: trafo_dict[trafo] for trafo in trafo_dict.keys() if config.time_series.trafos[trafo]}
    models_dict = {
        'KNN': KNeighborsTimeSeriesClassifier(),
        'arsenal': Arsenal(random_state=seed),
        'rocket': RocketClassifier(random_state=seed),
        'SVC': TimeSeriesSVC(random_state=seed, probability=True),
        'forest': CanonicalIntervalForest(random_state=seed),
        'catch22': Catch22Classifier(random_state=seed),
        'tsfresh': TSFreshClassifier(random_state=seed),
        'random_interval': RandomIntervalClassifier(random_state=seed),
        'CNN': CNNClassifier(random_state=seed),
        'FCN': FCNClassifier(random_state=seed),
        'LSTMFCN': LSTMFCNClassifier(random_state=seed),
        'MACNN': MACNNClassifier(random_state=seed),
        'MCDCNN': MCDCNNClassifier(random_state=seed),
        'MLP': MLPClassifier(random_state=seed),
        'ResNet': ResNetClassifier(random_state=seed),
        'SimpleRNN': SimpleRNNClassifier(random_state=seed),
        'TapNet': TapNetClassifier(random_state=seed),
    }
    models = {model: models_dict[model] for model in models_dict.keys() if config.time_series.models[model]}

    return trafos, models


class IdentityTransformer(BaseTransformer):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X
