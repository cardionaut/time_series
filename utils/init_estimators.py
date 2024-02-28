from sktime.transformations.panel.pca import PCATransformer
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import Arsenal, RocketClassifier, TimeSeriesSVC
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.feature_based import Catch22Classifier, TSFreshClassifier, RandomIntervalClassifier
from sktime.classification.deep_learning import CNNClassifier


def init_estimators(config):
    seed = config.time_series.seed
    
    trafo_dict = {
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
    }
    models = {model: models_dict[model] for model in models_dict.keys() if config.time_series.models[model]}

    return trafos, models
