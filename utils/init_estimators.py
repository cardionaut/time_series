from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


def init_estimators(config):
    models_dict = {
        'KNN': KNeighborsTimeSeriesClassifier(),
    }
    models = [models_dict[model] for model in models_dict.keys() if config.time_series.model[model]]

    return models