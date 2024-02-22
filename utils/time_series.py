import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sktime.transformations.panel.pca import PCATransformer
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


class TimeSeries:
    def __init__(self, config, data) -> None:
        self.config = config
        self.n_workers = config.n_workers
        self.seed = config.time_series.seed
        self.scoring = config.time_series.scoring
        np.random.seed(self.seed)
        self.prepare_data(data)
        self.param_grid = {
            'pca__n_components': [1, 2, 3],
            'kneighbors__n_neighbors': [3, 5, 7],
        }

        self.results = None

    def __call__(self) -> pd.DataFrame:
        self.split_data()
        self.fit_and_evaluate()

        return self.results

    def prepare_data(self, data) -> None:
        data = data.sort_values(by=['id', 'phase'])
        self.ids = data[['id', 'ATTR_Amyloidose']].drop_duplicates()
        self.data_x = data.drop(columns=['ATTR_Amyloidose'])
        self.data_x = self.data_x.set_index(['id', 'phase'])  # sktime requires a MultiIndex
        logger.debug(self.data_x.shape)
        self.data_y = data['ATTR_Amyloidose']

        from sktime.datasets import load_basic_motions

        x_train, y_train = load_basic_motions(split='train', return_type='pd-multiindex', return_X_y=True)
        logger.debug(x_train.shape)
        logger.debug(y_train.shape)

    def split_data(self) -> None:
        train_indices, test_indices = train_test_split(  # split by patient ID
            self.ids,
            test_size=self.config.time_series.test_size,
            stratify=self.ids['ATTR_Amyloidose'],
            random_state=self.seed,
        )
        train_indices = train_indices['id']
        test_indices = test_indices['id']

        self.x_train = self.data_x.loc[train_indices, :]
        self.x_test = self.data_x.loc[test_indices, :]
        self.y_train = self.data_y.loc[train_indices]
        self.y_train = np.array(self.y_train)
        self.y_test = self.data_y.loc[test_indices]
        self.y_test = np.array(self.y_test)

    def fit_and_evaluate(self) -> None:
        logger.info("Fitting and evaluating the model")
        pipe = StandardScaler() * PCATransformer() * KNeighborsTimeSeriesClassifier()
        cv = KFold(n_splits=10, random_state=self.seed, shuffle=True)
        gcv = GridSearchCV(
            pipe,
            param_grid=self.param_grid,
            return_train_score=True,
            cv=cv,
            n_jobs=self.n_workers,
            scoring=self.scoring,
        )
        logger.debug(self.x_train.shape)
        gcv.fit(self.x_train, self.y_train)

        prediction = gcv.predict(self.x_test)

        self.results = {
            'roc_auc': roc_auc_score(self.y_test, prediction),
            'recall': recall_score(self.y_test, prediction),
            'precision': precision_score(self.y_test, prediction),
            'f1': f1_score(self.y_test, prediction),
        }
        logger.info(f"Results: {self.results}")
