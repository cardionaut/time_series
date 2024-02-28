import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import sklearn.metrics as metrics
from sktime.classification.compose import ClassifierPipeline
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

from utils.init_estimators import init_estimators


class TimeSeries:
    def __init__(self, config, data, results_path) -> None:
        self.config = config
        self.results_path = results_path
        self.n_workers = config.n_workers
        self.seed = config.time_series.seed
        self.scoring = config.time_series.scoring
        np.random.seed(self.seed)
        self.prepare_data(data)
        self.trafos, self.models = init_estimators(config)
        num_combinations = len(self.trafos) * len(self.models)
        self.metrics = ['roc_auc', 'recall', 'precision', 'f1']
        try:  # load existing results
            self.results = pd.read_csv(self.results_path)
            self.results = self.results.dropna(axis=0, how='all')
            self.row_to_write = self.results.shape[0]
            to_concat = pd.DataFrame(index=range(num_combinations - self.row_to_write), columns=self.results.columns)
            self.results = pd.concat([self.results, to_concat], ignore_index=True)
        except FileNotFoundError:
            self.results = pd.DataFrame(index=range(num_combinations), columns=['trafo', 'model'] + self.metrics)
            self.row_to_write = 0

    def __call__(self) -> pd.DataFrame:
        # from sktime.registry import all_estimators
        # logger.info(all_estimators('classifier', filter_tags={'capability:multivariate': True}, as_dataframe=True))
        # return
        self.split_data()
    
        for trafo_name, trafo in self.trafos.items():
            trafo_params = self.config.time_series.trafo_params[trafo_name]
            for model_name, model in self.models.items():
                if ((self.results['trafo'] == trafo_name) & (self.results['model'] == model_name)).any():
                    continue
                model_params = self.config.time_series.model_params[model_name]
                param_grid = {**trafo_params, **model_params}
                results = self.fit_and_evaluate(trafo, model, param_grid)
                results = [trafo_name, model_name] + results
                self.results.iloc[self.row_to_write] = results
                try:  # ensure that results are saved in case of a KeyboardInterrupt
                    self.results.to_csv(self.results_path, index=False, float_format='%.2f')
                except KeyboardInterrupt:
                    logger.warning('KeyboardInterrupt: Saving results and exiting')
                    self.results.to_csv(self.results_path, index=False, float_format='%.2f')
                    sys.exit(130)
                self.row_to_write += 1

        return self.results

    def prepare_data(self, data) -> None:
        data = data.sort_values(by=['id', 'phase'])
        self.num_phases = data['phase'].max()
        self.ids = data[['id', 'ATTR_Amyloidose']].drop_duplicates()
        self.data_x = data.drop(columns=['ATTR_Amyloidose'])
        self.data_x = self.data_x.set_index(['id', 'phase'])
        self.data_y = data[['id', 'ATTR_Amyloidose']].drop_duplicates()
        self.data_y = self.data_y.set_index('id')

    def split_data(self) -> None:
        train_indices, test_indices = train_test_split(  # split by patient ID
            self.ids,
            test_size=self.config.time_series.test_size,
            stratify=self.ids['ATTR_Amyloidose'],
            random_state=self.seed,
        )
        train_indices = train_indices['id'].values
        test_indices = test_indices['id'].values

        scaler = StandardScaler()
        x_train = self.data_x.loc[train_indices, :]
        x_test = self.data_x.loc[test_indices, :]
        x_train.loc[:, :] = scaler.fit_transform(x_train)
        x_test.loc[:, :] = scaler.transform(x_test)
        self.x_train = x_train.values.reshape(len(train_indices), x_train.shape[1], self.num_phases)
        self.x_test = x_test.values.reshape(len(test_indices), x_test.shape[1], self.num_phases)
        self.y_train = self.data_y.loc[train_indices]
        self.y_test = self.data_y.loc[test_indices]
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

    def fit_and_evaluate(self, trafo, model, param_grid) -> None:
        logger.info(f"Fitting and evaluating {trafo} * {model}")
        pipe = ClassifierPipeline(
            model,
            [
                ('trafo', trafo),
            ],
        )
        cv = KFold(n_splits=3, random_state=self.seed, shuffle=True)
        gcv = GridSearchCV(
            pipe,
            param_grid=param_grid,
            return_train_score=True,
            cv=cv,
            n_jobs=self.n_workers,
            scoring=self.scoring,
            verbose=2,
        )
        gcv.fit(self.x_train, self.y_train)
        prediction = gcv.predict(self.x_test)
        results = [getattr(metrics, f'{metric}_score')(self.y_test, prediction) for metric in self.metrics]

        return results
