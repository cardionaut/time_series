import pandas as pd
from loguru import logger
from sktime.classification.compose import TimeSeriesForestClassifier


class TimeSeries:
    def __init__(self, config, data) -> None:
        self.config = config
        self.data = data.set_index(['id', 'phase'])
        self.results = None

    def __call__(self) -> pd.DataFrame:
        self.split_data()

        return self.results
    
    def split_data(self) -> None:
        pass