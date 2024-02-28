import os
import hydra

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from utils.preprocessing import Preprocessing
from utils.time_series import TimeSeries


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig) -> None:
    path, extension = os.path.splitext(config.preprocessing.input_file)
    if config.preprocessing.active:
        data = Preprocessing(config)()
    else:  # data already preprocessed
        data = pd.read_csv(path + '_preprocessed' + extension)

    TimeSeries(config, data, results_path=path + '_results.csv')()


if __name__ == '__main__':
    main()
