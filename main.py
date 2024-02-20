import os
import hydra

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from utils.preprocessing import Preprocessing
from utils.time_series import TimeSeries


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig) -> None:
    if config.preprocessing.active:
        data = Preprocessing(config)()
    else:  # data already preprocessed
        path, extension = os.path.splitext(config.preprocessing.input_file)
        data = pd.read_csv(path + '_preprocessed' + extension)

    results = TimeSeries(config, data)()
    


if __name__ == '__main__':
    main()
