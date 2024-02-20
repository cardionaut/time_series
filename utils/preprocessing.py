import os

import pandas as pd
from loguru import logger


class Preprocessing:
    def __init__(self, config) -> None:
        self.config = config
        self.input_file = self.config.preprocessing.input_file
        self.label_file = self.config.preprocessing.label_file
        path, extension = os.path.splitext(self.input_file)
        self.out_path = path + '_preprocessed' + extension

    def __call__(self) -> pd.DataFrame:
        self.read_data()
        self.extract_id_and_phase()
        self.add_labels()
        self.save_preprocessed_data()

        return self.data

    def read_data(self) -> None:
        logger.info(f"Reading data from {self.input_file}")
        self.data = pd.read_csv(self.input_file)
        self.labels = pd.read_excel(self.label_file)
        self.labels = self.labels.rename(columns={'ID_Imaging': 'id'})
        self.labels = self.labels[['id', 'ATTR_Amyloidose']]

    def extract_id_and_phase(self) -> None:
        self.data['id'] = self.data['Image Name'].apply(lambda x: x.split('_')[1])
        self.data['id'] = self.data['id'].astype(int)
        self.data['phase'] = self.data['Image Name'].apply(lambda x: x.split('.')[-3].split('_')[-1])
        self.data = self.data.drop(columns=['Image Name'])

    def add_labels(self) -> None:
        data_ids = set(self.data['id'])
        label_ids = set(self.labels.loc[self.labels['ATTR_Amyloidose'] == 1, 'id'])
        missing_ids = label_ids - data_ids
        if missing_ids:
            logger.warning(f'out of {len(label_ids)} labels, {len(missing_ids)} are missing in the data file')
            logger.warning(f"IDs {missing_ids} are in the label file but not in the data file")

        self.data = self.data.merge(self.labels, on='id', how='left')
        self.data = self.data.dropna(subset=['ATTR_Amyloidose'])

    def save_preprocessed_data(self) -> None:
        logger.info(f"Saving preprocessed data to {self.out_path}")
        self.data.to_csv(self.out_path, index=False)
