import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rotating_logger import RotatingLogger

import configparser
import os


class HeatmapBuilder:
    def __init__(self, config_path) -> None:
        """
        Initialize HeatmapBuilder object
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.logger = RotatingLogger()
        self.logger.info('Initializing HeatmapBuilder')

    @staticmethod
    def __clear_dir(path: str | os.PathLike[str]) -> None:
        """
        Clear directory contents if it exists
        :param path: directory path
        """
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def __clear(self) -> None:
        """
        Clear matrices and heatmaps
        """
        self.logger.info(f'--> Clearing current heatmaps and matrices contents')
        self.__clear_dir(self.config.get('builder.paths', 'matrices'))
        self.__clear_dir(self.config.get('builder.paths', 'heatmaps'))

    def __load_data(self) -> None:
        """
        Load data into object
        """
        self.logger.info(f'--> Loading processed data')
        self.dataframes = {}
        for filename in os.listdir(self.config.get('builder.paths', 'processed_currency')):
            ticker = filename.split('.csv')[0]
            self.dataframes[ticker] = pd.read_csv(f'{self.config.get('builder.paths', 'processed_currency')}{filename}',
                                                  index_col='Date')
        self.target = pd.read_csv(f'{self.config.get('builder.paths', 'processed_index')}DX-Y.NYB.csv',
                                  index_col='Date')

    def __build_matrices(self) -> None:
        """
        Build heatmap matrices
        """
        self.logger.info(f'--> Building matrices and linking them to targets')
        # Keeping full data
        features = pd.concat(self.dataframes, axis=1)
        features = features.dropna()
        # Extracting target
        y = self.target[['regime']].copy()
        y_shifted = y.shift(1)
        # Joining on common indices
        common_idx = features.index.intersection(y_shifted.index)
        features = features.loc[common_idx]
        y_shifted = y_shifted.loc[common_idx]
        # Building matrices
        for t in features.index:
            regime_next = y_shifted.loc[t, "regime"]
            if pd.notna(regime_next):
                self.logger.info(f'------> Processing timestep {t}')
                key = f"{t}_{int(regime_next)}"
                value = features.loc[t].unstack()
                value.index.name = 'currencies'
                value = (value - value.mean()) / value.std()
                value.to_csv(f'{self.config.get('builder.paths', 'matrices')}{key}.csv')

    @staticmethod
    def _build_heatmap(matrix: pd.DataFrame, path: str | os.PathLike[str]) -> None:
        """
        Method for building and saving a heatmap from a matrix
        :param matrix: the data matrix
        :param path: output path
        """
        plt.axis('off')
        sns.heatmap(matrix, cmap='magma', annot=False, cbar=False)
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=256)

    def __build_heatmaps(self) -> None:
        """
        Build heatmaps from matrices
        """
        self.logger.info(f'--> Building heatmaps from matrices')
        for filename in sorted(os.listdir(self.config.get('builder.paths', 'matrices')), reverse=True):
            target = filename.split('.csv')[0]
            self.logger.info(f'------> Building heatmap for {target}')
            matrix = pd.read_csv(f'{self.config.get('builder.paths', 'matrices')}{filename}', index_col='currencies')
            self._build_heatmap(matrix, f'{self.config.get('builder.paths', 'heatmaps')}{target}.png')

    def build(self):
        """
        Build heatmaps
        """
        self.logger.info(f'Building')
        self.__clear()
        self.__load_data()
        self.__build_matrices()
        self.__build_heatmaps()
