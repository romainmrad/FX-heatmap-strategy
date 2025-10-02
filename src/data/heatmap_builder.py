from src.utils.dir_utils import clear_dir

import pandas as pd
import matplotlib.pyplot as plt
from rotating_logger import RotatingLogger
from concurrent.futures import ProcessPoolExecutor, as_completed
import configparser
import os


def build_heatmap(args: tuple[str | os.PathLike[str], str | os.PathLike[str]]) -> str:
    """
    Heatmap constructor for distributed computing
    :param args: CSV path and output PNG path
    """
    mat_path, png_path = args
    matrix = pd.read_csv(mat_path, index_col="currencies").to_numpy()
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.axis("off")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=256)
    plt.close()
    return os.path.basename(png_path).replace(".png", "")


class HeatmapBuilder:
    def __init__(self, config_path: str) -> None:
        """
        Initialize HeatmapBuilder object
        :param config_path: path to config file
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.logger = RotatingLogger()
        self.logger.info('Initializing HeatmapBuilder')

    def __clear(self) -> None:
        """
        Clear matrices and heatmaps
        """
        self.logger.info(f'--> Clearing current heatmaps and matrices contents')
        for r in ['negative', 'positive']:
            clear_dir(f'{self.config.get("builder.paths", "matrices")}{r}')
            clear_dir(f'{self.config.get("builder.paths", "heatmaps")}{r}')

    def __load_data(self) -> None:
        """
        Load data into object
        """
        self.logger.info(f'--> Loading processed data')
        self.dataframes = {}
        for filename in os.listdir(self.config.get('builder.paths', 'processed_currency')):
            ticker = filename.replace(".csv", "")
            self.dataframes[ticker] = pd.read_csv(
                os.path.join(self.config.get("builder.paths", "processed_currency"), filename),
                index_col='Date')
        self.target = pd.read_csv(os.path.join(self.config.get("builder.paths", "processed_index"), 'DX-Y.NYB.csv'),
                                  index_col='Date')

    @staticmethod
    def __get_regime_dir(r: int) -> str:
        """
        Get regime directory from integer value
        :param r: regime integer value
        :return: corresponding regime directory
        """
        return 'positive' if r == 1 else 'negative'

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
                self.logger.debug(f'------> Building matrix for {t}')
                value = features.loc[t].unstack()
                value.index.name = 'currencies'
                value.to_csv(
                    f'{self.config.get("builder.paths", "matrices")}{self.__get_regime_dir(regime_next)}/{t}.csv')
        self.logger.info('--> Finished building matrices')

    def __build_heatmaps_serial(self) -> None:
        """
        Build heatmaps from matrices
        """
        self.logger.info(f'--> Building heatmaps from matrices')
        for regime in os.listdir(self.config.get('builder.paths', 'matrices')):
            for filename in sorted(os.listdir(f'{self.config.get("builder.paths", "matrices")}{regime}/'),
                                   reverse=True):
                target = filename.split('.csv')[0]
                build_heatmap((f'{self.config.get("builder.paths", "matrices")}{regime}/{filename}',
                               f'{self.config.get("builder.paths", "heatmaps")}{regime}/{target}.png'))
                self.logger.debug(f'------> Heatmap built for {target}')

    def __build_heatmaps_distributed(self) -> None:
        """
        Build heatmaps from matrices using multiprocessing
        """
        self.logger.info(f'--> Executing distributed heatmap construction')
        tasks = []
        for regime in os.listdir(self.config.get("builder.paths", "matrices")):
            matrix_path = f"{self.config.get('builder.paths', 'matrices')}{regime}/"
            heatmap_path = f"{self.config.get('builder.paths', 'heatmaps')}{regime}/"
            for filename in sorted(os.listdir(matrix_path), reverse=True):
                target = filename.split(".csv")[0]
                curr_mat_path = f"{matrix_path}{filename}"
                curr_png_path = f"{heatmap_path}{target}.png"
                tasks.append((curr_mat_path, curr_png_path))
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(build_heatmap, task) for task in tasks]
            for future in as_completed(futures):
                result_path = future.result()
                self.logger.debug(f"------> Heatmap built for {result_path}")
        self.logger.info('--> Finished building heatmaps')

    def build(self):
        """
        Build heatmaps
        """
        self.logger.info(f'Building')
        self.__clear()
        self.__load_data()
        self.__build_matrices()
        self.__build_heatmaps_distributed()
