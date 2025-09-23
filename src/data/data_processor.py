import pandas as pd
import numpy as np
from rotating_logger import RotatingLogger
from ta import trend
from src.utils.dir_utils import clear_dir
from src.utils.plot_utils import plot_feature
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import configparser
import os


class DataProcessor:
    def __init__(self, config_path: str) -> None:
        """
        Initialize the data processor
        :param config_path: config file path
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.logger = RotatingLogger()
        self.logger.info('Initializing DataProcessor')
        self.plot_tasks = []

    def __clear(self) -> None:
        """
        Clear processed data
        """
        clear_dir(self.config.get('processor.paths', 'processed_index'))
        clear_dir(self.config.get('processor.paths', 'processed_currency'))

    def build_RET(self, currency_data: pd.DataFrame) -> np.ndarray:
        """
        Build return feature on currency data
        :param currency_data: exchange rate OHLC
        :return: Returns feature
        """
        SMA = currency_data['Close'].rolling(
            self.config.getint('processor.currency.RET_Feature', 'SMA_window')).mean()
        SMA_log_ret = pd.Series(np.log(SMA / SMA.shift()))
        SMA_log_ret_mean = SMA_log_ret.rolling(
            self.config.getint('processor.currency.RET_Feature', 'SMA_log_ret_mean_window')).mean()
        past_mean = SMA_log_ret_mean.expanding().mean().shift(1)
        return np.where(SMA_log_ret_mean > past_mean, 1, 0)

    def build_AROON(self, currency_data: pd.DataFrame) -> np.ndarray:
        """
        Build Aroon feature on currency data
        :param currency_data: exchange rate OHLC
        :return: Aroon feature
        """
        up = trend.aroon_up(currency_data['High'], currency_data['Low'],
                            window=self.config.getint('processor.currency.AROON_Feature',
                                                      'UP_window'))
        down = trend.aroon_down(currency_data['High'], currency_data['Low'],
                                window=self.config.getint('processor.currency.AROON_Feature',
                                                          'DOWN_window'))
        data = up - down
        return np.where(data > 30, 1, 0)

    def build_MACD(self, currency_data: pd.DataFrame) -> np.ndarray:
        """
        Build MACD feature on currency data
        :param currency_data: exchange rate OHLC
        :return: MACD feature
        """
        data = trend.macd(currency_data['Close'],
                          window_slow=self.config.getint('processor.currency.MACD_Feature',
                                                         'MACD_window_slow'),
                          window_fast=self.config.getint('processor.currency.MACD_Feature',
                                                         'MACD_window_fast'))
        return np.where(np.sign(data) > 0, 1, 0)

    def __build_regimes(self) -> None:
        """
        Build USD regimes data
        """
        self.logger.info('--> Building USD regimes data')
        # Loading data
        dxy = pd.read_csv(
            os.path.join(self.config.get("processor.paths", "raw_index"),
                         f'{self.config.get("processor.dxy", "ticker")}.csv'),
            index_col='Date'
        )
        # Computing regime
        dxy['RET_Feature'] = self.build_RET(dxy)
        dxy['AROON_Feature'] = self.build_AROON(dxy)
        dxy['MACD_Feature'] = self.build_MACD(dxy)
        dxy['regime'] = np.where(dxy['AROON_Feature'] + dxy['MACD_Feature'] + dxy['RET_Feature'] > 1, 1, -1)
        self.plot_tasks.append(
            (self.config.get("processor.dxy", "ticker"), 'RET_Feature', dxy['Close'], dxy['RET_Feature'],
             os.path.join(self.config.get("processor.dxy", "plot_feature_path"), 'RET_Feature.pdf')))
        self.plot_tasks.append((self.config.get("processor.dxy", "ticker"), 'AROON_Feature', dxy['Close'],
                                dxy['AROON_Feature'],
                                os.path.join(self.config.get("processor.dxy", "plot_feature_path"),
                                             'AROON_Feature.pdf')))
        self.plot_tasks.append((self.config.get("processor.dxy", "ticker"), 'MACD_Feature', dxy['Close'],
                                dxy['MACD_Feature'],
                                os.path.join(self.config.get("processor.dxy", "plot_feature_path"),
                                             'MACD_Feature.pdf')))
        self.plot_tasks.append((self.config.get("processor.dxy", "ticker"), 'Regime', dxy['Close'],
                                dxy['regime'],
                                os.path.join(self.config.get("processor.dxy", "plot_feature_path"), 'regime.pdf')))
        dxy.drop(columns=['Close', 'High', 'Low'], inplace=True, axis=1)
        # Output processed data
        dxy.to_csv(os.path.join(self.config.get("processor.paths", "processed_index"),
                                f'{self.config.get("processor.dxy", "ticker")}.csv'))
        self.logger.info(
            f"--> Built USD regimes data: {len(dxy[dxy['regime'] == 1])} positive and {len(dxy[dxy['regime'] == -1])} negative")

    def __build_currency_features(self, ticker: str, currency_data: pd.DataFrame) -> None:
        """
        Build currency features data
        :param ticker: ticker symbol
        :param currency_data: currency data
        :return: currency features data
        """
        self.logger.info(f'------> Processing {ticker}')
        currency_data['RET_Feature'] = self.build_RET(currency_data)
        currency_data['AROON_Feature'] = self.build_AROON(currency_data)
        currency_data['MACD_Feature'] = self.build_MACD(currency_data)
        self.plot_tasks.append((ticker, 'RET', currency_data['Close'], currency_data['RET_Feature'],
                                os.path.join(self.config.get("processor.currency", "plot_feature_path"),
                                             f'{ticker}_RET.pdf')))
        self.plot_tasks.append((ticker, 'RET', currency_data['Close'], currency_data['AROON_Feature'],
                                os.path.join(self.config.get("processor.currency", "plot_feature_path"),
                                             f'{ticker}_AROON.pdf')))
        self.plot_tasks.append((ticker, 'RET', currency_data['Close'], currency_data['AROON_Feature'],
                                os.path.join(self.config.get("processor.currency", "plot_feature_path"),
                                             f'{ticker}_MACD.pdf')))
        currency_data.drop(columns=['Close', 'High', 'Low'], inplace=True, axis=1)
        currency_data = currency_data[::-1].dropna()
        currency_data.to_csv(os.path.join(self.config.get("processor.paths", "processed_currency"), f'{ticker}.csv'))

    def __build_dataset(self) -> None:
        """
        Build all currency features data
        """
        self.logger.info('--> Building all currency features data')
        for filename in os.listdir(self.config.get('processor.paths', 'raw_currency')):
            ticker = filename.split('.csv')[0]
            current_currency = pd.read_csv(os.path.join(self.config.get("processor.paths", "raw_currency"), filename),
                                           index_col='Date')
            self.__build_currency_features(ticker=ticker, currency_data=current_currency)

    def __plot_features_distributed(self):
        """
        Plot features
        """
        with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=get_context("fork")) as executor:
            futures = [executor.submit(plot_feature, task) for task in self.plot_tasks]
            for future in as_completed(futures):
                result_path = future.result()
                self.logger.debug(f"--> Feature plotted: {result_path}")

    def process(self) -> None:
        """
        Process the data
        """
        self.logger.info('Processing data')
        self.__clear()
        self.__build_regimes()
        self.__build_dataset()
        self.__plot_features_distributed()
        self.logger.info('Processing complete')
