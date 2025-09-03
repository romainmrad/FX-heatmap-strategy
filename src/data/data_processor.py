import pandas as pd
import numpy as np
from rotating_logger import RotatingLogger
from ta import momentum, trend

import configparser
import os


class DataProcessor:
    def __init__(self, config_path) -> None:
        """
        Initialize the data processor
        :param config_path: config file path
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.logger = RotatingLogger()
        self.logger.info('Initializing DataProcessor')

    def __build_regimes(self) -> None:
        """
        Build USD regimes data
        """
        self.logger.info('--> Building USD regimes data')
        # Loading data
        dxy = pd.read_csv(
            f'{self.config.get('processor.paths', 'raw_index')}{self.config.get('processor.dxy', 'ticker')}.csv',
            index_col='Date'
        )
        # Computing log-returns, returns mean and volatility
        dxy_shifted = dxy.shift(1)
        dxy['log_returns'] = np.log(dxy / dxy_shifted)
        dxy['ret_means'] = dxy['log_returns'].rolling(self.config.getint('processor.dxy', 'mean_rw')).mean()
        dxy['volatility'] = dxy['log_returns'].rolling(self.config.getint('processor.dxy', 'std_rw')).std()
        dxy = dxy[::-1].dropna()
        # Computing regime thresholds
        mu = dxy['ret_means'].mean()
        sigma = dxy['ret_means'].std()
        factor = self.config.getfloat('processor.dxy', 'regime_scaling_factor')
        lb = mu - factor * sigma
        ub = mu + factor * sigma
        vol_bound = dxy['volatility'].quantile(0.7)
        # Classifying regimes
        conditions = [
            (dxy['ret_means'] >= ub) & (dxy['volatility'] < vol_bound),
            (dxy['ret_means'] >= ub) & (dxy['volatility'] >= vol_bound),
            (dxy['ret_means'] <= lb) & (dxy['volatility'] < vol_bound),
            (dxy['ret_means'] <= lb) & (dxy['volatility'] >= vol_bound),
        ]
        dxy['regime'] = np.select(conditions, [1, 2, -1, -2], default=0)
        # Output processed data
        dxy.to_csv(
            f'{self.config.get('processor.paths', 'processed_index')}{self.config.get('processor.dxy', 'ticker')}.csv')

    def __build_currency_features(self, ticker: str, currency_data: pd.DataFrame) -> None:
        """
        Build currency features data
        :param ticker: ticker symbol
        :param currency_data: currency data
        :return: currency features data
        """
        self.logger.info(f'------> Processing {ticker}')
        # Adding RSI indicator
        currency_data['RSI'] = momentum.rsi(currency_data.iloc[:, 0],
                                            window=self.config.getint('processor.currency', 'RSI_window'))
        # Adding MACD indicator
        currency_data['MACD'] = trend.macd(currency_data.iloc[:, 0],
                                           window_slow=self.config.getint('processor.currency', 'MACD_window_slow'),
                                           window_fast=self.config.getint('processor.currency', 'MACD_window_fast'))
        # Computing rolling quadratic variation
        currency_data['quad_var'] = currency_data.iloc[:, 0].rolling(
            self.config.getint('processor.currency', 'QV_window')).apply(lambda x: sum(np.diff(x) ** 2), raw=True)
        # Outputting data
        currency_data = currency_data[::-1].dropna()
        currency_data.drop(currency_data.columns[0], inplace=True, axis=1)
        currency_data.to_csv(f'{self.config.get('processor.paths', 'processed_currency')}{ticker}.csv')

    def __build_all_currency_features(self) -> None:
        """
        Build all currency features data
        """
        self.logger.info('--> Building all currency features data')
        for filename in os.listdir(self.config.get('processor.paths', 'raw_currency')):
            ticker = filename.split('.csv')[0]
            current_currency = pd.read_csv(f'{self.config.get('processor.paths', 'raw_currency')}{filename}',
                                           index_col='Date')
            self.__build_currency_features(ticker=ticker, currency_data=current_currency)

    @staticmethod
    def __compute_quadratic_variation(values: np.ndarray) -> float:
        return sum(np.diff(values) ** 2)

    def process(self) -> None:
        self.logger.info('Processing data')
        self.__build_regimes()
        self.__build_all_currency_features()
