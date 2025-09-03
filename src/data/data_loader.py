import yfinance as yf
from rotating_logger import RotatingLogger

import configparser
import os
import sys

from yfinance.exceptions import YFPricesMissingError


class DataLoader:
    def __init__(self, config_path) -> None:
        """
        Initialize the data loader
        :param config_path: path to the configuration file
        """
        self.config = configparser.ConfigParser()
        self.logger = RotatingLogger()
        self.logger.info('Initializing DataLoader')
        if not os.path.isfile(config_path):
            self.logger.error(f'Configuration file not found at {config_path}')
            sys.exit(1)
        self.logger.info('Reading configuration file')
        try:
            self.start = self.config.get('timeline', 'start')
            self.end = self.config.get('timeline', 'end')
        except configparser.NoSectionError as e:
            self.logger.error(f'DataLoader error: {str(e)}')
            sys.exit(1)

    def __download_indices(self) -> None:
        """
        Download index data
        """
        self.logger.info('--> Downloading index data')
        for t in self.config.get('tickers.index', 'list').split(','):
            self.logger.info(f'------> Downloading {t}')
            try:
                data = yf.download(t, start=self.start, end=self.end, progress=False, auto_adjust=True)['Close']
                data.to_csv(f'data/raw/index/{t}.csv')
            except YFPricesMissingError:
                self.logger.error(f'Invalid ticker {t}')
                sys.exit(1)


    def __download_currency(self) -> None:
        """
        Download currency exchange rates data
        """
        self.logger.info('--> Downloading currency exchange rates')
        for t in self.config.get('tickers.currency', 'non_inverted_list').split(','):
            self.logger.info(f'------> Downloading {t}')
            try:
                data = yf.download(t, start=self.start, end=self.end, progress=False, auto_adjust=True)['Close']
                data.to_csv(f'data/raw/currency/{t}.csv')
            except YFPricesMissingError:
                self.logger.error(f'Invalid ticker {t}')
                sys.exit(1)
        for t in self.config.get('tickers.currency', 'inverted_list').split(','):
            self.logger.info(f'------> Downloading {t}')
            try:
                data = yf.download(t, start=self.start, end=self.end, progress=False, auto_adjust=True)['Close']
                data = 1 / data
                data.to_csv(f'data/raw/currency/{t}.csv')
            except YFPricesMissingError:
                self.logger.error(f'Invalid ticker {t}')
                sys.exit(1)

    def download_data(self) -> None:
        """
        Download data
        """
        self.logger.info('Downloading data')
        self.__download_indices()
        self.__download_currency()
