import yfinance as yf
from rotating_logger import RotatingLogger
from src.utils.dir_utils import clear_dir
from src.utils.plot_utils import plot_fx_rate

import configparser
import os
import sys

from yfinance.exceptions import YFPricesMissingError


class DataLoader:
    def __init__(self, config_path: str) -> None:
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
        self.config.read(config_path)
        self.logger.info('Reading configuration file')
        try:
            self.start = self.config.get('timeline', 'start')
            self.end = self.config.get('timeline', 'end')
        except configparser.NoSectionError as e:
            self.logger.error(f'DataLoader: {str(e)}')
            sys.exit(1)
        except configparser.NoOptionError as e:
            self.logger.error(f'DataLoader: {str(e)}')
            sys.exit(1)

    def __clear(self) -> None:
        """
        Clear raw data
        """
        self.logger.info('--> Clearing raw data')
        clear_dir(self.config.get("loader", "index_path"))
        clear_dir(self.config.get("loader", "currency_path"))
        clear_dir(self.config.get("loader", "image_path"))

    def __download_indices(self) -> None:
        """
        Download index data
        """
        self.logger.info('--> Downloading index data')
        target_columns = ['Close', 'High', 'Low']
        for t in self.config.get('tickers.index', 'list').split(','):
            try:
                data = yf.download(t, start=self.start, end=self.end, progress=False, auto_adjust=True)[target_columns]
                data.columns = target_columns
                plot_fx_rate(t, data['Close'], os.path.join(self.config.get("loader", "image_path"), f'{t}.pdf'))
                data.to_csv(os.path.join(self.config.get("loader", "index_path"), f'{t}.csv'))
                self.logger.info(f'------> Download complete for {t}')
            except YFPricesMissingError:
                self.logger.error(f'Invalid ticker {t}')
                sys.exit(1)

    def __download_currency(self) -> None:
        """
        Download currency exchange rates data
        """
        self.logger.info('--> Downloading currency exchange rates')
        target_columns = ['Close', 'High', 'Low']
        for t in self.config.get('tickers.currency', 'list').split(','):
            try:
                data = yf.download(t, start=self.start, end=self.end, progress=False, auto_adjust=True)[target_columns]
                data.columns = target_columns
                plot_fx_rate(t, data['Close'], os.path.join(self.config.get("loader", "image_path"), f'{t}.pdf'))
                data.to_csv(os.path.join(self.config.get("loader", "currency_path"), f'{t}.csv'))
                self.logger.info(f'------> Download complete for {t}')
            except YFPricesMissingError:
                self.logger.error(f'Invalid ticker {t}')
                sys.exit(1)

    def load(self) -> None:
        """
        Download data
        """
        self.logger.info('Downloading data')
        self.__clear()
        self.__download_indices()
        self.__download_currency()
