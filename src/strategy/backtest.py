import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
from datetime import datetime
from rotating_logger import RotatingLogger

from src.strategy.thermal_vision import ThermalVision


class BackTest:
    def __init__(self, config_path: str, vision_model: ThermalVision):
        """
        Initialize the backtest
        :param config_path: path to config file
        :param vision_model: vision model
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.initial_capital = self.config.getfloat('backtest', 'initial_capital')
        self.vision = vision_model
        self.logger = RotatingLogger()
        self.logger.info("Initializing BackTest")
        self.rate = self.config.getfloat('backtest', 'rate')

    def __load_data(self):
        """
        Load DXY and exchange rates as DataFrames
        """
        self.logger.info("Loading data")
        idx_path = self.config.get("data", "index_path")
        cur_path = self.config.get("data", "currency_path")
        self.logger.info("--> Loading DXY")
        dxy = pd.read_csv(
            os.path.join(idx_path, "DX-Y.NYB.csv"),
            index_col="Date",
            parse_dates=True
        )["Close"]
        currencies = {}
        for f in os.listdir(cur_path):
            name = f.replace(".csv", "")
            self.logger.info(f"--> Loading {name}")
            df = pd.read_csv(os.path.join(cur_path, f), index_col="Date", parse_dates=True)["Close"]
            currencies[name] = df
        prices = pd.DataFrame(currencies)
        prices["USD"] = 1.0
        self.dxy = dxy
        self.prices = prices

    def __backtest(self) -> None:
        """
        Run backtest and return portfolio history with a single constant interest rate
        applied to all currencies
        """
        from tensorflow.keras.utils import load_img, img_to_array

        self.logger.info("Running backtest")
        common_index = self.dxy.index.intersection(self.prices.index)
        self.prices = self.prices.loc[common_index]
        self.dxy = self.dxy.loc[common_index]
        currencies = [c for c in self.prices.columns if c != "USD"]
        self.logger.info("--> Initializing Portfolio")
        # Initial portfolio
        portfolio = {"USD": self.initial_capital / 2}
        first_row = self.prices.iloc[0]
        for c in currencies:
            usd_alloc = (self.initial_capital / 2) / len(currencies)
            portfolio[c] = usd_alloc * first_row[c]

        history = []
        predictions = {'Date': [], 'USD': [], 'FX': []}
        heatmap_root = self.config.get("backtest", "heatmaps_path")

        self.logger.info("--> Walking through time")
        prev_t = None
        for t, row in self.prices.iterrows():
            if prev_t is not None:
                dt = (t - prev_t).days
                for k in portfolio:
                    portfolio[k] *= np.exp(self.rate * dt / 365)
                if dt < self.config.getint('backtest', 'n_days_rebalance'):
                    continue
            prev_t = t
            # Compute portfolio value in USD
            total_value = portfolio["USD"]
            for c in currencies:
                total_value += portfolio[c] / row[c]
            self.logger.debug(f"--> Rebalancing at {t.strftime('%Y-%m-%d')}: {total_value:.4f} USD")

            # Heatmap prediction
            pr_path = os.path.join(heatmap_root, self.config.get('backtest', 'positive_path'),
                                   f"{t.strftime('%Y-%m-%d')}.png")
            nr_path = os.path.join(heatmap_root, self.config.get('backtest', 'negative_path'),
                                   f"{t.strftime('%Y-%m-%d')}.png")
            if not os.path.exists(pr_path) and not os.path.exists(nr_path):
                continue
            current_heatmap_path = pr_path if os.path.exists(pr_path) else nr_path
            img = load_img(current_heatmap_path, color_mode="grayscale", target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array / 255.0, axis=0)
            prediction = self.vision.predict(img_array)
            predictions['Date'].append(t)
            predictions['USD'].append(prediction[0][0])
            predictions['FX'].append(prediction[0][1])
            usd_capital = prediction[0][0] * total_value
            fx_capital = prediction[0][1] * total_value
            # Rebalance portfolio
            portfolio = {"USD": usd_capital}
            for c in currencies:
                fx_alloc = fx_capital / len(currencies)
                portfolio[c] = fx_alloc * row[c]
            history.append({"Date": t, "Total": total_value, **portfolio})

        history_df = pd.DataFrame(history).set_index("Date")
        history_df.to_csv(os.path.join(self.config.get('backtest', 'history_path'),
                                       f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_backtest_history.csv"))
        self.history_df = history_df
        self.predictions = pd.DataFrame(predictions).set_index("Date")
        self.logger.info("Backtest complete")
        returns = history_df["Total"].pct_change()
        SR = returns.mean() / returns.std() * np.sqrt(12)
        self.logger.info(f"Strategy Sharpe Ratio: {SR}")

    def __plot_results(self):
        """
        Plot equity and drawdown
        """
        equity = self.history_df["Total"]
        start_date = equity.index[0]
        benchmark = []
        for t in equity.index:
            dt = (t - start_date).days / 365.0
            benchmark.append(self.initial_capital * np.exp(self.rate * dt))
        benchmark = pd.Series(benchmark, index=equity.index)
        dxy_norm = (self.dxy / self.dxy.iloc[0]) * self.initial_capital
        dxy_norm = dxy_norm.reindex(equity.index).ffill()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axs[0].plot(equity, label="Equity Curve", color="blue")
        axs[0].plot(benchmark, label="Benchmark (cash at interest)", color="green", linestyle="--")
        axs[0].plot(dxy_norm, label="Benchmark (DXY index)", color="orange", linestyle="--")
        axs[0].set_ylabel("Portfolio Value")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(drawdown, label="Drawdown", color="red")
        axs[1].set_ylabel("Drawdown")
        axs[1].legend()
        axs[1].grid(True)
        sns.lineplot(ax=axs[2], data=self.predictions, linestyle="-", markers='o')
        axs[2].set_ylabel("CNN predictions")
        axs[2].set_xlabel("Date")
        axs[2].legend()
        axs[2].grid(True)
        plt.suptitle("Backtest Results", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.get('backtest', 'plot_path'),
                                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_backtest.pdf"))
        plt.close()

    def run(self):
        self.__load_data()
        self.__backtest()
        self.__plot_results()
