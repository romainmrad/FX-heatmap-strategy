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
    def __init__(self, config_path: str, vision_model: ThermalVision = None):
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
        self.trading_days = self.config.getint('backtest', 'trading_days')
        self.currencies = self.config.get('backtest', 'list').split(',')

    def __load_data(self):
        """
        Load DXY all data
        """
        self.logger.info("Loading data")
        idx_path = self.config.get("data", "index_path")
        rts_path = self.config.get("data", "rates_path")
        spr_path = self.config.get("data", "bid_ask_spread_path")
        # Loading DXY
        self.logger.info("--> Loading DXY index")
        dxy = pd.read_csv(
            os.path.join(idx_path, "DX-Y.NYB.csv"),
            index_col="Date",
            parse_dates=True
        )["Close"]
        # Loading exchange rates
        self.logger.info("--> Loading Bid-Ask values")
        self.ask = pd.read_csv(os.path.join(spr_path, 'ask.csv'), index_col="date", parse_dates=True)
        self.bid = pd.read_csv(os.path.join(spr_path, 'bid.csv'), index_col="date", parse_dates=True)
        # Loading interest rates
        rates = {}
        for f in os.listdir(rts_path):
            name = f.replace(".csv", "")
            self.logger.info(f"--> Loading {name} risk-free rate")
            df = pd.read_csv(os.path.join(rts_path, f), index_col="date", parse_dates=True)['value']
            if name != 'USD':
                rates[f'USD{name}=X'] = df
            else:
                rates[name] = df
        rates = pd.DataFrame(rates).ffill().bfill()
        self.dxy = dxy
        # self.prices = prices
        self.rates = rates / 100

    def __initialise_portfolio(self) -> None:
        """
        Initialise the currencies portfolio with half USD and half FX
        """
        self.portfolio = {"USD": self.initial_capital / 2}
        first_row = self.ask.iloc[0]
        for c in self.currencies:
            usd_alloc = (self.initial_capital / 2) / len(self.currencies)
            self.portfolio[c] = usd_alloc * first_row[c]

    def __compute_interest(self, t, n_days) -> None:
        """
        Compute interest rate added value to portfolio
        """
        for k in self.portfolio.keys():
            self.portfolio[k] *= np.exp(self.rates.loc[t][k] * n_days / self.trading_days)

    def __compute_portfolio_value(self, t) -> float:
        """
        Compute USD value of portfolio
        """
        total = self.portfolio["USD"]
        for c in self.currencies:
            total += self.portfolio[c] / self.bid.loc[t][c]
        return total

    def __rebalance_portfolio(self, t, usd_cap) -> None:
        """
        Rebalance portfolio at time t
        :param t: timestep
        :param usd_cap: USD capital allocation
        """
        current_usd = self.portfolio["USD"]
        current_fx_val = sum(self.portfolio[c] / self.bid.loc[t][c] for c in self.currencies)
        # Difference in USD allocation
        delta_usd = usd_cap - current_usd
        if delta_usd > 0:
            # Need more USD -> sell FX at bid
            sell_val = min(delta_usd, current_fx_val)
            per_currency = sell_val / len(self.currencies)
            for c in self.currencies:
                units_to_sell = per_currency * self.bid.loc[t][c]
                self.portfolio[c] -= units_to_sell
                self.portfolio["USD"] += per_currency
        elif delta_usd < 0:
            # Need more FX -> buy FX at ask
            buy_val = -delta_usd
            per_currency = buy_val / len(self.currencies)
            for c in self.currencies:
                units_bought = per_currency * self.ask.loc[t][c]
                self.portfolio[c] += units_bought
                self.portfolio["USD"] -= per_currency

    def __get_heatmap_path(self, t) -> str | None:
        """
        Get heatmap path for timestep t
        :param t: timestep
        :return: path to heatmap
        """
        pr_path = os.path.join(self.config.get("backtest", "heatmaps_path"),
                               self.config.get('backtest', 'positive_path'),
                               f"{t.strftime('%Y-%m-%d')}.png")
        if os.path.exists(pr_path):
            return pr_path
        nr_path = os.path.join(self.config.get("backtest", "heatmaps_path"),
                               self.config.get('backtest', 'negative_path'),
                               f"{t.strftime('%Y-%m-%d')}.png")
        if os.path.exists(nr_path):
            return nr_path
        return None

    def __backtest(self) -> None:
        """
        Run backtest and return portfolio history with a single constant interest rate
        applied to all currencies
        """
        from tensorflow.keras.utils import load_img, img_to_array

        self.logger.info("Running backtest")
        common_index = self.dxy.index.intersection(self.ask.index)
        self.dxy = self.dxy.loc[common_index]
        self.rates = self.rates.loc[common_index]
        self.ask = self.ask.loc[common_index]
        self.bid = self.bid.loc[common_index]
        self.ask.sort_index(inplace=True)
        self.logger.info("--> Initializing Portfolio")
        # Initial portfolio
        self.__initialise_portfolio()
        # History and predictions list
        history = []
        predictions = {'Date': [], 'USD': [], 'FX': []}
        prev_t = self.dxy.index[0]
        # Running backtest
        self.logger.info("--> Walking through time")
        for t in self.dxy.index:
            t = t
            dt = (t - prev_t).days
            # Compute interest on portfolio
            self.__compute_interest(t=t, n_days=dt)
            # Compute portfolio value in USD
            total_value = self.__compute_portfolio_value(t=t)
            # Check if rebalance possible
            if dt < self.config.getint('backtest', 'n_days_rebalance'):
                history.append({"Date": t, "Total": total_value, **self.portfolio})
                self.logger.debug(f"------> Value at {t.strftime('%Y-%m-%d')}: {total_value:.4f} USD")
                continue
            prev_t = t
            self.logger.debug(f"--> Rebalance at {t.strftime('%Y-%m-%d')}: {total_value:.4f} USD")
            # Load heatmap
            heatmap_path = self.__get_heatmap_path(t=t)
            if heatmap_path is None:
                continue
            img = load_img(heatmap_path, color_mode="grayscale", target_size=(256, 256))
            img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
            # Predict next regime
            prediction = self.vision.predict(img_array)
            predictions['Date'].append(t)
            predictions['USD'].append(prediction[0][0])
            predictions['FX'].append(prediction[0][1])
            usd_capital = float(prediction[0][0]) * total_value
            # Rebalance portfolio
            self.__rebalance_portfolio(t, usd_capital)
            history.append({"Date": t, "Total": total_value, **self.portfolio})

        history_df = pd.DataFrame(history).set_index("Date")
        history_df.to_csv(os.path.join(self.config.get('backtest', 'history_path'),
                                       f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_backtest_history.csv"))
        self.history_df = history_df
        self.predictions = pd.DataFrame(predictions).set_index("Date")
        self.logger.info("Backtest complete")
        returns = history_df["Total"].pct_change()
        mu = returns.mean()
        sigma = returns.std()
        rfr = self.rates['USD'].mean()
        SR = np.sqrt(self.trading_days) * (mu - ((1 + rfr) ** (1 / 252) - 1)) / sigma
        self.logger.info(f"Strategy Sharpe Ratio: {SR}")

    def __plot_results(self):
        """
        Plot equity and drawdown
        """
        equity = self.history_df["Total"]
        start_date = equity.index[0]
        benchmark = []
        for t in equity.index:
            dt = (t - start_date).days
            benchmark.append(self.initial_capital * np.exp(self.rates.loc[t]['USD'] * dt / self.trading_days))
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
        axs[2].set_ylabel("CNN predictions -- Portfolio Weights")
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
