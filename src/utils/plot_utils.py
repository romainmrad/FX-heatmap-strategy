import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


def plot_fx_rate(ticker: str, data: pd.Series, path: str | os.PathLike[str]) -> None:
    """
    Plot a time series
    :param ticker: ticker symbol
    :param data: the data to plot
    :param path: the output path
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data, legend=False)
    plt.title(f'Evolution of {ticker}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature(args: tuple[str, str, pd.Series, pd.Series, str]) -> str:
    """
    Plot a feature
    :param args: arguments
    """
    ticker, feature_name, rate, feature_data, path = args
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))
    data = pd.concat([rate, feature_data], axis=1)
    data.reset_index(inplace=True)
    data.index = pd.to_datetime(data['Date'])
    data.drop('Date', axis=1, inplace=True)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    sns.lineplot(data, legend=False)
    plt.title(f'Visualisation of {feature_name} for {ticker}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return f'{ticker}_{feature_name}'
