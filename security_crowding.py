import numpy as np
import pandas as pd

from glob import glob
import glob
import os
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import math
from math import sqrt, exp
from datetime import date
import datetime as dt
from datetime import date
import seaborn as sns
from datetime import datetime
from tabulate import tabulate
from functools import reduce
import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rework_backtrader.simulation.alpha_backbone import VietQuantBaseStrategy
from rework_backtrader.simulation.alpha_backbone.abstract_strategy_backbone import AbstractStrategy
from rework_backtrader.simulation import SimulationEngine
from backtrader.indicators.rsi import RSI_EMA
from rework_backtrader.utils.file_accessor import FileAccessor
from backtrader import Order
from scipy.stats.mstats import winsorize
from pathlib import Path
from rework_backtrader.simulation.funda_column_selector import FundaColumnSelector
from rework_backtrader.utils.category_elastic_reader import categoryElasticReader
from pandas.api.types import is_numeric_dtype
import sys
from datetime import timedelta
import factor_processing
from factor_processing import *
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax

#------------------------------------------------

# importing package
import matplotlib.pyplot as plt
import numpy as np
 
# create data
x = integrated_crowding.sort_values(by = integrated_crowding.index[-1], axis = 1).iloc[:,-20:].columns
time = time_crowding.loc[integrated_crowding.index[-1],x]
cross = cross_crowding.loc[integrated_crowding.index[-1],x]
# liquidity = np.array([20, 25, 15, 25])
# volatility = np.array([12, 15, 19, 6])
# momentum = np.array([10, 29, 13, 19])
 # Create an array to store the bottom values for positive and negative bars
bottoms_positive = np.zeros(len(x))
bottoms_negative = np.zeros(len(x))
plt.figure(figsize=(21,10))
# Plot positive values
plt.bar(x, np.maximum(0, time), color='r', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, time)

plt.bar(x, np.maximum(0, cross), color='b', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, cross)

# Plot negative values
plt.bar(x, np.minimum(0, time), color='r', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, time)

plt.bar(x, np.minimum(0, cross), color='b', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, cross)

plt.plot(x, integrated_crowding.loc[integrated_crowding.index[-1],x], color='purple', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='Sum of Components')

plt.xlabel("Securities")
plt.ylabel("Score")
plt.legend(["integrated", "Time Crowding", "Cross Crowding"])
plt.title("Scores by Security")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# create data
x = integrated_crowding.sort_values(by = integrated_crowding.index[-1], axis = 1).iloc[:,-20:].columns
# time = time_crowding.loc[integrated_crowding.index[-1],x]
value_time = value_time_crowding().loc[integrated_crowding.index[-1],x]
liquidity_time = liquidity_time_crowding(cross_crowding.columns).loc[integrated_crowding.index[-1],x]
volatility_time = volatility_time_crowding(cross_crowding.columns).loc[integrated_crowding.index[-1],x]
momentum_time = momentum_time_crowding(cross_crowding.columns).loc[integrated_crowding.index[-1],x]
 
# Create an array to store the bottom values for positive and negative bars
bottoms_positive = np.zeros(len(x))
bottoms_negative = np.zeros(len(x))

# Plot bars in a stacked manner with positive and negative values separated
plt.figure(figsize=(21,10))

# Plot positive values
plt.bar(x, np.maximum(0, 0.7*value_time), color='r', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, 0.7*value_time)

plt.bar(x, np.maximum(0, 0.1*liquidity_time), color='b', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, 0.1*liquidity_time)

plt.bar(x, np.maximum(0, 0.1*volatility_time), color='y', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, 0.1*volatility_time)

plt.bar(x, np.maximum(0, 0.1*momentum_time), color='g', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, 0.1*momentum_time)

# Plot negative values
plt.bar(x, np.minimum(0, 0.7*value_time), color='r', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, 0.7*value_time)

plt.bar(x, np.minimum(0, 0.1*liquidity_time), color='b', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, 0.1*liquidity_time)

plt.bar(x, np.minimum(0, 0.1*volatility_time), color='y', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, 0.1*volatility_time)

plt.bar(x, np.minimum(0, 0.1*momentum_time), color='g', bottom=bottoms_negative)

plt.plot(x, time_crowding.loc[integrated_crowding.index[-1],x], color='purple', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='Sum of Components')


plt.xlabel("Securities")
plt.ylabel("Score")
plt.legend(['time',"value", 'liquidity', 'volatility', 'momentum'])
plt.title("Time Scores by Security")
plt.axhline(0, color='black', linewidth=0.8)  # Add a line at y=0 for reference
plt.show()


# CROSS-SECTIONAL CROWDING
all_scores_grouped = factor_processing.all_scores_grouped
value_cross = {}
liquidity_cross = {}
volatility_cross = {}
momentum_cross = {}
for ticker in all_scores.Ticker.unique():
    ticker_scores = all_scores_grouped.get_group(ticker)
    if ticker_scores.count().max() >= 500:
        value_cross[ticker] = 0.7*value_cross_crowding(ticker_scores)
        liquidity_cross[ticker] = 0.1*liquidity_cross_crowding(ticker_scores)
        volatility_cross[ticker] = 0.1*volatility_cross_crowding(ticker_scores)
        momentum_cross[ticker] = 0.1*momentum_cross_crowding(ticker_scores)
value_cross = pd.DataFrame(value_cross)
liquidity_cross = pd.DataFrame(liquidity_cross)
volatility_cross = pd.DataFrame(volatility_cross)
momentum_cross = pd.DataFrame(momentum_cross)

# create data
x = integrated_crowding.sort_values(by = integrated_crowding.index[-1], axis = 1).iloc[:,-20:].columns
# time = time_crowding.loc[integrated_crowding.index[-1],x]
value_cross = value_cross.loc[integrated_crowding.index[-1],x]
liquidity_cross = liquidity_cross.loc[integrated_crowding.index[-1],x]
volatility_cross = volatility_cross.loc[integrated_crowding.index[-1],x]
momentum_cross = momentum_cross.loc[integrated_crowding.index[-1],x]
 
# Create an array to store the bottom values for positive and negative bars
bottoms_positive = np.zeros(len(x))
bottoms_negative = np.zeros(len(x))

# Plot bars in a stacked manner with positive and negative values separated
plt.figure(figsize=(21,10))

# Plot positive values
plt.bar(x, np.maximum(0, value_cross), color='r', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, value_cross)

plt.bar(x, np.maximum(0, liquidity_cross), color='b', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, liquidity_cross)

plt.bar(x, np.maximum(0, volatility_cross), color='y', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, volatility_cross)

plt.bar(x, np.maximum(0, momentum_cross), color='g', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, momentum_cross)

# Plot negative values
plt.bar(x, np.minimum(0, value_cross), color='r', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, value_cross)

plt.bar(x, np.minimum(0, liquidity_cross), color='b', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, liquidity_cross)

plt.bar(x, np.minimum(0, volatility_cross), color='y', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, volatility_cross)

plt.bar(x, np.minimum(0, momentum_cross), color='g', bottom=bottoms_negative)
plt.plot(x, cross, color='purple', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='Sum of Components')

plt.xlabel("Securities")
plt.ylabel("Score")
plt.legend(['cross',"value", 'liquidity', 'volatility', 'momentum'])
plt.title("Cross Scores by Security")
plt.axhline(0, color='black', linewidth=0.8)  # Add a line at y=0 for reference
plt.show()



import matplotlib.pyplot as plt
import numpy as np
# CROSS-SECTIONAL CROWDING
liquidity_cross_scores = pd.pivot_table(all_scores, values = liquidity_factors, index=all_scores.index, columns = 'Ticker')
value_cross = liquidity_cross_scores['STOM']
liquidity_cross = liquidity_cross_scores['STOQ']
volatility_cross = liquidity_cross_scores['STOA']
momentum_cross = liquidity_cross_scores['ATVR']

# create data
x = integrated_crowding.sort_values(by = integrated_crowding.index[-1], axis = 1).iloc[:,:].columns
# time = time_crowding.loc[integrated_crowding.index[-1],x]
value_cross = value_cross.loc[integrated_crowding.index[-1],x]
liquidity_cross = liquidity_cross.loc[integrated_crowding.index[-1],x]
volatility_cross = volatility_cross.loc[integrated_crowding.index[-1],x]
momentum_cross = momentum_cross.loc[integrated_crowding.index[-1],x]
liquidity_cross_scores = (value_cross+liquidity_cross+volatility_cross+momentum_cross)/4

 
# Create an array to store the bottom values for positive and negative bars
bottoms_positive = np.zeros(len(x))
bottoms_negative = np.zeros(len(x))

# Plot bars in a stacked manner with positive and negative values separated
plt.figure(figsize=(21,10))

# Plot positive values
plt.bar(x, np.maximum(0, value_cross), color='r', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, value_cross)

plt.bar(x, np.maximum(0, liquidity_cross), color='b', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, liquidity_cross)

plt.bar(x, np.maximum(0, volatility_cross), color='y', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, volatility_cross)

plt.bar(x, np.maximum(0, momentum_cross), color='g', bottom=bottoms_positive)
bottoms_positive += np.maximum(0, momentum_cross)

# Plot negative values
plt.bar(x, np.minimum(0, value_cross), color='r', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, value_cross)

plt.bar(x, np.minimum(0, liquidity_cross), color='b', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, liquidity_cross)

plt.bar(x, np.minimum(0, volatility_cross), color='y', bottom=bottoms_negative)
bottoms_negative += np.minimum(0, volatility_cross)

plt.bar(x, np.minimum(0, momentum_cross), color='g', bottom=bottoms_negative)
plt.plot(x, liquidity_cross_scores, color='purple', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='Sum of Components')

plt.xlabel("Securities")
plt.ylabel("Score")
plt.legend(['liquidity_scores',"stom", 'stoq', 'stoa', 'atvr'])
plt.title("Liquidity Scores by Security")
plt.axhline(0, color='black', linewidth=0.8)  # Add a line at y=0 for reference
plt.show()


####################PERFORMANCE EVALUATION

def create_quintile_portfolios(stocks, scores):
    '''
    Return: 
    {'quintile_1': [('StockD', 9), ('StockA', 12)],
    'quintile_2': [('StockI', 14), ('StockF', 18)],
    'quintile_3': [('StockC', 21), ('StockH', 27)],
    'quintile_4': [('StockG', 32), ('StockB', 34)],
    'quintile_5': [('StockJ', 40), ('StockE', 45)]}

    Example data: list of stocks and their scores
    stocks = ['StockA', 'StockB', 'StockC', 'StockD', 'StockE', 'StockF', 'StockG', 'StockH', 'StockI', 'StockJ']
    scores = [12, 34, 21, 9, 45, 18, 32, 27, 14, 40]
    '''
    combined_data = list(zip(stocks, scores))
    sorted_data = sorted(combined_data, key=lambda x: x[1])

    num_stocks = len(sorted_data)
    stocks_per_quintile = num_stocks // 5

    quintile_portfolios = {}
    for i in range(5):
        start_idx = i * stocks_per_quintile
        end_idx = start_idx + stocks_per_quintile
        quintile = i + 1
        stocks_in_quintile = sorted_data[start_idx:end_idx]
        quintile_portfolios[f'quintile_{quintile}'] = stocks_in_quintile
    
    return quintile_portfolios

weekly_averages = integrated_crowding.resample('W-MON').mean()
def create_weekly_quintile_portfolio(weekly_averages):
    # Create a DataFrame to store portfolio assignments
    portfolio_assignments = pd.DataFrame(index=weekly_averages.index)
    week_starts = []
    quintile_portfolios = {}
    # Calculate ranks for each week
    for week_start, week_scores in weekly_averages.iterrows():
        quintile_portfolios[week_start] = create_quintile_portfolios(stocks=week_scores.index, scores=week_scores.values)
    return quintile_portfolios

quintile_portfolios = create_weekly_quintile_portfolio(weekly_averages)









dfs = {}
for name in integrated_crowding.columns:
    df = pd.read_csv(f'/data/live_data/daily/{name}.csv', index_col=0, parse_dates=[0])
    df = df.fillna(method="ffill")
    df = df.loc[from_time:to_time, 'Close']
    dfs[name] = df

daily_data = pd.DataFrame(dfs)
stock_returns = np.log(daily_data) - np.log(daily_data).shift(1)
stock_returns = stock_returns[1:].fillna(0)

def calculate_next_week_cumulative_return(factor_portfolios, stock_returns):
    '''
    Calculate the next week cumulative return of portfolios based on given factor portfolios and daily stock returns.
    
    Args:
        factor_portfolios (dict): Weekly factor portfolios for each factor.
        stock_returns (DataFrame): Daily stock returns with dates as index and stock symbols as columns.
        
    Returns:
        next_week_cumulative_returns (dict): Next week cumulative returns for each portfolio and factor.
    '''
    next_week_cumulative_returns = {}

    for week_start, portfolio_data in factor_portfolios.items(): # week_start and quintile portfolios
        cumulative_returns = pd.DataFrame(columns=portfolio_data.keys()) # quintile_1 
        for quintile, stocks in portfolio_data.items(): # quintile, stock and scores
            cumulative_portfolio_return = []
            for stock, _ in stocks: # stock and score
                stock_return = stock_returns.loc[(stock_returns.index >= week_start+timedelta(days=7)) & 
                                                 (stock_returns.index < week_start+timedelta(days=14))][stock]
                cumulative_portfolio_return.append(stock_return)
            if cumulative_portfolio_return:  # Ensure there are returns to calculate cumulative return
                cumulative_portfolio_return = pd.concat(cumulative_portfolio_return, axis=1).mean(axis=1)
            else:
                cumulative_portfolio_return = pd.Series(index=stock_returns.index, data=1)  # Default to 1 if no returns
            cumulative_returns[quintile] = cumulative_portfolio_return.cumsum()
        next_week_cumulative_returns[week_start.strftime('%Y-%m-%d')] = cumulative_returns

    return next_week_cumulative_returns

next_week_cumret = calculate_next_week_cumulative_return(quintile_portfolios, stock_returns)

portfolio_rets = pd.DataFrame(columns=next_week_cumret['2023-07-03'].columns)
for week_start in next_week_cumret.keys():
    portfolio_rets = pd.concat([portfolio_rets, next_week_cumret[week_start]], axis = 0)

portfolio_rets






dfs = []

for week_start, quintiles in quintile_portfolios.items(): # week start, quintile
    if week_start-timedelta(days=7) in quintile_portfolios.keys():
        prev_week_data = quintile_portfolios[week_start-timedelta(days=7)]
    else:
        prev_week_data = None
    if prev_week_data is not None:
        for quintile, stocks in quintiles.items(): # quintile, stocks (with score)
            for stock, _ in stocks: # stock, score

                old_quintile = None
                for prev_quintile, prev_stocks in prev_week_data.items():
                    for prev_stock, prev_old_quintile in prev_stocks:
                        if prev_stock == stock:
                            old_quintile = prev_quintile
                            break

                new_quintile = quintile
                dfs.append({
                    'week_start': week_start,
                    'stock': stock,
                    'old_quintile': old_quintile,
                    'new_quintile': new_quintile,
                })
    prev_week_data = quintiles

trace_stock = pd.DataFrame(dfs)
trace_stock[trace_stock.old_quintile!=trace_stock.new_quintile]
