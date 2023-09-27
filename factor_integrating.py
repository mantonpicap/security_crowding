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
sys.path.append(os.getcwd())
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
from config import *
from factor_processing import *


# INTEGRATING FACTOR SCORES___________________________________

## CROSS-SECTIONAL CROWDING
cross_crowding = {}
tickers = top120_industry[top120_industry.name.isin(all_scores.Ticker.unique())].name.tolist()
for ticker in tickers:
    ticker_scores = all_scores_grouped.get_group(ticker)
    if ticker_scores.count().max() >= 500:
        cross_crowding[ticker] = (liquidity_cross_crowding(ticker_scores)+ 7*value_cross_crowding(ticker_scores)+
                                volatility_cross_crowding(ticker_scores)+ momentum_cross_crowding(ticker_scores))/10
    # else:
    #     print(ticker)
    #     print(ticker_scores.count().max())
cross_crowding = pd.DataFrame(cross_crowding)
cross_crowding = cross_crowding.dropna(axis=1, subset=cross_crowding.index[-1])
cross_crowding.to_csv(f'{DAILY_DATA_FOLDER}/cross_scores_{to_time}.csv')

## TIME-SERIES CROWDING
time_crowding = (7*value_time_crowding() + liquidity_time_crowding(cross_crowding.columns) + 
                 volatility_time_crowding(cross_crowding.columns) + momentum_time_crowding(cross_crowding.columns))/10
# time_crowding = time_crowding[cross_crowding.columns]
time_crowding = time_crowding[cross_crowding.columns]
time_crowding = time_crowding.dropna(axis=1, subset=time_crowding.index[-1])
time_crowding.to_csv(f'{DAILY_DATA_FOLDER}/time_scores_{to_time}.csv')


## INTEGRATED CROWDING
integrated_crowding = (cross_crowding + time_crowding)
integrated_crowding = integrated_crowding.dropna(axis=1, subset=integrated_crowding.index[-1])

integrated_crowding.to_csv(f'{DAILY_DATA_FOLDER}/integrated_scores_{to_time}.csv')
