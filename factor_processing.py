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

# IMPORT ALL_SCORES AND PREPROCESSING___________________________________

n=120
from_time = '2018-01-01'
to_time = (date.today()).strftime("%Y-%m-%d")
all_scores = pd.read_csv(f'{DAILY_DATA_FOLDER}/all_scores_{n}_{to_time}.csv', parse_dates=['Date'], index_col=['Date'])
all_scores_grouped = all_scores.groupby('Ticker')

top120_industry = pd.read_csv(f'{DAILY_DATA_FOLDER}/top{n}_industry_{to_time}.csv',index_col=0)
top120_industry = top120_industry[top120_industry.name != 'BAF']
stocks_industry = {}
industries = top120_industry[top120_industry.name.isin(all_scores.Ticker.unique())].lv1.unique()
for industry in industries:
    stocks_industry[industry] = top120_industry[top120_industry.name.isin(all_scores.Ticker.unique())].groupby('lv1').get_group(industry)

value_factors = ['BTOP','ETOP','STOP']
liquidity_factors = ['STOM', 'STOQ', 'STOA', 'ATVR']
# volatility_factors = ['HSIGMA','DASTD','CMRA']
volatility_factors = ['HSIGMA', 'DASTD','beta']
momentum_factors = ['relative_strength','LTRS','STRS']

marketcap = {}
for ticker in top120_industry.name.values:
    marketcap[ticker] = pd.read_csv(f'/data/VIETSTOCK_STS/sts_price/{ticker}.csv', parse_dates = ['Date'], index_col='Date')['MarketCap']
marketcap = pd.DataFrame(marketcap)

## assign weights for value crowding scores by market cap
# value_weights = {}
# for industry in industries:
#     stocks = stocks_industry[industry].name.values
#     value_weights[industry] = [(marketcap[stocks].sum(axis=1).iloc[-1])/marketcap.sum(axis=1).iloc[-1]]
# value_weights = list(value_weights.values())

value_weights = [1]*len(industries)
# FACTOR PROCESSING___________________________________

## VALUE CROSS-SECTIONAL CROWDING
def value_cross_crowding(ticker_scores):
    ticker_scores_value = ticker_scores[value_factors]
    value_cross_crowding = ticker_scores_value.mean(axis = 1)
    return value_cross_crowding

## LIQUIDITY CROSS-SECTIONAL CROWDING
def liquidity_cross_crowding(ticker_scores):
    ticker_scores_value = ticker_scores[liquidity_factors]
    value_cross_crowding = ticker_scores_value.mean(axis = 1)
    return value_cross_crowding

## VOLATILITY CROSS-SECTIONAL CROWDING
def volatility_cross_crowding(ticker_scores):
    ticker_scores_value = ticker_scores[volatility_factors]
    value_cross_crowding = ticker_scores_value.mean(axis = 1)
    return value_cross_crowding

## MOMENTUM CROSS-SECTIONAL CROWDING
def momentum_cross_crowding(ticker_scores):
    ticker_scores_value = ticker_scores[momentum_factors]
    value_cross_crowding = ticker_scores_value.mean(axis = 1)
    return value_cross_crowding

## VALUE TIME-SERIES CROWDING

def finance_value_time_crowding(tickers = stocks_industry['Tài chính và bảo hiểm'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_time_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_time_crowding.index = pd.to_datetime(value_time_crowding.index)
    return value_time_crowding

def agri_value_time_crowding(tickers=stocks_industry['Sản xuất nông nghiệp'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding

def oil_value_time_crowding(tickers=stocks_industry['Khai khoáng'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding

def ulti_value_time_crowding(tickers=stocks_industry['Tiện ích'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding

def estate_value_time_crowding(tickers=stocks_industry['Xây dựng và Bất động sản'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding

def manu_value_time_crowding(tickers=stocks_industry['Sản xuất'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding
def wholesale_value_time_crowding(tickers=stocks_industry['Bán buôn'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding
def retail_value_time_crowding(tickers=stocks_industry['Bán lẻ'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding
def transport_value_time_crowding(tickers=stocks_industry['Vận tải và kho bãi'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding
def it_value_time_crowding(tickers=stocks_industry['Công nghệ và thông tin'].name.values):
    all_stop_value = pd.DataFrame()
    for ticker in tickers:
        all_stop_value = pd.concat([all_stop_value, all_scores_grouped.get_group(ticker)['value']], axis = 1)
    all_stop_value = all_stop_value.sort_index()
    all_stop_value.columns = tickers
    value_cross_crowding = all_stop_value.sub(all_stop_value.median(axis=1), axis=0).div(all_stop_value.rolling(window=252).std(), axis=0)
    value_cross_crowding.index = pd.to_datetime(value_cross_crowding.index)
    return value_cross_crowding

def value_time_crowding():
    finance = finance_value_time_crowding()*value_weights[0][0]
    agri = agri_value_time_crowding()*value_weights[1][0]
    oil = oil_value_time_crowding()*value_weights[2][0]
    ulti = ulti_value_time_crowding()*value_weights[3][0]
    estate = estate_value_time_crowding()*value_weights[4][0]
    manu = manu_value_time_crowding()*value_weights[5][0]
    wholesale = wholesale_value_time_crowding()*value_weights[6][0]
    retail = retail_value_time_crowding()*value_weights[7][0]
    transport = transport_value_time_crowding()*value_weights[8][0]
    it = it_value_time_crowding()*value_weights[9][0]

    value_time_crowding = pd.concat([finance,agri,oil,ulti,estate,manu,wholesale,retail,transport,it], axis = 1)
    return value_time_crowding
    
## LIQUIDITY CROSS-SECTIONAL CROWDING

def liquidity_time_crowding(tickers):
    all_atvr_value = pd.DataFrame()
    for ticker in tickers:
        all_atvr_value = pd.concat([all_atvr_value, all_scores_grouped.get_group(ticker)['liquidity']], axis = 1)
    all_atvr_value = all_atvr_value.sort_index()
    all_atvr_value.columns = tickers
    liquidity_time_crowding = all_atvr_value.shift(22).sub(all_atvr_value.shift(22).median(axis=1), axis=0).div(all_atvr_value.shift(22).rolling(window=252).std(), axis=0)
    liquidity_time_crowding.index = pd.to_datetime(liquidity_time_crowding.index)
    return liquidity_time_crowding

## VOLATILITY TIME-SERIES CROWDING
def volatility_time_crowding(tickers):
    all_dastd_value = pd.DataFrame()
    for ticker in tickers:
        all_dastd_value = pd.concat([all_dastd_value, all_scores_grouped.get_group(ticker)['volatility']], axis = 1)
    all_dastd_value = all_dastd_value.sort_index()
    all_dastd_value.columns = tickers
    dastd_time_crowding = all_dastd_value.sub(all_dastd_value.median(axis=1), axis=0).div(all_dastd_value.rolling(window=252).std(), axis=0)
    dastd_time_crowding.index = pd.to_datetime(dastd_time_crowding.index)
    return dastd_time_crowding

## MOMENTUM TIME-SERIES CROWDING

def momentum_time_crowding(tickers):
    all_volume_value = pd.DataFrame()
    for ticker in tickers:
        all_volume_value = pd.concat([all_volume_value, all_scores_grouped.get_group(ticker)['momentum']], axis = 1)
    all_volume_value = all_volume_value.sort_index()
    all_volume_value.columns = tickers
    momentum_time_crowding = all_volume_value.sub(all_volume_value.median(axis=1), axis=0).div(all_volume_value.rolling(window=252).std(), axis=0)
    momentum_time_crowding.index = pd.to_datetime(momentum_time_crowding.index)
    return momentum_time_crowding


