from datetime import date
from glob import glob
import os
from functools import reduce
from time import sleep
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import copy
import pandas as pd
import numpy as np
import glob
from tabulate import tabulate
import requests
from datetime import timedelta
from datetime import date
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from vq_tools.utils.others import get_all_symbols
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
from pandas.tseries.offsets import BDay

def quarterly_to_yearly(series,quarterReport):
    # quarterReport=df_combine['quarterReport']

    df = pd.concat([quarterReport, series], axis=1)
    df.index = range(len(df))
    grouped = df.groupby((df['quarterReport'] != df['quarterReport'].shift()).cumsum())

    output = pd.DataFrame(columns=['quarterReport', 'value'])
    past = 0
    lengths = []

    for group_num, group_data in grouped:
        unique_values = group_data.iloc[:,1].drop_duplicates()
        group_output = pd.DataFrame({'quarterReport': group_data['quarterReport'], 'value': unique_values})
        output = output.append(group_output, ignore_index=True)
        if past < 4:
            windows = sum(lengths) + group_data.index - min(group_data.index) + 1
            for index, row in group_data.iterrows():
                for window in windows:
                    output['rolling_sum'] = output['value'].rolling(window=window, min_periods=1).sum()
        elif past >= 4:
            lengths.pop(0)
            windows = sum(lengths) + group_data.index - min(group_data.index) + 1
            for index, row in group_data.iterrows():
                for window in windows:
                    output['rolling_sum'] = output['value'].rolling(window=window, min_periods=1).sum()
        past = past + 1
        lengths.append(len(group_data))
    output.index = series.index
    return output['rolling_sum']
def halflife(half_life, length):
    t = np.arange(length)
    w = 2 ** (t / half_life) / sum(2 ** (t / half_life)) 
    return w

# Define the number of months to use for the calculation
NUM_MONTHS = 12
##################### MOMENTUM#####################
# Define a function to calculate the long-term momentum factor for a stock based on the Barra risk model
# def calculate_barra_momentum(df):
#     # Calculate the daily percentage change in closing price
#     df['daily_return'] = df['Close'].pct_change()
#     # Calculate the momentum score for each day using a rolling window
#     window_size = NUM_MONTHS * 20
#     momentum = df['daily_return'].rolling(window_size).apply(lambda x: (x[:-20].mean() - x[-20:].mean()) / x[:-20].std() if x[:-20].std() != 0 else np.nan, raw=False)
#     # Add the momentum score to the DataFrame
#     df['momentum'] = momentum
#     # Drop any rows with NaN values
#     df = df.dropna(subset=['momentum'])
#     return df[['Ticker', 'momentum']]
def calculate_barra_relative_strength(df_stock, df_market, T=252, half_life=252/4, L=3):
    # Calculate the daily percentage change in closing price
    df_stock['ret_stock'] = df_stock['Close'].pct_change()
    df_market['ret_mkt'] = df_market['Close'].pct_change()
    
    # Calculate the long-term relative strength score for each day using a rolling window
    window_size = NUM_MONTHS * 20
    half_life = window_size/2
    log_excess_returns = np.log(1 + df_stock['ret_stock']) - np.log(1 + df_market['ret_mkt'])
    # Calculate the exponential weights
    def halflife(half_life, length):
        t = np.arange(length)
        w = 2 ** (t / half_life) / sum(2 ** (t / half_life)) 
        return w
    w = halflife(half_life, T)
    # Calculate the exponentially-weighted sum of log excess returns
    rs = np.zeros(len(df_stock))
    for t in range(T, len(df_stock) + 1):
        rt = log_excess_returns[(t - T): t]
        rs[t - 1] = np.sum(rt * w)
    # Calculate the equal-weighted average over a lagged window and reverse the sign
    rstr = np.zeros(len(df_stock))
    for t in range(T + L + L, len(df_stock) + 1):
        rstr[t - 1] = -1 * np.mean(rs[(t - L - L): (t - L)])
    # Add the rstr values to the DataFrame
    df_stock['relative_strength'] = rstr
    return df_stock[['Ticker', 'relative_strength']]

# def HALPHA(self, T=int(252), L=11):
#     halpha = np.zeros(self.length)
#     _, _, alpha = self.beta(T)
#     for t in range(T + L + L, self.length + 1):
#         halpha[t-1] = np.mean(alpha[(t-L-L): (t-L)])
#     return halpha

##################### REVERSAL#####################
# Long-Term Relative Strength
def calculate_barra_long_term_relative_strength(df_stock, df_market):
    # Calculate the daily percentage change in closing price
    df_stock['ret_stock'] = df_stock['Close'].pct_change()
    df_market['ret_mkt'] = df_market['Close'].pct_change()
    
    # Calculate the long-term relative strength score for each day using a rolling window
    window_size = NUM_MONTHS * 20
    half_life = window_size/2
    log_excess_returns = np.log(1 + df_stock['ret_stock']) - np.log(1 + df_market['ret_mkt'])
    ewma = pd.Series.ewm(log_excess_returns, halflife=half_life, min_periods=window_size, adjust=True).mean()
    long_term_relative_strength = ewma.rolling(window_size).apply(lambda x: np.exp(x.sum()) - 1, raw=False)

    # Add the long-term relative strength score for the stock to the DataFrame
    df_stock['LTRS'] = long_term_relative_strength

    # Take the equal-weighted average of the non-lagged values over an 11-day window lagged by 273 days
    equal_weighted_average = df_stock['LTRS'].rolling(window=11).mean().shift(273)
    # Reverse the sign
    ltrstr = -1 * equal_weighted_average
    df_stock['LTRSTR'] = ltrstr

    # Drop any rows with NaN values
    df_stock = df_stock.dropna(subset=['LTRS'])
    return df_stock[['Ticker', 'LTRS']]
def calculate_barra_short_term_relative_strength(df_stock, df_market):
    # Calculate the daily percentage change in closing price
    df_stock['ret_stock'] = df_stock['Close'].pct_change()
    df_market['ret_mkt'] = df_market['Close'].pct_change()
    
    # Calculate the long-term relative strength score for each day using a rolling window
    window_size = int(NUM_MONTHS/4) * 20
    half_life = 10
    log_excess_returns = np.log(1 + df_stock['ret_stock']) - np.log(1 + df_market['ret_mkt'])
    ewma = pd.Series.ewm(log_excess_returns, halflife=half_life, min_periods=window_size, adjust=True).mean()
    long_term_relative_strength = ewma.rolling(window_size).apply(lambda x: np.exp(x.sum()) - 1, raw=False)

    # Add the long-term relative strength score for the stock to the DataFrame
    df_stock['STRS'] = long_term_relative_strength

    # Take the equal-weighted average of the non-lagged values over an 11-day window lagged by 273 days
    equal_weighted_average = df_stock['LTRS'].rolling(window=3).mean().shift(273)
    # Reverse the sign
    ltrstr = -1 * equal_weighted_average
    df_stock['STRSTR'] = ltrstr

    # Drop any rows with NaN values
    df_stock = df_stock.dropna(subset=['STRS'])
    return df_stock[['Ticker', 'STRS']]

##################### BETA
def calculate_barra_beta(df_stock, df_market):
    # Calculate the daily percentage change in closing price
    df_stock['ret_stock'] = df_stock['Close'].pct_change()
    df_market['ret_mkt'] = df_market['Close'].pct_change()

    # Calculate the beta score for each day using a rolling window
    window_size = NUM_MONTHS * 20
    daily_returns_stock = df_stock['Close'].pct_change()
    daily_returns_market = df_market['Close'].pct_change()
    rolling_covariance = daily_returns_stock.rolling(window_size).cov(daily_returns_market)
    rolling_variance = daily_returns_market.rolling(window_size).var()
    rolling_beta = rolling_covariance / rolling_variance
    df_stock['beta'] = rolling_beta

    # Drop any rows with NaN values
    df_stock = df_stock.dropna(subset=['beta'])
    return df_stock[['Ticker', 'beta']]

##################### PROFITABILITY
def calculate_barra_profitability_factor(df_combine):
    ## Asset Turnover
    # Mapping sales column for manufacturing, insurance, bank respectively
    df_combine['ATO'] = df_combine['sales'] / df_combine['totalAssets']

    ## Gross Profitability
    df_combine['costOfSales'] = df_combine['costOfSales'].apply(lambda x: -x if pd.notnull(x) and x > 0 else x)
    # Mapping costOfSales column for manufacturing, insurance, bank respectively
    # df_combine['GP'] = (quarterly_to_yearly(df_combine['sales'].fillna(df_combine['netSalesFromInsuranceBusiness'].fillna(df_combine['interestAndSimilarIncome']))) 
    #           + quarterly_to_yearly(df_combine['costOfSales'].fillna(df_combine['totalDirectInsuranceOperatingExpenses'].fillna(df_combine['interestAndSimilarExpenses'])))) / (df_combine['totalAssets'])

    ##Gross Profit Margin
    df_combine['GPM'] = (df_combine['sales']  
               + df_combine['costOfSales']) / (df_combine['sales'])

    ## Return on Assets
    df_combine['ROA'] = df_combine['attributableToParentCompany'] / df_combine['totalAssets']

    ## Return on Equity
    df_combine['ROE'] = df_combine['attributableToParentCompany'] / df_combine['ownersEquity']

    # Drop any rows with NaN values
    df_combine = df_combine.dropna(subset=['ATO', 'GPM', 'ROA', 'ROE'])
    return df_combine[['Ticker', 'ATO', 'GPM', 'ROA', 'ROE']]

##################### LIQUIDITY
def calculate_barra_liquidity_factors(df_combine):
    # Monthly share turnover
    df_combine['STOM'] = np.log(df_combine['M_TotalVol'].rolling(window=20).sum() / (df_combine['MarketCap']*1e6 / df_combine['ClosePrice']))

    # Quarterly share turnover
    df_combine['STOQ'] = np.log(df_combine['M_TotalVol'].rolling(window=60).sum() / (df_combine['MarketCap']*1e6 / df_combine['ClosePrice']))

    # Annual Share Turnover
    df_combine['STOA'] = np.log(df_combine['M_TotalVol'].rolling(window=240).sum() / (df_combine['MarketCap']*1e6 / df_combine['ClosePrice']))

    # Annualized Traded Value Ratio
    window_size = 252
    half_life = window_size/4
    df_combine['TVR'] = df_combine['M_TotalVol'] / (df_combine['MarketCap']*1e6 / df_combine['ClosePrice']) # turnover ratio
    df_combine['ATVR'] = df_combine['TVR'].ewm(halflife=half_life, adjust=True).mean().rolling(window_size).apply(lambda x: np.log(x[-1] / x[:-20].mean()), raw=False)

    df_combine = df_combine.dropna(subset=['STOM', 'STOQ', 'STOA', 'ATVR'])
    return df_combine[['Ticker', 'STOM', 'STOQ', 'STOA', 'ATVR']]

##################### LEVERAGE
def calculate_barra_leverage_factor(df_combine):
    # Debt-to-Assets
    df_combine['DTOA'] = df_combine['liabilities'] / df_combine['totalAssets']

    # Market Leverage
    df_combine['MLEV'] = (df_combine['MarketCap'] * 1e6 + df_combine['longtermLiabilites'].fillna(df_combine['liabilities']) + df_combine['capitalAndReservesPreferredShares'].fillna(0)) / (df_combine['MarketCap'] * 1e6)

    #Book Leverage
    df_combine['BLEV'] = (df_combine['ownersEquity']+ df_combine['longtermLiabilites'].fillna(df_combine['liabilities']) + df_combine['capitalAndReservesPreferredShares'].fillna(0)) / df_combine['ownersEquity']

    df_combine = df_combine.dropna(subset=['DTOA', 'MLEV', 'BLEV'])
    return df_combine[['Ticker', 'DTOA', 'MLEV', 'BLEV']]

##################### VALUE
def calculate_barra_value(df_combine):
    # Book to Price
    df_combine['BTOP'] = (df_combine['ownersEquity'])/ (df_combine['MarketCap']*1e6)

    # Sales to PriceFdf.ind
    df_combine['STOP'] = df_combine['sales'] / (df_combine['MarketCap']*1e6)

    # Earning to Price
    df_combine['ETOP'] = df_combine['attributableToParentCompany'] / (df_combine['MarketCap'] * 1e6)
    # Drop any rows with NaN values
    df_combine = df_combine.dropna(subset=['BTOP', 'STOP', 'ETOP'])
    return df_combine[['Ticker', 'BTOP', 'STOP', 'ETOP']]

##################### VOLATILITY
def calculate_barra_residual_volatility(df_stock, df_market):
    # Calculate the excess returns for the stock and market
    df_stock['ret_stock'] = df_stock['Close'].pct_change()
    df_market['ret_mkt'] = df_market['Close'].pct_change()
    excess_returns = df_stock['ret_stock'] - df_market['ret_mkt']
    excess_returns = excess_returns.dropna()
    df_stock['excess_return'] = excess_returns
    
    # Calculate the historical sigma factor
    window_size = NUM_MONTHS * 20
    rolling_resid = excess_returns.rolling(window_size).apply(lambda x: sm.OLS(x, sm.add_constant(df_market['ret_mkt'][x.index])).fit().resid.std(), raw=False)
    df_stock['HSIGMA'] = rolling_resid

    # Calculate the DASTD factor
    # rolling_std = excess_returns.rolling(window_size).std()
    # df_stock['DASTD'] = rolling_std
    dastd_values = np.zeros(len(df_stock))
    
    for t in range(window_size, len(df_stock) + 1):
        excess_returns = df_stock['ret_stock'][(t - window_size):t] - df_market['ret_mkt'][(t - window_size):t]
        dastd_values[t - 1] = np.std(excess_returns)
        
    df_stock['DASTD'] = dastd_values

    # Calculate the Cumulative Range factor
    # cum_log_returns = np.log(df_stock['Close'] / df_stock['Close'].shift(1)).rolling(window_size).sum()
    # df_stock['CMRA'] = cum_log_returns.max() - cum_log_returns.min()
    cmra_values = np.zeros(len(df_stock))
    
    for k in range(NUM_MONTHS * 21, len(df_stock) + 1):
        r = []
        for i in range(NUM_MONTHS):
            r_month = np.prod(df_stock['Close'].pct_change().iloc[k - (i + 1) * 21:k - i * 21] + 1) - 1
            r.append(np.log(1 + r_month))
        Z = np.cumsum(r)
        cmra_values[k - 1] = max(Z) - min(Z)
        
    df_stock['CMRA'] = cmra_values
    # Drop any rows with NaN values
    df_stock = df_stock.dropna(subset=['HSIGMA', 'DASTD', 'CMRA'])

    return df_stock[['Ticker', 'excess_return', 'HSIGMA', 'DASTD', 'CMRA']]

##################### GROWTH
def calculate_slope_to_mean(df):
    copy_vec = df.fillna(method='ffill').fillna(method='bfill')
    y = copy_vec.unique()
    X = np.arange(len(y)).reshape(-1, 1)
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(X, y)
    return regressor.coef_[0] / np.nanmean(y)

def calculate_barra_growth(df_combine):
    # Define the size of the rolling window
    window_size = NUM_MONTHS * 20
    # Calculate the earnings per share growth rate for the past window_size days
    df_combine['eps'] = df_combine['attributableToParentCompany'] / (df_combine['MarketCap'] * 1e6 / df_combine['ClosePrice'])
    eps = df_combine['eps']
    eps_growth_rate = eps.rolling(window_size).apply(calculate_slope_to_mean)
    # eps_growth_rate = slope / eps.rolling(window_size).mean()
    df_combine['EGRO'] = eps_growth_rate

    # Calculate the sales per share growth rate for the past window_size days
    df_combine['sps'] = df_combine['sales'] / (df_combine['MarketCap'] * 1e6 / df_combine['ClosePrice'])
    sps = df_combine['sps']
    sps_growth_rate = sps.rolling(window_size).apply(calculate_slope_to_mean)
    # sps_growth_rate = slope / sps.rolling(window_size).mean()
    df_combine['SGRO'] = sps_growth_rate

    # Drop any rows with NaN values
    df_combine = df_combine.dropna(subset=['EGRO', 'SGRO'])
    return df_combine[['Ticker', 'EGRO', 'SGRO']]

##################### INVESTMENT QUALITY
def calculate_barra_investment_quality(df_combine):
    # Define the size of the rolling window
    window_size = NUM_MONTHS * 20
    # Calculate the Total Assets Growth Rate for the past window_size days
    total_asset = df_combine['totalAssets']
    asset_growth_rate = total_asset.rolling(window_size).apply(calculate_slope_to_mean)
    df_combine['AGRO'] = -1 * asset_growth_rate

    # Calculate the Issuance Growth for the past window_size days
    shares_outstanding = df_combine['MarketCap'] * 1e6 / df_combine['ClosePrice']
    shares_outstanding.ffill(inplace=True)
    issuance_growth_rate = shares_outstanding.rolling(window_size).apply(calculate_slope_to_mean)
    df_combine['IGRO'] = -1 * issuance_growth_rate

    # Calculate the Capital Expenditure Growth for the past window_size days
    capex = df_combine['diff_PPE']
    capex_growth_rate = capex.rolling(window_size).apply(calculate_slope_to_mean)
    df_combine['CXGRO'] = -1 * capex_growth_rate

    # Drop any rows with NaN values
    df_combine = df_combine.dropna(subset=['AGRO', 'IGRO', 'CXGRO'])
    return df_combine[['Ticker', 'AGRO', 'IGRO', 'CXGRO']]

##################### SIZE
def calculate_barra_size(df_combine):
    # Log of Market Capitalization
    df_combine['LNCAP'] = np.log((df_combine['MarketCap'] * 1e6))
    # Drop any rows with NaN values
    df_combine = df_combine.dropna(subset=['LNCAP'])
    return df_combine[['Ticker', 'LNCAP']]

##################### EARNINGS
def calculate_barra_earnings(df_combine):
    window_size = NUM_MONTHS * 20
    ## Earnings Quality
    # Accruals - Balance Sheet Version
    df_combine['ABS'] = ((df_combine['diff_CA']-df_combine['diff_cash'])-df_combine['diff_CL']-(df_combine['depreciationAndAmortisation'].fillna(df_combine['nan_depreciationAndAmortisation'])))/df_combine['totalAssets']
    # Accruals – Cash Flow Version
    df_combine['ACF'] = (-(df_combine['diff_AR'] + df_combine['diff_inv'] + df_combine['diff_AP'] + df_combine['diff_tax'] + df_combine['diff_OL'] + df_combine['diff_OA']) - (df_combine['depreciationAndAmortisation'].fillna(df_combine['nan_depreciationAndAmortisation']))) / df_combine['totalAssets']

    ## Earnings Variability
    # Variability in Sales
    df_combine['VSAL'] = df_combine['sales'].rolling(window_size).std() / (df_combine['sales'].rolling(window_size).mean())
    # Variability in Earnings
    df_combine['VERN'] = df_combine['attributableToParentCompany'].rolling(window_size).std() / df_combine['attributableToParentCompany'].rolling(window_size).mean()
    # Variability in Cash-flows
    df_combine['VFLO'] = df_combine['cashAndCashEquivalents'].rolling(window_size).std() / df_combine['cashAndCashEquivalents'].rolling(window_size).mean()
    df_combine = df_combine.dropna(subset=['ABS', 'ACF', 'VSAL', 'VERN', 'VFLO'])
    return df_combine[['Ticker', 'ABS', 'ACF', 'VSAL', 'VERN', 'VFLO']]

##################### EXTRA
def calculate_eps_growth(df_combine):
    df_combine['eps'] = df_combine['attributableToParentCompany'] / (df_combine['MarketCap']*1e6/df_combine['ClosePrice'])
    eps_trailing = df_combine['eps_sum']
    df_combine['eps_growth'] = df_combine['eps']/eps_trailing
    # Drop any rows with NaN values
    df_combine = df_combine.dropna(subset=['eps_growth'])
    return df_combine[['Ticker', 'eps_growth']]

def calculate_income_from_core_business(df_combine):
    df_combine['core_business'] = df_combine['grossProfit']/df_combine['core_sales']
    # Drop any rows with NaN values
    df_combine = df_combine.dropna(subset=['core_business'])
    return df_combine[['Ticker', 'core_business']]

def change_weekday(date):
    if date.weekday() == 5:  # Saturday
        return date - pd.Timedelta(days=1)
    elif date.weekday() == 6:  # Sunday
        return date - pd.Timedelta(days=2)
    else:
        return date

# Define a function to read in the CSV files and calculate the long-term momentum score for each stock on each day
def calculate_barra_descriptor_for_all_stocks(symbols: list, from_time: str, to_time: str):
    # Create an empty DataFrame to hold the all scores for all stocks on each day
    stock_scores = pd.DataFrame()
    all_scores = pd.DataFrame()
    # symbols = get_stock_symbols('hose')
    symbols = symbols
    print(symbols)
    print(len(symbols))
    # symbols = symbol
    # Loop through each CSV file
    for symbol in symbols:
        if len(symbol) == 3:
            # print(symbol)
            try:
                ### Read in the OHLCV data for the stock
                ## Daily data
                df = pd.read_csv('/data/live_data/daily/{}.csv'.format(symbol), parse_dates=['Date'], index_col=0)
                # df.index=pd.to_datetime(df.index)
                df = df.loc[from_time:to_time]
                # df.set_index('Date', inplace=True)
                df['Ticker'] = symbol

                ## Market data
                df_market = pd.read_csv('/data/live_data/daily/VNINDEX.csv'.format(symbol), parse_dates=['Date'], index_col=0)
                # df_market.index=pd.to_datetime(df_market.index)
                df_market = df_market.loc[from_time:to_time]
                df_market = df_market[df_market.index.isin(df.index)]
                df = df[df.index.isin(df_market.index)]

                ## STS data
                df_sts = pd.read_csv('/data/VIETSTOCK_STS/sts_price/{}.csv'.format(symbol), parse_dates=['Date'], index_col=0)
                # df_sts.index=pd.to_datetime(df_sts.index)
                df_sts['Ticker'] = symbol
                df_sts['MarketCap'] = df_sts['MarketCap'].replace(0, np.nan)
                df_sts['MarketCap'].ffill(inplace=True)
                df_sts = df_sts.loc[from_time:to_time]

                ## Fundamental data
                df_fdmt = pd.read_csv('/data/FIINTRADE_FDMT/{}.csv'.format(symbol), parse_dates=['Date'], index_col=0)
                df_fdmt.index = df_fdmt.index.map(change_weekday)

                        #Tangible fixed assets
                df_fdmt['diff_PPE'] = df_fdmt['tangibleFixedAssets'].shift(0) - df_fdmt['tangibleFixedAssets'].shift(1)

                        #Current Assets _ fillna with others terms expression
                calculated_current_assets = ( df_fdmt['totalAssets'] -
                                              df_fdmt['longtermFinancialAssets'].fillna(df_fdmt['longtermInvestments'].fillna(0)) -
                                              df_fdmt['fixedAssets'] -
                                              df_fdmt['provisionForLongTermAssets'].fillna(0)
                                            )
                df_fdmt['currentAssets'] = df_fdmt['currentAssets'].fillna(df_fdmt['totalAssets']   
                                                                            -df_fdmt['longtermFinancialAssets'].fillna(df_fdmt['longtermInvestments'].fillna(0))
                                                                            -df_fdmt['fixedAssets']
                                                                            -df_fdmt['provisionForLongTermAssets'].fillna(0))
                df_fdmt['currentAssets'] = np.where(df_fdmt['currentAssets'] == 0, calculated_current_assets, df_fdmt['currentAssets'])
                df_fdmt['diff_CA'] = (df_fdmt['currentAssets']).shift(0) - (df_fdmt['currentAssets']).shift(1)

                        #Cash _ fillna with equivalant term
                df_fdmt['diff_cash'] = (df_fdmt['cashAndCashEquivalents'].fillna(0)).shift(0) - (df_fdmt['cashAndCashEquivalents'].fillna(0).shift(1))

                        #CurrentLiabilities _ fillna with others terms expression
                calculated_liabilities = (df_fdmt['liabilities'] - df_fdmt['otherLiabilities'].fillna(0))
                df_fdmt['currentLiabilities'] = df_fdmt['currentLiabilities'].fillna(df_fdmt['liabilities'] - df_fdmt['otherLiabilities'].fillna(0))
                df_fdmt['currentLiabilities'] = np.where(df_fdmt['currentLiabilities'] == 0, calculated_liabilities, df_fdmt['currentLiabilities'])
                df_fdmt['diff_CL'] = df_fdmt['currentLiabilities'].shift(0) - df_fdmt['currentLiabilities'].shift(1)
                        
                    
                        #Account receivable _ fillna with 0 (update later)
                df_fdmt['diff_AR'] = (df_fdmt['accountsReceivables'].fillna(0)).shift(0) - (df_fdmt['accountsReceivables'].fillna(0)).shift(1)
                        #Inventories _ fillna with 0
                df_fdmt['diff_inv'] = (df_fdmt['inventories'].fillna(0)).shift(0) - (df_fdmt['inventories'].fillna(0)).shift(1)
                        #Trade account payaple
                df_fdmt['diff_AP'] = (df_fdmt['tradeAccountsPayable'].fillna(0)).shift(0) - (df_fdmt['tradeAccountsPayable'].fillna(0)).shift(1)

                        #Taxes expense
                calculated_taxes = (df_fdmt['businessIncomeTaxExpenses'].fillna(0))
                df_fdmt['taxesAndOtherPayableToStateBudget'] = df_fdmt['taxesAndOtherPayableToStateBudget'].fillna(df_fdmt['businessIncomeTaxExpenses'].fillna(0))
                df_fdmt['taxesAndOtherPayableToStateBudget'] = np.where(df_fdmt['taxesAndOtherPayableToStateBudget'] == 0, calculated_taxes, df_fdmt['taxesAndOtherPayableToStateBudget'])
                df_fdmt['diff_tax'] = df_fdmt['taxesAndOtherPayableToStateBudget'].shift(0) - (df_fdmt['taxesAndOtherPayableToStateBudget'].shift(1))

                        #Other Liabilities
                df_fdmt['diff_OL'] = df_fdmt['otherLiabilities'].shift(0) - df_fdmt['otherLiabilities'].shift(1)
                df_fdmt['diff_OL'] = df_fdmt['diff_OL'].fillna(0)
                        #Depreciation & Amortisation
                df_fdmt['nan_depreciationAndAmortisation']= df_fdmt['tangibleFixedAssetsAccumulatedDepreciation'].fillna(0) +df_fdmt['intagibleFixedAssetsAccumulatedDepreciation'].fillna(0) +df_fdmt['investmentPropertiesAccumulatedDepreciation'].fillna(0) + df_fdmt['amortisationOfGoodwill'].fillna(0)
                df_fdmt['nan_depreciationAndAmortisation'] = df_fdmt['nan_depreciationAndAmortisation'].apply(lambda x: -x if pd.notnull(x) and x < 0 else x)
                df_fdmt['nan_depreciationAndAmortisation'] = df_fdmt['nan_depreciationAndAmortisation'].fillna(0)

                df_fdmt['diff_OA'] = (df_fdmt['otherCurrentAssets'].fillna(0)).shift(0) - (df_fdmt['otherCurrentAssets'].fillna(0)).shift(1)
                shares_outstanding = (df_sts['MarketCap'] * 1e6) / df_sts['ClosePrice']
                df_fdmt['eps_sum'] = df_fdmt['attributableToParentCompany'].div(shares_outstanding, axis=0).dropna().rolling(4).sum()
                df_fdmt = df_fdmt.loc[from_time:to_time]

        

                ##  THE COMPLETED DAILY INFORCRITERIA  ##--------------------------------------------------------------------------------------------------------------------

                df_fdmt = df_fdmt.drop_duplicates(subset=['quarterReport','yearReport'] , keep='last')
                df_combine = pd.concat([df_sts, df_fdmt], axis=1)
                df_combine = df_combine[df_combine.index.isin(df.index)]
                df_combine['sales'] = df_combine['sales'].replace(0, np.nan)
                gross_profit_insurance = df_combine['grossInsuranceOperatingProfit']
                gross_profit_bank = df_combine['netInterestIncome']+df_combine['netFeeAndCommissionIncome']+df_combine['netGainLossFromTradingOfTradingSecurities']+df_combine['netGainLossFromDisposalOfInvestmentSecurities']
                sales_insurance = df_combine['netSalesFromInsuranceBusiness']
                sales_bank = gross_profit_bank + df_combine['otherIncomes']+df_combine['dividendsIncome'].fillna(0)
                df_combine['core_sales'] = quarterly_to_yearly(df_combine['sales'].fillna(sales_insurance.fillna(sales_bank)),quarterReport=df_combine['quarterReport'])
                
                df_combine['sales'] = quarterly_to_yearly(df_combine['sales'].fillna(df_combine['netSalesFromInsuranceBusiness'].fillna(df_combine['interestAndSimilarIncome'])),quarterReport=df_combine['quarterReport'])
                df_combine['costOfSales'] = df_combine['costOfSales'].replace(0, np.nan)
                df_combine['costOfSales'] = quarterly_to_yearly(df_combine['costOfSales'].fillna(df_combine['totalDirectInsuranceOperatingExpenses'].fillna(df_combine['interestAndSimilarExpenses'])),quarterReport=df_combine['quarterReport'])
                df_combine['attributableToParentCompany'] = quarterly_to_yearly(df_combine['attributableToParentCompany'],quarterReport=df_combine['quarterReport'])
                df_combine['grossProfit'] = df_combine['grossProfit'].replace(0, np.nan)
                df_combine['grossProfit'] = quarterly_to_yearly(df_combine['grossProfit'].fillna(gross_profit_insurance.fillna(gross_profit_bank)),quarterReport=df_combine['quarterReport'])
                df_combine.ffill(inplace=True)
                df_combine = df_combine.dropna(subset=['diff_CA'])

                df = df[df.index.isin(df_combine.index)]
                df_market = df_market[df_market.index.isin(df_combine.index)]

                ##################### SIZE
                size_scores = calculate_barra_size(df_combine)

                ##################### VALUE
                value_scores = calculate_barra_value(df_combine)
                value_scores['value'] = value_scores[['BTOP', 'STOP', 'ETOP']].mean(axis=1)

                ##################### MOMENTUM & REVERSAL
                relative_strength = calculate_barra_relative_strength(df, df_market)
                lt_reversal_scores = calculate_barra_long_term_relative_strength(df, df_market)
                st_reversal_scores = calculate_barra_short_term_relative_strength(df, df_market)
                # momentum_scores['momentum'] = momentum_scores[['relative_strength','lt_reversal_scores','st_reversal_scores']].mean(axis=1)
                
                ##################### QUALITY: PROFITABILITY, LEVERAGE, INVESTMENT QUALITY & EARNINGS
                profitability_scores = calculate_barra_profitability_factor(df_combine)
                # profitability_scores['profitability'] = profitability_scores[['ATO', 'GPM', 'ROA', 'ROE']].mean(axis=1)
                leverage_scores = calculate_barra_leverage_factor(df_combine)
                # leverage_scores['leverage'] = leverage_scores[['DTOA', 'MLEV', 'BLEV']].mean(axis=1)
                investment_scores = calculate_barra_investment_quality(df_combine)
                # investment_scores['investment_quality'] = quality_scores[['AGRO', 'IGRO', 'CXGRO']].mean(axis=1)
                 
                earnings_quality_scores = calculate_barra_earnings(df_combine)
                # earnings_quality_scores['earnings_quality'] = earnings_quality_scores[['ABS', 'ACF']].mean(axis=1)
                # earnings_quality_scores['earnings_variability'] = earnings_quality_scores[['VSAL', 'VERN', 'VFLO']].mean(axis=1)

                ##################### LIQUIDITY
                liquidity_scores = calculate_barra_liquidity_factors(df_combine)
                liquidity_scores['liquidity'] = liquidity_scores[['STOM', 'STOQ', 'STOA', 'ATVR']].mean(axis=1)

                ##################### VOLATILITY & BETA
                residual_scores = calculate_barra_residual_volatility(df, df_market)
                beta_scores = calculate_barra_beta(df, df_market)
                # volatility_scores['volatility'] = volatility_scores[['HSIGMA', 'DASTD', 'CMRA','beta']].mean(axis=1)

                ##################### GROWTH
                growth_scores = calculate_barra_growth(df_combine)
                growth_scores['growth'] = growth_scores[['EGRO', 'SGRO']].mean(axis=1)

                ##################### EXTRA
                eps_scores = calculate_eps_growth(df_combine)
                core_business_scores = calculate_income_from_core_business(df_combine)

                to_merge = [size_scores, value_scores, relative_strength, lt_reversal_scores, st_reversal_scores,
                            profitability_scores, leverage_scores, investment_scores, earnings_quality_scores,
                            liquidity_scores, residual_scores, beta_scores, growth_scores, eps_scores, core_business_scores]
                
                stock_scores = reduce(lambda left,right: pd.merge(left,right,on=['Date', 'Ticker']), to_merge)

                # stock_scores['momentum'] = stock_scores[['relative_strength','LTRS','STRS']].mean(axis=1)
                # stock_scores['quality'] = stock_scores[['ATO', 'GPM', 'ROA', 'ROE','DTOA', 'MLEV', 'BLEV',
                #                                         'AGRO', 'IGRO', 'CXGRO','ABS', 'ACF','VSAL', 'VERN', 'VFLO']].mean(axis=1)
                # stock_scores['volatility'] = stock_scores[['HSIGMA', 'DASTD', 'CMRA','beta']].mean(axis=1)
                if len(stock_scores) == 0:
                    print(f"This {symbol} cannot merge")

                all_scores = all_scores.append(stock_scores)

                print(symbol)

            except FileNotFoundError:
                print(f"File not found for stock {symbol}")
                continue

    return all_scores

#####################################################
current = glob.glob("/data/VIETSTOCK_STS/sts_price/*.csv")
current_name = [os.path.splitext(os.path.basename(path))[0] for path in current]
print(current_name)
symbols = sorted(get_all_symbols(exclude=['upcom']))
names = list(set(symbols) & set(current_name))

# Pick top N stocks
n=120
from_time = '2018-01-01'
from_time_1 = (date.today() - BDay(252)).strftime("%Y-%m-%d") 
to_time = (date.today()).strftime("%Y-%m-%d")
dfs = {}
for name in names:
    if len(name) == 3:
        df = pd.read_csv(f"/data/VIETSTOCK_STS/sts_price/{name}.csv", index_col=0, parse_dates=[0])  # , usecols = ['StockCode','MarketCap']
        df = df.fillna(method="ffill")

        df = df[from_time_1:to_time]
        dfs[name] =(df)

        if dfs[name]['TotalVal'].median() < 5000:
            dfs.pop(name)
    else:
        continue
dict_mktcap = {}
for name in dfs.keys():
    df = pd.read_csv(f"/data/VIETSTOCK_STS/sts_price/{name}.csv", index_col=0, parse_dates=[0]).fillna(method='ffill')
    mktc= np.median(df['MarketCap'][-22*12:])
    dict_mktcap[name] = mktc

sorted_mktcap = dict(sorted(dict_mktcap.items(), key=lambda item: item[1],reverse=True)[:n])
top_n = [i for i in sorted_mktcap.keys()]
top_n = pd.DataFrame(top_n)
top_n.columns = ['name']
top_n.to_csv(f'/home/trieu-man/Documents/Code/output/top_{n}_{to_time}.csv')
print(top_n)

all_stocks_industry_vietstock = pd.read_csv('/home/trieu-man/Documents/Code/multi_factor_risk/category_vietstock.csv',index_col='number')
# all_stocks_industry = pd.merge(all_stocks_industry_mirae, all_stocks_industry_vietstock, on=['name'])
all_stocks_industry = all_stocks_industry_vietstock
top120_industry = pd.merge(all_stocks_industry, top_n, on='name')
top120_industry.to_csv(f'/home/trieu-man/Documents/Code/output/top{n}_industry_{to_time}.csv')

all_scores = calculate_barra_descriptor_for_all_stocks(top_n['name'].values, from_time=from_time, to_time=to_time)
# Compute global mean and standard deviation scores
global_mean_scores = all_scores.groupby('Ticker').expanding().mean()
global_std_scores = all_scores.groupby('Ticker').expanding().std().dropna()
#  Reindex and sort data by Ticker and Date
all_scores = all_scores.reset_index().sort_values(by=['Ticker','Date']).set_index(['Ticker','Date'])
# Match indices of global stats with sorted_data
matched_indices = global_std_scores.index.intersection(all_scores.index)
global_mean_scores = global_mean_scores.loc[matched_indices]
all_scores = all_scores.loc[matched_indices]
# Standardize scores
all_scores = ((all_scores-global_mean_scores)/global_std_scores).reset_index(level='Ticker')
# Drop unnecessary columns
'''
    Incorporating the two columns in all_scores leads to frequent NaN values, due to prolonged similar entries. Given that both columns pertain to volatility,
    it might be redundant to include them, especially when they introduce issues.
# '''
# all_scores = all_scores.drop(columns='CMRA')
# all_scores = all_scores.drop(columns='DASTD')

# Handle infinite values and drop rows with NaN
all_scores.replace([-np.inf,np.inf], np.nan, inplace=True)
# all_scores=all_scores.dropna()
# 8. MinMaxScaler: Normalize while ignoring NaN values
min_max_scaler = MinMaxScaler((-10,10))
all_scores.iloc[:,1:] = min_max_scaler.fit_transform(all_scores.iloc[:,1:])

all_scores['momentum'] = all_scores[['relative_strength','LTRS','STRS']].mean(axis=1)
all_scores['quality'] = all_scores[['ATO', 'GPM', 'ROA', 'ROE','DTOA', 'MLEV', 'BLEV',
                                        'AGRO', 'IGRO', 'CXGRO','ABS', 'ACF','VSAL', 'VERN', 'VFLO']].mean(axis=1)
all_scores['volatility'] = all_scores[['HSIGMA', 'DASTD', 'CMRA','beta']].mean(axis=1)
all_scores.to_csv(f'/home/trieu-man/Documents/Code/multi_factor_risk/security_crowding/all_scores_{n}_{to_time}.csv')
print(all_scores)

# # Refresh backend after calculation
# requests.post('http://backend-barra.web-test/api/refresh')
