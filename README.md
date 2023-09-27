This file is for security crowding, logging all variables and functions.

Order of files:
1. run_daily_fixed_.py - calculate all the lowest level descriptors
2. factor_processing.py - calculate level 2 factors anf integrating all factor scores
3. security_crowding.py - plots
4. performance.py - evaluate performance of stocks by quintile portfolios

Functions:
- value_cross_crowding:
    + input: ticker_scores
    + output: value_cross_crowding
    + calculate the cross crowding score of the value factor
- liquidity_cross_crowding:
    + input: ticker_scores
    + output: liquidity_cross_crowding
    + calculate the cross crowding score of the liquidity factor
- volatility_cross_crowding:
    + input: ticker_scores
    + output: volatility_cross_crowding
    + calculate the cross crowding score of the volatility factor
- momentum_cross_crowding:
    + input: ticker_scores
    + output: momentum_cross_crowding
    + calculate the cross crowding score of the momentum factor

- value_time_crowding
    + input:
    + output: value_time_crowding
    + compute the time crowding scores of value factor. this factor requires standardization cross sectionally. because of the differences between the financial structure of firms in various industries, in this case, we standardized the factor by industry.

- liquidity_time_crowding:
    + input: tickers
    + output: liquidity_time_crowding

- volatility_time_crowding:
    + input: tickers
    + output: volatility_time_crowding

- momentum_time_crowding:
    + input: tickers
    + output: momentum_time_crowding

- create_quintile_portfolios:
    + input: 
        stocks = ['StockA', 'StockB', 'StockC', 'StockD', 'StockE', 'StockF', 'StockG', 'StockH', 'StockI', 'StockJ']
        scores = [12, 34, 21, 9, 45, 18, 32, 27, 14, 40]
    + ouput: a dictionary of quintile porfolios 
        {'quintile_1': [('StockD', 9), ('StockA', 12)],
        'quintile_2': [('StockI', 14), ('StockF', 18)],
        'quintile_3': [('StockC', 21), ('StockH', 27)],
        'quintile_4': [('StockG', 32), ('StockB', 34)],
        'quintile_5': [('StockJ', 40), ('StockE', 45)]}
- create_weekly_quintile_portfolio
    + input: weekly_averages
    + output: a dictionary of quintile porfolios in all weeks avaiable. keys are the week start (on monday)
        {Timestamp('2019-12-23 00:00:00', freq='W-MON'): {'quintile_1': [('ACB', nan),
        ('BID', nan),
        ('CTG', nan),
        ('EIB', nan),
        ('HDB', nan),
        ('LPB', nan),
        ('MBB', nan),
        ('SHB', nan),
        ('STB', nan),
        ('TCB', nan),
        ('TPB', nan),
        ('VCB', nan),
        ('VIB', nan),
        ('VPB', nan),
        ('BSI', nan),
        ('FTS', nan),
        ('HCM', nan),
        ('MBS', nan),
        ('OGC', nan),
        ('SHS', nan),
        ('SSI', nan),
        ('VCI', nan)],
        'quintile_2': [('VIX', nan),
        ('VND', nan),
        ('BMI', nan),
        ...
        ('NVL', 2.5338661485680687),
        ('ANV', 2.8259493684745745),
        ('BCG', 2.869602253127168),
        ('SAB', 2.918681493145707),
        ('VJC', 3.61077313756787)]}}
- calculate_next_week_cumulative_return:
    + input: 
        factor_portfolios (dict): Weekly factor portfolios for each factor.
        stock_returns (DataFrame): Daily stock returns with dates as index and stock symbols as columns.
    + output: 
        next_week_cumulative_returns (dict): Next week cumulative returns for each portfolio.



Variables:
- all_scores: a dataframe of all factor scores computed for all tickers
- all_scores_grouped: all_scores grouped by ticker
- top120_industry: tickers classified by industry
- tickers: unique tickers in both top120_industry and all_scores
- cross_crowding: weighted average of liquidity_cross_crowding, value_cross_crowding, volatility_cross_crowding, and momentum_cross_crowding with weights 0.1, 0.7, 0.1, and 0.1, respectively. cross_crowding has excluded tickers with less than 500 observations.
- time_crowing: weighted average of liquidity_time_crowding, value_time_crowding, volatility_time_crowding, and momentum_time_crowding with weights 0.1, 0.7, 0.1, and 0.1, respectively. 
- integrated_crowding: sum of cross_crowding and time_crowding.
- ranks: ranked integrated_crowding by column (ticker)
- weekly_averages: weekly average of integrated_crowding
- quintile_portfolios: quintile portfolios of weekly average integrated crowding scores.
- daily_data: daily price volume data of all tickers.
- stock_returns: log returns of all tickers.
- next_week_cumret: adict of Next week cumulative returns for each portfolio of all weeks.
- portfolio_rets: a df of Next week cumulative returns for each portfolio of all weeks.
- trace_stock: a record of old quintile and new quintile of all stocks for all week.
- softmax_rets: portfolio returns after softmax function.
- stocks_daily_volume: share turnover of all tickers in all trading days.
- next_week_cumsto: cumulative share turnover next week.
- portfolio_stos: average values of next_week_cumsto by portfolio.
- softmax_stos: portfolio_stos after softmax function.
- softmax_total: weighted sum of softmax_rets and softmax_stos



