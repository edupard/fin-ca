from pandas import read_csv


def get_supported_tickers():
    return read_csv('data/supported_tickers.csv')


def filter_nasdaq_tickers(t_df):
    return t_df[t_df.exchange == u'NASDAQ']


def filter_nyse_tickers(t_df):
    return t_df[t_df.exchange == u'NYSE']


def filter_stock_tickers(t_df):
    return t_df[t_df.assetType == u'Stock']


def filter_stock_related_tickers(t_df):
    # filter warrants, pref stocks and other
    sel = t_df['ticker'].str.contains('-')
    sel = ~sel
    return t_df[sel]


def get_stock_related_tickers():
    t_df = get_supported_tickers()
    t_df = filter_stock_tickers(t_df)
    t_df = filter_stock_related_tickers(t_df)
    return t_df


def get_nyse_nasdaq_tickers():
    t_df = get_stock_related_tickers()

    t_df_nyse = filter_nyse_tickers(t_df)
    t_df_nasdaq = filter_nasdaq_tickers(t_df)

    t_df_stocks = t_df_nyse.append(t_df_nasdaq)

    tickers = t_df_stocks['ticker'].tolist()
    return tickers


def get_nyse_tickers():
    t_df = get_stock_related_tickers()

    t_df_nyse = filter_nyse_tickers(t_df)

    tickers = t_df_nyse['ticker'].tolist()
    return tickers


def get_nasdaq_tickers():
    t_df = get_stock_related_tickers()

    t_df_nasdaq = filter_nasdaq_tickers(t_df)

    tickers = t_df_nasdaq['ticker'].tolist()
    return tickers