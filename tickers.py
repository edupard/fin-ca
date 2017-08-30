from pandas import read_csv


def get_supported_tickers():
    return read_csv('data/supported_tickers_corrected.csv')


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


def get_snp_tickers():
    tickers = []
    snp_curr_df = read_csv('data/snp500.csv')
    for index, row in snp_curr_df.iterrows():
        ticker = row.ticker
        ticker = ticker.replace('.', '-')
        tickers.append(ticker)

    snp_changes_df = read_csv('data/snp500_changes.csv')
    for index, row in snp_changes_df.iterrows():
        ticker_add = row.Added
        if type(ticker_add) is not str or ticker_add == "":
            ticker_add = None
        ticker_rem = row.Removed
        if type(ticker_rem) is not str or ticker_rem == "":
            ticker_rem = None
        if ticker_add is not None:
            tickers.append(ticker_add)
        if ticker_rem is not None:
            tickers.append(ticker_rem)

    return list(set(tickers))

def get_snp_tickers_incorrect():
    tickers = []
    snp_curr_df = read_csv('data/snp500.csv')
    for index, row in snp_curr_df.iterrows():
        ticker = row.ticker
        ticker = ticker.replace('.', '-')
        tickers.append(ticker)

    snp_changes_df = read_csv('data/snp500_changes.csv')
    for index, row in snp_changes_df.iterrows():
        ticker_add = row.Added
        if type(ticker_add) is not str or ticker_add == "":
            ticker_add = None
        ticker_rem = row.Removed
        if type(ticker_rem) is not str or ticker_rem == "":
            ticker_rem = None
        if ticker_add is not None:
            tickers.append(ticker)
        if ticker_rem is not None:
            tickers.append(ticker)

    return list(set(tickers))

def get_snp_tickers_exch_map():
    exch_map = {}
    snp_curr_df = read_csv('data/snp500_exchange.csv')
    for index, row in snp_curr_df.iterrows():
        ticker = row.ticker
        ticker = ticker.replace('.', '-')
        exchange = row.exchange
        exch_map[ticker] = exchange
    return exch_map
