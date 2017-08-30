import pandas as pd

snp_df = pd.read_csv('data/snp500.csv')
nyse_df = pd.read_csv('data/tickers_nyse.csv')
nasdaq_df = pd.read_csv('data/tickers_nasdaq.csv')

nyse_symbols = set(nyse_df.Symbol.unique())
nasdaq_symbols = set(nasdaq_df.Symbol.unique())

result = pd.DataFrame(columns=('ticker', 'exchange'))
i = 0
for index, row in snp_df.iterrows():
    ticker = row.ticker
    ticker = ticker.replace('.', '-')
    exchange = ''
    if ticker in nyse_symbols:
        exchange = 'NYSE'
    if ticker in nasdaq_symbols:
        exchange = 'NASDAQ'
    row = [ticker, exchange]
    result.loc[i] = row
    i += 1
result.to_csv('data/snp500_exchange.csv', index=False)
