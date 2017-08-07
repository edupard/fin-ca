from pandas import read_csv

df = read_csv('data/Jul-28/compare/history.csv')
df_fri = read_csv('data/Jul-28/compare/history_fri.csv')


tickers = df.ticker.unique()
ib_tickers = df_fri.ticker.unique()

tickers = set(tickers)
ib_tickers = set(ib_tickers)
ib_absent_tickers = set.difference(tickers,ib_tickers)

print(ib_absent_tickers)
