import pandas as pd

nasdaq = pd.read_csv('rec/tickers_nasdaq.csv')
nasdaq.Symbol = nasdaq.Symbol.apply(lambda x: str(x).strip())
nasdaq = nasdaq[~nasdaq.Symbol.str.contains('\^')]
nasdaq = nasdaq[~nasdaq.Symbol.str.contains('\.')]
nyse = pd.read_csv('rec/tickers_nyse.csv')
nyse.Symbol = nyse.Symbol.apply(lambda x: str(x).strip())
nyse = nyse[~nyse.Symbol.str.contains('\^')]
nyse = nyse[~nyse.Symbol.str.contains('\.')]
nyse_nasdaq = nyse.append(nasdaq)

tng = pd.read_csv('rec/supported_tickers.csv')
# filter warrants, -A,-B,-C etc
tng = tng[~tng.ticker.str.contains('-')]

tng_nasdaq = tng[tng.exchange == u"NASDAQ"]
tng_nyse = tng[tng.exchange == u"NYSE"]
tng_nyse_nasdaq = tng_nyse.append(tng_nasdaq)

# currently listed tickers which we can find in tiingo supported_tickers.csv but absent in exchange files
absent_mask = ~tng_nyse_nasdaq.ticker.isin(nyse_nasdaq.Symbol)
listed_mask = tng_nyse_nasdaq.endDate == u"2017-07-05"
mask = absent_mask & listed_mask
output_1 = tng_nyse_nasdaq.loc[mask]
output_1.to_csv('rec/1.csv')

# tickers we can find in exchange files but absent in supported_tickers.csv
mask = ~nyse_nasdaq.Symbol.isin(tng_nyse_nasdaq.ticker)
output_2 = nyse_nasdaq.loc[mask]
output_2.to_csv('rec/2.csv')

# tickers marked as NASDAQ(supported_tickers.csv), but looks like they actually listed on nyse
mask = tng_nasdaq.ticker.isin(nyse.Symbol)
output_3 = tng_nasdaq.loc[mask]
output_3.to_csv('rec/3.csv')

# tickers marked as NYSE(supported_tickers.csv), but looks like they actually listed on nasdaq
mask = tng_nyse.ticker.isin(nasdaq.Symbol)
output_4 = tng_nyse.loc[mask]
output_4.to_csv('rec/4.csv')

mask = tng_nasdaq.ticker.isin(nyse.Symbol)
tng_nasdaq.loc[mask, 'exchange'] = u'NYSE'
mask = tng_nyse.ticker.isin(nasdaq.Symbol)
tng_nyse.loc[mask, 'exchange'] = u'NASDAQ'

supported_tickers_corrected = tng_nasdaq.append(tng_nyse)
supported_tickers_corrected.to_csv("rec/supported_tickers_corrected.csv", index=False)
