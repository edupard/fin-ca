import csv
import datetime
import pandas as pd

DATE = datetime.datetime.strptime('2017-08-28', '%Y-%m-%d')
DT_COLUMN = 'Ib time'


df = pd.read_csv('data/executions.csv')
df[DT_COLUMN] =  pd.to_datetime(df[DT_COLUMN], format='%Y-%m-%d %H:%M:%S')
mask = (df[DT_COLUMN] > DATE) & (df[DT_COLUMN] < (DATE + datetime.timedelta(days=1)))
df = df.loc[mask]

px_df = pd.read_csv('data/prediction.csv')

result = pd.DataFrame(columns=('ticker', 'pos', 'px','fri px','mon px'))

i = 0
tickers = df.LocalSymbol.unique()
for ticker in tickers:
    mask = df.LocalSymbol == ticker
    t_df = df.loc[mask]
    pos = 0
    gross_px = 0.0
    for index, row in t_df.iterrows():
        pos += row.Position
        gross_px += row.Position * row.Price
    avg_px = gross_px / pos

    ticker_px_df = px_df[px_df.ticker==ticker]

    fri_px = ticker_px_df.iloc[0]['* px']
    mon_px = ticker_px_df.iloc[0]['hp px']

    result.loc[i] = [ticker, pos, avg_px, fri_px, mon_px]
    i += 1

result = result.sort_values('ticker')

result.to_csv('data/avg_px.csv', index=False)