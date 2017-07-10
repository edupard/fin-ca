import csv
import datetime
import pandas as pd

DATE = datetime.datetime.strptime('2017-07-03', '%Y-%m-%d')
DT_COLUMN = 'Ib time'


df = pd.read_csv('executions.csv')
df[DT_COLUMN] =  pd.to_datetime(df[DT_COLUMN], format='%Y-%m-%d %H:%M:%S')
mask = (df[DT_COLUMN] > DATE) & (df[DT_COLUMN] < (DATE + datetime.timedelta(days=1)))
df = df.loc[mask]

result = pd.DataFrame(columns=('ticker', 'pos', 'px'))

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

    result.loc[i] = [ticker, pos, avg_px]
    i += 1

result = result.sort_values('ticker')
result.to_csv('avg_px.csv')