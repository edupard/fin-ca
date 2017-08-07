import pandas as pd
from pandas import read_csv
import datetime
import numpy as np
import csv

PREDICTION_DATE = datetime.datetime.strptime('2017-07-28', '%Y-%m-%d').date()

tng_df = read_csv('data/history.csv')
ib_df = read_csv('data/px_tod.csv')

# find tickers intersection and filter out other

tng_tickers = tng_df.ticker.unique()
ib_tickers = ib_df.ticker.unique()

common_tickers = set.intersection(set(tng_tickers), set(ib_tickers))

df = tng_df[tng_df.ticker.isin(common_tickers)]
df.to_csv('data/history.csv', index=False)
