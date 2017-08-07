import pandas as pd
from pandas import read_csv
import datetime
import numpy as np
import csv

# PREDICTION_DATE = datetime.datetime.strptime('2017-07-28', '%Y-%m-%d').date()
PREDICTION_DATE = datetime.datetime.strptime('2017-08-04', '%Y-%m-%d').date()

tng_df = read_csv('data/px_hist.csv')
ib_df = read_csv('data/px_tod.csv')

# find tickers intersection and filter out other

tng_tickers = tng_df.ticker.unique()
ib_tickers = ib_df.ticker.unique()

common_tickers = set.intersection(set(tng_tickers), set(ib_tickers))

ib_df['div'] = 0.0
ib_df['split'] = 1.0

adj_df = read_csv('data/adjustments.csv')
adj_tickers = adj_df.ticker.unique()
for ticker in adj_tickers:
    ticker_adj_df = adj_df[adj_df.ticker == ticker]
    div = ticker_adj_df.iloc[0]['div']
    split = ticker_adj_df.iloc[0].split

    ib_df.loc[ib_df.ticker == ticker, 'div'] = div
    ib_df.loc[ib_df.ticker == ticker, 'split'] = split

    ib_ticker_df = ib_df[ib_df.ticker == ticker]
    if len(ib_ticker_df) == 0:
        continue

    c = ib_ticker_df.iloc[0].c
    adj_r = (c / (c + div)) / split
    tng_df.loc[tng_df.ticker == ticker, 'adj_o'] *= adj_r
    tng_df.loc[tng_df.ticker == ticker, 'adj_c'] *= adj_r
    tng_df.loc[tng_df.ticker == ticker, 'adj_h'] *= adj_r
    tng_df.loc[tng_df.ticker == ticker, 'adj_l'] *= adj_r

ib_df = ib_df[ib_df.ticker.isin(common_tickers)]
ib_df['adj_o'] = ib_df.o
ib_df['adj_c'] = ib_df.c
ib_df['adj_h'] = ib_df.h
ib_df['adj_l'] = ib_df.l
ib_df['date'] = datetime.datetime.strftime(PREDICTION_DATE, '%Y-%m-%d')


df = pd.concat([tng_df, ib_df], ignore_index=True)
cols = ['ticker','date','o','c','h','l','v','adj_o','adj_c','adj_h','adj_l','div','split']
df = df[cols]
df.to_csv('data/history.csv', index=False)
