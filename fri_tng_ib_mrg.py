import pandas as pd
from pandas import read_csv
import datetime
import numpy as np
import csv

PREDICTION_DATE = datetime.datetime.strptime('2017-07-28', '%Y-%m-%d').date()

tng_df = read_csv('data/prev_px.csv')
ib_df = read_csv('data/daily_px.csv')

# find tickers intersection and filter out other

tng_tickers = tng_df.ticker.unique()
ib_tickers = ib_df.ticker.unique()

common_tickers = set.intersection(set(tng_tickers), set(ib_tickers))

with open('data/history.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(('ticker', 'date', 'o', 'c', 'h', 'l', 'v', 'adj_o', 'adj_c', 'adj_h', 'adj_l', 'div', 'split'))

    # df = pd.DataFrame(
    #     columns=('ticker', 'date', 'o', 'c', 'h', 'l', 'v', 'adj_o', 'adj_c', 'adj_h', 'adj_l', 'div', 'split'), index=np.arange(20000000))

    t_i = 0
    i = 0
    print("total tickers %d" % len(common_tickers))
    for ticker in common_tickers:
        t_i += 1
        print("%s %d" % (ticker, t_i))
        tng_stk_df = tng_df[tng_df.ticker == ticker]
        ib_stk_df = ib_df[ib_df.ticker == ticker]
        for index, row in ib_stk_df.iterrows():
            writer.writerow((ticker, PREDICTION_DATE.strftime('%Y-%m-%d'), row.o, row.c, row.h, row.l, row.v, row.o, row.c,
                         row.h, row.l, 0.0, 1.0))
            # df.loc[i] = [ticker, PREDICTION_DATE.strftime('%Y-%m-%d'), row.o, row.c, row.h, row.l, row.v, row.o, row.c,
            #              row.h, row.l, 0.0, 1.0]
            i += 1
        tng_stk_df = tng_stk_df.sort_values('date', ascending = False)
        adj_r = 1.0
        for index, row in tng_stk_df.iterrows():
            adj_o = row.o / adj_r
            adj_c = row.c / adj_r
            adj_h = row.h / adj_r
            adj_l = row.l / adj_r

            adj_d = row['div'] / adj_c
            adj_r = (adj_r + adj_d) * row.split

            writer.writerow((ticker, row.date, row.o, row.c, row.h, row.l, row.v, row.adj_o, row.adj_c, row.adj_h, row.adj_l, row['div'], row.split))
            # df.loc[i] = [ticker, row.date, row.o, row.c, row.h, row.l, row.v, row.adj_o,
            #              row.adj_c, row.adj_h, row.adj_l, row['div'], row.split]
            i += 1

    # df.to_csv('data/history.csv')
