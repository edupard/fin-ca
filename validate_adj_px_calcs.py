import pandas as pd
from pandas import read_csv
import datetime
import numpy as np
import csv

df = read_csv('data/prices.csv')

def check_prices_same(px1 , px2):
    return (px1 - px2) / px2 < 1e-6

tickers = df.ticker.unique()
for ticker in tickers:
    print(ticker)
    stock_df = df[df.ticker == ticker]
    stock_df = stock_df.sort_values('date', ascending=False)
    adj_r = 1.0
    for index, row in stock_df.iterrows():
        adj_o = row.o / adj_r
        adj_c = row.c / adj_r
        adj_h = row.h / adj_r
        adj_l = row.l / adj_r
        same = True
        same &= check_prices_same(adj_o, row.adj_o)
        same &= check_prices_same(adj_c, row.adj_c)
        same &= check_prices_same(adj_h, row.adj_l)
        same &= check_prices_same(adj_l, row.adj_l)
        if not same:
            print("validation failed")
            break

        adj_d = row['div'] / adj_c
        adj_r = (adj_r + adj_d) * row.split

