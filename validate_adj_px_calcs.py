import pandas as pd
from pandas import read_csv
import datetime
import numpy as np
import csv

df = read_csv('data/prices.csv')

def check_prices_same(px1 , px2):
    if px2 > 0:
        return (px1 - px2) / px2 < 0.01
    return px1 - px2 < 1e-6

tickers = df.ticker.unique()
for ticker in tickers:
    print(ticker)
    stock_df = df[df.ticker == ticker]
    div_or_split_df = stock_df[(stock_df['div'] != 0.0) | (stock_df.split != 1.0)]
    if len(div_or_split_df) == 0:
        continue
    stock_df = stock_df.sort_values('date', ascending=False)


    try:
        adj_r = 1.0
        adj_d_c_r = 1.0
        adj_s_c_r = 1.0
        for index, row in stock_df.iterrows():
            d_r = row.c / (row.c + row['div'])
            adj_d_c_r *= d_r
            adj_s_c_r *= row.split

            adj_o = row.o * adj_r
            adj_c = row.c * adj_r
            adj_h = row.h * adj_r
            adj_l = row.l * adj_r
            same = True
            same &= check_prices_same(adj_o, row.adj_o)
            same &= check_prices_same(adj_c, row.adj_c)
            same &= check_prices_same(adj_h, row.adj_h)
            same &= check_prices_same(adj_l, row.adj_l)
            if not same:
                print("validation failed")
                break

            adj_r = adj_d_c_r / adj_s_c_r
    except:
        print("validation failed")