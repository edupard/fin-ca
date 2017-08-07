import pandas as pd
from pandas import read_csv
import datetime
import numpy as np
import csv
from date_range import HIST_BEG, HIST_END

# DATE_BEG = datetime.datetime.strptime('2017-03-24', '%Y-%m-%d').date()
# DATE_END = datetime.datetime.strptime('2017-06-30', '%Y-%m-%d').date()
DATE_BEG = HIST_BEG
DATE_END = HIST_END

df = read_csv('data/prices_append.csv')
tickers = ['UONE']
df = df[df.ticker.isin(tickers)]
date_column = pd.to_datetime(df.date)
df = df[(date_column >= DATE_BEG) & (date_column <= DATE_END)]


df.to_csv('data/prices_UONE.csv', index=False)