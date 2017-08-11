import pandas as pd
from pandas import read_csv
import datetime
import numpy as np
import csv
from config import get_config

# DATE_BEG = datetime.datetime.strptime('2017-03-24', '%Y-%m-%d').date()
# DATE_END = datetime.datetime.strptime('2017-06-30', '%Y-%m-%d').date()
DATE_BEG = get_config().HIST_BEG
DATE_END = get_config().HIST_END

df = read_csv('data/prices_append.csv')
tickers = ['UONE']
df = df[df.ticker.isin(tickers)]
date_column = pd.to_datetime(df.date)
df = df[(date_column >= DATE_BEG) & (date_column <= DATE_END)]


df.to_csv('data/prices_UONE.csv', index=False)