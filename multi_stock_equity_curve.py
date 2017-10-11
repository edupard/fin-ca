import datetime
import numpy as np
import pandas as pd

from portfolio.multi_stock_config import get_config

def date_to_idx(date):
    if get_config().TRAIN_BEG <= date <= get_config().TRAIN_END:
        return (date - get_config().TRAIN_BEG).days
    return None


def idx_to_date(idx):
    return get_config().TRAIN_BEG + datetime.timedelta(days=idx)


days = (get_config().TRAIN_END - get_config().TRAIN_BEG).days
data = np.ones((len(stocks), days))

stk_idx = 0
for stock in stocks:
    get_config().TICKER = stock
    get_config().reload()

    df = pd.read_csv(get_config().TRAIN_EQ_PATH)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date

    idx_from = date_to_idx(get_config().TRAIN_BEG)
    capital = 1.0
    for index, row in df.iterrows():
        date = row['date']
        idx_to = date_to_idx(date)
        data[stk_idx, idx_from:idx_to] = capital
        capital = row['capital']
        idx_from = idx_to
    idx_to = date_to_idx(get_config().TRAIN_END)
    data[stk_idx, idx_from:idx_to] = capital

    stk_idx += 1

capital = np.mean(data, axis=0)
dt = []
for d in range(days):
    dt.append(get_config().TRAIN_BEG + datetime.timedelta(days=d))

df = pd.DataFrame({'date': dt, 'capital': capital})
df.to_csv('data/eq/eq_2007_tod_ms.csv', index=False)