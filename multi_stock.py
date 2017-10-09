from portfolio.multi_stock_train import train
from portfolio.multi_stock_config import get_config
from portfolio.net_shiva import NetShiva
import numpy as np
import pandas as pd
import datetime


stocks =[
    'ABT',
    'ARNC',
    'HON',
    'SHW',
    'CMI',
    'EMR',
    'SLB',
    'CSX',
    'CLX',
    'GIS',
    'NEM',
    'MCD',
    'LLY',
    'BAX',
    'BDX',
    'JNJ',
    'GPC',
    'HPQ',
    'WMB',
    'BCR',
    'JPM',
    'IFF',
    'AET',
    'AXP',
    'BAC',
    'CI',
    'DUK',
    'LNC',
    'TAP',
    'NEE',
    'DIS',
    'XRX',
    'IBM',
    'WFC',
    'INTC',
    'TGT',
    'TXT',
    'VFC',
    'WBA',
    'AIG',
    'FLR',
    'FDX',
    'PCAR',
    'ADP',
    'GWW',
    'MAS',
    'ADM',
    'MAT',
    'WMT',
    'SNA',
    'SWK',
    'AAPL',
    'OXY',
    'CAG',
    'LB',
    'T',
    'VZ',
    'LOW',
    'PHM',
    'HES',
    'LMT',
    'HAS',
    'BLL',
    'APD',
    'NUE',
    'PKI',
    'NOC',
    'CNP',
    'TJX',
    'DOV',
    'PH',
    'ITW',
    'GPS',
    'JWN',
    'MDT',
    'HRB',
    'SYY',
    'CA',
    'MMC',
    'AVY',
    'HD',
    'PNC',
    'C',
    'STI',
    'NKE',
    'ECL',
    'NWL',
    'TMK',
    'ORCL',
    'ADSK',
    'MRO',
    'AEE',
    'AMGN',
    'PX',
    'IPG',
    'COST',
    'CSCO',
    'EMN',
    'KEY',
    'UNM',
    'MSFT',
    'LUV',
    'UNH',
    'CBS',
    'MU',
    'BSX',
    'ADBE',
    'EFX',
    'PGR',
    'YUM',
    'RF',
    'SPLS',
    'NTAP',
    'BBY',
    'VMC',
    'XLNX',
    'A',
    'TIF',
    'DVN',
    'EOG',
    'INTU',
    'RHI',
    'SYK',
    'COP'
]

# net = NetShiva()
# for stock in stocks:
#     get_config().TICKER = stock
#     get_config().reload()
#     train(net)


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

capital = np.mean(data, axis = 0)
dt = []
for d in range(days):
    dt.append(get_config().TRAIN_BEG + datetime.timedelta(days=d))

df = pd.DataFrame({'date': dt, 'capital': capital})
df.to_csv('data/eq/eq_2007_tod_ms.csv', index=False)