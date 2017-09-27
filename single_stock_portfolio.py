from portfolio.single_stock_data import download_px, preprocess_px
from portfolio.single_stock_train import train, plot_equity_curve
from portfolio.stat import get_draw_down, get_sharpe_ratio
from portfolio.net_turtle import NetTurtle
from portfolio.single_stock_config import get_config
from portfolio.graphs import show_plots, plot_two_equity_curves
import pandas as pd
import numpy as np
import datetime

# net = NetTurtle()

stocks =[
    # 'ABT',
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
    # 'HPQ',
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


    # 'WFC',
    # 'INTC',
    # 'TGT',
    # 'TXT',
    # 'VFC',
    # 'WBA',
    # 'AIG',
    # 'FLR',
    # 'FDX',
    # 'PCAR',
    # 'ADP',
    # 'GWW',
    # 'MAS',
    # 'ADM',
    # 'MAT',
    # 'WMT',
    # 'SNA',
    # 'SWK',
    # 'BF-B',
    # 'AAPL',
    # 'OXY',
    # 'CAG',
    # 'LB',
    # 'T',
    # 'VZ',
    # 'LOW',
    # 'PHM',
    # 'HES',
    # 'LMT',
    # 'HAS',
    # 'BLL',
    # 'APD',
    # 'NUE',
    # 'PKI',
    # 'NOC',
    # 'CNP',
    # 'TJX',
    # 'DOV',
    # 'PH',
    # 'ITW',
    # 'GPS',
    # 'JWN',
    # 'MDT',
    # 'HRB',
    # 'SYY',
    # 'CA',
    # 'MMC',
    # 'AVY',
    # 'HD',
    # 'PNC',
    # 'C',
    # 'STI',
    # 'NKE',
    # 'ECL',
    # 'NWL',
    # 'TMK',
    # 'ORCL',
    # 'ADSK',
    # 'MRO',
    # 'AEE',
    # 'AMGN',
    # 'PX',
    # 'IPG',
    # 'COST',
    # 'CSCO',
    # 'EMN',
    # 'KEY',
    # 'UNM',
    # 'MSFT',
    # 'LUV',
    # 'UNH',
    # 'CBS',
    # 'MU',
    # 'BSX',
    # 'ADBE',
    # 'EFX',
    # 'PGR',
    # 'YUM',
    # 'RF',
    # 'SPLS',
    # 'NTAP',
    # 'BBY',
    # 'VMC',
    # 'XLNX',
    # 'A',
    # 'TIF',
    # 'DVN',
    # 'EOG',
    # 'INTU',
    # 'RHI',
    # 'SYK'
]


def date_to_idx(date):
    if get_config().TEST_BEG <= date <= get_config().TEST_END:
        return (date - get_config().TEST_BEG).days
    return None


def idx_to_date(idx):
    return get_config().TEST_BEG + datetime.timedelta(days=idx)

days = (get_config().TEST_END - get_config().TEST_BEG).days

data = np.ones((len(stocks), days))

stk_idx = 0
for stock in stocks:
    get_config().TICKER = stock
    get_config().reload()

    df = pd.read_csv(get_config().TEST_EQ_PATH)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date

    idx_from = date_to_idx(get_config().TEST_BEG)
    capital = 1.0
    for index, row in df.iterrows():
        date = row['date']
        idx_to = date_to_idx(date)
        data[stk_idx, idx_from:idx_to] = capital
        capital =  row['capital']
        idx_from = idx_to
    idx_to = date_to_idx(get_config().TEST_END)
    data[stk_idx, idx_from:idx_to] = capital

    stk_idx += 1


capital = np.mean(data, axis = 0)
dt = []
for d in range(days):
    dt.append(get_config().TEST_BEG + datetime.timedelta(days=d))

years = (get_config().TEST_END - get_config().TEST_BEG).days / 365
dd = get_draw_down(capital, False)
rets = capital[1:] - capital[:-1]
sharpe = get_sharpe_ratio(rets, years)
y_avg = (capital[-1] - capital[0]) / years
print('%s dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % ('test', dd * 100, y_avg * 100, sharpe))


from portfolio.snp_env import SnpEnv

snp_env = SnpEnv()

dt_snp = []
capital_snp = []
date = snp_env.find_trading_date(get_config().TEST_BEG)
i = 0
c = 1.0
bet = np.zeros((1))

ent_date = date
while date <= get_config().TEST_END:
    if date.year == 2009 and date.month == 9 and date.day == 21:
        _debug = 0

    if i % get_config().REBALANCE_FREQ == 0:
        bet_usd = np.mean(bet, axis=0)
        c += bet_usd

        stk_mask = snp_env.get_tradeable_snp_components_mask(date)
        total = np.sum(stk_mask)
        bet = np.ones(total, dtype=np.float32)

        c -= 1

        ent_date = date

    bet_usd = np.mean(bet, axis=0)
    capital_snp.append(c + bet_usd)
    dt_snp.append(date)

    date = snp_env.find_trading_date(date + datetime.timedelta(days=1))
    if date is not None:
        rets = snp_env.get_ret_lbl(stk_mask, ent_date, date)
        bet = np.ones(total, dtype=np.float32)
        bet += bet * rets
        i+=1
    else:
        break

capital_snp = np.array(capital_snp)
dd = get_draw_down(capital_snp, False)
rets = capital_snp[1:] - capital_snp[:-1]
sharpe = get_sharpe_ratio(rets, years)
y_avg = (capital_snp[-1] - capital_snp[0]) / years
print('%s dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % ('snp long', dd * 100, y_avg * 100, sharpe))

fig = plot_two_equity_curves("Test portfolio vs snp components long", dt, capital, dt_snp, capital_snp)
fig.savefig('data/test_vs_snp_long.png')

df = pd.DataFrame({'date': dt, 'capital': capital})
df.to_csv('data/eq/eq_avg_stocks.csv', index=False)

df = pd.DataFrame({'date': dt_snp, 'capital': capital_snp})
df.to_csv('data/eq/eq_snp.csv', index=False)

show_plots()