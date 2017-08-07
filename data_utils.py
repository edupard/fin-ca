import matplotlib
import datetime
import numpy as np
from pandas import read_csv
from enum import Enum

CAP = 50


def get_tradable_stocks_mask(raw_data):
    mask = raw_data[:, :, 3] > 0.0
    return mask


def get_snp_mask(tickers, raw_data, start_date, end_date):
    def get_ticker_idx(ticker, tickers):
        ticker_idxs = np.nonzero(tickers == ticker)
        if ticker_idxs[0].shape[0] > 0:
            return ticker_idxs[0][0]
        return None

    snp_mask = np.full((raw_data.shape[0], raw_data.shape[1]), False)
    snp_curr_df = read_csv('data/snp500.csv')
    for index, row in snp_curr_df.iterrows():
        ticker = row.ticker
        ticker_idx = get_ticker_idx(ticker, tickers)
        if ticker_idx is not None:
            snp_mask[ticker_idx, :] = True

    dt_idx_up_to = raw_data.shape[1]
    snp_changes_df = read_csv('data/snp500_changes.csv')
    curr_date = None
    for index, row in snp_changes_df.iterrows():
        ticker_add = row.Added
        ticker_rem = row.Removed
        s_date = row.Date
        t_d = datetime.datetime.strptime(s_date, '%B %d, %Y').date()
        dt_idx_from = get_data_idx(t_d, start_date, end_date)
        if curr_date is not None and t_d != curr_date:
            dt_idx_up_to = get_data_idx(curr_date, start_date, end_date)
        curr_date = t_d
        ticker_add_idx = get_ticker_idx(ticker_add, tickers)
        if ticker_add_idx is not None:
            snp_mask[ticker_add_idx, :dt_idx_from] = False
            snp_mask[ticker_add_idx, dt_idx_from:dt_idx_up_to] = True
        ticker_rem_idx = get_ticker_idx(ticker_rem, tickers)
        if ticker_rem_idx is not None:
            snp_mask[ticker_rem_idx, dt_idx_from:dt_idx_up_to] = False

    return snp_mask


def filter_activelly_tradeable_stocks(raw_data):
    g_a = raw_data[:, :, 4] * raw_data[:, :, 3]
    mask = g_a[:, :] > 10000000
    # alternative tradable stock selection algos
    # g_a_a = np.average(g_a, axis=0)
    # mask = (g_a[:, :] > (g_a_a[:] / 2.))
    # mask = (g_a[:, :] > 100000) & (raw_data[:, :, 3] > 5.)
    # mask = raw_data[:,:,4] != 0

    # calc traded stocks per day num
    traded_stocks = mask[:, :].sum(0)

    return mask, traded_stocks


def convert_to_mpl_time(raw_dt):
    def reduce_time(arr):
        for idx in range(arr.shape[0]):
            dt = datetime.datetime.fromtimestamp(raw_dt[idx])
            yield matplotlib.dates.date2num(dt)

    raw_mpl_dt = np.fromiter(reduce_time(raw_dt), dtype=np.float64)
    return raw_mpl_dt


def get_data_idx(dt, start_date, end_date):
    if dt < start_date or dt > end_date:
        return None
    return (dt - start_date).days


def get_dates_for_weekly_return(start_date, end_date, traded_stocks, date, n_w):
    dates = []
    t_d = date
    populated = 0
    while populated < n_w + 1:
        data_idx = get_data_idx(t_d, start_date, end_date)
        if data_idx is None:
            return None
        for j in range(date.isoweekday()):
            if traded_stocks[data_idx] > CAP:
                dates.append(data_idx)
                populated += 1
                break
            data_idx -= 1
            if data_idx < 0:
                return None
        t_d = t_d - datetime.timedelta(days=7) + datetime.timedelta(days=(7 - t_d.isoweekday()))
    return dates[::-1]


def get_dates_for_daily_return(start_date, end_date, traded_stocks, date, n_d):
    dates = []
    data_idx = get_data_idx(date, start_date, end_date)
    if data_idx is None:
        return None
    populated = 0
    while populated < n_d + 1:
        if traded_stocks[data_idx] > CAP:
            dates.append(data_idx)
            populated += 1
        data_idx -= 1
        if data_idx < 0:
            return None
    return dates[::-1]


def get_one_trading_date(start_date, end_date, traded_stocks, date):
    dates = []
    data_idx = get_data_idx(date, start_date, end_date)
    end_data_idx = get_data_idx(end_date, start_date, end_date)
    if data_idx is None:
        return None
    populated = 0
    while populated < 1:
        if traded_stocks[data_idx] > CAP:
            dates.append(data_idx)
            populated += 1
        data_idx += 1
        if data_idx > end_data_idx:
            return None
    return dates[::-1]


def get_intermediate_dates(start_date, end_date, traded_stocks, ent_r_i, ext_r_i):
    dates = []
    data_idx = ent_r_i[0] + 1
    while data_idx <= ext_r_i[0]:
        if traded_stocks[data_idx] > CAP:
            dates.append(data_idx)
        data_idx += 1
    return dates


def get_tradable_stock_indexes(mask, r_i):
    # stocks slice on days used to calculate returns
    s_s = mask[:, r_i]
    # tradable stocks slice
    t_s = np.all(s_s, axis=1)
    # get tradable stocks indices
    t_s_i = np.where(t_s)[0]
    return t_s_i


class PxType(Enum):
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3


def get_prices(raw_data, t_s_i, r_i, px_type: PxType):
    type_to_idx = {
        PxType.OPEN: 0,
        PxType.HIGH: 1,
        PxType.LOW: 2,
        PxType.CLOSE: 3
    }

    px_idx = type_to_idx.get(px_type, 3)
    c = raw_data[:, r_i, :]
    c = c[t_s_i, :, :]
    c = c[:, :, px_idx]
    return c


def calc_z_score(c):
    # calc returns
    r = (c[:, 1:] - c[:, :-1]) / c[:, :-1]
    # accumulate returns
    c_r = np.cumsum(r, axis=1)
    # calculate accumulated return mean
    r_m = np.average(c_r, axis=0)
    # calculate accumulated return std
    r_std = np.std(c_r, axis=0)
    # calc z score
    z_score = (c_r - r_m) / r_std
    return z_score


def calc_z_score_alt(c):
    # calc returns
    r = (c[:, 1:] - c[:, :-1]) / c[:, :-1]
    # accumulate returns
    c_r = np.cumsum(r, axis=1)
    # calculate accumulated return mean
    r_m = np.average(c_r, axis=0)
    # calculate accumulated return std
    r_std = np.std(c_r, axis=0)
    # calc z score
    z_score = (c_r - r_m) / r_std
    return z_score, r, c_r, r_m, r_std
