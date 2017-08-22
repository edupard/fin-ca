import matplotlib
import datetime
import numpy as np
from pandas import read_csv
from enum import Enum

DATA_OPEN_IDX = 0
DATA_HIGH_IDX = 1
DATA_LOW_IDX = 2
DATA_CLOSE_IDX = 3
DATA_VOLUME_IDX = 4
DATA_TO_IDX = 5


def get_tradable_stocks_mask(raw_data):
    mask = np.all(raw_data > 0.0, axis=2)
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
        ticker = ticker.replace('.','-')
        ticker_idx = get_ticker_idx(ticker, tickers)
        if ticker_idx is not None:
            snp_mask[ticker_idx, :] = True
        else:
            print('Miss ticker %s' % ticker)

    dt_idx_up_to = raw_data.shape[1]
    snp_changes_df = read_csv('data/snp500_changes.csv')
    curr_date = None
    for index, row in snp_changes_df.iterrows():
        ticker_add = row.Added
        if type(ticker_add) is not str or ticker_add == "":
            ticker_add = None
        ticker_rem = row.Removed
        if type(ticker_rem) is not str or ticker_rem == "":
            ticker_rem = None
        s_date = row.Date
        t_d = datetime.datetime.strptime(s_date, '%B %d, %Y').date()
        dt_idx_from = get_data_idx(t_d, start_date, end_date)
        if curr_date is not None and t_d != curr_date:
            dt_idx_up_to = get_data_idx(curr_date, start_date, end_date)
        curr_date = t_d
        if ticker_add is not None:
            ticker_add_idx = get_ticker_idx(ticker_add, tickers)
            if ticker_add_idx is not None:
                snp_mask[ticker_add_idx, :dt_idx_from] = False
                snp_mask[ticker_add_idx, dt_idx_from:dt_idx_up_to] = True
            else:
                print('Miss add ticker %s' % ticker_add)
        if ticker_rem is not None:
            ticker_rem_idx = get_ticker_idx(ticker_rem, tickers)
            if ticker_rem_idx is not None:
                snp_mask[ticker_rem_idx, dt_idx_from:dt_idx_up_to] = False
            else:
                print('Miss rem ticker %s' % ticker_rem)

    return snp_mask


def get_snp_mask_v1(tickers, raw_data, start_date, end_date):
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
        else:
            print('Miss ticker %s' % ticker)

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
        else:
            print('Miss add ticker %s' % ticker_add)
        ticker_rem_idx = get_ticker_idx(ticker_rem, tickers)
        if ticker_rem_idx is not None:
            snp_mask[ticker_rem_idx, dt_idx_from:dt_idx_up_to] = False
        else:
            print('Miss rem ticker %s' % ticker_rem)

    return snp_mask


def filter_activelly_tradeable_stocks(raw_data, DAY_TO_FILTER):
    g_a = raw_data[:, :, DATA_TO_IDX]
    mask = g_a[:, :] > DAY_TO_FILTER
    return mask


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


def get_dates_for_weekly_return(start_date, end_date, trading_day_mask, date, n_w):
    dates = []
    t_d = date
    populated = 0
    while populated < n_w + 1:
        data_idx = get_data_idx(t_d, start_date, end_date)
        if data_idx is None:
            return None
        for j in range(date.isoweekday()):
            if trading_day_mask[data_idx]:
                dates.append(data_idx)
                populated += 1
                break
            data_idx -= 1
            if data_idx < 0:
                return None
        t_d = t_d - datetime.timedelta(days=7) + datetime.timedelta(days=(7 - t_d.isoweekday()))
    return dates[::-1]


def get_dates_for_daily_return(start_date, end_date, trading_day_mask, date, n_d):
    dates = []
    data_idx = get_data_idx(date, start_date, end_date)
    if data_idx is None:
        return None
    populated = 0
    while populated < n_d + 1:
        if trading_day_mask[data_idx]:
            dates.append(data_idx)
            populated += 1
        data_idx -= 1
        if data_idx < 0:
            return None
    return dates[::-1]


def get_one_trading_date(start_date, end_date, trading_day_mask, date):
    dates = []
    data_idx = get_data_idx(date, start_date, end_date)
    end_data_idx = get_data_idx(end_date, start_date, end_date)
    if data_idx is None:
        return None
    populated = 0
    while populated < 1:
        if trading_day_mask[data_idx]:
            dates.append(data_idx)
            populated += 1
        data_idx += 1
        if data_idx > end_data_idx:
            return None
    return dates[::-1]


def get_intermediate_dates(trading_day_mask, ent_r_i, ext_r_i):
    dates = []
    data_idx = ent_r_i + 1
    while data_idx <= ext_r_i:
        if trading_day_mask[data_idx]:
            dates.append(data_idx)
        data_idx += 1
    return dates

def get_top_tradable_stks(raw_data, trading_day_mask, s_i, e_i, limit):
    dts = get_intermediate_dates(trading_day_mask, s_i, e_i)
    raw_data = raw_data[:, dts, :]
    g_a = raw_data[:, :, DATA_TO_IDX]
    avg_g_a = np.mean(g_a, axis=1)
    s_t_s_i = np.argsort(avg_g_a)
    t_s_i = s_t_s_i[-limit:]
    return t_s_i

def get_stks_with_price_above(raw_data, e_i, limit):
    active_stk_mask = (raw_data[:, e_i, DATA_TO_IDX] / raw_data[:, e_i, DATA_VOLUME_IDX]) > limit
    # active_stk_mask = raw_data[:, e_i, DATA_CLOSE_IDX] > limit
    t_s_i = np.where(active_stk_mask)[0]
    return t_s_i

def get_avg_stk_to(raw_data, tsi, trading_day_mask, s_i, e_i):
    dts = get_intermediate_dates(trading_day_mask, s_i, e_i)
    raw_data = raw_data[:, dts, :]
    raw_data = raw_data[tsi,:,:]
    s_to = raw_data[:, :, DATA_TO_IDX]
    avg_s_to = np.mean(s_to, axis=1)
    return avg_s_to

def get_active_stks(raw_data, trading_day_mask, s_i, e_i, limit):
    dts = get_intermediate_dates(trading_day_mask, s_i, e_i)
    raw_data = raw_data[:, dts, :]
    g_a = raw_data[:, :, DATA_TO_IDX]
    avg_g_a = np.mean(g_a, axis= 1)
    active_stk_mask = avg_g_a > limit
    t_s_i = np.where(active_stk_mask)[0]
    return t_s_i


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


def get_price_idx(px_type: PxType):
    type_to_idx = {
        PxType.OPEN: DATA_OPEN_IDX,
        PxType.HIGH: DATA_HIGH_IDX,
        PxType.LOW: DATA_LOW_IDX,
        PxType.CLOSE: DATA_CLOSE_IDX
    }
    return type_to_idx.get(px_type)


def get_price(raw_data, s_i, r_i, px_type: PxType):
    px_idx = get_price_idx(px_type)
    return raw_data[s_i, r_i, px_idx]


def get_prices(raw_data, t_s_i, r_i, px_type: PxType):
    px_idx = get_price_idx(px_type)
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
