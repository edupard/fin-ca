import matplotlib
import datetime
import numpy as np

from enum import Enum

def filter_tradeable_stocks(raw_data):
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


def get_dates_for_weekly_return(start_date, end_date, traded_stocks, sunday, n_w):
    dates = []
    t_d = sunday
    populated = 0
    while populated < n_w + 1:
        data_idx = get_data_idx(t_d, start_date, end_date)
        if data_idx is None:
            return None
        for j in range(7):
            if traded_stocks[data_idx] > 0:
                dates.append(data_idx)
                populated += 1
                break
            data_idx -= 1
            if data_idx < 0:
                return None
        t_d = t_d - datetime.timedelta(days=7)
    return dates[::-1]


def get_dates_for_daily_return(start_date, end_date, traded_stocks, sunday, n_d):
    dates = []
    data_idx = get_data_idx(sunday, start_date, end_date)
    if data_idx is None:
        return None
    populated = 0
    while populated < n_d + 1:
        if traded_stocks[data_idx] > 0:
            dates.append(data_idx)
            populated += 1
        data_idx -= 1
        if data_idx < 0:
            return None
    return dates[::-1]

def get_date_for_enter_return(start_date, end_date, traded_stocks, monday):
    dates = []
    data_idx = get_data_idx(monday, start_date, end_date)
    end_data_idx = get_data_idx(end_date, start_date, end_date)
    if data_idx is None:
        return None
    populated = 0
    while populated < 1:
        if traded_stocks[data_idx] > 0:
            dates.append(data_idx)
            populated += 1
        data_idx += 1
        if data_idx > end_data_idx:
            return None
    return dates[::-1]

def get_tradeable_stock_indexes(mask, r_i):
    # stocks slice on days used to calculate returns
    s_s = mask[:, r_i]
    # tradable stocks slice
    t_s = np.all(s_s, axis=1)
    # get tradable stocks indices
    t_s_i = np.where(t_s)[0]
    return t_s_i

class PxType(Enum):
    OPEN = 0
    CLOSE = 1

def get_prices(raw_data, t_s_i, r_i, px_type: PxType):
    type_to_idx = {
        PxType.OPEN: 0,
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