import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

import math

from rbm import RBM
from au import AutoEncoder
from ffnn import FFNN
from data_utils import filter_tradeable_stocks, convert_to_mpl_time, get_dates_for_daily_return, \
    get_dates_for_weekly_return, get_tradeable_stock_indexes, get_close_prices, calc_z_score
from download_utils import load_npz_data
from visualization import wealth_graph, confusion_matrix, hpr_analysis
from visualization import plot_20_random_stock_prices, plot_traded_stocks_per_day
from nn import train_ae, train_ffnn, train_rbm, evaluate_ffnn

NUM_WEEKS = 12
NUM_DAYS = 5

TRAIN_RBM = False
RBM_EPOCH_TO_TRAIN = 50
RBM_BATCH_SIZE = 10

TRAIN_AU = False
LOAD_RBM_WEIGHTS = True
AU_EPOCH_TO_TRAIN = 30
AU_BATCH_SIZE = 10

TRAIN_FFNN = False
LOAD_AU_WEIGHTS = True
FFNN_EPOCH_TO_TRAIN = 1000
FFNN_BATCH_SIZE = 10

PERCENTILE = 10

raw_dt, raw_data = load_npz_data('data/nasdaq_raw_data.npz')

raw_mpl_dt = convert_to_mpl_time(raw_dt)

mask, traded_stocks = filter_tradeable_stocks(raw_data)

# plot_20_random_stock_prices(raw_data, raw_mpl_dt)
# plot_traded_stocks_per_day(traded_stocks, raw_mpl_dt)

TRAIN_UP_TO_DATE = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
START_DATE = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
END_DATE = datetime.datetime.strptime('2017-04-18', '%Y-%m-%d').date()
SUNDAY = START_DATE + datetime.timedelta(days=7 - START_DATE.isoweekday())

train_records = 0
train_weeks = 0
total_weeks = 0
data_set_records = 0

dr = None
wr = None
hpr = None
c_l = None
c_s = None
stocks = None
w_data_index = None
w_num_stocks = None
w_enter_index = None
w_exit_index = None


def append_data(data, _data):
    if data is None:
        return _data
    else:
        return np.concatenate([data, _data], axis=0)


def make_array(value):
    return np.array([value]).astype(np.int32)

while True:
    # iterate over weeks
    SUNDAY = SUNDAY + datetime.timedelta(days=7)
    # break when all availiable data processed
    if SUNDAY > END_DATE:
        break
    w_r_i = get_dates_for_weekly_return(START_DATE, END_DATE, traded_stocks, SUNDAY, NUM_WEEKS + 1)
    # continue if all data not availiable yet
    if w_r_i is None:
        continue
    # continue if all data not availiable yet
    d_r_i = get_dates_for_daily_return(START_DATE, END_DATE, traded_stocks, SUNDAY - datetime.timedelta(days=7),
                                       NUM_DAYS)
    if d_r_i is None:
        continue

    t_s_i = get_tradeable_stock_indexes(mask, w_r_i, d_r_i)
    d_c = get_close_prices(raw_data, t_s_i, d_r_i)
    w_c = get_close_prices(raw_data, t_s_i, w_r_i)

    # calc daily returns
    d_n_r = calc_z_score(d_c)
    w_n_r = calc_z_score(w_c[:, :-1])
    dr = append_data(dr, d_n_r)
    wr = append_data(wr, w_n_r)

    _hpr = (w_c[:, NUM_WEEKS + 1] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]
    hpr = append_data(hpr, _hpr)

    hpr_med = np.median(_hpr)
    _c_l = _hpr >= hpr_med
    _c_s = ~_c_l
    c_l = append_data(c_l, _c_l)
    c_s = append_data(c_s, _c_s)

    enter_date_idx = w_r_i[NUM_WEEKS]
    exit_date_idx = w_r_i[NUM_WEEKS + 1]

    stocks = append_data(stocks, t_s_i)

    # sample size
    num_stocks = t_s_i.shape[0]

    w_data_index = append_data(w_data_index, make_array(data_set_records))
    w_num_stocks = append_data(w_num_stocks, make_array(num_stocks))
    w_enter_index = append_data(w_enter_index, make_array(enter_date_idx))
    w_exit_index = append_data(w_exit_index, make_array(exit_date_idx))

    # record counts
    data_set_records += num_stocks
    total_weeks += 1
    if SUNDAY <= TRAIN_UP_TO_DATE:
        train_records += num_stocks
        train_weeks += 1

if TRAIN_RBM:
    train_rbm(train_records, dr, wr)

if TRAIN_AU:
    train_ae(train_records, dr, wr)

if TRAIN_FFNN:
    train_ffnn(train_records, dr, wr, c_l, c_s)

prob_l = np.zeros((data_set_records), dtype=np.float)
evaluate_ffnn(data_set_records, dr, wr, prob_l)


def calc_classes_and_decisions(data_set_records, total_weeks, data):
    c_l = np.zeros((data_set_records), dtype=np.bool)
    c_s = np.zeros((data_set_records), dtype=np.bool)

    s_l = np.zeros((data_set_records), dtype=np.bool)
    s_s = np.zeros((data_set_records), dtype=np.bool)

    top_hpr = np.zeros((total_weeks))
    bottom_hpr = np.zeros((total_weeks))
    top_stocks_num = np.zeros((total_weeks))
    bottom_stocks_num = np.zeros((total_weeks))

    for i in range(total_weeks):
        w_i = i
        beg = w_data_index[w_i]
        end = beg + w_num_stocks[w_i]

        _data = data[beg: end]

        median = np.median(_data)
        _s_c_l = c_l[beg: end]
        _s_c_s = c_s[beg: end]
        pred_long_cond = _data >= median
        _s_c_l |= pred_long_cond
        _s_c_s |= ~pred_long_cond

        top_bound = np.percentile(_data, 100 - PERCENTILE)
        bottom_bound = np.percentile(_data, PERCENTILE)
        _s_s_l = s_l[beg: end]
        _s_s_s = s_s[beg: end]
        long_cond = _data >= top_bound
        short_cond = _data <= bottom_bound
        _s_s_l |= long_cond
        _s_s_s |= short_cond
        _hpr = hpr[beg: end]
        l_hpr = _hpr[_s_s_l]
        s_hpr = _hpr[_s_s_s]
        top_hpr[w_i] = np.mean(l_hpr)
        bottom_hpr[w_i] = np.mean(s_hpr)
        top_stocks_num[w_i] = l_hpr.shape[0]
        bottom_stocks_num[w_i] = s_hpr.shape[0]
    return c_l, c_s, top_hpr, bottom_hpr, top_stocks_num, bottom_stocks_num

# s_c_l, s_c_s, t_hpr, b_hpr, t_stocks, b_stocks = calc_classes_and_decisions(
#     data_set_records, total_weeks, wr[:, NUM_WEEKS - 1]
# )

# confusion_matrix(c_l, c_s, s_c_l, s_c_s)
# hpr_analysis(t_hpr, b_hpr)
# wealth_graph(t_hpr, b_hpr, w_exit_index, raw_mpl_dt, raw_dt)

e_c_l, e_c_s, t_e_hpr, b_e_hpr, t_e_stocks, b_e_stocks = calc_classes_and_decisions(
    data_set_records, total_weeks, prob_l
)

confusion_matrix(c_l[train_records:], c_s[train_records:], e_c_l[train_records:], e_c_s[train_records:])
hpr_analysis(t_e_hpr[train_weeks:], b_e_hpr[train_weeks:])
wealth_graph(t_e_hpr[train_weeks:],
             b_e_hpr[train_weeks:],
             w_exit_index[train_weeks:],
             raw_mpl_dt,
             raw_dt)

plt.show(True)
