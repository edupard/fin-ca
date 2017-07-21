import datetime
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import csv

from tickers import get_nyse_nasdaq_tickers, get_nyse_tickers, get_nasdaq_tickers
from data_utils import filter_activelly_tradeable_stocks, convert_to_mpl_time, get_dates_for_daily_return, \
    get_one_trading_date, get_dates_for_weekly_return, get_tradable_stock_indexes, get_prices, \
    PxType, calc_z_score, get_tradable_stocks_mask, get_intermediate_dates
from download_utils import load_npz_data, load_npz_data_alt
from visualization import wealth_graph, confusion_matrix, hpr_analysis, wealth_csv
from visualization import plot_20_random_stock_prices, plot_traded_stocks_per_day
from nn import train_ae, train_ffnn, train_rbm, evaluate_ffnn
from date_range import HIST_BEG, HIST_END

NUM_WEEKS = 12
NUM_DAYS = 5

ENT_ON_MON = False
ENT_MON_OPEN = True
EXIT_ON_MON = False
EXIT_ON_MON_OPEN = True

STOP_LOSS_HPR = -0.05

class SelectionAlgo(Enum):
    TOP = 0
    BOTTOM = 1
    MIDDLE = 2
    MIDDLE_ALT = 3


class SelectionType(Enum):
    PCT = 0
    FIXED = 1


SLCT_TYPE = SelectionType.FIXED
SLCT_VAL = 2

SLCT_PCT = 100
SLCT_ALG = SelectionAlgo.TOP

tickers, raw_dt, raw_data = load_npz_data_alt('data/nasdaq_adj.npz')

raw_mpl_dt = convert_to_mpl_time(raw_dt)

mask, traded_stocks = filter_activelly_tradeable_stocks(raw_data)
tradable_mask = get_tradable_stocks_mask(raw_data)

# plot_20_random_stock_prices(raw_data, raw_mpl_dt)
# plot_traded_stocks_per_day(traded_stocks, raw_mpl_dt)

TRAIN_BEG = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
# TRAIN_END = HIST_END
# TRAIN_END = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
# TRAIN_END = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()
TRAIN_END = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
SUNDAY = TRAIN_BEG + datetime.timedelta(days=7 - TRAIN_BEG.isoweekday())

train_records = 0
train_weeks = 0
total_weeks = 0
data_set_records = 0

dr = None
wr = None
t_w_s_hpr = None
t_w_s_h_hpr = None
t_w_s_l_hpr = None
s_hpr = None
s_hpr_model = None
s_int_r = None
c_l = None
c_s = None
stocks = None
w_data_index = None
w_num_stocks = None
w_enter_index = None
w_exit_index = None


def append_data_and_pad_with_zeros(data, _data):
    if data is None:
        return _data
    else:
        d_l = data.shape[1]
        s_l = _data.shape[1]
        if d_l > s_l:
            _data = np.pad(_data, ((0, 0), (0, d_l - s_l)), mode='constant', constant_values=0.0)
        elif s_l > d_l:
            data = np.pad(data, ((0, 0), (0, s_l - d_l)), mode='constant', constant_values=0.0)

        return np.concatenate([data, _data], axis=0)


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
    if SUNDAY > HIST_END:
        break
    w_r_i = get_dates_for_weekly_return(HIST_BEG, HIST_END, traded_stocks, SUNDAY, NUM_WEEKS + 1)
    # continue if all data not availiable yet
    if w_r_i is None:
        continue
    d_r_i = get_dates_for_daily_return(HIST_BEG, HIST_END, traded_stocks, SUNDAY - datetime.timedelta(days=7),
                                       NUM_DAYS)
    # continue if all data not availiable yet
    if d_r_i is None:
        continue
    if ENT_ON_MON:
        ent_r_i = get_one_trading_date(HIST_BEG, HIST_END, traded_stocks,
                                       SUNDAY - datetime.timedelta(days=7) + datetime.timedelta(days=1))
        if ent_r_i is None:
            continue
    else:
        ent_r_i = d_r_i[-1:]

    if EXIT_ON_MON:
        ext_r_i = get_one_trading_date(HIST_BEG, HIST_END, traded_stocks,
                                       SUNDAY + datetime.timedelta(days=1))
        if ext_r_i is None:
            continue
    else:
        ext_r_i = w_r_i[-1:]

    tw_r_i = get_intermediate_dates(HIST_BEG, HIST_END, traded_stocks, ent_r_i, ext_r_i)

    # t_s_i = get_tradable_stock_indexes(mask, w_r_i + d_r_i + ent_r_i + ext_r_i)
    t_s_i = get_tradable_stock_indexes(mask, w_r_i[:-1] + d_r_i)
    t_s_i_e_e = get_tradable_stock_indexes(tradable_mask, w_r_i[-1:] + ent_r_i + ext_r_i + tw_r_i)
    t_s_i = np.intersect1d(t_s_i, t_s_i_e_e)

    d_c = get_prices(raw_data, t_s_i, d_r_i, PxType.CLOSE)
    w_c = get_prices(raw_data, t_s_i, w_r_i, PxType.CLOSE)

    # exit_date = datetime.datetime.fromtimestamp(raw_dt[ext_r_i[0]]).date()
    # if exit_date == datetime.datetime.strptime('2017-07-07', '%Y-%m-%d').date():
    #     with open('data/test_analyze.csv', 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         row = ['ticker']
    #         for dt_idx in w_r_i[:-1]:
    #             dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
    #             row.append(dt.strftime('%Y-%m-%d'))
    #         for dt_idx in d_r_i:
    #             dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
    #             row.append(dt.strftime('%Y-%m-%d'))
    #         writer.writerow(row)
    #
    #         idx = 0
    #         for ticker_idx in t_s_i:
    #             ticker = tickers[ticker_idx]
    #             row = []
    #             row.append(ticker)
    #             for v in w_c[idx, :-1]:
    #                 row.append(v)
    #             for v in d_c[idx, :]:
    #                 row.append(v)
    #             writer.writerow(row)
    #             idx += 1

    px_type = PxType.CLOSE
    if ENT_ON_MON and ENT_MON_OPEN:
        px_type = PxType.OPEN
    ent_px = get_prices(raw_data, t_s_i, ent_r_i, px_type)
    px_type = PxType.CLOSE
    if EXIT_ON_MON and EXIT_ON_MON_OPEN:
        px_type = PxType.OPEN
    ext_px = get_prices(raw_data, t_s_i, ext_r_i, px_type)

    # calc daily returns
    d_n_r = calc_z_score(d_c)
    w_n_r = calc_z_score(w_c[:, :-1])

    _s_hpr = (w_c[:, NUM_WEEKS + 1] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]
    _s_hpr_model = (ext_px[:, 0] - ent_px[:, 0]) / ent_px[:, 0]

    tw_s_px = get_prices(raw_data, t_s_i, tw_r_i, PxType.CLOSE)
    _t_w_s_hpr = ((tw_s_px.transpose() - ent_px[:, 0]) / ent_px[:, 0]).transpose()

    tw_s_h_px = get_prices(raw_data, t_s_i, tw_r_i, PxType.HIGH)
    _t_w_s_h_hpr = ((tw_s_h_px.transpose() - ent_px[:, 0]) / ent_px[:, 0]).transpose()

    tw_s_l_px = get_prices(raw_data, t_s_i, tw_r_i, PxType.LOW)
    _t_w_s_l_hpr = ((tw_s_l_px.transpose() - ent_px[:, 0]) / ent_px[:, 0]).transpose()

    _s_int_r = (ent_px[:, 0] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]

    hpr_med = np.median(_s_hpr)
    _c_l = _s_hpr >= hpr_med
    _c_s = ~_c_l

    enter_date_idx = w_r_i[NUM_WEEKS]
    exit_date_idx = w_r_i[NUM_WEEKS + 1]

    # sample size
    num_stocks = t_s_i.shape[0]

    stocks = append_data(stocks, t_s_i)
    dr = append_data(dr, d_n_r)
    wr = append_data(wr, w_n_r)
    t_w_s_hpr = append_data_and_pad_with_zeros(t_w_s_hpr, _t_w_s_hpr)
    t_w_s_h_hpr = append_data_and_pad_with_zeros(t_w_s_h_hpr, _t_w_s_h_hpr)
    t_w_s_l_hpr = append_data_and_pad_with_zeros(t_w_s_l_hpr, _t_w_s_l_hpr)
    s_hpr = append_data(s_hpr, _s_hpr)
    s_hpr_model = append_data(s_hpr_model, _s_hpr_model)
    s_int_r = append_data(s_int_r, _s_int_r)
    c_l = append_data(c_l, _c_l)
    c_s = append_data(c_s, _c_s)
    w_data_index = append_data(w_data_index, make_array(data_set_records))
    w_num_stocks = append_data(w_num_stocks, make_array(num_stocks))
    w_enter_index = append_data(w_enter_index, make_array(enter_date_idx))
    w_exit_index = append_data(w_exit_index, make_array(exit_date_idx))

    # record counts
    data_set_records += num_stocks
    total_weeks += 1
    if SUNDAY <= TRAIN_END:
        train_records += num_stocks
        train_weeks += 1

train_rbm(train_records, dr, wr)
train_ae(train_records, dr, wr)
train_ffnn(train_records, dr, wr, c_l, c_s, w_data_index, w_num_stocks)

prob_l = np.zeros((data_set_records), dtype=np.float)
evaluate_ffnn(data_set_records, dr, wr, prob_l)


def calc_classes_and_decisions(data_set_records, total_weeks, prob_l):
    c_l = np.zeros((data_set_records), dtype=np.bool)
    c_s = np.zeros((data_set_records), dtype=np.bool)

    s_l = np.zeros((data_set_records), dtype=np.bool)
    s_s = np.zeros((data_set_records), dtype=np.bool)

    top_hpr = np.zeros((total_weeks))
    bottom_hpr = np.zeros((total_weeks))
    model_hpr = np.zeros((total_weeks))

    model_no_sl_hpr = np.zeros((total_weeks))
    model_eod_sl_hpr = np.zeros((total_weeks))
    model_lb_sl_hpr = np.zeros((total_weeks))
    model_s_sl_hpr = np.zeros((total_weeks))

    min_w_eod_hpr = np.zeros((total_weeks))
    min_w_lb_hpr = np.zeros((total_weeks))
    l_port = np.empty((total_weeks), dtype=np.object)
    s_port = np.empty((total_weeks), dtype=np.object)

    for i in range(total_weeks):
        w_i = i
        beg = w_data_index[w_i]
        end = beg + w_num_stocks[w_i]

        _prob_l = prob_l[beg: end]

        prob_median = np.median(_prob_l)
        _s_c_l = c_l[beg: end]
        _s_c_s = c_s[beg: end]
        pred_long_cond = _prob_l >= prob_median
        _s_c_l |= pred_long_cond
        _s_c_s |= ~pred_long_cond

        if SLCT_TYPE == SelectionType.PCT:
            top_bound = np.percentile(_prob_l, 100 - SLCT_VAL)
            bottom_bound = np.percentile(_prob_l, SLCT_VAL)
        else:
            _prob_l_sorted = np.sort(_prob_l)
            bottom_bound = _prob_l_sorted[SLCT_VAL - 1]
            top_bound = _prob_l_sorted[-SLCT_VAL]

        _s_s_l = s_l[beg: end]
        _s_s_s = s_s[beg: end]
        long_cond = _prob_l >= top_bound
        short_cond = _prob_l <= bottom_bound
        _s_s_l |= long_cond
        _s_s_s |= short_cond
        _int_r = s_int_r[beg:end]
        l_s_int_r = _int_r[_s_s_l]
        s_s_int_r = _int_r[_s_s_s]
        l_s_int_r_sorted = np.sort(l_s_int_r)
        s_s_int_r_sorted = np.sort(s_s_int_r)

        if SLCT_ALG == SelectionAlgo.TOP:
            l_int_r_t_b = np.max(l_s_int_r)
            l_int_r_b_b = np.percentile(l_s_int_r, 100 - SLCT_PCT)
        elif SLCT_ALG == SelectionAlgo.BOTTOM:
            l_int_r_t_b = np.percentile(l_s_int_r, SLCT_PCT)
            l_int_r_b_b = np.min(l_s_int_r)
        elif SLCT_ALG == SelectionAlgo.MIDDLE:
            l_int_r_t_b = np.percentile(l_s_int_r, 100 - SLCT_PCT / 2)
            l_int_r_b_b = np.percentile(l_s_int_r, SLCT_PCT / 2)

        if SLCT_ALG == SelectionAlgo.TOP:
            s_int_r_t_b = np.percentile(s_s_int_r, SLCT_PCT)
            s_int_r_b_b = np.min(s_s_int_r)
        elif SLCT_ALG == SelectionAlgo.BOTTOM:
            s_int_r_t_b = np.max(s_s_int_r)
            s_int_r_b_b = np.percentile(s_s_int_r, 100 - SLCT_PCT)
        elif SLCT_ALG == SelectionAlgo.MIDDLE:
            s_int_r_t_b = np.percentile(s_s_int_r, 100 - SLCT_PCT / 2)
            s_int_r_b_b = np.percentile(s_s_int_r, SLCT_PCT / 2)

        sel_l_cond = _s_s_l
        sel_l_cond &= _int_r >= l_int_r_b_b
        sel_l_cond &= _int_r <= l_int_r_t_b

        sel_s_cond = _s_s_s
        sel_s_cond &= _int_r <= s_int_r_t_b
        sel_s_cond &= _int_r >= s_int_r_b_b
        _s_hpr_model = s_hpr_model[beg: end]
        l_s_hpr = _s_hpr_model[sel_l_cond]
        s_s_hpr = _s_hpr_model[sel_s_cond]

        _stocks = stocks[beg:end]
        _l_stocks = _stocks[sel_l_cond]
        _s_stocks = _stocks[sel_s_cond]
        s_longs = ""
        s_shorts = ""
        idx = 0
        for _stock_idx in _l_stocks:
            if s_longs != "":
                s_longs += " "
            s_longs += tickers[_stock_idx]
            s_longs += " "
            s_longs += str(l_s_hpr[idx])
            idx += 1
        idx = 0
        for _stock_idx in _s_stocks:
            if s_shorts != "":
                s_shorts += " "
            s_shorts += tickers[_stock_idx]
            s_shorts += " "
            s_shorts += str(s_s_hpr[idx])
            idx += 1

        l_port[w_i] = s_longs
        s_port[w_i] = s_shorts
        l_hpr = np.mean(l_s_hpr)
        s_hpr = np.mean(s_s_hpr)
        top_hpr[w_i] = l_hpr
        bottom_hpr[w_i] = s_hpr
        w_hpr = (l_hpr - s_hpr) / 2

        _t_w_s_s_hpr = t_w_s_hpr[beg: end, :]
        _t_w_l_s_hpr = _t_w_s_s_hpr[sel_l_cond, :]
        _t_w_s_s_hpr = _t_w_s_s_hpr[sel_s_cond, :]
        _t_w_l_s_hpr_mean = np.mean(_t_w_l_s_hpr, axis=0)
        _t_w_s_s_hpr_mean = np.mean(_t_w_s_s_hpr, axis=0)
        _t_w_hpr = (_t_w_l_s_hpr_mean - _t_w_s_s_hpr_mean) / 2
        # calc min w eod hpr
        _min_w_eod_hpr = np.min(_t_w_hpr)
        min_w_eod_hpr[w_i] = _min_w_eod_hpr
        # calc no sl model
        _model_no_sl_hpr = w_hpr
        model_no_sl_hpr[w_i] = _model_no_sl_hpr
        # calc eod sl model
        _model_eod_sl_hpr = _model_no_sl_hpr
        sl_idxs = np.nonzero(_t_w_hpr <= STOP_LOSS_HPR)
        if sl_idxs[0].shape[0] > 0:
            _model_eod_sl_hpr = _t_w_hpr[sl_idxs[0][0]]
        model_eod_sl_hpr[w_i] = _model_eod_sl_hpr

        # calc lower bound hpr
        _t_w_s_s_h_hpr = t_w_s_h_hpr[beg: end, :]
        _t_w_s_s_l_hpr = t_w_s_l_hpr[beg: end, :]
        _t_w_l_s_lb_hpr = _t_w_s_s_l_hpr[sel_l_cond, :]
        _t_w_s_s_lb_hpr = _t_w_s_s_h_hpr[sel_s_cond, :]
        _t_w_l_s_lb_hpr_mean = np.mean(_t_w_l_s_lb_hpr, axis=0)
        _t_w_s_s_lb_hpr_mean = np.mean(_t_w_s_s_lb_hpr, axis=0)
        _t_w_lb_hpr = (_t_w_l_s_lb_hpr_mean - _t_w_s_s_lb_hpr_mean) / 2
        # calc lower bound weekly min
        _min_w_lb_hpr = np.min(_t_w_lb_hpr)
        min_w_lb_hpr[w_i] = _min_w_lb_hpr
        # calc lower bound sl
        _model_lb_hpr = w_hpr
        if _min_w_lb_hpr <= STOP_LOSS_HPR:
            _model_lb_hpr = STOP_LOSS_HPR
        model_lb_sl_hpr[w_i] = _model_lb_hpr
        # calc sl by stock model
        l_condition = _t_w_l_s_lb_hpr > STOP_LOSS_HPR
        l_condition = np.all(l_condition, axis=1)
        l_s_sl_hpr = np.where(l_condition, l_s_hpr, STOP_LOSS_HPR)
        s_condition = _t_w_s_s_lb_hpr < -STOP_LOSS_HPR
        s_condition = np.all(s_condition, axis=1)
        s_s_sl_hpr = np.where(s_condition, s_s_hpr, -STOP_LOSS_HPR)
        l_s_sl_hpr_mean = np.mean(l_s_sl_hpr)
        s_s_sl_hpr_mean = np.mean(s_s_sl_hpr)
        _model_s_sl_hpr = (l_s_sl_hpr_mean - s_s_sl_hpr_mean) / 2
        model_s_sl_hpr[w_i] = _model_s_sl_hpr

        model_hpr[w_i] = _model_no_sl_hpr

    return c_l, c_s, model_no_sl_hpr, model_eod_sl_hpr, model_lb_sl_hpr, model_s_sl_hpr, top_hpr, bottom_hpr, min_w_eod_hpr, min_w_lb_hpr, l_port, s_port

e_c_l, e_c_s, e_model_no_sl_hpr, e_model_eod_sl_hpr, e_model_lb_sl_hpr, e_model_s_sl_hpr, e_t_hpr, e_b_hpr, e_min_w_hpr, e_min_w_lb_hpr, l_port, s_port = calc_classes_and_decisions(
    data_set_records, total_weeks, prob_l
)

confusion_matrix(c_l[train_records:], c_s[train_records:], e_c_l[train_records:], e_c_s[train_records:])
hpr_analysis(e_t_hpr[train_weeks:], e_b_hpr[train_weeks:])
wealth_graph(e_model_no_sl_hpr[train_weeks:],
             w_enter_index[train_weeks:],
             w_exit_index[train_weeks:],
             raw_mpl_dt,
             raw_dt)
wealth_csv(e_model_no_sl_hpr[train_weeks:],
           e_model_eod_sl_hpr[train_weeks:],
           e_model_lb_sl_hpr[train_weeks:],
           e_model_s_sl_hpr[train_weeks:],
           e_t_hpr[train_weeks:],
           e_b_hpr[train_weeks:],
           e_min_w_hpr[train_weeks:],
           e_min_w_lb_hpr[train_weeks:],
           w_enter_index[train_weeks:],
           w_exit_index[train_weeks:],
           raw_dt,
           l_port[train_weeks:],
           s_port[train_weeks:])

plt.show(True)
