import datetime

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import csv

from tickers import get_nyse_nasdaq_tickers, get_nyse_tickers, get_nasdaq_tickers
from data_utils import filter_activelly_tradeable_stocks, convert_to_mpl_time, get_dates_for_daily_return, \
    get_one_trading_date, get_dates_for_weekly_return, get_tradable_stock_indexes, get_prices, \
    PxType, calc_z_score, get_tradable_stocks_mask, get_intermediate_dates, get_snp_mask, get_active_stks, \
    get_top_tradable_stks
from download_utils import load_npz_data, load_npz_data_alt
from visualization import wealth_graph, confusion_matrix, wealth_csv, calc_wealth
from visualization import plot_20_random_stock_prices, plot_traded_stocks_per_day
from nn import train_ae, train_ffnn, train_rbm, evaluate_ffnn
from date_range import HIST_BEG, HIST_END

NUM_WEEKS = 12
NUM_DAYS = 5

ENT_ON_MON = False
ENT_MON_OPEN = True
EXIT_ON_MON = False
EXIT_ON_MON_OPEN = True


class SelectionAlgo(Enum):
    TOP = 0
    BOTTOM = 1
    MIDDLE = 2
    MIDDLE_ALT = 3


class SelectionType(Enum):
    PCT = 0
    FIXED = 1


class StopLossType(Enum):
    NO = 0
    EOD = 1
    LB = 2
    STOCK = 3


ADJ_PX = True

SLCT_TYPE = SelectionType.FIXED
# SLCT_VAL = 4
SLCT_VAL = 30

SLCT_PCT = 100
SLCT_ALG = SelectionAlgo.TOP

# STOP_LOSS_HPR = -0.19
# STOP_LOSS_TYPE = StopLossType.NO

STOP_LOSS_HPR = -0.12
STOP_LOSS_TYPE = StopLossType.NO

GRID_SEARCH = True

MIN_STOCKS_TRADABLE_PER_TRADING_DAY = 10

MIN_SELECTION_FILTER = False
MIN_SELECTION_STOCKS = 100

AVG_DAY_TO_FILTER = False
AVG_DAY_TO_LIMIT = 10000000
TOP_TRADABLE_FILTER = True
TOP_TRADABLE_STOCKS = 500
DAY_TO_FILTER = False
DAY_TO_LIMIT = 10000000

tickers, raw_dt, raw_data = load_npz_data_alt('data/nasdaq.npz')
# tickers, raw_dt, raw_data = load_npz_data_alt('data/nasdaq_adj.npz')

raw_mpl_dt = convert_to_mpl_time(raw_dt)

actively_tradeable_mask = filter_activelly_tradeable_stocks(raw_data, DAY_TO_FILTER)

tradable_mask = get_tradable_stocks_mask(raw_data)
tradable_stocks_per_day = tradable_mask[:, :].sum(0)
trading_day_mask = tradable_stocks_per_day > MIN_STOCKS_TRADABLE_PER_TRADING_DAY

snp_mask = get_snp_mask(tickers, raw_data, HIST_BEG, HIST_END)

# plot_20_random_stock_prices(raw_data, raw_mpl_dt)
# plot_traded_stocks_per_day(tradable_stocks_per_day, raw_mpl_dt)

TRAIN_BEG = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
TRAIN_END = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
# TRAIN_END = HIST_END

CV_BEG = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
CV_END = HIST_END
# CV_BEG = TRAIN_BEG
# CV_END = TRAIN_END

SUNDAY = TRAIN_BEG + datetime.timedelta(days=7 - TRAIN_BEG.isoweekday())

cv_beg_idx = None
cv_end_idx = None
cv_wk_beg_idx = None
cv_wk_end_idx = None

tr_beg_idx = None
tr_end_idx = None
tr_wk_beg_idx = None
tr_wk_end_idx = None

total_weeks = 0
data_set_records = 0

dr = None
wr = None
t_w_s_hpr = None
t_w_s_h_hpr = None
t_w_s_l_hpr = None
t_w_s_o_hpr = None
t_w_eod_num = None
t_w_eods = None
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


def update_indexes():
    global cv_beg_idx
    global cv_end_idx
    global cv_wk_beg_idx
    global cv_wk_end_idx

    global tr_beg_idx
    global tr_end_idx
    global tr_wk_beg_idx
    global tr_wk_end_idx

    if SUNDAY >= TRAIN_BEG and SUNDAY <= TRAIN_END:
        if tr_beg_idx is None:
            tr_beg_idx = data_set_records
        if tr_wk_beg_idx is None:
            tr_wk_beg_idx = total_weeks

    if SUNDAY > TRAIN_END:
        if tr_end_idx is None:
            tr_end_idx = data_set_records
        if tr_wk_end_idx is None:
            tr_wk_end_idx = total_weeks

    if SUNDAY >= CV_BEG and SUNDAY <= CV_END:
        if cv_beg_idx is None:
            cv_beg_idx = data_set_records
        if cv_wk_beg_idx is None:
            cv_wk_beg_idx = total_weeks

    if SUNDAY > CV_END:
        if cv_end_idx is None:
            cv_end_idx = data_set_records
        if cv_wk_end_idx is None:
            cv_wk_end_idx = total_weeks


while True:
    # iterate over weeks
    SUNDAY = SUNDAY + datetime.timedelta(days=7)
    # break when all availiable data processed
    if SUNDAY > HIST_END:
        break
    w_r_i = get_dates_for_weekly_return(HIST_BEG, HIST_END, trading_day_mask, SUNDAY, NUM_WEEKS + 1)
    # continue if all data not availiable yet
    if w_r_i is None:
        continue
    d_r_i = get_dates_for_daily_return(HIST_BEG, HIST_END, trading_day_mask, SUNDAY - datetime.timedelta(days=7),
                                       NUM_DAYS)
    # continue if all data not availiable yet
    if d_r_i is None:
        continue
    if ENT_ON_MON:
        ent_r_i = get_one_trading_date(HIST_BEG, HIST_END, trading_day_mask,
                                       SUNDAY - datetime.timedelta(days=7) + datetime.timedelta(days=1))
        if ent_r_i is None:
            continue
    else:
        ent_r_i = d_r_i[-1:]

    if EXIT_ON_MON:
        ext_r_i = get_one_trading_date(HIST_BEG, HIST_END, trading_day_mask,
                                       SUNDAY + datetime.timedelta(days=1))
        if ext_r_i is None:
            continue
    else:
        ext_r_i = w_r_i[-1:]

    tw_r_i = get_intermediate_dates(trading_day_mask, ent_r_i[0], ext_r_i[0])
    # stocks should be tradeable on all dates we need for calcs
    t_s_i = get_tradable_stock_indexes(tradable_mask, w_r_i + d_r_i + ent_r_i + ext_r_i + tw_r_i)

    if AVG_DAY_TO_FILTER:
        a_s_i = get_active_stks(raw_data, trading_day_mask, w_r_i[0], d_r_i[-1], AVG_DAY_TO_LIMIT)
        t_s_i = np.intersect1d(t_s_i, a_s_i)
    if TOP_TRADABLE_FILTER:
        t_t_s_i = get_top_tradable_stks(raw_data, trading_day_mask, w_r_i[0], d_r_i[-1], TOP_TRADABLE_STOCKS)
        t_s_i = np.intersect1d(t_s_i, t_t_s_i)
    if DAY_TO_FILTER:
        a_t_s_i = get_tradable_stock_indexes(actively_tradeable_mask, w_r_i[:-1] + d_r_i)
        t_s_i = np.intersect1d(t_s_i, a_t_s_i)

    # t_s_i_e_e = get_tradable_stock_indexes(tradable_mask, w_r_i[-1:] + ent_r_i + ext_r_i + tw_r_i)
    # t_s_i_old = np.intersect1d(t_s_i_old, t_s_i_e_e)
    #
    # print("new: %d vs old: %d" % (t_s_i.shape[0], t_s_i_old.shape[0]))

    if MIN_SELECTION_FILTER and t_s_i.shape[0] < MIN_SELECTION_STOCKS:
        continue

    d_c = get_prices(raw_data, t_s_i, d_r_i, PxType.CLOSE, ADJ_PX)
    w_c = get_prices(raw_data, t_s_i, w_r_i, PxType.CLOSE, ADJ_PX)

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
    ent_px = get_prices(raw_data, t_s_i, ent_r_i, px_type, ADJ_PX)
    px_type = PxType.CLOSE
    if EXIT_ON_MON and EXIT_ON_MON_OPEN:
        px_type = PxType.OPEN
    ext_px = get_prices(raw_data, t_s_i, ext_r_i, px_type, ADJ_PX)

    # calc daily returns
    d_n_r = calc_z_score(d_c)
    w_n_r = calc_z_score(w_c[:, :-1])

    _s_hpr = (w_c[:, NUM_WEEKS + 1] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]
    _s_hpr_model = (ext_px[:, 0] - ent_px[:, 0]) / ent_px[:, 0]

    tw_s_px = get_prices(raw_data, t_s_i, tw_r_i, PxType.CLOSE, ADJ_PX)
    _t_w_s_hpr = ((tw_s_px.transpose() - ent_px[:, 0]) / ent_px[:, 0]).transpose()

    tw_s_h_px = get_prices(raw_data, t_s_i, tw_r_i, PxType.HIGH, ADJ_PX)
    _t_w_s_h_hpr = ((tw_s_h_px.transpose() - ent_px[:, 0]) / ent_px[:, 0]).transpose()

    tw_s_o_px = get_prices(raw_data, t_s_i, tw_r_i, PxType.OPEN, ADJ_PX)
    _t_w_s_o_hpr = ((tw_s_o_px.transpose() - ent_px[:, 0]) / ent_px[:, 0]).transpose()

    tw_s_l_px = get_prices(raw_data, t_s_i, tw_r_i, PxType.LOW, ADJ_PX)
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
    # intermediate week points num
    _t_w_eod_num = _t_w_s_hpr.shape[1]
    t_w_eod_num = append_data(t_w_eod_num, np.array([_t_w_eod_num]))
    _t_w_eods = np.array(tw_r_i).reshape((1, -1))
    t_w_eods = append_data_and_pad_with_zeros(t_w_eods, _t_w_eods)
    t_w_s_hpr = append_data_and_pad_with_zeros(t_w_s_hpr, _t_w_s_hpr)
    t_w_s_o_hpr = append_data_and_pad_with_zeros(t_w_s_o_hpr, _t_w_s_o_hpr)
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

    update_indexes()

    # record counts
    data_set_records += num_stocks
    total_weeks += 1

update_indexes()

train_rbm(dr, wr, tr_beg_idx, tr_end_idx)
train_ae(dr, wr, tr_beg_idx, tr_end_idx)
train_ffnn(dr, wr, c_l, c_s, w_data_index, w_num_stocks, tr_beg_idx, tr_end_idx, tr_wk_beg_idx, tr_wk_end_idx)

prob_l = np.zeros((data_set_records), dtype=np.float)
evaluate_ffnn(data_set_records, dr, wr, prob_l)


def calc_classes_and_decisions(data_set_records, total_weeks, prob_l):
    c_l = np.zeros((data_set_records), dtype=np.bool)
    c_s = np.zeros((data_set_records), dtype=np.bool)

    s_l = np.zeros((data_set_records), dtype=np.bool)
    s_s = np.zeros((data_set_records), dtype=np.bool)

    model_no_sl_hpr = np.zeros((total_weeks))
    min_w_eod_hpr_no_sl = np.zeros((total_weeks))
    min_w_lb_hpr_no_sl = np.zeros((total_weeks))
    l_port_no_sl = np.empty((total_weeks), dtype=np.object)
    s_port_no_sl = np.empty((total_weeks), dtype=np.object)
    l_stops_no_sl = np.zeros((total_weeks))
    s_stops_no_sl = np.zeros((total_weeks))

    model_eod_sl_hpr = np.zeros((total_weeks))
    min_w_eod_hpr_eod_sl = np.zeros((total_weeks))
    min_w_lb_hpr_eod_sl = np.zeros((total_weeks))
    l_port_eod_sl = np.empty((total_weeks), dtype=np.object)
    s_port_eod_sl = np.empty((total_weeks), dtype=np.object)
    l_stops_eod_sl = np.zeros((total_weeks))
    s_stops_eod_sl = np.zeros((total_weeks))

    model_lb_sl_hpr = np.zeros((total_weeks))
    min_w_eod_hpr_lb_sl = np.zeros((total_weeks))
    min_w_lb_hpr_lb_sl = np.zeros((total_weeks))
    l_port_lb_sl = np.empty((total_weeks), dtype=np.object)
    s_port_lb_sl = np.empty((total_weeks), dtype=np.object)
    l_stops_lb_sl = np.zeros((total_weeks))
    s_stops_lb_sl = np.zeros((total_weeks))

    model_s_sl_hpr = np.zeros((total_weeks))
    min_w_eod_hpr_s_sl = np.zeros((total_weeks))
    min_w_lb_hpr_s_sl = np.zeros((total_weeks))
    l_port_s_sl = np.empty((total_weeks), dtype=np.object)
    s_port_s_sl = np.empty((total_weeks), dtype=np.object)
    l_stops_s_sl = np.zeros((total_weeks))
    s_stops_s_sl = np.zeros((total_weeks))

    for i in range(total_weeks):
        w_i = i
        beg = w_data_index[w_i]
        end = beg + w_num_stocks[w_i]

        _prob_l = prob_l[beg: end]

        prob_median = np.median(_prob_l)
        prob_median = 0.5
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

        # select long and short stocks in portfolio
        _stocks = stocks[beg:end]
        _l_stocks = _stocks[sel_l_cond]
        _s_stocks = _stocks[sel_s_cond]

        _t_w_eod_num = t_w_eod_num[w_i]
        _t_w_eods = t_w_eods[w_i, :_t_w_eod_num]
        # select eod hpr by stock during the week
        _t_w_s_s_hpr = t_w_s_hpr[beg: end, :_t_w_eod_num]
        _t_w_l_s_hpr = _t_w_s_s_hpr[sel_l_cond, :]
        _t_w_s_s_hpr = _t_w_s_s_hpr[sel_s_cond, :]
        # select lb hpr by stock during the week
        _t_w_s_s_h_hpr = t_w_s_h_hpr[beg: end, :_t_w_eod_num]
        _t_w_s_s_l_hpr = t_w_s_l_hpr[beg: end, :_t_w_eod_num]
        _t_w_l_s_lb_hpr = _t_w_s_s_l_hpr[sel_l_cond, :]
        _t_w_s_s_lb_hpr = _t_w_s_s_h_hpr[sel_s_cond, :]
        # select hpr by stock during the week using open px
        _t_w_ss_o_hpr = t_w_s_o_hpr[beg: end, :_t_w_eod_num]
        _t_w_l_s_o_hpr = _t_w_ss_o_hpr[sel_l_cond, :]
        _t_w_s_s_o_hpr = _t_w_ss_o_hpr[sel_s_cond, :]

        def calc_params(override_ext_lb_hpr,
                        _t_w_eods,
                        t_w_l_s_hpr,
                        t_w_s_s_hpr,
                        t_w_l_s_lb_hpr,
                        t_w_s_s_lb_hpr,
                        _s_l_ext_idx,
                        _s_s_ext_idx,
                        _s_l_stop,
                        _s_s_stop,
                        _s_l_ext_hpr,
                        _s_s_ext_hpr):
            # create arrays copy
            _t_w_l_s_hpr = np.copy(t_w_l_s_hpr)
            _t_w_s_s_hpr = np.copy(t_w_s_s_hpr)
            _t_w_l_s_lb_hpr = np.copy(t_w_l_s_lb_hpr)
            _t_w_s_s_lb_hpr = np.copy(t_w_s_s_lb_hpr)
            # fill proper array values
            for idx in range(_s_l_ext_idx.shape[0]):
                _ext_idx = _s_l_ext_idx[idx]
                _ext_hpr = _s_l_ext_hpr[idx]
                _t_w_l_s_hpr[idx, _ext_idx:] = _ext_hpr
                _t_w_l_s_lb_hpr[idx, _ext_idx if override_ext_lb_hpr else _ext_idx + 1:] = _ext_hpr
            for idx in range(_s_s_ext_idx.shape[0]):
                _ext_idx = _s_s_ext_idx[idx]
                _ext_hpr = _s_s_ext_hpr[idx]
                _t_w_s_s_hpr[idx, _ext_idx:] = _ext_hpr
                _t_w_s_s_lb_hpr[idx, _ext_idx if override_ext_lb_hpr else _ext_idx + 1:] = _ext_hpr

            # calc portfolio hpr
            _l_s_hpr = _t_w_l_s_hpr[:, -1]
            _s_s_hpr = _t_w_s_s_hpr[:, -1]

            _l_hpr = np.mean(_l_s_hpr)
            _s_hpr = np.mean(_s_s_hpr)
            _w_hpr = (_l_hpr - _s_hpr) / 2

            # calc min w eod hpr
            _t_w_l_s_hpr_mean = np.mean(_t_w_l_s_hpr, axis=0)
            _t_w_s_s_hpr_mean = np.mean(_t_w_s_s_hpr, axis=0)
            _t_w_hpr = (_t_w_l_s_hpr_mean - _t_w_s_s_hpr_mean) / 2
            _min_w_eod_hpr = np.min(_t_w_hpr)

            # calc lower bound weekly min
            _t_w_l_s_lb_hpr_mean = np.mean(_t_w_l_s_lb_hpr, axis=0)
            _t_w_s_s_lb_hpr_mean = np.mean(_t_w_s_s_lb_hpr, axis=0)
            _t_w_lb_hpr = (_t_w_l_s_lb_hpr_mean - _t_w_s_s_lb_hpr_mean) / 2
            _min_w_lb_hpr = np.min(_t_w_lb_hpr)

            # calc portfolio string
            _s_longs = ""
            _s_shorts = ""
            if not GRID_SEARCH:
                idx = 0
                for _stock_idx in _l_stocks:
                    if _s_longs != "":
                        _s_longs += " "
                    _s_longs += tickers[_stock_idx]
                    _s_longs += " "
                    _s_longs += "%.2f%%" % (_l_s_hpr[idx] * 100.0)
                    _s_longs += " "
                    dt_ext = datetime.datetime.fromtimestamp(raw_dt[_t_w_eods[_s_l_ext_idx[idx]]])
                    _s_longs += dt_ext.strftime('%Y-%m-%d')
                    idx += 1
                idx = 0
                for _stock_idx in _s_stocks:
                    if _s_shorts != "":
                        _s_shorts += " "
                    _s_shorts += tickers[_stock_idx]
                    _s_shorts += " "
                    _s_shorts += "%.2f%%" % (_s_s_hpr[idx] * 100.0)
                    _s_shorts += " "
                    dt_ext = datetime.datetime.fromtimestamp(raw_dt[_t_w_eods[_s_s_ext_idx[idx]]])
                    _s_shorts += dt_ext.strftime('%Y-%m-%d')
                    idx += 1

            # calc long and short stops
            _l_stops = np.sum(_s_l_stop)
            _s_stops = np.sum(_s_s_stop)

            return _l_hpr, _s_hpr, _w_hpr, _min_w_eod_hpr, _min_w_lb_hpr, _l_stops, _s_stops, _s_longs, _s_shorts

        _default_ext_idx = _t_w_eod_num - 1

        # calc no sl model
        _s_l_ext_idx = np.full(_l_stocks.shape, _t_w_eod_num - 1)
        _s_s_ext_idx = np.full(_s_stocks.shape, _t_w_eod_num - 1)
        _s_l_stop = np.full(_l_stocks.shape, False)
        _s_s_stop = np.full(_l_stocks.shape, False)
        _s_l_ext_hpr = _t_w_l_s_hpr[:, _t_w_eod_num - 1]
        _s_s_ext_hpr = _t_w_s_s_hpr[:, _t_w_eod_num - 1]

        _l_hpr, _s_hpr, _w_hpr, _min_w_eod_hpr, _min_w_lb_hpr, _l_stops, _s_stops, _s_longs, _s_shorts = calc_params(
            False,
            _t_w_eods,
            _t_w_l_s_hpr,
            _t_w_s_s_hpr,
            _t_w_l_s_lb_hpr,
            _t_w_s_s_lb_hpr,
            _s_l_ext_idx,
            _s_s_ext_idx,
            _s_l_stop,
            _s_s_stop,
            _s_l_ext_hpr,
            _s_s_ext_hpr)
        model_no_sl_hpr[w_i] = _w_hpr
        min_w_eod_hpr_no_sl[w_i] = _min_w_eod_hpr
        min_w_lb_hpr_no_sl[w_i] = _min_w_lb_hpr
        l_port_no_sl[w_i] = _s_longs
        s_port_no_sl[w_i] = _s_shorts
        l_stops_no_sl[w_i] = _l_stops
        s_stops_no_sl[w_i] = _s_stops

        # calc eod model
        _t_w_l_s_hpr_mean = np.mean(_t_w_l_s_hpr, axis=0)
        _t_w_s_s_hpr_mean = np.mean(_t_w_s_s_hpr, axis=0)
        _t_w_hpr = (_t_w_l_s_hpr_mean - _t_w_s_s_hpr_mean) / 2
        sl_idxs = np.nonzero(_t_w_hpr <= STOP_LOSS_HPR)
        _ext_idx = _default_ext_idx
        _stop = False
        if sl_idxs[0].shape[0] > 0:
            _ext_idx = sl_idxs[0][0]
            _stop = True

        _s_l_ext_idx = np.full(_l_stocks.shape, _ext_idx)
        _s_s_ext_idx = np.full(_s_stocks.shape, _ext_idx)
        _s_l_stop = np.full(_l_stocks.shape, _stop)
        _s_s_stop = np.full(_l_stocks.shape, _stop)
        _s_l_ext_hpr = _t_w_l_s_hpr[:, _ext_idx]
        _s_s_ext_hpr = _t_w_s_s_hpr[:, _ext_idx]

        _l_hpr, _s_hpr, _w_hpr, _min_w_eod_hpr, _min_w_lb_hpr, _l_stops, _s_stops, _s_longs, _s_shorts = calc_params(
            False,
            _t_w_eods,
            _t_w_l_s_hpr,
            _t_w_s_s_hpr,
            _t_w_l_s_lb_hpr,
            _t_w_s_s_lb_hpr,
            _s_l_ext_idx,
            _s_s_ext_idx,
            _s_l_stop,
            _s_s_stop,
            _s_l_ext_hpr,
            _s_s_ext_hpr)

        model_eod_sl_hpr[w_i] = _w_hpr
        min_w_eod_hpr_eod_sl[w_i] = _min_w_eod_hpr
        min_w_lb_hpr_eod_sl[w_i] = _min_w_lb_hpr
        l_port_eod_sl[w_i] = _s_longs
        s_port_eod_sl[w_i] = _s_shorts
        l_stops_eod_sl[w_i] = _l_stops
        s_stops_eod_sl[w_i] = _s_stops

        # calc lower bound hpr
        _t_w_l_s_lb_hpr_mean = np.mean(_t_w_l_s_lb_hpr, axis=0)
        _t_w_s_s_lb_hpr_mean = np.mean(_t_w_s_s_lb_hpr, axis=0)
        _t_w_lb_hpr = (_t_w_l_s_lb_hpr_mean - _t_w_s_s_lb_hpr_mean) / 2

        _t_w_l_s_o_hpr_mean = np.mean(_t_w_l_s_o_hpr, axis=0)
        _t_w_s_s_o_hpr_mean = np.mean(_t_w_s_s_o_hpr, axis=0)
        _t_w_o_hpr = (_t_w_l_s_o_hpr_mean - _t_w_s_s_o_hpr_mean) / 2

        sl_idxs = np.nonzero(_t_w_lb_hpr <= STOP_LOSS_HPR)
        _ext_idx = _t_w_eod_num - 1
        _stop = False
        _stop_on_open = False
        if sl_idxs[0].shape[0] > 0:
            _ext_idx = sl_idxs[0][0]
            _stop = True
            if _t_w_o_hpr[_ext_idx] <= STOP_LOSS_HPR:
                _stop_on_open = True

        _s_l_ext_idx = np.full(_l_stocks.shape, _ext_idx)
        _s_s_ext_idx = np.full(_s_stocks.shape, _ext_idx)
        _s_l_stop = np.full(_l_stocks.shape, _stop)
        _s_s_stop = np.full(_s_stocks.shape, _stop)
        if _stop:
            if _stop_on_open:
                _s_l_ext_hpr = _t_w_l_s_o_hpr[:,_ext_idx]
                _s_s_ext_hpr = _t_w_s_s_o_hpr[:,_ext_idx]
            else:
                _s_l_ext_hpr = np.full(_l_stocks.shape, STOP_LOSS_HPR)
                _s_s_ext_hpr = np.full(_s_stocks.shape, -STOP_LOSS_HPR)
        else:
            _s_l_ext_hpr = _t_w_l_s_hpr[:, _ext_idx]
            _s_s_ext_hpr = _t_w_s_s_hpr[:, _ext_idx]

        _l_hpr, _s_hpr, _w_hpr, _min_w_eod_hpr, _min_w_lb_hpr, _l_stops, _s_stops, _s_longs, _s_shorts = calc_params(
            True,
            _t_w_eods,
            _t_w_l_s_hpr,
            _t_w_s_s_hpr,
            _t_w_l_s_lb_hpr,
            _t_w_s_s_lb_hpr,
            _s_l_ext_idx,
            _s_s_ext_idx,
            _s_l_stop,
            _s_s_stop,
            _s_l_ext_hpr,
            _s_s_ext_hpr)

        model_lb_sl_hpr[w_i] = _w_hpr
        min_w_eod_hpr_lb_sl[w_i] = _min_w_eod_hpr
        min_w_lb_hpr_lb_sl[w_i] = _min_w_lb_hpr
        l_port_lb_sl[w_i] = _s_longs
        s_port_lb_sl[w_i] = _s_shorts
        l_stops_lb_sl[w_i] = _l_stops
        s_stops_lb_sl[w_i] = _s_stops

        # calc sl by stock model
        def first_true_idx_by_row(mask, default_idx):
            idxs = np.full(mask.shape[0], default_idx)
            for i, ele in enumerate(np.argmax(mask, axis=1)):
                if ele == 0 and mask[i][0] == 0:
                    idxs[i] = default_idx
                else:
                    idxs[i] = ele

            return idxs

        # _t_w_l_s_o_hpr = _t_w_s_s_o_hpr[sel_l_cond, :]
        # _t_w_s_s_o_hpr = _t_w_s_s_o_hpr[sel_s_cond, :]
        # long
        _s_l_stop_mask = _t_w_l_s_lb_hpr <= STOP_LOSS_HPR
        _s_l_stop = np.any(_s_l_stop_mask, axis=1)
        _s_l_ext_idx = first_true_idx_by_row(_s_l_stop_mask, _default_ext_idx)
        # by default ext hpr == hpr no sl
        _s_l_ext_hpr = _t_w_l_s_hpr[:, _default_ext_idx]
        # calc hpr for stocks with sl
        # exit idx for stocks with sl
        _s_l_sl_ext_idx = _s_l_ext_idx[_s_l_stop]
        # stocks with sl hpr by open px during the week
        _t_w_l_sl_s_o_hrp = _t_w_l_s_o_hpr[_s_l_stop, :]
        _aaa = _t_w_l_sl_s_o_hrp
        _xxx = np.arange(_aaa.shape[0])
        _yyy = _s_l_sl_ext_idx
        # hpr for stocks with sl by open px
        _l_s_sl_o_hrp = _aaa[_xxx, _yyy]
        # condition
        _use_o_px = _l_s_sl_o_hrp <= STOP_LOSS_HPR
        # stock with sl hpr
        _s_l_sl_hpr = np.where(_use_o_px, _l_s_sl_o_hrp, STOP_LOSS_HPR)
        # override default hpr for stocks with sl
        _s_l_ext_hpr[_s_l_stop] = _s_l_sl_hpr

        # short
        _s_s_stop_mask = _t_w_s_s_lb_hpr >= -STOP_LOSS_HPR
        _s_s_stop = np.any(_s_s_stop_mask, axis=1)
        _s_s_ext_idx = first_true_idx_by_row(_s_s_stop_mask, _default_ext_idx)

        # by default ext hpr == hpr no sl
        _s_s_ext_hpr = _t_w_s_s_hpr[:, _default_ext_idx]
        # calc hpr for stocks with sl
        # exit idx for stocks with sl
        _s_s_sl_ext_idx = _s_s_ext_idx[_s_s_stop]
        # stocks with sl hpr by open px during the week
        _t_w_s_sl_s_o_hrp = _t_w_s_s_o_hpr[_s_s_stop, :]
        _aaa = _t_w_s_sl_s_o_hrp
        _xxx = np.arange(_aaa.shape[0])
        _yyy = _s_s_sl_ext_idx
        # hpr for stocks with sl by open px
        _s_s_sl_o_hrp = _aaa[_xxx, _yyy]
        # condition
        _use_o_px = _s_s_sl_o_hrp >= -STOP_LOSS_HPR
        # stock with sl hpr
        _s_s_sl_hpr = np.where(_use_o_px, _s_s_sl_o_hrp, -STOP_LOSS_HPR)
        # override default hpr for stocks with sl
        _s_s_ext_hpr[_s_s_stop] = _s_s_sl_hpr

        _l_hpr, _s_hpr, _w_hpr, _min_w_eod_hpr, _min_w_lb_hpr, _l_stops, _s_stops, _s_longs, _s_shorts = calc_params(
            True,
            _t_w_eods,
            _t_w_l_s_hpr,
            _t_w_s_s_hpr,
            _t_w_l_s_lb_hpr,
            _t_w_s_s_lb_hpr,
            _s_l_ext_idx,
            _s_s_ext_idx,
            _s_l_stop,
            _s_s_stop,
            _s_l_ext_hpr,
            _s_s_ext_hpr)
        model_s_sl_hpr[w_i] = _w_hpr
        min_w_eod_hpr_s_sl[w_i] = _min_w_eod_hpr
        min_w_lb_hpr_s_sl[w_i] = _min_w_lb_hpr
        l_port_s_sl[w_i] = _s_longs
        s_port_s_sl[w_i] = _s_shorts
        l_stops_s_sl[w_i] = _l_stops
        s_stops_s_sl[w_i] = _s_stops

        model_no_sl = (
            model_no_sl_hpr, min_w_eod_hpr_no_sl, min_w_lb_hpr_no_sl, l_stops_no_sl, s_stops_no_sl, l_port_no_sl,
            s_port_no_sl)
        model_eod_sl = (
            model_eod_sl_hpr, min_w_eod_hpr_eod_sl, min_w_lb_hpr_eod_sl, l_stops_eod_sl, s_stops_eod_sl, l_port_eod_sl,
            s_port_eod_sl)
        model_lb_sl = (
            model_lb_sl_hpr, min_w_eod_hpr_lb_sl, min_w_lb_hpr_lb_sl, l_stops_lb_sl, s_stops_lb_sl, l_port_lb_sl,
            s_port_lb_sl)
        model_s_sl = (
            model_s_sl_hpr, min_w_eod_hpr_s_sl, min_w_lb_hpr_s_sl, l_stops_s_sl, s_stops_s_sl, l_port_s_sl, s_port_s_sl)

    return c_l, \
           c_s, \
           model_no_sl, \
           model_eod_sl, \
           model_lb_sl, \
           model_s_sl


if GRID_SEARCH:
    with open('./data/grid_search.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                'stop loss type',
                'stop loss',
                'selection',
                'type',
                'wealth',
                'min w eod',
                'min w lb',
                'dd',
                'w dd',
                'w avg',
                'w best',
                'sharpe',
                'y avg',
                'recap wealth',
                'recap max dd'
            ))

        SLCT_TYPE = SelectionType.FIXED
        SLCT_VAL = 2
        STOP_LOSS_HPR = -0.05


        def print_rows_for_fixed_params():

            model_c_l, model_c_s, model_no_sl, model_eod_sl, model_lb_sl, model_s_sl = calc_classes_and_decisions(
                data_set_records, total_weeks, prob_l
            )

            def print_row(model, sl_name):
                model_hpr, model_min_w_eod_hpr, model_min_w_lb_hpr, model_l_stops, model_s_stops, model_l_port, model_s_port = model

                wealth, dd, sharpe, rc_wealth, rc_dd, rc_sharpe, yr, years = calc_wealth(
                    model_hpr[cv_wk_beg_idx:cv_wk_end_idx],
                    w_enter_index[cv_wk_beg_idx:cv_wk_end_idx],
                    raw_dt)

                yr_avg = np.mean(yr)
                w_dd = np.min(model_hpr[cv_wk_beg_idx:cv_wk_end_idx])
                w_avg = np.mean(model_hpr[cv_wk_beg_idx:cv_wk_end_idx])
                w_best = np.max(model_hpr[cv_wk_beg_idx:cv_wk_end_idx])

                min_min_w_eod = np.min(model_min_w_eod_hpr)
                min_min_w_lb = np.min(model_min_w_lb_hpr)

                writer.writerow(
                    (
                        sl_name,
                        "%f" % STOP_LOSS_HPR,
                        SLCT_VAL,
                        'pct' if SLCT_TYPE == SelectionType.PCT else 'fixed',
                        wealth[-1],
                        "%f" % min_min_w_eod,
                        "%f" % min_min_w_lb,
                        "%f" % dd,
                        "%f" % w_dd,
                        "%f" % w_avg,
                        "%f" % w_best,
                        sharpe,
                        "%f" % yr_avg,
                        rc_wealth[-1],
                        "%f" % rc_dd
                    ))

            print_row(model_no_sl, 'no')
            print_row(model_eod_sl, 'eod')
            print_row(model_lb_sl, 'lower bound')
            print_row(model_s_sl, 'stock')


        # grid search
        SLCT_TYPE = SelectionType.FIXED
        for SLCT_VAL in range(1, 40):
            print("FIXED %d" % SLCT_VAL)
            # for STOP_LOSS_HPR in np.linspace(-0.01, -0.30, 29 * 2 + 1):
            #     print("SL %.2f" % STOP_LOSS_HPR)
            #     print_rows_for_fixed_params()
            # no sl grid search
            STOP_LOSS_HPR = 0.0
            print("SL %.2f" % STOP_LOSS_HPR)
            print_rows_for_fixed_params()
        # SLCT_TYPE = SelectionType.PCT
        # for SLCT_VAL in np.linspace(0.5, 15, 15 * 2):
        #     for STOP_LOSS_HPR in np.linspace(-0.01, -0.50, 49 * 2 + 1):
        #         print_rows_for_fixed_params()

else:
    # plot hpr vs prob
    _prob_l = prob_l[cv_beg_idx:cv_end_idx]
    _s_hpr = s_hpr[cv_beg_idx:cv_end_idx]
    z = np.polyfit(_prob_l, _s_hpr, 2)
    p = np.poly1d(z)
    x = np.linspace(0, 1.0, 1000)
    y = p(x) * 100.0

    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, linestyle='-', color='0.75')
    ax.plot(_prob_l, _s_hpr * 100.0, 'bo')
    ax.plot(x, y, 'g-')

    model_c_l, model_c_s, model_no_sl, model_eod_sl, model_lb_sl, model_s_sl = calc_classes_and_decisions(
        data_set_records, total_weeks, prob_l
    )

    confusion_matrix(c_l[cv_beg_idx:cv_end_idx],
                     c_s[cv_beg_idx:cv_end_idx],
                     model_c_l[cv_beg_idx:cv_end_idx],
                     model_c_s[cv_beg_idx:cv_end_idx])

    type_to_idx = {
        StopLossType.NO: model_no_sl,
        StopLossType.EOD: model_eod_sl,
        StopLossType.LB: model_lb_sl,
        StopLossType.STOCK: model_s_sl
    }

    model = type_to_idx.get(STOP_LOSS_TYPE, model_no_sl)
    model_hpr, model_min_w_eod_hpr, model_min_w_lb_hpr, model_l_stops, model_s_stops, model_l_port, model_s_port = model

    wealth, dd, sharpe, rc_wealth, rc_dd, rc_sharpe, yr, years = calc_wealth(model_hpr[cv_wk_beg_idx:cv_wk_end_idx],
                                                                             w_enter_index[cv_wk_beg_idx:cv_wk_end_idx],
                                                                             raw_dt)

    yr_avg = np.mean(yr)
    w_dd = np.min(model_hpr[cv_wk_beg_idx:cv_wk_end_idx])
    w_avg = np.mean(model_hpr[cv_wk_beg_idx:cv_wk_end_idx])
    w_best = np.max(model_hpr[cv_wk_beg_idx:cv_wk_end_idx])

    print(
        "F: {:.2f} DD: {:.2f} W_DD: {:.2f} W_AVG: {:.2f} W_BEST: {:.2f} SHARPE: {:.2f} AVG_YEAR: {:.2f} F_R: {:.2f} DD_R: {:.2f}".format(
            wealth[-1],
            dd * 100.0,
            w_dd * 100.0,
            w_avg * 100.0,
            w_best * 100.0,
            sharpe,
            yr_avg * 100.0,
            rc_wealth[-1],
            rc_dd * 100.0
        ))

    wealth_graph(wealth,
                 dd,
                 sharpe,
                 rc_wealth,
                 rc_dd,
                 rc_sharpe,
                 yr,
                 years,
                 w_exit_index[cv_wk_beg_idx:cv_wk_end_idx],
                 raw_mpl_dt)

    wealth_csv("no",
               cv_wk_beg_idx,
               cv_wk_end_idx,
               w_enter_index,
               w_exit_index,
               raw_dt,
               model_no_sl
               )
    wealth_csv("eod",
               cv_wk_beg_idx,
               cv_wk_end_idx,
               w_enter_index,
               w_exit_index,
               raw_dt,
               model_eod_sl
               )
    wealth_csv("lb",
               cv_wk_beg_idx,
               cv_wk_end_idx,
               w_enter_index,
               w_exit_index,
               raw_dt,
               model_lb_sl
               )
    wealth_csv("stk",
               cv_wk_beg_idx,
               cv_wk_end_idx,
               w_enter_index,
               w_exit_index,
               raw_dt,
               model_s_sl
               )

    plt.show(True)
