import datetime
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from data_utils import filter_tradeable_stocks, convert_to_mpl_time, get_dates_for_daily_return, \
    get_date_for_enter_return, get_dates_for_weekly_return, get_tradeable_stock_indexes, get_prices, \
    PxType, calc_z_score
from download_utils import load_npz_data
from visualization import wealth_graph, confusion_matrix, hpr_analysis
from visualization import plot_20_random_stock_prices, plot_traded_stocks_per_day
from nn import train_ae, train_ffnn, train_rbm, evaluate_ffnn

NUM_WEEKS = 12
NUM_DAYS = 5

ENT_ON_MON = True
ENT_MON_OPEN = True
EXIT_ON_MON = False
EXIT_ON_MON_OPEN = True


class SelectionAlgo(Enum):
    TOP = 0
    BOTTOM = 1
    MIDDLE = 2
    MIDDLE_ALT = 3


BET_PCT = 2
SLCT_PCT = 100
SLCT_ALG = SelectionAlgo.TOP
# TOP_N_STOCKS = 2
# TOP_N_STOCKS = 8
TOP_N_STOCKS = None

raw_dt, raw_data = load_npz_data('data/nasdaq_raw_data.npz')

raw_mpl_dt = convert_to_mpl_time(raw_dt)

mask, traded_stocks = filter_tradeable_stocks(raw_data)

# plot_20_random_stock_prices(raw_data, raw_mpl_dt)
# plot_traded_stocks_per_day(traded_stocks, raw_mpl_dt)

# TRAIN_UP_TO_DATE = datetime.datetime.strptime('2008-01-01', '%Y-%m-%d').date()
TRAIN_UP_TO_DATE = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
# TRAIN_UP_TO_DATE = datetime.datetime.strptime('2013-01-01', '%Y-%m-%d').date()
# TRAIN_UP_TO_DATE = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()
# TRAIN_UP_TO_DATE = datetime.datetime.strptime('2017-04-18', '%Y-%m-%d').date()
# TRAIN_UP_TO_DATE = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
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
hpr_model = None
int_r = None
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
    d_r_i = get_dates_for_daily_return(START_DATE, END_DATE, traded_stocks, SUNDAY - datetime.timedelta(days=7),
                                       NUM_DAYS)
    # continue if all data not availiable yet
    if d_r_i is None:
        continue
    if ENT_ON_MON:
        ent_r_i = get_date_for_enter_return(START_DATE, END_DATE, traded_stocks,
                                            SUNDAY - datetime.timedelta(days=7) + datetime.timedelta(days=1))
        if ent_r_i is None:
            continue
    else:
        ent_r_i = d_r_i[-1:]

    if EXIT_ON_MON:
        ext_r_i = get_date_for_enter_return(START_DATE, END_DATE, traded_stocks,
                                            SUNDAY + datetime.timedelta(days=1))
        if ext_r_i is None:
            continue
    else:
        ext_r_i = w_r_i[-1:]

    # t_s_i = get_tradeable_stock_indexes(mask, w_r_i + d_r_i)
    t_s_i = get_tradeable_stock_indexes(mask, w_r_i + d_r_i + ent_r_i + ext_r_i)
    d_c = get_prices(raw_data, t_s_i, d_r_i, PxType.CLOSE)
    w_c = get_prices(raw_data, t_s_i, w_r_i, PxType.CLOSE)

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
    dr = append_data(dr, d_n_r)
    wr = append_data(wr, w_n_r)

    _hpr = (w_c[:, NUM_WEEKS + 1] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]
    hpr = append_data(hpr, _hpr)
    _hpr_model = (ext_px[:, 0] - ent_px[:, 0]) / ent_px[:, 0]
    hpr_model = append_data(hpr_model, _hpr_model)
    _int_r = (ent_px[:, 0] - w_c[:, NUM_WEEKS]) / w_c[:, NUM_WEEKS]
    int_r = append_data(int_r, _int_r)

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
    top_stocks_num = np.zeros((total_weeks))
    bottom_stocks_num = np.zeros((total_weeks))

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

        top_bound = np.percentile(_prob_l, 100 - BET_PCT)
        bottom_bound = np.percentile(_prob_l, BET_PCT)

        if TOP_N_STOCKS is not None:
            _prob_l_sorted = np.sort(_prob_l)
            bottom_bound = _prob_l_sorted[TOP_N_STOCKS - 1]
            top_bound = _prob_l_sorted[-TOP_N_STOCKS]

        _s_s_l = s_l[beg: end]
        _s_s_s = s_s[beg: end]
        long_cond = _prob_l >= top_bound
        short_cond = _prob_l <= bottom_bound
        _s_s_l |= long_cond
        _s_s_s |= short_cond
        # _hpr = hpr[beg: end]
        # l_hpr = _hpr[_s_s_l]
        # s_hpr = _hpr[_s_s_s]
        _int_r = int_r[beg:end]
        l_int_r = _int_r[_s_s_l]
        s_int_r = _int_r[_s_s_s]
        l_int_r_sorted = np.sort(l_int_r)
        s_int_r_sorted = np.sort(s_int_r)

        if SLCT_ALG == SelectionAlgo.TOP:
            l_int_r_t_b = np.max(l_int_r)
            l_int_r_b_b = np.percentile(l_int_r, 100 - SLCT_PCT)
        elif SLCT_ALG == SelectionAlgo.BOTTOM:
            l_int_r_t_b = np.percentile(l_int_r, SLCT_PCT)
            l_int_r_b_b = np.min(l_int_r)
        elif SLCT_ALG == SelectionAlgo.MIDDLE:
            l_int_r_t_b = np.percentile(l_int_r, 100 - SLCT_PCT / 2)
            l_int_r_b_b = np.percentile(l_int_r, SLCT_PCT / 2)

        if SLCT_ALG == SelectionAlgo.TOP:
            s_int_r_t_b = np.percentile(s_int_r, SLCT_PCT)
            s_int_r_b_b = np.min(s_int_r)
        elif SLCT_ALG == SelectionAlgo.BOTTOM:
            s_int_r_t_b = np.max(s_int_r)
            s_int_r_b_b = np.percentile(s_int_r, 100 - SLCT_PCT)
        elif SLCT_ALG == SelectionAlgo.MIDDLE:
            s_int_r_t_b = np.percentile(s_int_r, 100 - SLCT_PCT / 2)
            s_int_r_b_b = np.percentile(s_int_r, SLCT_PCT / 2)

        sel_l_cond = _s_s_l
        sel_l_cond &= _int_r >= l_int_r_b_b
        sel_l_cond &= _int_r <= l_int_r_t_b

        sel_s_cond = _s_s_s
        sel_s_cond &= _int_r <= s_int_r_t_b
        sel_s_cond &= _int_r >= s_int_r_b_b
        _hpr_model = hpr_model[beg: end]
        l_hpr = _hpr_model[sel_l_cond]
        s_hpr = _hpr_model[sel_s_cond]

        # _hpr_model = hpr_model[beg: end]
        # l_hpr = _hpr_model[_s_s_l]
        # s_hpr = _hpr_model[_s_s_s]

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
             w_enter_index[train_weeks:],
             w_exit_index[train_weeks:],
             raw_mpl_dt,
             raw_dt)

plt.show(True)
