import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np
import math
import datetime
import csv

from config import get_config

DDMMMYY_FMT = matplotlib.dates.DateFormatter('%y %b %d')
YYYY_FMT = matplotlib.dates.DateFormatter('%Y')

def plot_stock_returns(r, mpl_dt):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle='-', color='0.75')
    ax.xaxis.set_major_formatter(DDMMMYY_FMT)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    pos_mask = r >= 0
    pos_r = r[pos_mask]
    pos_mpl_dt = mpl_dt[pos_mask]
    ax.plot_date(pos_mpl_dt, pos_r, color='g', fmt='o')
    neg_mask = ~pos_mask
    neg_r = r[neg_mask]
    neg_mpl_dt = mpl_dt[neg_mask]
    ax.plot_date(neg_mpl_dt, neg_r, color='r', fmt='o')

def plot_20_random_stock_prices(raw_data, raw_mpl_dt):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle='-', color='0.75')
    ax.xaxis.set_major_formatter(DDMMMYY_FMT)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)

    for i in range(20):
        idx = random.randrange(0, raw_data.shape[0])
        px = raw_data[idx, :, 3]
        ax.plot_date(raw_mpl_dt, px, fmt='-')


def plot_gross_avg_amount(g_a_a, raw_mpl_dt):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle='-', color='0.75')
    ax.xaxis.set_major_formatter(DDMMMYY_FMT)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.plot_date(raw_mpl_dt, g_a_a, color='b', fmt='o')


def plot_traded_stocks_per_day(traded_stocks, raw_mpl_dt):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle='-', color='0.75')
    ax.xaxis.set_major_formatter(DDMMMYY_FMT)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.plot_date(raw_mpl_dt, traded_stocks, color='b', fmt='o')


# def hpr_analysis(t_hpr, b_hpr):
#     d_hpr = t_hpr - b_hpr
#     t_hpr_mean = np.mean(t_hpr)
#     b_hpr_mean = np.mean(b_hpr)
#     d_hpr_mean = np.mean(d_hpr)
#
#     t_t, p_t = stats.ttest_1samp(t_hpr, t_hpr_mean)
#     t_b, p_b = stats.ttest_1samp(b_hpr, b_hpr_mean)
#     t_d, p_d = stats.ttest_1samp(d_hpr, d_hpr_mean)
#     print(
#         "T: {:.4f} t-stat: {:.2f} p: {:.2f} B: {:.4f} t-stat: {:.2f} p: {:.2f} D: {:.4f} t-stat: {:.2f} p: {:.2f}".format(
#             t_hpr_mean,
#             t_t,
#             p_t,
#             b_hpr_mean,
#             t_b,
#             p_b,
#             d_hpr_mean,
#             t_d,
#             p_d
#         ))


def confusion_matrix(a_l, a_s, p_l, p_s):
    p_l_a_l = (p_l & a_l).sum(0)
    p_l_a_s = (p_l & a_s).sum(0)
    p_s_a_s = (p_s & a_s).sum(0)
    p_s_a_l = (p_s & a_l).sum(0)
    total = p_l_a_l + p_l_a_s + p_s_a_s + p_s_a_l
    print('L +: {:.2f} -: {:.2f} accuracy: {:.2f}'
        .format(
        100. * p_l_a_l / total,
        100. * p_l_a_s / total,
        100. * p_l_a_l / (p_l_a_l + p_l_a_s)
    ))
    print('S +: {:.2f} -: {:.2f} accuracy: {:.2f}'
        .format(
        100. * p_s_a_s / total,
        100. * p_s_a_l / total,
        100. * p_s_a_s / (p_s_a_s + p_s_a_l)
    ))
    print('Total accuracy: {:.2f}'
        .format(
        100. * (p_l_a_l + p_s_a_s) / total
    ))


def wealth_graph(yr_avg,
                 w_dd,
                 w_avg,
                 w_best,
                 wealth,
                 dd,
                 sharpe,
                 rc_wealth,
                 rc_dd,
                 rc_sharpe,
                 yr,
                 years,
                 w_exit_index,
                 raw_mpl_dt):
    def format_time_labels(ax, fmt):
        ax.xaxis.set_major_formatter(fmt)
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(45)

    def draw_grid(ax):
        ax.grid(True, linestyle='-', color='0.75')

    def hide_time_labels(ax):
        plt.setp(ax.get_xticklabels(), visible=False)

    yr = np.array(yr)
    yts = []
    for y in years:
        dt = datetime.date(year=y, day=1, month=1)
        yts.append(matplotlib.dates.date2num(dt))
    yts = np.array(yts)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=DDMMMYY_FMT)
    ax.set_title(
        "1 usd pl sharpe: %.2f dd: %.2f%% avg y: %.2f%% avg w: %.2f%%" % (sharpe, dd * 100, yr_avg * 100, w_avg * 100))
    ax.plot_date(raw_mpl_dt[w_exit_index], wealth, color='b', fmt='-')

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, sharex=ax)
    # draw_grid(ax)
    # # hide_time_labels(ax)
    # ax.set_title("1 USD PL RECAP Draw down: %.2f" % (rc_dd * 100))
    # ax.plot_date(raw_mpl_dt[w_exit_index], rc_wealth, color='b', fmt='-')
    # format_time_labels(ax, fmt=DDMMMYY_FMT)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=YYYY_FMT)
    ax.bar(yts, yr * 100, color='b', width=100)
    ax.xaxis_date()
    ax.set_title("Year pct return")


def calc_wealth(model_hpr, w_enter_index, raw_dt):
    def calc_dd(c, recap):
        # c == capital in time array
        # recap == flag indicating recapitalization or fixed bet
        def generate_previous_max():
            max = c[0]
            for idx in range(len(c)):
                # update max
                if c[idx] > max:
                    max = c[idx]
                yield max

        prev_max = np.fromiter(generate_previous_max(), dtype=np.float64)
        if recap:
            dd_a = (c - prev_max) / prev_max
        else:
            dd_a = c - prev_max

        return np.min(dd_a)

    def calc_sharp(r):
        return math.sqrt(r.shape[0]) * np.mean(r) / np.std(r)

    # calc dd, sharp
    progress = model_hpr
    wealth = np.cumsum(progress) + 1.0
    dd = calc_dd(wealth, False)
    sharpe = calc_sharp(model_hpr)

    # calc recap dd
    rc_progress = (model_hpr) + 1.00
    rc_wealth = np.cumprod(rc_progress)
    rc_dd = calc_dd(rc_wealth, True)

    # calc recap sharp: results are same as without recap
    rc_base = np.cumprod(rc_progress)
    rc_r = (rc_base[1:] - rc_base[:-1]) / rc_base[:-1]
    rc_r = np.concatenate([np.array([rc_base[0] - 1.]), rc_r])
    rc_sharpe = calc_sharp(rc_r)

    # calc return by years
    yr = []
    years = []
    c_y = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[0]]).year
    c_y_w = 0
    c_y_g_r = 0.0

    MIN_WEEKS_TO_APPEND_YEAR = 365 // 7 * 0.8
    for w in range(model_hpr.shape[0]):
        y = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[w]]).year
        if y != c_y:
            # append year data
            if c_y_w > MIN_WEEKS_TO_APPEND_YEAR:
                yr.append(c_y_g_r)
                years.append(c_y)
            c_y_g_r = 0.0
            c_y_w = 1
            c_y = y
        else:
            c_y_w += 1
        c_y_g_r += model_hpr[w]

    return wealth, dd, sharpe, rc_wealth, rc_dd, rc_sharpe, yr, years


def wealth_csv(sl_name,
               wk_beg_idx,
               wk_end_idx,
               w_enter_index,
               w_exit_index,
               raw_dt,
               model,
               model_min_to,
               model_avg_to,
               model_longs,
               model_shorts,
               model_selection
               ):
    model_hpr, model_min_w_eod_hpr, model_min_w_lb_hpr, model_l_stops, model_s_stops, model_l_port, model_s_port = model
    model_hpr = model_hpr[wk_beg_idx:wk_end_idx]
    model_min_w_eod_hpr = model_min_w_eod_hpr[wk_beg_idx:wk_end_idx]
    model_min_w_lb_hpr = model_min_w_lb_hpr[wk_beg_idx:wk_end_idx]
    model_l_stops = model_l_stops[wk_beg_idx:wk_end_idx]
    model_s_stops = model_s_stops[wk_beg_idx:wk_end_idx]
    model_l_port = model_l_port[wk_beg_idx:wk_end_idx]
    model_s_port = model_s_port[wk_beg_idx:wk_end_idx]
    model_min_to = model_min_to[wk_beg_idx:wk_end_idx]
    model_avg_to = model_avg_to[wk_beg_idx:wk_end_idx]
    model_longs = model_longs[wk_beg_idx:wk_end_idx]
    model_shorts = model_shorts[wk_beg_idx:wk_end_idx]
    model_selection = model_selection[wk_beg_idx:wk_end_idx]

    w_enter_index = w_enter_index[wk_beg_idx:wk_end_idx]
    w_exit_index = w_exit_index[wk_beg_idx:wk_end_idx]

    progress = model_hpr
    wealth = np.cumsum(progress) + 1.0

    with open('./data/weekly_{}_sl.csv'.format(sl_name), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                'beg',
                'end',
                'wealth',
                'hpr',
                'min w',
                'min w lb',
                'l stops',
                's stops',
                'min w to',
                'avg w to',
                'stks',
                'l',
                's',)
        )
        for w in range(wealth.shape[0]):
            dt_enter = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[w]])
            dt_exit = datetime.datetime.fromtimestamp(raw_dt[w_exit_index[w]])
            writer.writerow(
                (dt_enter.strftime('%Y-%m-%d'),
                 dt_exit.strftime('%Y-%m-%d'),
                 wealth[w],
                 "%.2f%%" % (model_hpr[w] * 100.0),
                 "%.2f%%" % (model_min_w_eod_hpr[w] * 100.0),
                 "%.2f%%" % (model_min_w_lb_hpr[w] * 100.0),
                 model_l_stops[w],
                 model_s_stops[w],
                 model_min_to[w],
                 model_avg_to[w],
                 model_selection[w],
                 model_longs[w],
                 model_shorts[w]
                 )
            )
    if not get_config().PRINT_PORTFOLIO:
        return
    with open('./data/weekly_{}_sl_portfolio.csv'.format(sl_name), 'w', newline='') as f_p:
        p_writer = csv.writer(f_p)

        p_writer.writerow(
            (
                'beg',
                'end',
                'pos',
                'ticker',
                'l p',
                'ret',
                'ext',
                'wato')
        )

        for w in range(wealth.shape[0]):
            dt_enter = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[w]])
            dt_exit = datetime.datetime.fromtimestamp(raw_dt[w_exit_index[w]])
            _l_port_df = model_l_port[w]
            _s_port_df = model_s_port[w]
            # for index, row in _l_port_df.iterrows():
            #     p_writer.writerow(
            #         (
            #             dt_enter.strftime('%Y-%m-%d'),
            #             dt_exit.strftime('%Y-%m-%d'),
            #             'long',
            #             row.ticker,
            #             row.p,
            #             row.ret,
            #             row.ext,
            #             row.wato)
            #     )
            # for index, row in _s_port_df.iterrows():
            #     p_writer.writerow(
            #         (
            #             dt_enter.strftime('%Y-%m-%d'),
            #             dt_exit.strftime('%Y-%m-%d'),
            #             'short',
            #             row.ticker,
            #             row.p,
            #             row.ret,
            #             row.ext,
            #             row.wato)
            #     )

            for idx in range(_l_port_df.shape[0]):
                p_writer.writerow(
                    (
                        dt_enter.strftime('%Y-%m-%d'),
                        dt_exit.strftime('%Y-%m-%d'),
                        'long',
                        _l_port_df[idx, 0],
                        _l_port_df[idx, 4],
                        _l_port_df[idx, 1],
                        _l_port_df[idx, 2],
                        _l_port_df[idx, 3])
                )
            for idx in range(_s_port_df.shape[0]):
                p_writer.writerow(
                    (
                        dt_enter.strftime('%Y-%m-%d'),
                        dt_exit.strftime('%Y-%m-%d'),
                        'short',
                        _s_port_df[idx, 0],
                        _s_port_df[idx, 4],
                        _s_port_df[idx, 1],
                        _s_port_df[idx, 2],
                        _s_port_df[idx, 3])
                )
