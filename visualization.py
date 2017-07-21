import matplotlib.pyplot as plt
import matplotlib
import random
import scipy.stats as stats
import numpy as np
import math
import datetime
import csv

DDMMMYY_FMT = matplotlib.dates.DateFormatter('%y %b %d')
YYYY_FMT = matplotlib.dates.DateFormatter('%Y')


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


# for i in range(20):
#     idx = random.randrange(0, raw_data.shape[0])
#     traded_stocks = mask[idx,:].astype(int) * (i + 1)
#     ax.plot_date(raw_mpl_dt, traded_stocks, fmt='o')

# for i in range(20):
#     idx = random.randrange(0, raw_data.shape[0])
#     g = g_a[idx, :]
#     ax.plot_date(raw_mpl_dt, g, fmt='o')


def hpr_analysis(t_hpr, b_hpr):
    d_hpr = t_hpr - b_hpr
    t_hpr_mean = np.mean(t_hpr)
    b_hpr_mean = np.mean(b_hpr)
    d_hpr_mean = np.mean(d_hpr)

    t_t, p_t = stats.ttest_1samp(t_hpr, t_hpr_mean)
    t_b, p_b = stats.ttest_1samp(b_hpr, b_hpr_mean)
    t_d, p_d = stats.ttest_1samp(d_hpr, d_hpr_mean)
    print(
        "T: {:.4f} t-stat: {:.2f} p: {:.2f} B: {:.4f} t-stat: {:.2f} p: {:.2f} D: {:.4f} t-stat: {:.2f} p: {:.2f}".format(
            t_hpr_mean,
            t_t,
            p_t,
            b_hpr_mean,
            t_b,
            p_b,
            d_hpr_mean,
            t_d,
            p_d
        ))


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


def wealth_graph(model_hpr, w_enter_index, w_exit_index, raw_mpl_dt, raw_dt):
    def format_time_labels(ax, fmt):
        ax.xaxis.set_major_formatter(fmt)
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(45)

    def draw_grid(ax):
        ax.grid(True, linestyle='-', color='0.75')

    def hide_time_labels(ax):
        plt.setp(ax.get_xticklabels(), visible=False)

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

        return np.min(dd_a) * 100.0

    def calc_sharp(r):
        return math.sqrt(r.shape[0]) * np.mean(r) / np.std(r)

    fig = plt.figure()

    # diff = (t_hpr - b_hpr) / 2
    # diff = np.maximum(diff, STOP_LOSS_HPR)

    progress = model_hpr
    # wealth = np.cumsum(progress)
    wealth = np.cumsum(progress) + 1.0
    dd = calc_dd(wealth, False)
    sharp = calc_sharp(model_hpr)

    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=DDMMMYY_FMT)
    ax.set_title("1 USD PL Sharpe: %.2f Draw down: %.2f" % (sharp, dd))
    ax.plot_date(raw_mpl_dt[w_exit_index], wealth, color='b', fmt='-')

    rc_progress = (model_hpr) + 1.00
    # rc_wealth = np.cumprod(rc_progress) - 1.
    rc_wealth = np.cumprod(rc_progress)
    rc_dd = calc_dd(rc_wealth, True)

    rc_base = np.cumprod(rc_progress)
    rc_r = (rc_base[1:] - rc_base[:-1]) / rc_base[:-1]
    rc_r = np.concatenate([np.array([rc_base[0] - 1.]), rc_r])
    rc_sharp = calc_sharp(rc_r)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, sharex=ax)
    draw_grid(ax)
    # hide_time_labels(ax)
    ax.set_title("1 USD PL RECAP Draw down: %.2f" % (rc_dd))
    ax.plot_date(raw_mpl_dt[w_exit_index], rc_wealth, color='b', fmt='-')
    format_time_labels(ax, fmt=DDMMMYY_FMT)
    # ax = fig.add_subplot(3, 1, 3, sharex=ax)
    # draw_grid(ax)
    # format_time_labels(ax)
    # ax.plot_date(raw_mpl_dt[w_exit_index], t_stocks, color='g', fmt='o')
    # ax.plot_date(raw_mpl_dt[w_exit_index], b_stocks, color='r', fmt='o')

    fig = plt.figure()

    yr = []
    yts = []
    c_y = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[0]]).year
    c_y_w = 0
    c_y_g_r = 0.0

    MIN_WEEKS_TO_APPEND_YEAR = 365 // 7 * 0.8
    for w in range(model_hpr.shape[0]):
        y = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[w]]).year
        if y != c_y:
            # append year data
            if c_y_w > MIN_WEEKS_TO_APPEND_YEAR:
                yr.append(c_y_g_r * 100.0)
                dt = datetime.date(year=c_y, day=1, month=1)
                yts.append(matplotlib.dates.date2num(dt))
            c_y_g_r = 0.0
            c_y_w = 1
            c_y = y
        else:
            c_y_w += 1
        c_y_g_r += model_hpr[w]

    yr_mean = np.mean(yr)

    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=YYYY_FMT)
    ax.bar(yts, yr, color='b', width=100)
    ax.xaxis_date()
    ax.set_title("Year pct return")

    w_dd = np.min(model_hpr) * 100.0
    w_r_avg = np.mean(model_hpr) * 100.0
    w_r_best = np.max(model_hpr) * 100.0
    print(
        "F: {:.2f} DD: {:.2f} W_DD: {:.2f} W_AVG: {:.2f} W_BEST: {:.2f} SHARPE: {:.2f} AVG_YEAR: {:.2f} F_R: {:.2f} DD_R: {:.2f}".format(
            wealth[-1],
            dd,
            w_dd,
            w_r_avg,
            w_r_best,
            sharp,
            yr_mean,
            rc_wealth[-1],
            rc_dd
        ))


def wealth_params(model_hpr, w_enter_index, w_exit_index, raw_mpl_dt, raw_dt):
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

        return np.min(dd_a) * 100.0

    def calc_sharp(r):
        return math.sqrt(r.shape[0]) * np.mean(r) / np.std(r)

    # calc dd, sharp
    progress = model_hpr
    wealth = np.cumsum(progress) + 1.0
    dd = calc_dd(wealth, False)
    sharp = calc_sharp(model_hpr)

    # calc recap dd
    rc_progress = (model_hpr) + 1.00
    rc_wealth = np.cumprod(rc_progress)
    rc_dd = calc_dd(rc_wealth, True)

    # calc recap sharp: results are same as without recap
    rc_base = np.cumprod(rc_progress)
    rc_r = (rc_base[1:] - rc_base[:-1]) / rc_base[:-1]
    rc_r = np.concatenate([np.array([rc_base[0] - 1.]), rc_r])
    rc_sharp = calc_sharp(rc_r)

    # calc return by years
    yr = []
    yts = []
    c_y = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[0]]).year
    c_y_w = 0
    c_y_g_r = 0.0

    MIN_WEEKS_TO_APPEND_YEAR = 365 // 7 * 0.8
    for w in range(model_hpr.shape[0]):
        y = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[w]]).year
        if y != c_y:
            # append year data
            if c_y_w > MIN_WEEKS_TO_APPEND_YEAR:
                yr.append(c_y_g_r * 100.0)
                dt = datetime.date(year=c_y, day=1, month=1)
                yts.append(matplotlib.dates.date2num(dt))
            c_y_g_r = 0.0
            c_y_w = 1
            c_y = y
        else:
            c_y_w += 1
        c_y_g_r += model_hpr[w]

    yr_mean = np.mean(yr)

    w_dd = np.min(model_hpr) * 100.0
    w_r_avg = np.mean(model_hpr) * 100.0
    w_r_best = np.max(model_hpr) * 100.0
    print(
        "F: {:.2f} DD: {:.2f} W_DD: {:.2f} W_AVG: {:.2f} W_BEST: {:.2f} SHARPE: {:.2f} AVG_YEAR: {:.2f} F_R: {:.2f} DD_R: {:.2f}".format(
            wealth[-1],
            dd,
            w_dd,
            w_r_avg,
            w_r_best,
            sharp,
            yr_mean,
            rc_wealth[-1],
            rc_dd
        ))


def wealth_csv(model_no_sl_hpr,
               model_eod_sl_hpr,
               model_lb_sl_hpr,
               model_s_sl_hpr,
               t_hpr,
               b_hpr,
               min_w_hpr,
               min_w_lb_hpr,
               w_enter_index,
               w_exit_index,
               raw_dt,
               l_port,
               s_port):
    progress = model_no_sl_hpr
    wealth = np.cumsum(progress) + 1.0

    with open('./data/weekly.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                'beg',
                'end',
                'wealth',
                'long stocks ret',
                'short stocks ret',
                'no sl',
                'eod sl',
                'lb sl',
                'stock sl',
                'min w hpr',
                'min w lb hpr',
                'longs', 'shorts'))
        for w in range(wealth.shape[0]):
            dt_enter = datetime.datetime.fromtimestamp(raw_dt[w_enter_index[w]])
            dt_exit = datetime.datetime.fromtimestamp(raw_dt[w_exit_index[w]])
            writer.writerow(
                (dt_enter.strftime('%Y-%m-%d'),
                 dt_exit.strftime('%Y-%m-%d'),
                 wealth[w],
                 t_hpr[w],
                 b_hpr[w],
                 model_no_sl_hpr[w],
                 model_eod_sl_hpr[w],
                 model_lb_sl_hpr[w],
                 model_s_sl_hpr[w],
                 min_w_hpr[w],
                 min_w_lb_hpr[w],
                 l_port[w],
                 s_port[w]))
