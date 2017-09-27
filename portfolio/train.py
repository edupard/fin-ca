import numpy as np
import datetime
import pandas as pd
import math
import random
import csv
import os.path
import timeit

from download_utils import load_npz_data
from portfolio.config import get_config, NetVersion, Mode, TradingFrequency
from portfolio.net_apple import NetApple
from portfolio.net_banana import NetBanana
from portfolio.net_worm import NetWorm
from portfolio.net_snake import NetSnake
from portfolio.net_anti_snake import NetAntiSnake
from portfolio.net_cat import NetCat
from portfolio.net_cow import NetCow
import progress
from portfolio.stat import print_alloc, get_draw_down, get_sharpe_ratio, get_capital, get_avg_yeat_ret
from portfolio.graphs import plot_equity_curve, show_plots
from portfolio.snp_env import SnpEnv


def is_same_week(date1, date2):
    _min = min(date1, date2)
    _max = max(date1, date2)
    if (_max - _min).days < (7 - _min.isoweekday()):
        return True
    return False


def is_same_month(date1, date2):
    return date1.year == date2.year and date1.month == date2.month


def get_curr_week_mon(date):
    return date - datetime.timedelta(days=date.isoweekday() - 1)


def get_curr_week_sun(date):
    return date + datetime.timedelta(days=7 - date.isoweekday())


def calc_variance(x):
    return np.maximum(np.var(x[:, :, 1], axis=1), get_config().MIN_VARIANCE)


def train():
    snp_env = SnpEnv()
    if get_config().NET_VER == NetVersion.APPLE:
        net = NetApple()
    elif get_config().NET_VER == NetVersion.BANANA:
        net = NetBanana()
    elif get_config().NET_VER == NetVersion.WORM:
        net = NetWorm()
    elif get_config().NET_VER == NetVersion.SNAKE:
        net = NetSnake()
    elif get_config().NET_VER == NetVersion.ANTI_SNAKE:
        net = NetAntiSnake()
    elif get_config().NET_VER == NetVersion.CAT:
        net = NetCat()
    elif get_config().NET_VER == NetVersion.COW:
        net = NetCow()
    net.init()
    train_trading_schedule = []
    test_trading_schedule = []
    for ent, ext in snp_env.trading_schedule_generator(get_config().TRAIN_BEG, get_config().TRAIN_END,
                                                       get_config().TRADING_PERIOD_DAYS):
        train_trading_schedule.append((ent, ext))
    for ent, ext in snp_env.trading_schedule_generator(get_config().TEST_BEG, get_config().TEST_END,
                                                       get_config().TRADING_PERIOD_DAYS):
        test_trading_schedule.append((ent, ext))

    if not os.path.exists(get_config().TRAIN_STAT_PATH):
        with open(get_config().TRAIN_STAT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    'epoch',
                    'train dd',
                    'train y avg',
                    'train sharpe',
                    'test dd',
                    'test y avg',
                    'test sharpe'
                ))

    with open(get_config().TRAIN_STAT_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if get_config().EPOCH_WEIGHTS_TO_LOAD != 0:
            net.load_weights(get_config().WEIGHTS_PATH, get_config().EPOCH_WEIGHTS_TO_LOAD)
            epoch = get_config().EPOCH_WEIGHTS_TO_LOAD
            if get_config().MODE == Mode.TRAIN:
                epoch += 1
        else:
            epoch = 0
        while True:
            print("Epoch %d" % epoch)

            # train
            if get_config().MODE == Mode.TRAIN:
                print("Training...")
                if get_config().SHUFFLE:
                    train_schedule = random.sample(train_trading_schedule, len(train_trading_schedule))
                else:
                    train_schedule = train_trading_schedule
                dataset_size = len(train_schedule)
                curr_progress = 0
                passed = 0
                for ent, ext in train_schedule:
                    stk_mask = snp_env.get_tradeable_snp_components_mask(ent)
                    # print("%d %s %s" % (np.sum(stk_mask), ent.strftime("%Y-%m-%d"), ext.strftime("%Y-%m-%d")))
                    x = snp_env.get_input(stk_mask, ent)
                    labels = snp_env.get_ret_lbl(stk_mask, ent, ext)
                    if get_config().NET_VER == NetVersion.COW:
                        variances = calc_variance(x)
                    if get_config().NET_VER == NetVersion.COW:
                        net.fit(x, labels, variances)
                    else:
                        net.fit(x, labels)
                    if passed == 0:
                        if get_config().NET_VER == NetVersion.COW:
                            pl, weights = net.eval(x, labels, variances)
                        else:
                            pl, weights = net.eval(x, labels)
                        print(pl)
                    curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                    passed += 1
                progress.print_progess_end()

            # eval train
            print("Eval train...")
            ret = np.zeros((len(train_trading_schedule)))

            dataset_size = len(train_trading_schedule)
            curr_progress = 0
            passed = 0
            dt = []
            for ent, ext in train_trading_schedule:
                if ext is None:
                    break
                if len(dt) == 0:
                    dt.append(ent)
                dt.append(ext)
                stk_mask = snp_env.get_tradeable_snp_components_mask(ent)
                x = snp_env.get_input(stk_mask, ent)
                labels = snp_env.get_ret_lbl(stk_mask, ent, ext)

                if get_config().NET_VER == NetVersion.COW:
                    variances = calc_variance(x)
                    pl, weights = net.eval(x, labels, variances)
                else:
                    pl, weights = net.eval(x, labels)

                if get_config().NET_VER == NetVersion.APPLE:
                    # # net
                    # long_mask = weights >= 0.0
                    # short_mask = weights <= 0.0
                    #
                    # int_date = snp_env.find_trading_date(ent + datetime.timedelta(days=1))
                    # int_r = snp_env.get_ret_lbl(stk_mask, ent, int_date)
                    # port_ret = labels - int_r
                    # # sorted_int_r_idxs = np.argsort(int_r)
                    # long_idxs = np.nonzero(long_mask)[0]
                    # long_int_r = int_r[long_idxs]
                    # sorted_long_int_r_idxs = np.argsort(long_int_r)
                    # long_sel_idxs = long_idxs[sorted_long_int_r_idxs[:get_config().SELECTTION]]
                    #
                    # short_idxs = np.nonzero(short_mask)[0]
                    # short_int_r = int_r[short_idxs]
                    # sorted_short_int_r_idxs = np.argsort(short_int_r)
                    # short_sel_idxs = short_idxs[sorted_short_int_r_idxs[-get_config().SELECTTION:]]
                    #
                    # long_pl = np.mean(port_ret[long_sel_idxs])
                    # short_pl = np.mean(port_ret[short_sel_idxs])

                    # # no net
                    # int_date = snp_env.find_trading_date(ent + datetime.timedelta(days=1))
                    # int_r = snp_env.get_ret_lbl(stk_mask, ent, int_date)
                    # port_ret = labels - int_r
                    # sorted_int_r_idxs = np.argsort(int_r)
                    # long_pl = np.mean(port_ret[sorted_int_r_idxs[:get_config().SELECTTION]])
                    # short_pl = np.mean(port_ret[sorted_int_r_idxs[-get_config().SELECTTION:]])

                    # 1d no net
                    prev_date = snp_env.find_prev_trading_date(ent - datetime.timedelta(days=1))
                    int_stk_mask = snp_env.get_tradeable_snp_components_mask(prev_date)
                    stk_mask &= int_stk_mask

                    int_r = snp_env.get_ret_lbl(stk_mask, prev_date, ent)
                    port_ret = snp_env.get_ret_lbl(stk_mask, ent, ext)
                    sorted_int_r_idxs = np.argsort(int_r)
                    long_pl = np.mean(port_ret[sorted_int_r_idxs[:get_config().SELECTTION]])
                    short_pl = np.mean(port_ret[sorted_int_r_idxs[-get_config().SELECTTION:]])

                    # real_ext = snp_env.find_trading_date(ext + datetime.timedelta(days=1))
                    # if real_ext is None:
                    #     real_ext = ext
                    # int_r = labels
                    # port_ret = snp_env.get_ret_lbl(stk_mask, ext, real_ext)
                    # sorted_int_r_idxs = np.argsort(int_r)
                    # long_pl = np.mean(port_ret[sorted_int_r_idxs[:get_config().SELECTTION]])
                    # short_pl = np.mean(port_ret[sorted_int_r_idxs[-get_config().SELECTTION:]])

                    pl = 0.7 * long_pl - 0.3 * short_pl


                # if get_config().NET_VER == NetVersion.SNAKE or get_config().NET_VER == NetVersion.ANTI_SNAKE:
                #     sorted_weights_idxs = np.argsort(weights)
                #     selected_weights_idxs = sorted_weights_idxs[-get_config().SELECTTION:]
                #     pl = np.mean(labels[selected_weights_idxs])
                #     if get_config().NET_VER == NetVersion.ANTI_SNAKE:
                #         pl = -pl

                if get_config().PRINT_PREDICTION:
                    print_alloc(pl, ent, snp_env.tickers, stk_mask, weights)
                if abs(pl) >= 0.3:
                    pl = 0
                ret[passed] = pl
                curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                passed += 1
            progress.print_progess_end()

            # print("Train loss: %.4f" % (np.mean(np.sqrt(ret)) * 100))
            years = (get_config().TRAIN_END - get_config().TRAIN_BEG).days / 365
            capital = get_capital(ret, False)
            train_dd = get_draw_down(capital, False)
            train_sharpe = get_sharpe_ratio(ret, years)
            train_y_avg = get_avg_yeat_ret(ret, years)
            print('Train dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (train_dd * 100, train_y_avg * 100, train_sharpe))
            if get_config().MODE == Mode.TEST:
                plot_equity_curve("Train equity curve", dt, capital)
                df = pd.DataFrame({'date': dt, 'capital': capital})
                df.to_csv('data/tr_eq.csv', index=False)

            # eval test
            print("Eval test...")
            ret = np.zeros((len(test_trading_schedule)))

            dataset_size = len(test_trading_schedule)
            curr_progress = 0
            passed = 0
            dt = []
            for ent, ext in test_trading_schedule:
                if ext.month == 3 and ext.day == 11 and ext.year == 2015:
                    _debug = 0
                if ext is None:
                    break
                if len(dt) == 0:
                    dt.append(ent)
                dt.append(ext)
                stk_mask = snp_env.get_tradeable_snp_components_mask(ent)
                x = snp_env.get_input(stk_mask, ent)
                labels = snp_env.get_ret_lbl(stk_mask, ent, ext)
                if get_config().NET_VER == NetVersion.COW:
                    variances = calc_variance(x)
                    pl, weights = net.eval(x, labels, variances)
                else:
                    pl, weights = net.eval(x, labels)

                if get_config().NET_VER == NetVersion.APPLE:
                    # # net
                    # long_mask = weights >= 0.0
                    # short_mask = weights <= 0.0
                    #
                    # int_date = snp_env.find_trading_date(ent + datetime.timedelta(days=1))
                    # int_r = snp_env.get_ret_lbl(stk_mask, ent, int_date)
                    # port_ret = labels - int_r
                    # # sorted_int_r_idxs = np.argsort(int_r)
                    # long_idxs = np.nonzero(long_mask)[0]
                    # long_int_r = int_r[long_idxs]
                    # sorted_long_int_r_idxs = np.argsort(long_int_r)
                    # long_sel_idxs = long_idxs[sorted_long_int_r_idxs[:get_config().SELECTTION]]
                    #
                    # short_idxs = np.nonzero(short_mask)[0]
                    # short_int_r = int_r[short_idxs]
                    # sorted_short_int_r_idxs = np.argsort(short_int_r)
                    # short_sel_idxs =  short_idxs[sorted_short_int_r_idxs[-get_config().SELECTTION:]]
                    #
                    # long_pl = np.mean(port_ret[long_sel_idxs])
                    # short_pl = np.mean(port_ret[short_sel_idxs])

                    # # no net
                    # int_date = snp_env.find_trading_date(ent + datetime.timedelta(days=1))
                    # int_r = snp_env.get_ret_lbl(stk_mask, ent, int_date)
                    # port_ret = labels - int_r
                    # sorted_int_r_idxs = np.argsort(int_r)
                    # long_pl = np.mean(port_ret[sorted_int_r_idxs[:get_config().SELECTTION]])
                    # short_pl = np.mean(port_ret[sorted_int_r_idxs[-get_config().SELECTTION:]])

                    # 1d no net
                    prev_date = snp_env.find_prev_trading_date(ent - datetime.timedelta(days=1))
                    int_stk_mask = snp_env.get_tradeable_snp_components_mask(prev_date)
                    stk_mask &= int_stk_mask

                    int_r = snp_env.get_ret_lbl(stk_mask, prev_date, ent)
                    port_ret = snp_env.get_ret_lbl(stk_mask, ent, ext)
                    sorted_int_r_idxs = np.argsort(int_r)
                    long_pl = np.mean(port_ret[sorted_int_r_idxs[:get_config().SELECTTION]])
                    short_pl = np.mean(port_ret[sorted_int_r_idxs[-get_config().SELECTTION:]])

                    pl = 0.7 * long_pl - 0.3 * short_pl

                # if get_config().NET_VER == NetVersion.SNAKE or get_config().NET_VER == NetVersion.ANTI_SNAKE:
                #     sorted_weights_idxs = np.argsort(weights)
                #     selected_weights_idxs = sorted_weights_idxs[-get_config().SELECTTION:]
                #     pl = np.mean(labels[selected_weights_idxs])
                #     if get_config().NET_VER == NetVersion.ANTI_SNAKE:
                #         pl = -pl

                if get_config().PRINT_PREDICTION:
                    print_alloc(pl, ent, snp_env.tickers, stk_mask, weights)
                if abs(pl) >= 0.3:
                    pl = 0
                ret[passed] = pl
                curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                passed += 1
            progress.print_progess_end()

            # print("Test loss: %.4f" % (np.mean(np.sqrt(ret)) * 100))

            years = (get_config().TRAIN_END - get_config().TRAIN_BEG).days / 365
            capital = get_capital(ret, False)
            test_dd = get_draw_down(capital, False)
            test_sharpe = get_sharpe_ratio(ret, years)
            test_y_avg = get_avg_yeat_ret(ret, years)
            print('Test dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (test_dd * 100, test_y_avg * 100, test_sharpe))

            if get_config().MODE == Mode.TEST:
                plot_equity_curve("Test equity curve", dt, capital)
                df = pd.DataFrame({'date': dt, 'capital': capital})
                df.to_csv('data/tst_eq.csv', index=False)


            if get_config().MODE == Mode.TRAIN:
                net.save_weights(get_config().WEIGHTS_PATH, epoch)
                writer.writerow(
                    (
                        epoch,
                        train_dd,
                        train_y_avg,
                        train_sharpe,
                        test_dd,
                        test_y_avg,
                        test_sharpe
                    ))
                epoch += 1
                f.flush()
            else:
                show_plots()
                break
