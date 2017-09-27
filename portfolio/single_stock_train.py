import numpy as np
import csv
import os.path

from portfolio.net_turtle import NetTurtle
from portfolio.single_stock_config import get_config, Mode
from portfolio.stat import print_alloc, get_draw_down, get_sharpe_ratio, get_capital, get_avg_yeat_ret
from portfolio.graphs import plot_equity_curve, show_plots, create_time_serie_fig, plot_time_serie
import progress
import matplotlib.pyplot as plt
import pandas as pd

from portfolio.single_stock_env import Env, date_from_timestamp

if get_config().MODE == Mode.TRAIN:
    plt.ioff()

def create_folders():
    if not os.path.exists(get_config().WEIGHTS_FOLDER_PATH):
        os.makedirs(get_config().WEIGHTS_FOLDER_PATH)
    if not os.path.exists(get_config().TRAIN_FIG_PATH):
        os.makedirs(get_config().TRAIN_FIG_PATH)
    if not os.path.exists(get_config().TEST_FIG_PATH):
        os.makedirs(get_config().TEST_FIG_PATH)

def get_batches_num(ds_sz, bptt_steps):
    return ds_sz // bptt_steps + (0 if ds_sz % bptt_steps == 0 else 1)


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def build_time_axis(raw_dates):
    dt = []
    for raw_dt in np.nditer(raw_dates):
        dt.append(date_from_timestamp(raw_dt))
    return dt


def plot_prediction(name, dt, px, pred_px_series, pred_dt_series):
    ax = create_time_serie_fig(name)
    plot_time_serie(ax, dt, px[0, :], color='b')
    for i in range(len(pred_px_series)):
        pred_px = pred_px_series[i]
        pred_px = pred_px[0, :]
        pred_dt = pred_dt_series[i]
        plot_time_serie(ax, pred_dt, pred_px, color='r')


def plot_eq(name, BEG, END, dt, capital):
    years = (END - BEG).days / 365
    dd = get_draw_down(capital[0], False)
    rets = capital[0, 1:] - capital[0, :-1]
    sharpe = get_sharpe_ratio(rets, years)
    y_avg = (capital[0, -1] - capital[0, 0]) / years
    print('%s dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (name, dd * 100, y_avg * 100, sharpe))
    return plot_equity_curve("%s equity curve" % name, dt, capital[0, :])


def train(net):
    create_folders()

    env = Env()

    if not os.path.exists(get_config().TRAIN_STAT_PATH):
        with open(get_config().TRAIN_STAT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    'epoch',
                    'train loss',
                    'test loss',
                ))

    tickers_to_plot = [get_config().TICKER]
    total = len(tickers_to_plot)
    ticker_to_plot_idxs = np.zeros((total), dtype=np.int32)

    for i in range(total):
        ticker_to_plot_idxs[i] = env._ticker_to_idx(tickers_to_plot[i])

    def open_train_stat_file():
        return open(get_config().TRAIN_STAT_PATH, 'a', newline='')

    def is_train():
        return get_config().MODE == Mode.TRAIN

    with open_train_stat_file() if is_train() else dummy_context_mgr() as f:
        if is_train():
            writer = csv.writer(f)
        if get_config().EPOCH_WEIGHTS_TO_LOAD != 0:
            net.load_weights(get_config().WEIGHTS_PATH, get_config().EPOCH_WEIGHTS_TO_LOAD)
            epoch = get_config().EPOCH_WEIGHTS_TO_LOAD
            if is_train():
                epoch += 1
        else:
            net.init()
            epoch = 0

        def get_net_data(BEG, END):
            beg_idx = env.get_next_trading_day_data_idx(BEG)
            end_idx = env.get_prev_trading_day_data_idx(END)

            raw_dates = env.get_raw_dates(beg_idx, end_idx)
            input = env.get_input(beg_idx, end_idx)
            px = env.get_adj_close_px(beg_idx, end_idx)
            px_pred_hor = env.get_adj_close_px(beg_idx + get_config().PRED_HORIZON, end_idx + get_config().PRED_HORIZON)
            px_t1 = env.get_adj_close_px(beg_idx + 1, end_idx + 1)
            ds_sz = px_pred_hor.shape[1]
            raw_dates = raw_dates[:ds_sz]
            input = input[:, :ds_sz, :]
            px = px[:, :ds_sz]
            px_t1 = px_t1[:, :ds_sz]
            labels = (px_pred_hor - px) / px
            rets = (px_t1 - px) / px
            batch_num = get_batches_num(ds_sz, get_config().BPTT_STEPS)

            return ds_sz, batch_num, raw_dates, px, input, labels, rets

        tr_ds_sz, tr_batch_num, tr_raw_dates, tr_px, tr_input, tr_labels, tr_rets = get_net_data(get_config().TRAIN_BEG,
                                                                                                 get_config().TRAIN_END)

        tst_ds_sz, tst_batch_num, tst_raw_dates, tst_px, tst_input, tst_labels, tst_rets = get_net_data(
            get_config().TEST_BEG,
            get_config().TEST_END)

        if get_config().DRAW_PREDICTIONS:
            tr_pred_px_series = []
            tr_pred_dt_series = []
            tst_pred_px_series = []
            tst_pred_dt_series = []

        tr_eq = np.zeros((total, tr_ds_sz))
        tst_eq = np.zeros((total, tst_ds_sz))

        def get_batch_input_and_lables(input, labels, b):
            b_i = b * get_config().BPTT_STEPS
            e_i = (b + 1) * get_config().BPTT_STEPS
            return input[:, b_i: e_i, :], labels[:, b_i: e_i]

        def predict_price_series(px, raw_dates, predictions, b, pred_px, pred_dt, pred_px_series,
                                 pred_dt_series, curr_pred_px):
            for i in range(predictions.shape[1]):

                data_idx = b * get_config().BPTT_STEPS + i
                serie_idx = data_idx % get_config().RESET_PRED_PX_EACH_N_DAYS

                if serie_idx == 0:
                    # finish old serie is exists
                    if pred_px is not None:
                        pred_px[ticker_to_plot_idxs, get_config().RESET_PRED_PX_EACH_N_DAYS] = curr_pred_px
                        pred_dt[get_config().RESET_PRED_PX_EACH_N_DAYS] = date_from_timestamp(
                            raw_dates[data_idx])

                        pred_px_series.append(pred_px)
                        pred_dt_series.append(pred_dt)
                    # reset price
                    curr_pred_px = px[ticker_to_plot_idxs, data_idx]
                    # create new serie
                    pred_px = np.zeros((total, get_config().RESET_PRED_PX_EACH_N_DAYS + 1))
                    pred_dt = [None] * (get_config().RESET_PRED_PX_EACH_N_DAYS + 1)

                # fill values
                pred_px[ticker_to_plot_idxs, serie_idx] = curr_pred_px
                pred_dt[serie_idx] = date_from_timestamp(raw_dates[data_idx])

                # update pred px
                curr_pred_px += (predictions[ticker_to_plot_idxs, i, 0] / get_config().PRED_HORIZON) * curr_pred_px

            return curr_pred_px, pred_px, pred_dt

        def fill_eq(eq, rets, raw_dates):
            nonlocal capital, bet, position
            if capital is None:
                capital = np.ones(total, dtype=np.float32)
            if bet is None:
                bet = np.zeros(total, dtype=np.float32)
            if position is None:
                position = 0

            for i in range(predictions.shape[1]):
                data_idx = b * get_config().BPTT_STEPS + i

                date = date_from_timestamp(raw_dates[data_idx])
                if date.year == 2012 and date.month == 12 and date.day == 31:
                    _debug = 0

                if data_idx % get_config().REBALANCE_FREQ == 0:
                    position = np.sign(predictions[ticker_to_plot_idxs, i, 0])
                    capital += bet
                    bet = np.ones(total)
                    capital -= bet
                eq[ticker_to_plot_idxs, data_idx] = (capital + bet)

                bet += bet * rets[ticker_to_plot_idxs, data_idx] * position

        while epoch <= get_config().MAX_EPOCH:

            print("Eval %d epoch on train set..." % epoch)
            curr_progress = 0
            state = None
            losses = np.zeros((tr_batch_num))
            curr_pred_px = None
            pred_px = None
            pred_dt = None

            capital = None
            bet = None
            position = None
            for b in range(tr_batch_num):
                if state is None:
                    state = net.zero_state(len(env.tickers))

                input, labels = get_batch_input_and_lables(tr_input, tr_labels, b)
                state, loss, predictions = net.eval(state, input, labels)

                if get_config().DRAW_PREDICTIONS:
                    curr_pred_px, pred_px, pred_dt = predict_price_series(tr_px, tr_raw_dates, predictions, b, pred_px,
                                                                          pred_dt, tr_pred_px_series,
                                                                          tr_pred_dt_series, curr_pred_px)
                fill_eq(tr_eq, tr_rets, tr_raw_dates)

                losses[b] = loss
                curr_progress = progress.print_progress(curr_progress, b, tr_batch_num)

            progress.print_progess_end()
            train_avg_loss = np.mean(np.sqrt(losses))
            print("Train loss: %.4f%%" % (train_avg_loss * 100))

            print("Eval %d epoch on test set..." % epoch)
            curr_progress = 0
            state = None
            losses = np.zeros((tst_batch_num))
            curr_pred_px = None
            pred_px = None
            pred_dt = None

            capital = None
            bet = None
            position = None
            for b in range(tst_batch_num):
                if state is None:
                    state = net.zero_state(len(env.tickers))

                input, labels = get_batch_input_and_lables(tst_input, tst_labels, b)
                state, loss, predictions = net.eval(state, input, labels)

                if get_config().DRAW_PREDICTIONS:
                    curr_pred_px, pred_px, pred_dt = predict_price_series(tst_px, tst_raw_dates, predictions, b,
                                                                          pred_px, pred_dt,
                                                                          tst_pred_px_series,
                                                                          tst_pred_dt_series, curr_pred_px)
                fill_eq(tst_eq, tst_rets, tst_raw_dates)

                losses[b] = loss
                curr_progress = progress.print_progress(curr_progress, b, tst_batch_num)

            progress.print_progess_end()
            tst_avg_loss = np.mean(np.sqrt(losses))
            print("Test loss: %.4f%%" % (tst_avg_loss * 100))

            # draw plots in test mode and break
            if not is_train():
                dt = build_time_axis(tr_raw_dates)

                if get_config().DRAW_PREDICTIONS:
                    plot_prediction('Train', dt, tr_px, tr_pred_px_series, tr_pred_dt_series)
                plot_eq('Train', get_config().TRAIN_BEG, get_config().TRAIN_END, dt, tr_eq)

                dt = build_time_axis(tst_raw_dates)
                if get_config().DRAW_PREDICTIONS:
                    plot_prediction('Test', dt, tst_px, tst_pred_px_series, tst_pred_dt_series)
                plot_eq('Test', get_config().TEST_BEG, get_config().TEST_END, dt, tst_eq)

                show_plots()
                break

            # train
            if is_train() and epoch <= get_config().MAX_EPOCH:
                # save train progress
                writer.writerow(
                    (
                        epoch,
                        train_avg_loss,
                        tst_avg_loss
                    ))
                f.flush()
                # plot and save graphs
                dt = build_time_axis(tr_raw_dates)
                fig = plot_eq('Train', get_config().TRAIN_BEG, get_config().TRAIN_END, dt, tr_eq)
                fig.savefig('%s/%04d.png' % (get_config().TRAIN_FIG_PATH, epoch))
                plt.close(fig)
                if epoch == get_config().MAX_EPOCH:
                    tr_df = pd.DataFrame({'date': dt, 'capital': tr_eq[0,:]})
                    tr_df.to_csv(get_config().TRAIN_EQ_PATH, index=False)

                dt = build_time_axis(tst_raw_dates)
                fig = plot_eq('Test', get_config().TEST_BEG, get_config().TEST_END, dt, tst_eq)
                fig.savefig('%s/%04d.png' % (get_config().TEST_FIG_PATH, epoch))
                plt.close(fig)
                if epoch == get_config().MAX_EPOCH:
                    tr_df = pd.DataFrame({'date': dt, 'capital': tst_eq[0, :]})
                    tr_df.to_csv(get_config().TEST_EQ_PATH, index=False)

                epoch += 1
                if epoch > get_config().MAX_EPOCH:
                    break
                print("Training %d epoch..." % epoch)

                curr_progress = 0
                state = None
                for b in range(tr_batch_num):
                    if state is None:
                        state = net.zero_state(len(env.tickers))

                    input, labels = get_batch_input_and_lables(tr_input, tr_labels, b)
                    state, loss, predictions = net.fit(state, input, labels)

                    curr_progress = progress.print_progress(curr_progress, b, tr_batch_num)

                progress.print_progess_end()

                net.save_weights(get_config().WEIGHTS_PATH, epoch)
