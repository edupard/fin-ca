import numpy as np
import csv
import os.path
import pandas as pd
import datetime

from portfolio.net_shiva import NetShiva
from portfolio.multi_stock_config import get_config, Mode
from portfolio.stat import print_alloc, get_draw_down, get_sharpe_ratio, get_capital, get_avg_yeat_ret
from portfolio.graphs import plot_equity_curve, show_plots, create_time_serie_fig, plot_time_serie
import progress
import matplotlib.pyplot as plt
from portfolio.capm import Capm

from portfolio.multi_stock_env import Env, date_from_timestamp

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

def plot_eq(name, BEG, END, dt, capital):
    years = (END - BEG).days / 365
    dd = get_draw_down(capital, False)
    rets = capital[1:] - capital[:-1]
    sharpe = get_sharpe_ratio(rets, years)
    y_avg = (capital[-1] - capital[0]) / years
    print('%s dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (name, dd * 100, y_avg * 100, sharpe))
    return plot_equity_curve("%s equity curve" % name, dt, capital[:])


def train():
    create_folders()

    env = Env()
    net = NetShiva()

    if not os.path.exists(get_config().TRAIN_STAT_PATH):
        with open(get_config().TRAIN_STAT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    'epoch',
                    'train loss',
                    'test loss',
                ))

    total_tickers = len(env.tickers)

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
            beg_idx, end_idx = env.get_data_idxs_range(BEG, END)

            raw_dates = env.get_raw_dates(beg_idx, end_idx)
            input = env.get_input(beg_idx, end_idx)
            px = env.get_adj_close_px(beg_idx, end_idx)
            px_pred_hor = env.get_adj_close_px(beg_idx + get_config().PRED_HORIZON, end_idx + get_config().PRED_HORIZON)
            tradeable_mask = env.get_tradeable_mask(beg_idx, end_idx)
            port_mask = env.get_portfolio_mask(beg_idx, end_idx)

            ds_sz = px_pred_hor.shape[1]

            raw_dates = raw_dates[:ds_sz]
            input = input[:, :ds_sz, :]
            tradeable_mask = tradeable_mask[:,:ds_sz]
            port_mask = port_mask[:, :ds_sz]
            px = px[:, :ds_sz]

            labels = (px_pred_hor - px) / px
            batch_num = get_batches_num(ds_sz, get_config().BPTT_STEPS)

            return beg_idx, ds_sz, batch_num, raw_dates, tradeable_mask, port_mask, px, input, labels

        tr_beg_data_idx, tr_ds_sz, tr_batch_num, tr_raw_dates, tr_tradeable_mask, tr_port_mask, tr_px, tr_input, tr_labels = get_net_data(get_config().TRAIN_BEG,
                                                                                                 get_config().TRAIN_END)

        tst_beg_data_idx, tst_ds_sz, tst_batch_num, tst_raw_dates, tst_tradeable_mask, tst_port_mask, tst_px, tst_input, tst_labels = get_net_data(
            get_config().TEST_BEG,
            get_config().TEST_END)

        tr_eq = np.zeros((tr_ds_sz))
        tst_eq = np.zeros((tst_ds_sz))

        def get_batch_slice(input, labels, mask, b):
            b_i = b * get_config().BPTT_STEPS
            e_i = (b + 1) * get_config().BPTT_STEPS
            return input[:, b_i: e_i, :], labels[:, b_i: e_i], mask[:, b_i: e_i].astype(np.float32)

        while epoch <= get_config().MAX_EPOCH:

            print("Eval %d epoch on train set..." % epoch)
            batch_num = tr_batch_num
            input = tr_input
            labels = tr_labels
            px = tr_px
            mask = tr_tradeable_mask
            port_mask = tr_port_mask
            eq = tr_eq
            beg_data_idx = tr_beg_data_idx
            raw_dates = tr_raw_dates
            state = None

            def eval():
                nonlocal raw_dates, beg_data_idx, batch_num, input, labels, px, mask, port_mask, eq, state
                curr_progress = 0
                cash = 1
                pos = np.zeros((total_tickers))
                pos_px = np.zeros((total_tickers))
                losses = np.zeros((batch_num))

                for b in range(batch_num):
                    if state is None:
                        state = net.zero_state(total_tickers)

                    _input, _labels, _mask = get_batch_slice(input, labels, mask, b)
                    state, loss, predictions = net.eval(state, _input, _labels, _mask)


                    for i in range(predictions.shape[1]):
                        data_idx = b * get_config().BPTT_STEPS + i
                        curr_px = px[:, data_idx]
                        global_data_idx = beg_data_idx + data_idx

                        date = datetime.datetime.fromtimestamp(raw_dates[data_idx]).date()
                        open_pos = False
                        close_pos = False
                        if get_config().REBALANCE_MON_FRI:
                            if date.isoweekday() == 1:
                                open_pos = True
                            if date.isoweekday() == 5:
                                close_pos = True
                        else:
                            if data_idx % get_config().REBALANCE_FREQ == 0:
                                close_pos = True
                                open_pos = True

                        if close_pos:
                            rpl = np.sum(pos * (curr_px - pos_px))
                            cash += rpl
                            pos[:] = 0
                        if open_pos:
                            pos_px = curr_px
                            pos_mask = port_mask[:, data_idx]
                            num_stks = np.sum(pos_mask)
                            if get_config().CAPM:
                                exp, cov = env.get_exp_and_cov(pos_mask,
                                                               global_data_idx - get_config().COVARIANCE_LENGTH + 1,
                                                               global_data_idx)
                                exp = get_config().REBALANCE_FREQ * exp
                                cov = get_config().REBALANCE_FREQ * get_config().REBALANCE_FREQ * cov
                                if get_config().CAPM_USE_NET_PREDICTIONS:
                                    exp = predictions[:, i, 0][pos_mask]

                                capm = Capm(num_stks)
                                capm.init()

                                best_sharpe = None
                                best_weights = None
                                best_constriant = None
                                while i <= 10000:
                                    w, sharpe, constraint = capm.get_params(exp, cov)
                                    # print("Iteration: %d Sharpe: %.2f Constraint: %.6f" % (i, sharpe, constraint))
                                    if w is None:
                                        break
                                    if best_sharpe is None or sharpe >= best_sharpe:
                                        best_weights = w
                                        best_sharpe = sharpe
                                        best_constriant = constraint
                                    capm.fit(exp, cov)
                                    capm.rescale_weights()

                                    i += 1
                                date = datetime.datetime.fromtimestamp(raw_dates[data_idx]).date()
                                print("Date: %s sharpe: %.2f constraint: %.6f" %
                                      (date.strftime('%Y-%m-%d'),
                                       best_sharpe,
                                       best_constriant)
                                      )

                                pos[pos_mask] = best_weights / curr_px[pos_mask]
                            else:
                                pos[pos_mask] = 1 / num_stks / curr_px[pos_mask] * np.sign(predictions[pos_mask, i, 0])

                        urpl = np.sum(pos * (curr_px - pos_px))
                        nlv = cash + urpl

                        eq[data_idx] = nlv

                    losses[b] = loss
                    curr_progress = progress.print_progress(curr_progress, b, tr_batch_num)

                progress.print_progess_end()
                avg_loss = np.mean(np.sqrt(losses))
                return avg_loss

            tr_avg_loss = eval()
            print("Train loss: %.4f%%" % (tr_avg_loss * 100))

            print("Eval %d epoch on train set..." % epoch)
            batch_num = tst_batch_num
            input = tst_input
            labels = tst_labels
            px = tst_px
            mask = tst_tradeable_mask
            port_mask = tst_port_mask
            eq = tst_eq
            beg_data_idx = tst_beg_data_idx
            raw_dates = tst_raw_dates
            state = None

            tst_avg_loss = eval()
            print("Test loss: %.4f%%" % (tst_avg_loss * 100))

            if not is_train():
                dt = build_time_axis(tr_raw_dates)
                plot_eq('Train', get_config().TRAIN_BEG, get_config().TRAIN_END, dt, tr_eq)

                dt = build_time_axis(tst_raw_dates)
                plot_eq('Test', get_config().TEST_BEG, get_config().TEST_END, dt, tst_eq)

                show_plots()
                break

            if is_train() and epoch <= get_config().MAX_EPOCH:
                writer.writerow(
                    (
                        epoch,
                        tr_avg_loss,
                        tst_avg_loss
                    ))

                f.flush()

                # plot and save graphs
                dt = build_time_axis(tr_raw_dates)
                fig = plot_eq('Train', get_config().TRAIN_BEG, get_config().TRAIN_END, dt, tr_eq)
                fig.savefig('%s/%04d.png' % (get_config().TRAIN_FIG_PATH, epoch))
                plt.close(fig)
                if epoch == get_config().MAX_EPOCH:
                    tr_df = pd.DataFrame({'date': dt, 'capital': tr_eq[:]})
                    tr_df.to_csv(get_config().TRAIN_EQ_PATH, index=False)

                dt = build_time_axis(tst_raw_dates)
                fig = plot_eq('Test', get_config().TEST_BEG, get_config().TEST_END, dt, tst_eq)
                fig.savefig('%s/%04d.png' % (get_config().TEST_FIG_PATH, epoch))
                plt.close(fig)
                if epoch == get_config().MAX_EPOCH:
                    tr_df = pd.DataFrame({'date': dt, 'capital': tst_eq[:]})
                    tr_df.to_csv(get_config().TEST_EQ_PATH, index=False)

                epoch += 1
                if epoch > get_config().MAX_EPOCH:
                    break
                print("Training %d epoch..." % epoch)

                curr_progress = 0
                state = None
                for b in range(tr_batch_num):
                    if state is None:
                        state = net.zero_state(total_tickers)

                    input, labels, mask = get_batch_slice(tr_input, tr_labels, tr_tradeable_mask, b)
                    state, loss, predictions = net.fit(state, input, labels, mask)

                    curr_progress = progress.print_progress(curr_progress, b, tr_batch_num)

                progress.print_progess_end()

                net.save_weights(get_config().WEIGHTS_PATH, epoch)