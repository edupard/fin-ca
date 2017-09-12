import numpy as np
import csv
import os.path

from portfolio.net_turtle import NetTurtle
from portfolio.single_stock_config import get_config, Mode
from portfolio.stat import print_alloc, get_draw_down, get_sharpe_ratio, get_capital, get_avg_yeat_ret
from portfolio.graphs import plot_equity_curve, show_plots
import progress

from portfolio.single_stock_env import Env


def flatten(l):
    return [item for sublist in l for item in sublist]


def train():
    if not os.path.exists(get_config().WEIGHTS_FOLDER_PATH):
        os.makedirs(get_config().WEIGHTS_FOLDER_PATH)

    env = Env()
    net = NetTurtle()
    net.init()

    if not os.path.exists(get_config().TRAIN_STAT_PATH):
        with open(get_config().TRAIN_STAT_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    'epoch',
                    'train loss',
                    'test loss',
                ))

    with open(get_config().TRAIN_STAT_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if get_config().EPOCH_WEIGHTS_TO_LOAD is not None:
            net.load_weights(get_config().WEIGHTS_PATH, get_config().EPOCH_WEIGHTS_TO_LOAD)
            epoch = get_config().EPOCH_WEIGHTS_TO_LOAD
            if get_config().MODE == Mode.TRAIN:
                epoch += 1
        else:
            epoch = 0

        train_raw_dates = env.get_raw_dates(get_config().TRAIN_BEG, get_config().TRAIN_END)
        train_input = env.get_input(get_config().TRAIN_BEG, get_config().TRAIN_END)
        train_px = env.get_px(get_config().TRAIN_BEG, get_config().TRAIN_END)
        train_px_t5 = env.get_px(get_config().TRAIN_BEG, get_config().TRAIN_END, delay_days=5)




        train_data = []
        for dt, px, daily_rets, input, labels in env.get_input_generator(get_config().TRAIN_BEG, get_config().TRAIN_END,
                                                                         get_config().BPTT_STEPS,
                                                                         get_config().PRED_HORIZON):
            train_data.append((dt, px, daily_rets, input, labels))

        test_data = []
        for dt, px, daily_rets, input, labels in env.get_input_generator(get_config().TEST_BEG, get_config().TEST_END,
                                                                         get_config().BPTT_STEPS,
                                                                         get_config().PRED_HORIZON):
            test_data.append((dt, px, daily_rets, input, labels))
        while True:

            print("Epoch %d" % epoch)

            print("Eval train...")
            dataset_size = len(train_data)
            curr_progress = 0
            passed = 0
            losses = np.zeros((dataset_size))
            state = None
            # test_px = []
            # test_pred_px = []
            # test_ret = []
            # last_pred_px = None
            # dts = []
            # stk_idx = 0
            for dt, px, daily_rets, input, labels in train_data:
                if state is None:
                    state = net.zero_state(input.shape[0])
                new_state, loss, predicted_returns = net.eval(state, input, labels)
                state = new_state
                losses[passed] = loss

                # # real px process
                # test_px.append(px[stk_idx,:])
                # # predicted px process
                # if last_pred_px is None:
                #     last_pred_px = px[0,0]
                # daily_pred_ret = predicted_returns[stk_idx,:,0] / get_config().PRED_HORIZON
                # for idx in range(daily_pred_ret.shape[0]):
                #     dpr = daily_pred_ret[idx]
                #     last_pred_px += dpr * last_pred_px
                #     test_pred_px.append(last_pred_px)
                # # eq returns
                # test_ret.append(daily_rets[stk_idx,:] * np.sign(predicted_returns[stk_idx,:,0]))
                # # time
                # dts.append(dt)

                curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                passed += 1
            progress.print_progess_end()
            train_avg_loss = np.mean(np.sqrt(losses))
            print("Train loss: %.4f%%" % (train_avg_loss * 100))

            # test_px = flatten(test_px)
            # test_ret = flatten(test_ret)
            # dts = flatten(dts)
            #
            # years = (get_config().TRAIN_END - get_config().TRAIN_BEG).days / 365
            # capital = get_capital(test_ret, False)
            # train_dd = get_draw_down(capital, False)
            # train_sharpe = get_sharpe_ratio(test_ret, years)
            # train_y_avg = get_avg_yeat_ret(test_ret, years)
            # print('Train dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (train_dd * 100, train_y_avg * 100, train_sharpe))
            # plot_equity_curve("Train equity curve", dts, capital)

            print("Eval test...")
            dataset_size = len(test_data)
            curr_progress = 0
            passed = 0
            losses = np.zeros((dataset_size))
            state = None
            for dt, px, daily_rets, input, labels in test_data:
                if state is None:
                    state = net.zero_state(input.shape[0])
                new_state, loss, predicted_returns = net.eval(state, input, labels)
                state = new_state
                losses[passed] = loss
                curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                passed += 1
            progress.print_progess_end()
            test_avg_loss = np.mean(np.sqrt(losses))
            print("Test loss: %.4f%%" % (test_avg_loss * 100))

            # train
            if get_config().MODE == Mode.TRAIN:
                print("Training...")
                dataset_size = len(train_data)
                curr_progress = 0
                passed = 0
                state = None
                for dt, px, daily_rets, input, labels in train_data:
                    if state is None:
                        state = net.zero_state(input.shape[0])
                    new_state, loss, predicted_returns = net.fit(state, input, labels)
                    state = new_state
                    curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                    passed += 1
                progress.print_progess_end()

                writer.writerow(
                    (
                        epoch,
                        train_avg_loss,
                        test_avg_loss
                    ))

                f.flush()
                net.save_weights(get_config().WEIGHTS_PATH, epoch)
                epoch += 1
            else:
                show_plots()
                break
