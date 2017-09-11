import numpy as np
import csv
import os.path

from portfolio.net_turtle import NetTurtle
from portfolio.single_stock_config import get_config, Mode
from portfolio.graphs import show_plots
import progress

from portfolio.single_stock_env import Env


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

        train_data = []
        for input, labels in env.get_input_generator(get_config().TRAIN_BEG, get_config().TRAIN_END,
                                                     get_config().BPTT_STEPS, get_config().PRED_HORIZON):
            train_data.append((input, labels))

        test_data = []
        for input, labels in env.get_input_generator(get_config().TEST_BEG, get_config().TEST_END,
                                                     get_config().BPTT_STEPS, get_config().PRED_HORIZON):
            test_data.append((input, labels))
        while True:

            print("Epoch %d" % epoch)

            print("Eval train...")
            dataset_size = len(train_data)
            curr_progress = 0
            passed = 0
            losses = np.zeros((dataset_size))
            state = None
            for input, labels in train_data:
                if state is None:
                    state = net.zero_state(input.shape[0])
                new_state, loss, predicted_returns = net.eval(state, input, labels)
                state = new_state
                losses[passed] = loss
                curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                passed += 1
            progress.print_progess_end()
            train_avg_loss = np.mean(np.sqrt(losses))
            print("Train loss: %.4f%%" % (train_avg_loss * 100))

            print("Eval test...")
            dataset_size = len(test_data)
            curr_progress = 0
            passed = 0
            losses = np.zeros((dataset_size))
            state = None
            for input, labels in test_data:
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
                for input, labels in train_data:
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
