import numpy as np
import datetime
import pandas as pd
import math
import random
import csv
import os.path

from download_utils import load_npz_data
from portfolio.config import get_config, NetVersion, Mode, TradingFrequency
from portfolio.net_apple import NetApple
from portfolio.net_banana import NetBanana
import progress


class SnpEnv(object):
    def __init__(self):
        print('loading data...')
        self._tickers, self.raw_dt, self.raw_data = load_npz_data('data/snp/snp_px.npz')
        print('data load complete')

        # calc data dimensions
        self.stks = self.raw_data.shape[0]
        self.days = self.raw_data.shape[1]

        # calc data dates range
        self.HIST_BEG = self._idx_to_date(0)
        self.HIST_END = self._idx_to_date(-1)

        # calc tradable_mask, traded_stocks_per_day, trading_day_mask
        self.tradable_mask = np.all(self.raw_data > 0.0, axis=2)
        self.traded_stocks_per_day = self.tradable_mask[:, :].sum(0)
        self.trading_day_mask = self.traded_stocks_per_day > get_config().MIN_STOCKS_TRADABLE_PER_TRADING_DAY

        # calc snp_mask
        self.snp_mask = np.full((self.stks, self.days), False)

        snp_mask_df = pd.read_csv('data/snp/snp_mask.csv')

        for idx, row in snp_mask_df.iterrows():
            _from = datetime.datetime.strptime(row['from'], '%Y-%m-%d').date()
            _to = datetime.datetime.strptime(row['to'], '%Y-%m-%d').date()
            _ticker = row['ticker']
            stk_idx = self._ticker_to_idx(_ticker)
            if stk_idx is None:
                continue
            _from = max(_from, self.HIST_BEG)
            _to = min(_to, self.HIST_END)
            _from_idx = self._date_to_idx(_from)
            _to_idx = self._date_to_idx(_to)

            self.snp_mask[stk_idx, _from_idx:_to_idx + 1] = True

        # prepare input array
        x = np.zeros((self.stks, self.days, 5))
        # fill array for each stock
        for stk_idx in range(self.stks):
            stk_raw_data = self.raw_data[stk_idx, :, :]
            tradable_mask = self.tradable_mask[stk_idx]

            # #there are some errors in data when let's say we have volume but no prices, or when one px is zero
            # #threat such days as not tradable to minimize errors - looks like we really can skip such days
            # vol_tradable_mask = stk_raw_data[:, get_config().VOLUME_DATA_IDX] > 0.0
            # xor = tradable_mask != vol_tradable_mask
            # missmatches = np.sum(xor)
            # if missmatches > 0:
            #     dt_idx = np.where(xor)[0]
            #     print(self.tickers[stk_idx])
            #     for idx in range(dt_idx.shape[0]):
            #         date = datetime.datetime.fromtimestamp(self.raw_dt[dt_idx[idx]]).date()
            #         print(date)

            stk_data = stk_raw_data[tradable_mask, :]
            a_o = stk_data[:, get_config().ADJ_OPEN_DATA_IDX]
            a_c = stk_data[:, get_config().ADJ_CLOSE_DATA_IDX]
            a_h = stk_data[:, get_config().ADJ_HIGH_DATA_IDX]
            a_l = stk_data[:, get_config().ADJ_LOW_DATA_IDX]
            a_v = stk_data[:, get_config().ADJ_VOLUME_DATA_IDX]

            if a_c.shape[0] == 0:
                # print(self._idx_to_ticker(stk_idx))
                continue

            prev_a_c = np.roll(a_c, 1)
            prev_a_c[0] = a_c[0]
            prev_a_v = np.roll(a_v, 1)
            prev_a_v[0] = a_v[0]

            # other variant is to use ln(a_c / prev_a_c) - they are almost identical
            # plus we can scale input by multiplier (100%)
            x_o = np.log(a_o / prev_a_c) if get_config().LOG_RET else (a_o - prev_a_c) / prev_a_c
            x_c = np.log(a_c / prev_a_c) if get_config().LOG_RET else (a_c - prev_a_c) / prev_a_c
            x_h = np.log(a_h / prev_a_c) if get_config().LOG_RET else (a_h - prev_a_c) / prev_a_c
            x_l = np.log(a_l / prev_a_c) if get_config().LOG_RET else (a_l - prev_a_c) / prev_a_c
            x_v = np.log(a_v / prev_a_v) if get_config().LOG_VOL_CHG else (a_v - prev_a_v) / prev_a_v

            x[stk_idx, tradable_mask, 0] = get_config().RET_MUL * x_o
            x[stk_idx, tradable_mask, 1] = get_config().RET_MUL * x_c
            x[stk_idx, tradable_mask, 2] = get_config().RET_MUL * x_h
            x[stk_idx, tradable_mask, 3] = get_config().RET_MUL * x_l
            x[stk_idx, tradable_mask, 4] = get_config().VOL_CHG_MUL * x_v
        self.x = x

    @property
    def tickers(self):
        return self._tickers

    def _date_to_idx(self, date):
        if self.HIST_BEG <= date <= self.HIST_END:
            return (date - self.HIST_BEG).days
        return None

    def _idx_to_date(self, idx):
        return datetime.datetime.fromtimestamp(self.raw_dt[idx]).date()

    def _ticker_to_idx(self, ticker):
        ticker_idxs = np.nonzero(self.tickers == ticker)
        if ticker_idxs[0].shape[0] > 0:
            return ticker_idxs[0][0]
        return None

    def _idx_to_ticker(self, idx):
        return self.tickers[idx]

    def get_input(self, stk_mask, beg, end):
        return self.x[stk_mask, self._date_to_idx(beg): self._date_to_idx(end), :]

    def get_snp_components_mask(self, date):
        # return self.snp_mask[:, self._date_to_idx(date)]
        return self.snp_mask[:, self._date_to_idx(date)] & self.tradable_mask[:, self._date_to_idx(date)]

    def get_ret_lbl(self, stk_mask, ent, ext):
        ent_px = self.raw_data[stk_mask, self._date_to_idx(ent), get_config().ADJ_CLOSE_DATA_IDX]
        ext_px = self.raw_data[stk_mask, self._date_to_idx(ext), get_config().ADJ_CLOSE_DATA_IDX]
        return (ext_px - ent_px) / ent_px

    def find_trading_date(self, candidate_date):
        if not (self.HIST_BEG <= candidate_date <= self.HIST_END):
            return None
        candidate_date_idx = self._date_to_idx(candidate_date)
        while candidate_date_idx < self.raw_dt.shape[0]:
            if self.trading_day_mask[candidate_date_idx] == True:
                return self._idx_to_date(candidate_date_idx)
            candidate_date_idx += 1
        return None

    def trading_day_generator(self):
        for idx in np.nonzero(self.trading_day_mask)[0]:
            yield self._idx_to_date(idx)

    def trading_schedule_generator(self, BEG, END, TRADING_PERIOD_DAYS):
        ent_candidate = BEG
        while True:
            ent = self.find_trading_date(ent_candidate)
            if ent is None:
                break

            found_days = 0
            ext_candidate = ent + datetime.timedelta(days=1)
            while found_days < TRADING_PERIOD_DAYS:
                ext = self.find_trading_date(ext_candidate)
                if ext is None:
                    break
                else:
                    found_days += 1
            if BEG <= ent <= END:
                yield (ent, ext)
            else:
                break
            if ext is None:
                break
            ent_candidate = ext


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


def calc_sharpe(rets, years):
    return math.sqrt(rets.shape[0] / years) * np.mean(rets) / np.std(rets)


def print_alloc(ent, tickers, stk_mask, weights):
    abs_weights = np.abs(weights)
    alloc = np.sum(abs_weights) * 100
    idxs = np.argsort(abs_weights)
    top_5_idxs = idxs[-5:]
    top_5_idxs = top_5_idxs[::-1]
    top_5_weights = weights[top_5_idxs] * 100
    all_stk_idxs = np.nonzero(stk_mask)[0]
    stk_idxs = all_stk_idxs[top_5_idxs]

    print("%s: %.2f%% W1: %.2f%% [%s], W2: %.2f%% [%s], W3: %.2f%% [%s], W4: %.2f%% [%s], W5: %.2f%% [%s]" % (
        ent.strftime('%Y.%m.%d'),
        alloc,
        top_5_weights[0],
        tickers[stk_idxs[0]],
        top_5_weights[1],
        tickers[stk_idxs[1]],
        top_5_weights[2],
        tickers[stk_idxs[2]],
        top_5_weights[3],
        tickers[stk_idxs[3]],
        top_5_weights[4],
        tickers[stk_idxs[4]]))


def train():
    snp_env = SnpEnv()
    if get_config().NET_VER == NetVersion.APPLE:
        net = NetApple()
    elif get_config().NET_VER == NetVersion.BANANA:
        net = NetBanana()
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
        if get_config().EPOCH_WEIGHTS_TO_LOAD is not None:
            net.load_weights(get_config().WEIGHTS_PATH, get_config().EPOCH_WEIGHTS_TO_LOAD)
            epoch = get_config().EPOCH_WEIGHTS_TO_LOAD
            if get_config().MODE == Mode.TRAIN:
                epoch += 1
        else:
            epoch = 0
        while True:
            print("Epoch %d" % epoch)

            if get_config().SHUFFLE:
                train_schedule = random.sample(train_trading_schedule, len(train_trading_schedule))
            else:
                train_schedule = train_trading_schedule

            if get_config().MODE == Mode.TRAIN:
                # train
                print("Training...")
                dataset_size = len(train_schedule)
                curr_progress = 0
                passed = 0
                for ent, ext in train_schedule:
                    stk_mask = snp_env.get_snp_components_mask(ent)
                    # print("%d %s %s" % (np.sum(stk_mask), ent.strftime("%Y-%m-%d"), ext.strftime("%Y-%m-%d")))
                    x = snp_env.get_input(stk_mask, ent - get_config().RNN_HISTORY, ent)
                    labels = snp_env.get_ret_lbl(stk_mask, ent, ext)
                    net.fit(x, labels)
                    curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                    passed += 1
                progress.print_progess_end()

            # eval train
            print("Eval train...")
            ret = np.zeros((len(train_trading_schedule)))

            dataset_size = len(train_trading_schedule)
            curr_progress = 0
            passed = 0
            for ent, ext in train_trading_schedule:
                stk_mask = snp_env.get_snp_components_mask(ent)
                x = snp_env.get_input(stk_mask, ent - get_config().RNN_HISTORY, ent)
                labels = snp_env.get_ret_lbl(stk_mask, ent, ext)
                pl, weights = net.eval(x, labels)
                if get_config().PRINT_PREDICTION:
                    print_alloc(ent, snp_env.tickers, stk_mask, weights)
                ret[passed] = pl
                curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                passed += 1
            progress.print_progess_end()

            years = (get_config().TRAIN_END - get_config().TRAIN_BEG).days / 365

            capital = np.cumsum(ret) + 1.0
            train_dd = calc_dd(capital, False)
            train_sharpe = calc_sharpe(ret, years)
            train_y_avg = np.sum(ret) / years

            print('Train dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (train_dd * 100, train_y_avg * 100, train_sharpe))

            # eval test
            print("Eval test...")
            ret = np.zeros((len(test_trading_schedule)))

            dataset_size = len(test_trading_schedule)
            curr_progress = 0
            passed = 0
            for ent, ext in test_trading_schedule:
                if ext is None:
                    break
                stk_mask = snp_env.get_snp_components_mask(ent)
                x = snp_env.get_input(stk_mask, ent - get_config().RNN_HISTORY, ent)
                labels = snp_env.get_ret_lbl(stk_mask, ent, ext)
                pl, weights = net.eval(x, labels)
                if get_config().PRINT_PREDICTION:
                    print_alloc(ent, snp_env.tickers, stk_mask, weights)
                ret[passed] = pl
                curr_progress = progress.print_progress(curr_progress, passed, dataset_size)
                passed += 1
            progress.print_progess_end()

            years = (get_config().TRAIN_END - get_config().TRAIN_BEG).days / 365

            capital = np.cumsum(ret) + 1.0
            test_dd = calc_dd(capital, False)
            test_sharpe = calc_sharpe(ret, years)
            test_y_avg = np.sum(ret) / years

            print('Test dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (test_dd * 100, test_y_avg * 100, test_sharpe))

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
                break
