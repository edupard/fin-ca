import numpy as np
import datetime

from download_utils import load_npz_data
from portfolio.single_stock_config import get_config


def date_from_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts).date()


class Env(object):
    def __init__(self):
        print('loading data...')
        self._tickers, self.raw_dt, self.raw_data = load_npz_data(get_config().DATA_NPZ_PATH)
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
        self.trading_day_mask = self.traded_stocks_per_day > 0
        self.trading_day_idxs = np.nonzero(self.trading_day_mask)[0]

        # prepare input array
        x = np.zeros((self.stks, self.days, 5))
        # fill array for each stock
        for stk_idx in range(self.stks):
            stk_raw_data = self.raw_data[stk_idx, :, :]
            tradable_mask = self.tradable_mask[stk_idx]

            stk_data = stk_raw_data[tradable_mask, :]
            a_o = stk_data[:, get_config().ADJ_OPEN_DATA_IDX]
            a_c = stk_data[:, get_config().ADJ_CLOSE_DATA_IDX]
            a_h = stk_data[:, get_config().ADJ_HIGH_DATA_IDX]
            a_l = stk_data[:, get_config().ADJ_LOW_DATA_IDX]
            a_v = stk_data[:, get_config().ADJ_VOLUME_DATA_IDX]

            if a_c.shape[0] == 0:
                continue

            prev_a_c = np.roll(a_c, 1)
            prev_a_c[0] = a_c[0]
            prev_a_v = np.roll(a_v, 1)
            prev_a_v[0] = a_v[0]

            # other variant is to use ln(a_c / prev_a_c) - they are almost identical
            # plus we can scale input by multiplier (100%)
            x_o = (a_o - prev_a_c) / prev_a_c
            x_c = (a_c - prev_a_c) / prev_a_c
            x_h = (a_h - prev_a_c) / prev_a_c
            x_l = (a_l - prev_a_c) / prev_a_c
            x_v = (a_v - prev_a_v) / prev_a_v

            x[stk_idx, tradable_mask, 0] = x_o
            x[stk_idx, tradable_mask, 1] = x_c
            x[stk_idx, tradable_mask, 2] = x_h
            x[stk_idx, tradable_mask, 3] = x_l
            x[stk_idx, tradable_mask, 4] = x_v
        self.x = x

    @property
    def tickers(self):
        return self._tickers

    def _date_to_idx(self, date):
        if self.HIST_BEG <= date <= self.HIST_END:
            return (date - self.HIST_BEG).days
        return None

    def _idx_to_date(self, idx):
        return date_from_timestamp(self.raw_dt[idx])

    def _ticker_to_idx(self, ticker):
        ticker_idxs = np.nonzero(self.tickers == ticker)
        if ticker_idxs[0].shape[0] > 0:
            return ticker_idxs[0][0]
        return None

    def _idx_to_ticker(self, idx):
        return self.tickers[idx]

    def get_input_generator(self, BEGIN, END, SEQ_LEN, PRED_HORIZON):
        x = self.x[:, self.trading_day_mask, :]
        raw_dt = self.raw_dt[self.trading_day_mask]
        raw_data = self.raw_data[:, self.trading_day_mask, :]

        data_len = raw_dt.shape[0]
        beg_idx = 0
        for idx in range(data_len):
            if date_from_timestamp(raw_dt[idx]) >= BEGIN:
                beg_idx = idx
                break
        end_idx = 0
        for idx in range(data_len):
            if date_from_timestamp(raw_dt[data_len - idx - 1]) <= END:
                end_idx = data_len - idx
                break
        end_idx = min(data_len - PRED_HORIZON, end_idx)
        data_points = end_idx - beg_idx - 1

        seqs = (data_points // SEQ_LEN) + (0 if data_points % SEQ_LEN == 0 else 1)

        for i in range(seqs):
            b_i = beg_idx + i * SEQ_LEN
            e_i = min(b_i + SEQ_LEN, end_idx)

            input = x[:, b_i: e_i, :]

            ent_px = raw_data[:, b_i: e_i, get_config().ADJ_CLOSE_DATA_IDX]
            ext_px = raw_data[:, b_i + PRED_HORIZON: e_i + PRED_HORIZON, get_config().ADJ_CLOSE_DATA_IDX]
            labels = (ext_px - ent_px) / ent_px
            yield (input, labels.reshape(1,-1,1))
