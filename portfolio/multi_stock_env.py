import numpy as np
import datetime
import pandas as pd

from download_utils import load_npz_data
from portfolio.multi_stock_config import get_config


def date_from_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts).date()


class Env(object):
    def __init__(self):
        print('loading data...')
        tickers, raw_dt, raw_data = load_npz_data(get_config().DATA_NPZ_PATH)
        # tickers, raw_dt, raw_data = load_npz_data('data/stocks/DIS/DIS.npz')
        print('data load complete')

        self._tickers = tickers
        self.stks = self.tickers.shape[0]

        days = raw_dt.shape[0]

        def _idx_to_date(idx):
            return datetime.datetime.fromtimestamp(raw_dt[idx]).date()

        # calc data dates range
        HIST_BEG = _idx_to_date(0)
        HIST_END = _idx_to_date(-1)

        def _date_to_idx(date):
            if HIST_BEG <= date <= HIST_END:
                return (date - HIST_BEG).days
            return None

        # calc snp_mask
        snp_mask = np.full((self.stks, days), False)

        snp_mask_df = pd.read_csv('data/snp/snp_mask.csv')

        for idx, row in snp_mask_df.iterrows():
            _from = datetime.datetime.strptime(row['from'], '%Y-%m-%d').date()
            _to = datetime.datetime.strptime(row['to'], '%Y-%m-%d').date()
            _ticker = row['ticker']
            stk_idx = self._ticker_to_idx(_ticker)
            if stk_idx is None:
                continue
            _from = max(_from, HIST_BEG)
            _to = min(_to, HIST_END)
            _from_idx = _date_to_idx(_from)
            _to_idx = _date_to_idx(_to)

            snp_mask[stk_idx, _from_idx:_to_idx + 1] = True

        # calc tradable_mask, traded_stocks_per_day, trading_day_mask
        tradable_mask = np.all(raw_data > 0.0, axis=2)
        traded_stocks_per_day = tradable_mask[:, :].sum(0)
        trading_day_mask = traded_stocks_per_day >= get_config().MIN_STOCKS_TRADABLE_PER_TRADING_DAY
        # trading_day_mask = traded_stocks_per_day > 0

        self.trading_days = np.sum(trading_day_mask)

        # leave tradeable days only
        self.raw_dt = raw_dt[trading_day_mask]
        self.raw_data = raw_data[:, trading_day_mask, :]
        self.traded_stocks_per_day = traded_stocks_per_day[trading_day_mask]
        self.tradable_mask = tradable_mask[:, trading_day_mask]
        self.snp_mask= snp_mask[:, trading_day_mask]

        # prepare input array
        input = np.zeros((self.stks, self.trading_days, 6))
        # fill array for each stock
        for stk_idx in range(self.stks):
            stk_raw_data = self.raw_data[stk_idx, :, :]
            tradable_mask = self.tradable_mask[stk_idx]

            # dirty hack: fill prices with first known px
            tr_days_idxs = np.nonzero(tradable_mask)[0]
            first_adj_volume = None
            first_volume = None
            if tr_days_idxs.shape[0] > 0:
                first_adj_close_px = stk_raw_data[tr_days_idxs[0], get_config().ADJ_CLOSE_DATA_IDX]
                first_adj_volume = stk_raw_data[tr_days_idxs[0], get_config().ADJ_VOLUME_DATA_IDX]
            if first_adj_close_px is None:
                stk_raw_data[:, :] = 1
            else:
                last_px = first_adj_close_px
                last_volume = first_adj_volume
                for day_idx in range(self.trading_days):
                    if tradable_mask[day_idx] == 0:
                        stk_raw_data[day_idx, get_config().ADJ_CLOSE_DATA_IDX] = last_px
                        stk_raw_data[day_idx, get_config().ADJ_OPEN_DATA_IDX] = last_px
                        stk_raw_data[day_idx, get_config().ADJ_HIGH_DATA_IDX] = last_px
                        stk_raw_data[day_idx, get_config().ADJ_LOW_DATA_IDX] = last_px
                        stk_raw_data[day_idx, get_config().ADJ_VOLUME_DATA_IDX] = last_volume
                    else:
                        last_px = stk_raw_data[day_idx, get_config().ADJ_CLOSE_DATA_IDX]
                        last_volume = stk_raw_data[day_idx, get_config().ADJ_VOLUME_DATA_IDX]

            self.raw_data[stk_idx, :, :] = stk_raw_data

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

            x_o = (a_o - prev_a_c) / prev_a_c
            x_c = (a_c - prev_a_c) / prev_a_c
            x_h = (a_h - prev_a_c) / prev_a_c
            x_l = (a_l - prev_a_c) / prev_a_c
            x_v = (a_v - prev_a_v) / prev_a_v

            input[stk_idx, tradable_mask, 0] = x_o
            input[stk_idx, tradable_mask, 1] = x_c
            input[stk_idx, tradable_mask, 2] = x_h
            input[stk_idx, tradable_mask, 3] = x_l
            input[stk_idx, tradable_mask, 4] = x_v
            input[stk_idx, tradable_mask, 5] = 1
        self.input = input

    def _ticker_to_idx(self, ticker):
        ticker_idxs = np.nonzero(self.tickers == ticker)
        if ticker_idxs[0].shape[0] > 0:
            return ticker_idxs[0][0]
        return None

    def _idx_to_ticker(self, idx):
        return self.tickers[idx]

    @property
    def total_trading_days(self):
        return self.trading_days

    @property
    def tickers(self):
        return self._tickers

    def get_timestamp(self, DATE):
        dt = datetime.datetime.combine(DATE, datetime.time.min)
        return dt.timestamp()

    def get_data_idxs_range(self, BEG, END):
        BEG_TS = self.get_timestamp(BEG)
        END_TS = self.get_timestamp(END)

        beg_idx = None
        end_idx = None
        for i in range(self.trading_days):
            ts = self.raw_dt[i]
            if ts >= BEG_TS and beg_idx is None:
                beg_idx = i
            if ts <= END_TS:
                end_idx = i
        return beg_idx, end_idx

    def get_data_idx(self, END):
        END_TS = self.get_timestamp(END)
        end_idx = None
        for i in range(self.trading_days):
            ts = self.raw_dt[i]
            if ts <= END_TS:
                end_idx = i
            else:
                break
        return end_idx


    def get_input(self, BEG_DATA_IDX, END_DATA_IDX):
        return self.input[:, BEG_DATA_IDX: END_DATA_IDX + 1, :]

    def get_adj_close_px(self, BEG_DATA_IDX, END_DATA_IDX):
        return self.raw_data[:, BEG_DATA_IDX: END_DATA_IDX + 1, get_config().ADJ_CLOSE_DATA_IDX]

    def get_raw_dates(self, BEG_DATA_IDX, END_DATA_IDX):
        return self.raw_dt[BEG_DATA_IDX: END_DATA_IDX + 1]

    def get_tradeable_mask(self, BEG_DATA_IDX, END_DATA_IDX):
        return self.tradable_mask[:,BEG_DATA_IDX: END_DATA_IDX + 1]

    def get_portfolio_mask(self, BEG_DATA_IDX, END_DATA_IDX):
        if get_config().is_snp_index():
            return self.tradable_mask[:, BEG_DATA_IDX: END_DATA_IDX + 1] & self.snp_mask[:, BEG_DATA_IDX: END_DATA_IDX + 1]
        else:
            return self.get_tradeable_mask(BEG_DATA_IDX, END_DATA_IDX)

    def get_exp_and_cov(self, mask, BEG_DATA_IDX, END_DATA_IDX):
        r = self.input[mask, BEG_DATA_IDX : END_DATA_IDX + 1, 1]
        exp = np.mean(r, axis=1)
        cov = np.cov(r)
        return exp, cov