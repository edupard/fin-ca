import numpy as np
import datetime
import pandas as pd

from download_utils import load_npz_data
from portfolio.config import get_config


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
        self.trading_day_idxs = np.nonzero(self.trading_day_mask)[0]

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

        # start_time = timeit.default_timer()
        # for i in range(self.days - get_config().COVARIANCE_LENGTH):
        #     idx = i + get_config().COVARIANCE_LENGTH - 1
        #     # print(self.raw_dt[idx])
        #     r = self.x[:, idx: idx + get_config().COVARIANCE_LENGTH, 1]
        #     var = np.var(r, axis=1)
        #     # cov = np.cov(r)
        # print(timeit.default_timer() - start_time)

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

    def get_input(self, stk_mask, end):
        end_idx = self._date_to_idx(end)
        if get_config().SKIP_NON_TRADING_DAYS:
            history = get_config().RNN_HISTORY.days
            valid_trading_day_idxs = np.nonzero(self.trading_day_idxs<=end_idx)[0]
            history_trading_day_idxs = self.trading_day_idxs[valid_trading_day_idxs[-history:]]
            x = self.x[:, history_trading_day_idxs, :]
            return x[stk_mask,:,:]
        else:
            beg = end - get_config().RNN_HISTORY
            beg_idx = self._date_to_idx(beg)
            return self.x[stk_mask, beg_idx : end_idx, :]

    def get_tradeable_snp_components_mask(self, date):
        return self.snp_mask[:, self._date_to_idx(date)] & self.tradable_mask[:, self._date_to_idx(date)]

    def get_ret_lbl(self, stk_mask, ent, ext):
        ent_idx = self._date_to_idx(ent)
        ext_idx = self._date_to_idx(ext)
        ent_px = self.raw_data[stk_mask, ent_idx, get_config().ADJ_CLOSE_DATA_IDX]
        ext_px = self.raw_data[stk_mask, ext_idx, get_config().ADJ_CLOSE_DATA_IDX]
        not_tradeable_mask = ~(self.tradable_mask[stk_mask, ext_idx])
        if np.sum(not_tradeable_mask) != 0:
            stk_idxs = np.nonzero(stk_mask)[0]
            not_tradeable_stk_idxs = stk_idxs[not_tradeable_mask]
            # stks = self.tickers[stk_mask]
            # not_tradeable_stks = stks[not_tradeable_mask]
            fill_idxs = np.nonzero(not_tradeable_mask)[0]
            for i in range(not_tradeable_stk_idxs.shape[0]):
                stk_idx = not_tradeable_stk_idxs[i]
                idx_to_fill = fill_idxs[i]
                stk_tradeable_mask = self.tradable_mask[stk_idx, :]
                stk_tradeable_day_idxs = np.nonzero(stk_tradeable_mask)[0]
                _ext_idx = ent_idx
                # first tradeable day after
                after_idxs = np.nonzero(stk_tradeable_day_idxs > ext_idx)[0]
                if after_idxs.shape[0] > 0:
                    _ext_idx = stk_tradeable_day_idxs[after_idxs[0]]
                else:
                    before_idxs = np.where(stk_tradeable_day_idxs > ent_idx)[0]
                    if before_idxs.shape[0] > 0:
                        _ext_idx = stk_tradeable_day_idxs[before_idxs[0]]
                # print("%s %s %d" % (ent.strftime('%Y-%m-%d'), not_tradeable_stks[i], _ext_idx - ent_idx))
                _ext_px = self.raw_data[stk_idx, _ext_idx, get_config().ADJ_CLOSE_DATA_IDX]
                ext_px[idx_to_fill] = _ext_px

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
            # be careful: ext can be none, i.e. no more trading
            if ext is None:
                break
            if BEG <= ent <= END:
                yield (ent, ext)
            else:
                break

            ent_candidate = ext
