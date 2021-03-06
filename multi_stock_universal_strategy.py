import datetime
import numpy as np
import pandas as pd
import os

from portfolio.multi_stock_config import get_config
from download_utils import download_data, preprocess_data
from portfolio.multi_stock_env import Env, date_from_timestamp
from portfolio.snp import get_snp_hitorical_components_tickers

stocks = get_snp_hitorical_components_tickers()

def download_data_for_all_stocks():

    def create_folders():
        if not os.path.exists(get_config().DATA_FOLDER_PATH):
            os.makedirs(get_config().DATA_FOLDER_PATH)

    get_config().TICKER = 'UNIVERSAL_NET'
    get_config().reload()

    create_folders()
    download_data(stocks,
                  get_config().DATA_PATH,
                  get_config().HIST_BEG,
                  get_config().HIST_END)
    preprocess_data(stocks,
                    get_config().DATA_PATH,
                    get_config().HIST_BEG,
                    get_config().HIST_END,
                    get_config().DATA_NPZ_PATH,
                    get_config().DATA_FEATURES)

predications = []

# download_data_for_all_stocks()

# EPOCH = 585
EPOCH = 600

pwd = os.getcwd()

for stock in stocks:
    print("Parsing %s predictions" % stock)
    # path = 'D:/Projects/portfolio/data/eval/petri/UNIVERSAL_NET/test/%s/prediction/%s_%d.csv' % (stock, stock, EPOCH)
    # path = 'D:/Projects/portfolio/data/eval/petri/UNIVERSAL_NET/test/%s/prediction/%s_%d.csv' % (stock, stock, EPOCH)
    path = 'data/ma/eval/%s/prediction/%s_%d.csv' % (stock, stock, EPOCH)
    try:
        os.chdir(os.path.dirname(path))
        df = pd.read_csv(os.path.basename(path))
        df['ticker'] = stock
        df['date'] = df['ts'].apply(date_from_timestamp)


        predications.append(df)
    except:
        pass

os.chdir(pwd)

pred_df = pd.concat(predications, ignore_index=True)
pred_df_length = len(pred_df['date'])

# add stock idx column
pred_df['stk_idx'] = pd.Series(np.zeros((pred_df_length), dtype=np.int32), index=pred_df.index)

get_config().TICKER = 'UNIVERSAL_NET'
get_config().reload()
env = Env()
total_stocks = len(env.tickers)

# update stk_idx column for all records
for stock in stocks:
    stk_idx = env._ticker_to_idx(stock)
    pred_df.loc[pred_df['ticker'] == stock, 'stk_idx'] = stk_idx

beg_idx, end_idx = env.get_data_idxs_range(get_config().TRAIN_BEG, get_config().TRAIN_END)
trading_days = end_idx + 1 - beg_idx
adj_px = env.get_adj_close_px(beg_idx, end_idx)
tradable_maks = env.get_portfolio_mask(beg_idx, end_idx)
# tradable_maks = env.get_tradeable_mask(beg_idx, end_idx)
raw_dates = env.get_raw_dates(beg_idx, end_idx)

def build_time_axis(raw_dates):
    dt = []
    for raw_dt in np.nditer(raw_dates):
        dt.append(date_from_timestamp(raw_dt))
    return dt

dt = build_time_axis(raw_dates)

# net factor weight, selection, long pct
NET_FACTOR_GRID = [1.0]
SELECTION_GRID = [100]
# SELECTION_GRID = [5,15,25,35]
# LONG_PCT_GRID = [0.0]
LONG_PCT_GRID = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# NET_FACTOR_GRID = [0.0, 1.0]
# SELECTION_GRID = [23]
# LONG_PCT_GRID = [0.7]

def calc_pl(pos,curr_px,pos_px):
    return np.sum(pos * (curr_px * (1 - np.sign(pos) * get_config().SLIPPAGE) - pos_px * (1 + np.sign(pos) * get_config().SLIPPAGE)))

for NET_FACTOR in NET_FACTOR_GRID:
    for SELECTION in SELECTION_GRID:
        for LONG_PCT in LONG_PCT_GRID:
            print('processing %.0f factor weight %d stocks %.0f long pct' % (NET_FACTOR * 100, SELECTION, LONG_PCT * 100))

            _pos_dt = []
            _pos_stocks = []
            _pos_pos = []

            SHORT_PCT = 1 - LONG_PCT
            YIELD_FACTOR = 1 - NET_FACTOR

            prediction = np.zeros((total_stocks))
            fri_px = None
            mon_px = None
            fri_tradable_mask = None
            mon_tradable_mask = None

            cash = 1
            pos = np.zeros((total_stocks))
            pos_px = np.zeros((total_stocks))

            eq = np.zeros((trading_days))
            long_pos_mask = np.full((total_stocks), False)
            short_pos_mask = np.full((total_stocks), False)

            for idx in range(trading_days):
                date = date_from_timestamp(raw_dates[idx])
                curr_px = adj_px[:, idx]
                _tradable_maks = tradable_maks[:,idx]

                open_pos = date.isoweekday() == 1 and fri_px is not None
                # open_pos = date.isoweekday() == 5
                close_pos = date.isoweekday() == 5
                predict = date.isoweekday() == 5

                if predict:
                    fri_px = curr_px
                    fri_tradable_mask = _tradable_maks

                    selection = pred_df['date'] == date
                    curr_pred_df = pred_df.loc[selection]
                    prediction[curr_pred_df['stk_idx'].values] = curr_pred_df['prediction'].values

                if close_pos:
                    rpl = calc_pl(pos,curr_px,pos_px)
                    cash += rpl
                    pos[:] = 0

                if open_pos:
                    try:
                        mon_px = curr_px
                        mon_tradable_mask = _tradable_maks

                        pos_mask = fri_tradable_mask & mon_tradable_mask

                        metric = np.zeros((total_stocks))

                        yield_factor = (mon_px[pos_mask] - fri_px[pos_mask]) / fri_px[pos_mask]
                        net_factor = prediction[pos_mask]
                        _metric = NET_FACTOR * net_factor - YIELD_FACTOR * yield_factor
                        metric[pos_mask] = _metric

                        # play vs net
                        _metric_sorted = np.sort(_metric)
                        _l_b_idx = max(-SELECTION, -_metric_sorted.shape[0])
                        _l_b = _metric_sorted[_l_b_idx]
                        _s_b_idx = min(SELECTION - 1, _metric_sorted.shape[0] - 1)
                        _s_b = _metric_sorted[_s_b_idx]

                        long_pos_mask[:] = pos_mask
                        short_pos_mask[:] = pos_mask
                        long_pos_mask &= metric >= _l_b
                        short_pos_mask &= metric <= _s_b

                        # # do not play vs net
                        # long_pos_mask[:] = pos_mask
                        # long_pos_mask &= prediction >= 0
                        # _long_metric = metric[long_pos_mask]
                        # long_metric_sorted = np.sort(_long_metric)
                        # _l_b_idx = max(-SELECTION, -long_metric_sorted.shape[0])
                        # _l_b = long_metric_sorted[_l_b_idx]
                        #
                        # short_pos_mask[:] = pos_mask
                        # short_pos_mask &= prediction < 0
                        # _short_metric = metric[short_pos_mask]
                        # short_metric_sorted = np.sort(_short_metric)
                        # _s_b_idx = min(SELECTION - 1, short_metric_sorted.shape[0] - 1)
                        # _s_b = short_metric_sorted[_s_b_idx]
                        #
                        # long_pos_mask &= metric >= _l_b
                        # short_pos_mask &= metric <= _s_b

                        pos_px = curr_px

                        # num_stks = np.sum(pos_mask)
                        # pos[pos_mask] = 1 / num_stks / curr_px[pos_mask] * np.sign(prediction[pos_mask])

                        long_num_stks = np.sum(long_pos_mask)
                        pos[long_pos_mask] = LONG_PCT / long_num_stks / curr_px[long_pos_mask]

                        short_num_stks = np.sum(short_pos_mask)
                        pos[short_pos_mask] = -SHORT_PCT / short_num_stks / curr_px[short_pos_mask]

                        # for stk_idx in np.nonzero(long_pos_mask)[0]:
                        #     _pos_dt.append(date)
                        #     _pos_pos.append(pos[stk_idx])
                        #     _pos_stocks.append(env._idx_to_ticker(stk_idx))
                        # for stk_idx in np.nonzero(short_pos_mask)[0]:
                        #     _pos_dt.append(date)
                        #     _pos_pos.append(pos[stk_idx])
                        #     _pos_stocks.append(env._idx_to_ticker(stk_idx))
                    except:
                        _debug = 0

                urpl = calc_pl(pos,curr_px,pos_px)
                nlv = cash + urpl

                eq[idx] = nlv

            eq_df = pd.DataFrame({'date': dt, 'capital': eq})
            eq_df.to_csv('data/eq/ma_100_%.0f_%d_%.0f.csv' % (NET_FACTOR * 100, SELECTION, LONG_PCT * 100), index=False)
            # eq_df.to_csv('data/eq/universal_%.0f_%d_%.0f.csv' % (NET_FACTOR * 100, SELECTION, LONG_PCT * 100), index=False)


            # pos_df = pd.DataFrame({'date': _pos_dt, 'ticker': _pos_stocks, 'pos': _pos_pos})
            # pos_df.to_csv('data/eq/poses_universal_%.0f_%d_%.0f.csv' % (NET_FACTOR * 100, SELECTION, LONG_PCT * 100),
            #              index=False)
