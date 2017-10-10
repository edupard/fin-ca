from download_utils import download_data, load_npz_data, preprocess_data
from portfolio.multi_stock_env import Env, date_from_timestamp
from tickers import get_snp_tickers_exch_map
from portfolio.multi_stock_config import get_config
from portfolio.net_shiva import NetShiva
import datetime
import numpy as np
import csv
import os


def ib_convert_ticker(ticker):
    ticker = ticker.replace('-', ' ')
    ticker = ticker.replace('.', ' ')
    return ticker


# YYYY-MM-DD
PREDICTION_T_MINUS_ONE = datetime.datetime.strptime('2017-10-05', '%Y-%m-%d').date()
PREDICTION_T_MINUS_WEEK = datetime.datetime.strptime('2017-09-29', '%Y-%m-%d').date()

PREDICTION_DATE = datetime.datetime.strptime('2017-10-06', '%Y-%m-%d').date()
OPEN_POS_DATE = datetime.datetime.strptime('2017-10-06', '%Y-%m-%d').date()
HPR_DATE = datetime.datetime.strptime('2017-10-06', '%Y-%m-%d').date()

START_DATE = PREDICTION_DATE - datetime.timedelta(days=365)
END_DATE = HPR_DATE

tickers = [
    'ABT',
    'ARNC',
    'HON',
    'SHW',
    'CMI',
    'EMR',
    'SLB',
    'CSX',
    'CLX',
    'GIS',
    'NEM',
    'MCD',
    'LLY',
    'BAX',
    'BDX',
    'JNJ',
    'GPC',
    'HPQ',
    'WMB',
    'BCR',
    'JPM',
    'IFF',
    'AET',
    'AXP',
    'BAC',
    'CI',
    'DUK',
    'LNC',
    'TAP',
    'NEE',
    'DIS',
    'XRX',
    'IBM',
    'WFC',
    'INTC',
    'TGT',
    'TXT',
    'VFC',
    'WBA',
    'AIG',
    'FLR',
    'FDX',
    'PCAR',
    'ADP',
    'GWW',
    'MAS',
    'ADM',
    'MAT',
    'WMT',
    'SNA',
    'SWK',
    'AAPL',
    'OXY',
    'CAG',
    'LB',
    'T',
    'VZ',
    'LOW',
    'PHM',
    'HES',
    'LMT',
    'HAS',
    'BLL',
    'APD',
    'NUE',
    'PKI',
    'NOC',
    'CNP',
    'TJX',
    'DOV',
    'PH',
    'ITW',
    'GPS',
    'JWN',
    'MDT',
    'HRB',
    'SYY',
    'CA',
    'MMC',
    'AVY',
    'HD',
    'PNC',
    'C',
    'STI',
    'NKE',
    'ECL',
    'NWL',
    'TMK',
    'ORCL',
    'ADSK',
    'MRO',
    'AEE',
    'AMGN',
    'PX',
    'IPG',
    'COST',
    'CSCO',
    'EMN',
    'KEY',
    'UNM',
    'MSFT',
    'LUV',
    'UNH',
    'CBS',
    'MU',
    'BSX',
    'ADBE',
    'EFX',
    'PGR',
    'YUM',
    'RF',
    'SPLS',
    'NTAP',
    'BBY',
    'VMC',
    'XLNX',
    'A',
    'TIF',
    'DVN',
    'EOG',
    'INTU',
    'RHI',
    'SYK',
    'COP'
]

get_config().PREDICTION_MODE = True


def create_folders():
    if not os.path.exists(get_config().DATA_FOLDER_PATH):
        os.makedirs(get_config().DATA_FOLDER_PATH)


ticker_exch_map = get_snp_tickers_exch_map()


def get_net_data(env, BEG, END):
    beg_idx, end_idx = env.get_data_idxs_range(BEG, END)

    raw_dates = env.get_raw_dates(beg_idx, end_idx)
    input = env.get_input(beg_idx, end_idx)
    px = env.get_adj_close_px(beg_idx, end_idx)
    tradeable_mask = env.get_tradeable_mask(beg_idx, end_idx)

    raw_week_days = np.full(raw_dates.shape, 0, dtype=np.int32)
    for i in range(raw_dates.shape[0]):
        date = date_from_timestamp(raw_dates[i])
        raw_week_days[i] = date.isoweekday()

    return beg_idx, end_idx, raw_dates, raw_week_days, tradeable_mask, px, input


def get_csv_date_string(date):
    return date.strftime('%Y-%m-%d')


CSV_ONE_WEEK_DATE = get_csv_date_string(PREDICTION_T_MINUS_WEEK)
CSV_ONE_DAY_DATE = get_csv_date_string(PREDICTION_T_MINUS_ONE)
CSV_PREDICATION_DATE = get_csv_date_string(PREDICTION_DATE)
CSV_HPR_DATE = get_csv_date_string(HPR_DATE)
CSV_OPEN_POS_DATE = get_csv_date_string(OPEN_POS_DATE)

net = NetShiva()

with open('./data/prediction.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ('ticker', 'exchange', 'long prob', 'class', '1w', '1d', '*', '#', 'hp', '1w px', '1d px', '* px', '# px',
         'hp px',
         '1wr pct', '1dr pct',
         'hpr pct', '1d v', '1w avg v'))

    idx = 0
    for ticker in tickers:
        print("Processing %s" % ticker)
        get_config().TICKER = ticker
        get_config().reload()
        create_folders()
        _tickers = [ticker]
        # download_data(_tickers,
        #               get_config().DATA_PATH,
        #               START_DATE,
        #               END_DATE)
        # preprocess_data(_tickers,
        #                 get_config().DATA_PATH,
        #                 START_DATE,
        #                 END_DATE,
        #                 get_config().DATA_NPZ_PATH,
        #                 get_config().DATA_FEATURES)
        env = Env()
        beg_idx, end_idx, raw_dates, raw_week_days, tradeable_mask, px, input = get_net_data(env, START_DATE,
                                                                                             PREDICTION_DATE)
        ds_size = end_idx - beg_idx + 1
        net.load_weights(get_config().WEIGHTS_PATH, get_config().MAX_EPOCH)
        state = net.zero_state(1)

        _input = input
        _labels = np.zeros((1, ds_size))
        _mask = np.zeros((1, ds_size))
        state, loss, predictions = net.eval(state, _input, _labels, _mask)

        pred_ret = predictions[0, -1, 0]

        _1d_data_idx = env.get_data_idx(PREDICTION_T_MINUS_ONE)
        _1w_data_idx = env.get_data_idx(PREDICTION_T_MINUS_WEEK)
        _pred_data_idx = env.get_data_idx(PREDICTION_DATE)
        _hpr_data_idx = env.get_data_idx(HPR_DATE)
        _open_pos_data_idx = env.get_data_idx(OPEN_POS_DATE)


        def get_adj_close_px(data_idx):
            return env.get_adj_close_px(data_idx, data_idx)[0, 0]


        _1w_px = get_adj_close_px(_1w_data_idx)
        _1d_px = get_adj_close_px(_1d_data_idx)
        _pred_px = get_adj_close_px(_pred_data_idx)
        _hp_px = get_adj_close_px(_hpr_data_idx)
        _open_px = get_adj_close_px(_open_pos_data_idx)


        # _week_gross_volume = raw_data[ticker_idx, d_r_i[1:], DATA_TO_IDX]
        # _last_day_gross_volume = _week_gross_volume[4]
        # _week_avg_gross_volume = np.average(_week_gross_volume)


        def get_pct(enter_px, exit_px):
            return (exit_px - enter_px) / enter_px


        _1wr_pct = get_pct(_1w_px, _pred_px)
        _1dr_pct = get_pct(_1d_px, _pred_px)
        _hpr_pct = get_pct(_open_px, _hp_px)

        writer.writerow((ib_convert_ticker(ticker),
                         ticker_exch_map[ticker],
                         pred_ret,
                         'L' if pred_ret > 0 else 'S',
                         CSV_ONE_WEEK_DATE,
                         CSV_ONE_DAY_DATE,
                         CSV_PREDICATION_DATE,
                         CSV_OPEN_POS_DATE,
                         CSV_HPR_DATE,
                         _1w_px,
                         _1d_px,
                         _pred_px,
                         _open_px,
                         _hp_px,
                         _1wr_pct,
                         _1dr_pct,
                         _hpr_pct,
                         0,
                         0
                         ))

        idx += 1





# print("Making prediction...")
# input = np.concatenate([wr, dr], axis=1)
# p_dist = ffnn.predict(input)
# p_l = p_dist[:, 0]
# p_l = p_l - np.median(p_l) + 0.5
# sorted_indexes = p_l.argsort()
#
# top_bound = np.percentile(p_l, 100 - PERCENTILE)
# bottom_bound = np.percentile(p_l, PERCENTILE)
#
# ONE_WEEK_RETURN_IDX = w_r_i[NUM_WEEKS - 1]
# ONE_DAY_RETURN_IDX = d_r_i[NUM_DAYS - 1]
# PREDICTION_DATE_IDX = d_r_i[NUM_DAYS]
# HPR_DATE_IDX = get_data_idx(HPR_DATE, START_DATE, END_DATE)
# OPEN_POS_DATE_IDX = get_data_idx(OPEN_POS_DATE, START_DATE, END_DATE)




# with open('./data/prediction.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(
#         ('ticker', 'exchange', 'long prob', 'class', '1w', '1d', '*', '#', 'hp', '1w px', '1d px', '* px', '# px', 'hp px',
#          '1wr pct', '1dr pct',
#          'hpr pct', '1d v', '1w avg v'))
#
#     for idx in sorted_indexes:
#         ticker_idx = t_s_i[idx]
#         ticker = tickers[ticker_idx]
#         long_prob = p_l[idx]
#         _class = ''
#         if long_prob > top_bound:
#             _class = 'L'
#         if long_prob < bottom_bound:
#             _class = 'S'
#
#
#
#         writer.writerow((ib_convert_ticker(ticker),
#                          ticker_exch_map[ticker],
#                          long_prob,
#                          _class,
#                          CSV_ONE_WEEK_DATE,
#                          CSV_ONE_DAY_DATE,
#                          CSV_PREDICATION_DATE,
#                          CSV_OPEN_POS_DATE,
#                          CSV_HPR_DATE,
#                          _1w_px,
#                          _1d_px,
#                          _pred_px,
#                          _open_px,
#                          _hp_px,
#                          _1wr_pct,
#                          _1dr_pct,
#                          _hpr_pct,
#                          _last_day_gross_volume,
#                          _week_avg_gross_volume
#                          ))
# print("Prediction saved")
