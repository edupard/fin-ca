from download_utils import download_data, parse_tickers, preprocess_data, load_npz_data, load_npz_data_alt, preprocess_data_alt
from data_utils import filter_activelly_tradeable_stocks, get_dates_for_daily_return, get_dates_for_weekly_return, \
    get_tradable_stock_indexes, get_prices, PxType, calc_z_score, get_data_idx, calc_z_score_alt
from date_range import HIST_BEG,HIST_END
from tickers import get_nasdaq_tickers
import datetime
import numpy as np
import csv

from nn import ffnn_instance

NUM_WEEKS = 12
NUM_DAYS = 5
PERCENTILE = 10

TODAY = datetime.datetime.today().date()

USE_ADJ_PX = True
# YYYY-MM-DD
PREDICTION_DATE = datetime.datetime.strptime('2017-07-21', '%Y-%m-%d').date()
OPEN_POS_DATE = datetime.datetime.strptime('2017-07-21', '%Y-%m-%d').date()
HPR_DATE = datetime.datetime.strptime('2017-07-28', '%Y-%m-%d').date()


START_DATE = PREDICTION_DATE - datetime.timedelta(days=(NUM_WEEKS + 2) * 7)
END_DATE = HPR_DATE

tickers = get_nasdaq_tickers()
download_data(tickers, 'data/history.csv', START_DATE, END_DATE, 50)
preprocess_data_alt(tickers, 'data/history.csv', START_DATE, END_DATE, 'data/history.npz', USE_ADJ_PX)
tickers, raw_dt, raw_data = load_npz_data_alt('data/history.npz')

mask, traded_stocks = filter_activelly_tradeable_stocks(raw_data)

w_r_i = get_dates_for_weekly_return(START_DATE, END_DATE, traded_stocks, PREDICTION_DATE, NUM_WEEKS)
d_r_i = get_dates_for_daily_return(START_DATE, END_DATE, traded_stocks, PREDICTION_DATE, NUM_DAYS)
t_s_i = get_tradable_stock_indexes(mask, w_r_i + d_r_i)
d_c = get_prices(raw_data, t_s_i, d_r_i, PxType.CLOSE)
w_c = get_prices(raw_data, t_s_i, w_r_i, PxType.CLOSE)
# dr = calc_z_score(d_c)
dr, d_r, d_c_r, d_r_m, d_r_std = calc_z_score_alt(d_c)
# wr = calc_z_score(w_c)
wr, w_r, w_c_r, w_r_m, w_r_std = calc_z_score_alt(w_c)

# with open('data/prediction_px.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#
#     row = ['ticker']
#     for dt_idx in w_r_i:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     for dt_idx in d_r_i:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     writer.writerow(row)
#
#     idx = 0
#     for ticker_idx in t_s_i:
#         ticker = tickers[ticker_idx]
#         row = []
#         row.append(ticker)
#         for v in w_c[idx,:]:
#             row.append(v)
#         for v in d_c[idx,:]:
#             row.append(v)
#         writer.writerow(row)
#         idx += 1
#
# with open('data/prediction_r.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#
#     row = ['ticker']
#     for dt_idx in w_r_i[1:]:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     for dt_idx in d_r_i[1:]:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     writer.writerow(row)
#
#     idx = 0
#     for ticker_idx in t_s_i:
#         ticker = tickers[ticker_idx]
#         row = []
#         row.append(ticker)
#         for v in w_r[idx,:]:
#             row.append(v)
#         for v in d_r[idx,:]:
#             row.append(v)
#         writer.writerow(row)
#         idx += 1
#
#
# with open('data/prediction_c_r.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#
#     row = ['ticker']
#     for dt_idx in w_r_i[1:]:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     for dt_idx in d_r_i[1:]:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     writer.writerow(row)
#
#     idx = 0
#     for ticker_idx in t_s_i:
#         ticker = tickers[ticker_idx]
#         row = []
#         row.append(ticker)
#         for v in w_c_r[idx,:]:
#             row.append(v)
#         for v in d_c_r[idx,:]:
#             row.append(v)
#         writer.writerow(row)
#         idx += 1
#
#     row = ['MEAN']
#     for v in w_r_m[:]:
#         row.append(v)
#     for v in d_r_m[:]:
#         row.append(v)
#     writer.writerow(row)
#
#     row = ['STDDEV']
#     for v in w_r_std[:]:
#         row.append(v)
#     for v in d_r_std[:]:
#         row.append(v)
#     writer.writerow(row)
#
#
# with open('data/prediction_z.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#
#     row = ['ticker']
#     for dt_idx in w_r_i[1:]:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     for dt_idx in d_r_i[1:]:
#         dt = datetime.datetime.fromtimestamp(raw_dt[dt_idx])
#         row.append(dt.strftime('%Y-%m-%d'))
#     writer.writerow(row)
#
#     idx = 0
#     for ticker_idx in t_s_i:
#         ticker = tickers[ticker_idx]
#         row = []
#         row.append(ticker)
#         for v in wr[idx,:]:
#             row.append(v)
#         for v in dr[idx,:]:
#             row.append(v)
#         writer.writerow(row)
#         idx += 1



print("Predicting optimal portfolio...")
ffnn = ffnn_instance()
ffnn.load_weights('./rbm/ffnn.chp')
input = np.concatenate([wr, dr], axis=1)
p_dist = ffnn.predict(input)
p_l = p_dist[:, 0]
p_l = p_l - np.median(p_l) + 0.5
sorted_indexes = p_l.argsort()

top_bound = np.percentile(p_l, 100 - PERCENTILE)
bottom_bound = np.percentile(p_l, PERCENTILE)

ONE_WEEK_RETURN_IDX = w_r_i[NUM_WEEKS - 1]
ONE_DAY_RETURN_IDX = d_r_i[NUM_DAYS - 1]
PREDICTION_DATE_IDX = d_r_i[NUM_DAYS]
HPR_DATE_IDX = get_data_idx(HPR_DATE, START_DATE, END_DATE)
OPEN_POS_DATE_IDX  = get_data_idx(OPEN_POS_DATE, START_DATE, END_DATE)


def get_csv_date_string(idx):
    date = datetime.datetime.fromtimestamp(raw_dt[idx]).date()
    return date.strftime('%Y-%m-%d')


CSV_ONE_WEEK_DATE = get_csv_date_string(ONE_WEEK_RETURN_IDX)
CSV_ONE_DAY_DATE = get_csv_date_string(ONE_DAY_RETURN_IDX)
CSV_PREDICATION_DATE = get_csv_date_string(PREDICTION_DATE_IDX)
CSV_HPR_DATE = get_csv_date_string(HPR_DATE_IDX)
CSV_OPEN_POS_DATE = get_csv_date_string(OPEN_POS_DATE_IDX)


with open('./data/prediction.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ('ticker', 'long prob', 'class', '1w', '1d', '*', '#', 'hp', '1w px', '1d px', '* px', '# px', 'hp px', '1wr pct', '1dr pct',
         'hpr pct', '1d v', '1w avg v'))

    for idx in sorted_indexes:
        ticker_idx = t_s_i[idx]
        ticker = tickers[ticker_idx]
        long_prob = p_l[idx]
        _class = ''
        if long_prob > top_bound:
            _class = 'L'
        if long_prob < bottom_bound:
            _class = 'S'

        _1w_px = raw_data[ticker_idx, ONE_WEEK_RETURN_IDX, 3]
        _1d_px = raw_data[ticker_idx, ONE_DAY_RETURN_IDX, 3]
        _pred_px = raw_data[ticker_idx, PREDICTION_DATE_IDX, 3]
        _hp_px = raw_data[ticker_idx, HPR_DATE_IDX, 3]

        _open_px = raw_data[ticker_idx, OPEN_POS_DATE_IDX, 0]
        if OPEN_POS_DATE <= PREDICTION_DATE:
            _open_px = raw_data[ticker_idx, OPEN_POS_DATE_IDX, 3]

        _week_gross_volume = raw_data[ticker_idx, d_r_i[1:], 3] * raw_data[ticker_idx, d_r_i[1:], 4]
        _last_day_gross_volume = _week_gross_volume[4]
        _week_avg_gross_volume = np.average(_week_gross_volume)


        def get_pct(enter_px, exit_px):
            return (exit_px - enter_px) / enter_px


        _1wr_pct = get_pct(_1w_px, _pred_px)
        _1dr_pct = get_pct(_1d_px, _pred_px)
        _hpr_pct = get_pct(_open_px, _hp_px)

        writer.writerow((ticker,
                         long_prob,
                         _class,
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
                         _last_day_gross_volume,
                         _week_avg_gross_volume
                         ))
print("Prediction saved")
