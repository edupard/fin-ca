from download_utils import download_data, parse_tickers, preprocess_data, load_npz_data
from data_utils import filter_tradeable_stocks, get_dates_for_daily_return, get_dates_for_weekly_return, \
    get_tradeable_stock_indexes, get_close_prices, calc_z_score, get_data_idx
import datetime
import numpy as np
import csv

from nn import ffnn_instance

NUM_WEEKS = 12
NUM_DAYS = 5
PERCENTILE = 10

TODAY = datetime.datetime.today().date()
# default values
HPR_DATE = TODAY - datetime.timedelta(days=1)
PREDICTION_DATE = HPR_DATE - datetime.timedelta(days=HPR_DATE.isoweekday() + 2)
# you can set prediction date(ie friday) and hpr date explicitly
# PREDICTION_DATE = datetime.datetime.strptime('2017-06-02', '%Y-%m-%d').date()
# HPR_DATE = datetime.datetime.strptime('2017-06-09', '%Y-%m-%d').date()
# PREDICTION_DATE = datetime.datetime.strptime('2017-06-09', '%Y-%m-%d').date()
# HPR_DATE = datetime.datetime.strptime('2017-06-16', '%Y-%m-%d').date()
PREDICTION_DATE = datetime.datetime.strptime('2017-06-16', '%Y-%m-%d').date()
HPR_DATE = datetime.datetime.strptime('2017-06-23', '%Y-%m-%d').date()
# PREDICTION_DATE = datetime.datetime.strptime('2017-06-23', '%Y-%m-%d').date()
# HPR_DATE = datetime.datetime.strptime('2017-06-27', '%Y-%m-%d').date()


START_DATE = PREDICTION_DATE - datetime.timedelta(days=(NUM_WEEKS + 2) * 7)
END_DATE = HPR_DATE

tickers, ticker_to_idx, idx_to_ticker = parse_tickers('data/tickers_nasdaq.csv')
# download_data(tickers, 'data/history.csv', START_DATE, END_DATE, 50)
# preprocess_data(ticker_to_idx, 'data/history.csv', START_DATE, END_DATE, 'data/history.npz')
raw_dt, raw_data = load_npz_data('data/history.npz')
mask, traded_stocks = filter_tradeable_stocks(raw_data)

w_r_i = get_dates_for_weekly_return(START_DATE, END_DATE, traded_stocks, PREDICTION_DATE, NUM_WEEKS)
d_r_i = get_dates_for_daily_return(START_DATE, END_DATE, traded_stocks, PREDICTION_DATE, NUM_DAYS)
t_s_i = get_tradeable_stock_indexes(mask, w_r_i + d_r_i)
d_c = get_close_prices(raw_data, t_s_i, d_r_i)
w_c = get_close_prices(raw_data, t_s_i, w_r_i)
dr = calc_z_score(d_c)
wr = calc_z_score(w_c)

print("Predicting optimal portfolio...")
ffnn = ffnn_instance()
ffnn.load_weights('./rbm/ffnn.chp')
input = np.concatenate([wr, dr], axis=1)
p_dist = ffnn.predict(input)
p_l = p_dist[:, 0]
# p_l = p_l - np.median(p_l) + 0.5
sorted_indexes = p_l.argsort()

top_bound = np.percentile(p_l, 100 - PERCENTILE)
bottom_bound = np.percentile(p_l, PERCENTILE)

ONE_WEEK_RETURN_IDX = w_r_i[NUM_WEEKS - 1]
ONE_DAY_RETURN_IDX = d_r_i[NUM_DAYS - 1]
PREDICTION_DATE_IDX = d_r_i[NUM_DAYS]
HPR_DATE_IDX = get_data_idx(HPR_DATE, START_DATE, END_DATE)


def get_csv_date_string(idx):
    date = datetime.datetime.fromtimestamp(raw_dt[idx]).date()
    return date.strftime('%Y-%m-%d')


CSV_ONE_WEEK_DATE = get_csv_date_string(ONE_WEEK_RETURN_IDX)
CSV_ONE_DAY_DATE = get_csv_date_string(ONE_DAY_RETURN_IDX)
CSV_PREDICATION_DATE = get_csv_date_string(PREDICTION_DATE_IDX)
CSV_HPR_DATE = get_csv_date_string(HPR_DATE_IDX)


with open('./data/prediction.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ('ticker', 'long prob', 'class', '1w', '1d', '*', 'hp', '1w px', '1d px', '* px', 'hp px', '1wr pct', '1dr pct', 'hpr pct','1d v', '1w avg v'))

    for idx in sorted_indexes:
        ticker_idx = t_s_i[idx]
        ticker = idx_to_ticker[ticker_idx]
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

        _week_gross_volume = raw_data[ticker_idx, d_r_i[1:], 3] * raw_data[ticker_idx, d_r_i[1:], 4]
        _last_day_gross_volume = _week_gross_volume[4]
        _week_avg_gross_volume = np.average(_week_gross_volume)



        def get_pct(enter_px, exit_px):
            return (exit_px - enter_px) / enter_px


        _1wr_pct = get_pct(_1w_px, _pred_px)
        _1dr_pct = get_pct(_1d_px, _pred_px)
        _hpr_pct = get_pct(_pred_px, _hp_px)

        writer.writerow((ticker,
                         long_prob,
                         _class,
                         CSV_ONE_WEEK_DATE,
                         CSV_ONE_DAY_DATE,
                         CSV_PREDICATION_DATE,
                         CSV_HPR_DATE,
                         _1w_px,
                         _1d_px,
                         _pred_px,
                         _hp_px,
                         _1wr_pct,
                         _1dr_pct,
                         _hpr_pct,
                         _last_day_gross_volume,
                         _week_avg_gross_volume
                         ))
print("Prediction saved")
