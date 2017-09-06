from download_utils import download_data, parse_tickers, load_npz_data_old, load_npz_data, preprocess_data
from data_utils import filter_activelly_tradeable_stocks, get_dates_for_daily_return, get_dates_for_weekly_return, \
    get_tradable_stock_indexes, get_prices, PxType, calc_z_score, get_data_idx, calc_z_score_alt
from tickers import get_nasdaq_tickers
import datetime
import numpy as np
import csv

from nn import ffnn_instance

NUM_WEEKS = 12

# YYYY-MM-DD
PREDICTION_DATE = datetime.datetime.strptime('2017-08-04', '%Y-%m-%d').date()
HPR_DATE = datetime.datetime.strptime('2017-08-04', '%Y-%m-%d').date()


START_DATE = PREDICTION_DATE - datetime.timedelta(days=(NUM_WEEKS + 2) * 7)
END_DATE = PREDICTION_DATE

tickers = get_nasdaq_tickers()
download_data(tickers, 'data/px_hist.csv', START_DATE, END_DATE)