from tickers import get_nyse_nasdaq_tickers
from download_utils import download_data
import datetime

START_DATE = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
END_DATE = datetime.datetime.strptime('2017-04-18', '%Y-%m-%d').date()


HIST_BEG = datetime.datetime.strptime('1990-01-01', '%Y-%m-%d').date()
HIST_END = datetime.datetime.strptime('2017-07-05', '%Y-%m-%d').date()

tickers = get_nyse_nasdaq_tickers()
download_data(tickers, 'data/prices.csv', HIST_BEG, HIST_END)
