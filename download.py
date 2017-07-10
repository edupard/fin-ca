from tickers import get_nyse_nasdaq_tickers
from download_utils import download_data
from date_range import HIST_BEG, HIST_END
import datetime

BEG = datetime.datetime.strptime('2017-07-06', '%Y-%m-%d').date()
END = datetime.datetime.strptime('2017-07-07', '%Y-%m-%d').date()

tickers = get_nyse_nasdaq_tickers()
download_data(tickers, 'data/prices_append.csv', BEG, END)
