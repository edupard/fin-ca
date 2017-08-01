from tickers import get_nyse_nasdaq_tickers
from download_utils import download_data
from date_range import HIST_BEG, HIST_END
import datetime

# BEG = datetime.datetime.strptime('2017-07-06', '%Y-%m-%d').date()
# END = datetime.datetime.strptime('2017-07-07', '%Y-%m-%d').date()
# BEG = datetime.datetime.strptime('2017-07-08', '%Y-%m-%d').date()
# END = datetime.datetime.strptime('2017-07-16', '%Y-%m-%d').date()
# BEG = datetime.datetime.strptime('2017-07-24', '%Y-%m-%d').date()
# END = datetime.datetime.strptime('2017-07-30', '%Y-%m-%d').date()
BEG = HIST_BEG
END = HIST_END

tickers = get_nyse_nasdaq_tickers()
# tickers = ['AAPL']
download_data(tickers, 'data/prices_right.csv', BEG, END)
