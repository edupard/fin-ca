from tickers import get_nyse_nasdaq_tickers
from download_utils import download_data
from date_range import HIST_BEG, HIST_END
import datetime

# BEG = datetime.datetime.strptime('2017-07-31', '%Y-%m-%d').date()
# END = datetime.datetime.strptime('2017-08-06', '%Y-%m-%d').date()
BEG = HIST_BEG
END = HIST_END


tickers = get_nyse_nasdaq_tickers()
# tickers = ['CSL']
download_data(tickers, 'data/prices.csv', BEG, END)
