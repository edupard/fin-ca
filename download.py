from tickers import get_nyse_nasdaq_tickers
from download_utils import download_data
from config import get_config

BEG = get_config().HIST_BEG
END = get_config().HIST_END

tickers = get_nyse_nasdaq_tickers()
download_data(tickers, 'data/prices.csv', BEG, END)
