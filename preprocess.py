from date_range import HIST_BEG, HIST_END
from tickers import get_nyse_nasdaq_tickers, get_nyse_tickers, get_nasdaq_tickers
from download_utils import preprocess_data


tickers = get_nasdaq_tickers()
preprocess_data(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nasdaq.npz", True)

tickers = get_nyse_tickers()
preprocess_data(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nyse.npz", True)

tickers = get_nyse_nasdaq_tickers()
preprocess_data(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nyse_nasdaq.npz", True)
