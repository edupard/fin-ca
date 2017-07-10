from date_range import HIST_BEG, HIST_END
from tickers import get_nyse_nasdaq_tickers, get_nyse_tickers, get_nasdaq_tickers
from download_utils import preprocess_data_alt


tickers = get_nasdaq_tickers()
preprocess_data_alt(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nasdaq.npz", False)
preprocess_data_alt(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nasdaq_adj.npz", True)

tickers = get_nyse_tickers()
preprocess_data_alt(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nyse.npz", False)
preprocess_data_alt(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nyse_adj.npz", True)

tickers = get_nyse_nasdaq_tickers()
preprocess_data_alt(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nyse_nasdaq.npz", False)
preprocess_data_alt(tickers, "data/prices.csv", HIST_BEG, HIST_END, "data/nyse_nasdaq_adj.npz", True)
