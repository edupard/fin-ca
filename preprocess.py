from config import get_config
from tickers import get_nyse_nasdaq_tickers, get_nyse_tickers, get_nasdaq_tickers, get_snp_tickers
from download_utils import preprocess_data


tickers = get_snp_tickers()
preprocess_data(tickers, "data/prices_snp.csv", get_config().HIST_BEG, get_config().HIST_END, "data/snp.npz", get_config().ADJ_PX_FEATURES)

# tickers = get_nasdaq_tickers()
# preprocess_data(tickers, "data/prices.csv", get_config().HIST_BEG, get_config().HIST_END, "data/nasdaq.npz", get_config().ADJ_PX_FEATURES)
#
# tickers = get_nyse_nasdaq_tickers()
# preprocess_data(tickers, "data/prices.csv", get_config().HIST_BEG, get_config().HIST_END, "data/nyse_nasdaq.npz", get_config().ADJ_PX_FEATURES)
#
# tickers = get_nyse_tickers()
# preprocess_data(tickers, "data/prices.csv", get_config().HIST_BEG, get_config().HIST_END, "data/nyse.npz", get_config().ADJ_PX_FEATURES)