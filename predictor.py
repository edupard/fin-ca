from download_utils import download_data, parse_tickers, preprocess_data
import datetime

NUM_WEEKS = 12
DAYS_TO_DOWNLOAD_DATA = datetime.timedelta(days=(NUM_WEEKS + 1) * 7)
ONE_DAY = datetime.timedelta(days=1)

END_DATE = datetime.datetime.today().date() - ONE_DAY
START_DATE = END_DATE - DAYS_TO_DOWNLOAD_DATA

tickers, ticker_to_idx, idx_to_ticker = parse_tickers('tickers_nasdaq.csv')
# download_data(tickers, 'history.csv', START_DATE, END_DATE, 50)
preprocess_data(ticker_to_idx, 'history.csv', START_DATE, END_DATE, 'history.npz')