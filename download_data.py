from download_utils import download_data, parse_tickers, preprocess_data
import datetime

START_DATE = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
END_DATE = datetime.datetime.strptime('2017-04-18', '%Y-%m-%d').date()

tickers, ticker_to_idx, idx_to_ticker  = parse_tickers('tickers_nasdaq.csv')
download_data(tickers, 'history_nasdaq.csv', START_DATE, END_DATE)
preprocess_data(ticker_to_idx, 'history_nasdaq.csv', START_DATE, END_DATE, 'history_nasdaq.npz')
