import csv
import datetime

from tickers import get_nyse_nasdaq_tickers, get_nyse_tickers, get_nasdaq_tickers

DATE_BEG = datetime.datetime.strptime('2010-03-18', '%Y-%m-%d').date()
DATE_END = datetime.datetime.strptime('2010-03-26', '%Y-%m-%d').date()
tickers = ['CTIC']

# DATE_BEG = datetime.datetime.strptime('2017-03-24', '%Y-%m-%d').date()
# DATE_END = datetime.datetime.strptime('2017-06-30', '%Y-%m-%d').date()
# tickers = get_nasdaq_tickers()
tickers_set = set(tickers)

with open('data/prices_filtered.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    with open('data/prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ticker = row[0]
            dt = datetime.datetime.strptime(row[1], '%Y-%m-%d').date()
            if dt >= DATE_BEG and dt <= DATE_END and ticker in tickers_set:
                writer.writerow(row)
