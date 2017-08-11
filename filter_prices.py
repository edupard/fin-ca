import csv
import datetime

from config import get_config

DATE_BEG = datetime.datetime.strptime('2010-03-18', '%Y-%m-%d').date()
DATE_END = datetime.datetime.strptime('2010-03-26', '%Y-%m-%d').date()
tickers = ['EVBN']

DATE_BEG = get_config().HIST_BEG
DATE_END = get_config().HIST_END

# tickers = get_nasdaq_tickers()
tickers_set = set(tickers)

with open('data/prices_filtered.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    with open('data/prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            if header:
                writer.writerow(row)
                header = False
                continue
            ticker = row[0]
            dt = datetime.datetime.strptime(row[1], '%Y-%m-%d').date()
            if dt >= DATE_BEG and dt <= DATE_END and ticker in tickers_set:
                writer.writerow(row)
