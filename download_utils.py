import csv
from yahoo_finance import Share
import datetime
import threading
import queue
from enum import Enum
import random

from tiingo import get_historical_data

tickers = []

ONE_DAY = datetime.timedelta(days=1)
BATCH_DOWNLOAD_DURATION = datetime.timedelta(days=100 * 365)
# BATCH_DOWNLOAD_DURATION = datetime.timedelta(days=30)

WORKERS = 300


class PayloadType(Enum):
    DATA = 0
    TASK_COMPLETED = 1


class Payload:
    def __init__(self, payload_type, ticker, payload):
        self.payload_type = payload_type
        self.ticker = ticker
        self.payload = payload


class Writer:
    def __init__(self, FILE_NAME):
        thread_func = lambda: self.task()
        self.thread = threading.Thread(target=(thread_func))
        self.queue = queue.Queue()
        self.tasks_completed = 0
        self.FILE_NAME = FILE_NAME

    def task(self):
        with open(self.FILE_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            while True:
                p = self.queue.get()
                if p.payload_type == PayloadType.TASK_COMPLETED:
                    self.tasks_completed += 1
                    if self.tasks_completed == WORKERS:
                        break
                elif p.payload_type == PayloadType.DATA:
                    try:
                        for d in p.payload:
                            o = d['open']
                            c = d['close']
                            h = d['high']
                            l = d['low']
                            v = d['volume']
                            dt = d['date']
                            t = p.ticker
                            writer.writerow((t, dt, o, c, h, l, v))
                    except:
                        pass
        print('write task completed')

    def start(self):
        self.thread.start()

    def write_data(self, ticker, data):
        self.queue.put(Payload(PayloadType.DATA, ticker, data))

    def task_completed(self):
        self.queue.put(Payload(PayloadType.TASK_COMPLETED, None, None))


class Worker:
    def __init__(self, writer, tickers, idx, START_DATE, END_DATE):
        self.idx = idx
        self.writer = writer
        self.tickers = tickers
        thread_func = lambda: self.task()
        self.thread = threading.Thread(target=(thread_func))
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE

    def start(self):
        self.thread.start()

    def task(self):
        shares = {}
        batch_beg = self.START_DATE
        last_year = 0
        while True:
            if batch_beg.year != last_year:
                last_year = batch_beg.year
                print('{} task - processing {} year'.format(self.idx, last_year))
            # break when all data gathered
            if batch_beg > self.END_DATE:
                break
            # calc batch end
            batch_end = batch_beg + BATCH_DOWNLOAD_DURATION - ONE_DAY
            # set batch end greater than today
            if batch_end > self.END_DATE:
                batch_end = self.END_DATE
            for ticker in self.tickers:
                data = get_historical_data(ticker, batch_beg, batch_end)
                if data is not None:
                    self.writer.write_data(ticker, data)

            batch_beg = batch_end + ONE_DAY
        print('{} task completed'.format(self.idx))
        self.writer.task_completed()


def download_data(FILE_NAME, START_DATE, END_DATE, NUM_WORKERS=300):
    with open('nasdaq_tickers.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            if not header:
                ticker = row[0]
                tickers.append(ticker.strip())
            else:
                header = False
    print(tickers)
    random.shuffle(tickers)

    writer = Writer(FILE_NAME)
    writer.start()

    tickers_per_worker = len(tickers) // NUM_WORKERS
    for idx in range(NUM_WORKERS):
        tickers_slice = tickers[idx * tickers_per_worker: min((idx + 1) * tickers_per_worker, len(tickers))]
        worker = Worker(writer, tickers_slice, idx, START_DATE, END_DATE)
        worker.start()
