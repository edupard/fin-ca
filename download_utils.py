import csv
import datetime
import threading
import queue
from enum import Enum
import numpy as np
import matplotlib

from tiingo import get_historical_data

tickers = []


class PayloadType(Enum):
    DATA = 0
    TASK_COMPLETED = 1


class Payload:
    def __init__(self, payload_type, ticker, payload):
        self.payload_type = payload_type
        self.ticker = ticker
        self.payload = payload


class Writer:
    def __init__(self, FILE_NAME, NUM_WORKERS):
        thread_func = lambda: self.task()
        self.thread = threading.Thread(target=(thread_func))
        self.queue = queue.Queue()
        self.tasks_completed = 0
        self.FILE_NAME = FILE_NAME
        self.NUM_WORKERS = NUM_WORKERS

    def task(self):
        print('downloading data...')
        with open(self.FILE_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            while True:
                p = self.queue.get()
                if p.payload_type == PayloadType.TASK_COMPLETED:
                    self.tasks_completed += 1
                    if self.tasks_completed == self.NUM_WORKERS:
                        break
                elif p.payload_type == PayloadType.DATA:
                    try:
                        for d in p.payload:
                            o = d['open']
                            c = d['close']
                            h = d['high']
                            l = d['low']
                            v = d['volume']
                            dt = d['date'].split("T")[0]
                            t = p.ticker
                            writer.writerow((t, dt, o, c, h, l, v))
                    except:
                        pass
        print('download task completed')

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

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
        for ticker in self.tickers:
            data = get_historical_data(ticker, self.START_DATE, self.END_DATE)
            if data is not None:
                self.writer.write_data(ticker, data)
        self.writer.task_completed()


def parse_tickers(file_name):
    tickers = []
    ticker_to_idx = {}
    idx_to_ticker = {}
    idx = 0

    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            if not header:
                ticker = row[0]
                ticker = ticker.replace(" ", "")
                ticker_to_idx[ticker] = idx
                idx_to_ticker[idx] = ticker
                tickers.append(ticker)
                idx += 1
            else:
                header = False

    return tickers, ticker_to_idx, idx_to_ticker


def download_data(tickers, FILE_NAME, START_DATE, END_DATE, NUM_WORKERS=50):
    writer = Writer(FILE_NAME, NUM_WORKERS)
    writer.start()

    tickers_per_worker = len(tickers) // NUM_WORKERS
    for idx in range(NUM_WORKERS):
        tickers_slice = tickers[idx * tickers_per_worker: min((idx + 1) * tickers_per_worker, len(tickers))]
        worker = Worker(writer, tickers_slice, idx, START_DATE, END_DATE)
        worker.start()

    writer.join()


def preprocess_data(ticker_to_idx, FILE_NAME, START_DATE, END_DATE, DUMP_FILE_NAME):
    print('preprocessing data...')
    num_tickers = len(ticker_to_idx)
    days = (END_DATE - START_DATE).days
    data_points = days + 1

    raw_data = np.zeros((num_tickers, data_points, 5))
    raw_dt = np.zeros((data_points))
    for idx in range(data_points):
        date = START_DATE + datetime.timedelta(days=idx)
        # convert date to datetime
        dt = datetime.datetime.combine(date, datetime.time.min)
        raw_dt[idx] = dt.timestamp()

    with open(FILE_NAME, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # idx = 0
        for row in reader:
            ticker = row[0]
            ticker_idx = ticker_to_idx[ticker]
            dt = datetime.datetime.strptime(row[1], '%Y-%m-%d').date()
            dt_idx = (dt - START_DATE).days
            o = float(row[2])
            c = float(row[3])
            h = float(row[4])
            l = float(row[5])
            v = float(row[6])
            raw_data[ticker_idx, dt_idx, 0] = o
            raw_data[ticker_idx, dt_idx, 1] = h
            raw_data[ticker_idx, dt_idx, 2] = l
            raw_data[ticker_idx, dt_idx, 3] = c
            raw_data[ticker_idx, dt_idx, 4] = v

    np.savez(DUMP_FILE_NAME, raw_dt=raw_dt, raw_data=raw_data)
    print('preprocess task completed')

def load_npz_data(DUMP_FILE_NAME):
    input = np.load(DUMP_FILE_NAME)
    raw_dt = input['raw_dt']
    raw_data = input['raw_data']
    return  raw_dt, raw_data

def convert_to_mpl_time(raw_dt):
    def reduce_time(arr):
        for idx in range(arr.shape[0]):
            dt = datetime.datetime.fromtimestamp(raw_dt[idx])
            yield matplotlib.dates.date2num(dt)

    raw_mpl_dt = np.fromiter(reduce_time(raw_dt), dtype=np.float64)
    return raw_mpl_dt
