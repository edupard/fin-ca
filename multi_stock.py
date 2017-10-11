from portfolio.multi_stock_train import train
from portfolio.multi_stock_config import get_config
from portfolio.net_shiva import NetShiva
from portfolio.snp import get_snp_tickers
import numpy as np
import pandas as pd
import datetime
from download_utils import download_data, load_npz_data, preprocess_data
import os

stocks = [
    'ABT',
    'ARNC',
    'HON',
    'SHW',
    'CMI',
    'EMR',
    'SLB',
    'CSX',
    'CLX',
    'GIS',
    'NEM',
    'MCD',
    'LLY',
    'BAX',
    'BDX',
    'JNJ',
    'GPC',
    'HPQ',
    'WMB',
    'BCR',
    'JPM',
    'IFF',
    'AET',
    'AXP',
    'BAC',
    'CI',
    'DUK',
    'LNC',
    'TAP',
    'NEE',
    'DIS',
    'XRX',
    'IBM',
    'WFC',
    'INTC',
    'TGT',
    'TXT',
    'VFC',
    'WBA',
    'AIG',
    'FLR',
    'FDX',
    'PCAR',
    'ADP',
    'GWW',
    'MAS',
    'ADM',
    'MAT',
    'WMT',
    'SNA',
    'SWK',
    'AAPL',
    'OXY',
    'CAG',
    'LB',
    'T',
    'VZ',
    'LOW',
    'PHM',
    'HES',
    'LMT',
    'HAS',
    'BLL',
    'APD',
    'NUE',
    'PKI',
    'NOC',
    'CNP',
    'TJX',
    'DOV',
    'PH',
    'ITW',
    'GPS',
    'JWN',
    'MDT',
    'HRB',
    'SYY',
    'CA',
    'MMC',
    'AVY',
    'HD',
    'PNC',
    'C',
    'STI',
    'NKE',
    'ECL',
    'NWL',
    'TMK',
    'ORCL',
    'ADSK',
    'MRO',
    'AEE',
    'AMGN',
    'PX',
    'IPG',
    'COST',
    'CSCO',
    'EMN',
    'KEY',
    'UNM',
    'MSFT',
    'LUV',
    'UNH',
    'CBS',
    'MU',
    'BSX',
    'ADBE',
    'EFX',
    'PGR',
    'YUM',
    'RF',
    'SPLS',
    'NTAP',
    'BBY',
    'VMC',
    'XLNX',
    'A',
    'TIF',
    'DVN',
    'EOG',
    'INTU',
    'RHI',
    'SYK',
    'COP'
]

stocks = get_snp_tickers()

def create_folders():
    if not os.path.exists(get_config().DATA_FOLDER_PATH):
        os.makedirs(get_config().DATA_FOLDER_PATH)

net = NetShiva()
for stock in stocks:
    print("Processing %s stock" % stock)
    get_config().TICKER = stock
    get_config().reload()

    _tickers = [stock]
    create_folders()
    download_data(_tickers,
                  get_config().DATA_PATH,
                  get_config().HIST_BEG,
                  get_config().HIST_END)
    preprocess_data(_tickers,
                    get_config().DATA_PATH,
                    get_config().HIST_BEG,
                    get_config().HIST_END,
                    get_config().DATA_NPZ_PATH,
                    get_config().DATA_FEATURES)
    train(net)


