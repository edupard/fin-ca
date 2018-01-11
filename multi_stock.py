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
'A',
'AAPL',
'ABBV',
'ABC',
'ABT',
'ACN',
'ADBE',
'ADI',
'ADM',
'ADP',
'ADS',
'ADSK',
'AEE',
'AEP',
'AES',
'AET',
'AFL',
'AGN',
'AIG',
'AIV',
'AIZ',
'AKAM',
'ALL',
'ALLE',
'ALXN',
'AMAT',
'AME',
'AMG',
'AMGN',
'AMP',
'AMT',
'AMZN',
'ANDV',
'ANTM',
'AON',
'APA',
'APC',
'APD',
'APH',
'ARNC',
'AVB',
'AVGO',
'AVY',
'AXP',
'AZO',
'BA',
'BAC',
'BAX',
'BBT',
'BBY',
'BCR',
'BDX',
'BEN',
'BF-B',
'BHGE',
'BIIB',
'BK',
'BLK',
'BLL',
'BMY',
'BRK-B',
'BSX',
'BWA',
'BXP',
'C',
'CA',
'CAG',
'CAH',
'CAT',
'CB',
'CBG',
'CBS',
'CCI',
'CCL',
'CELG',
'CERN',
'CF',
'CHK',
'CHRW',
'CI',
'CINF',
'CL',
'CLX',
'CMA',
'CME',
'CMG',
'CMI',
'CMS',
'CNP',
'COF',
'COG',
'COH',
'COL',
'COP',
'COST',
'CPB',
'CRM',
'CSCO',
'CSX',
'CTAS',
'CTL',
'CTSH',
'CTXS',
'CVS',
'CVX',
'D',
'DAL',
'DE',
'DFS',
'DG',
'DGX',
'DHI',
'DHR',
'DIS',
'DISCA',
'DISCK',
'DLPH',
'DLTR',
'DOV',
'DOW',
'DPS',
'DRI',
'DTE',
'DUK',
'DVA',
'DVN',
'EA',
'EBAY',
'ECL',
'ED',
'EFX',
'EIX',
'EL',
'EMN',
'EMR',
'EOG',
'EQR',
'EQT',
'ES',
'ESRX',
'ESS',
'ETFC',
'ETN',
'ETR',
'EW',
'EXC',
'EXPD',
'EXPE',
'F',
'FAST',
'FB',
'FCX',
'FDX',
'FE',
'FFIV',
'FIS',
'FISV',
'FITB',
'FLIR',
'FLR',
'FLS',
'FMC',
'FOXA',
'FTI',
'GD',
'GE',
'GGP',
'GILD',
'GIS',
'GLW',
'GM',
'GOOG',
'GOOGL',
'GPC',
'GPS',
'GRMN',
'GS',
'GT',
'GWW',
'HAL',
'HAS',
'HBAN',
'HCA',
'HCN',
'HCP',
'HD',
'HES',
'HIG',
'HOG',
'HON',
'HP',
'HPQ',
'HRB',
'HRL',
'HRS',
'HST',
'HSY',
'HUM',
'IBM',
'ICE',
'IFF',
'INTC',
'INTU',
'IP',
'IPG',
'IR',
'IRM',
'ISRG',
'ITW',
'IVZ',
'JCI',
'JEC',
'JNJ',
'JNPR',
'JPM',
'JWN',
'K',
'KEY',
'KIM',
'KLAC',
'KMB',
'KMI',
'KMX',
'KO',
'KORS',
'KR',
'KSS',
'KSU',
'L',
'LB',
'LEG',
'LEN',
'LH',
'LLL',
'LLY',
'LMT',
'LNC',
'LOW',
'LRCX',
'LUK',
'LUV',
'LVLT',
'LYB',
'M',
'MA',
'MAC',
'MAR',
'MAS',
'MAT',
'MCD',
'MCHP',
'MCK',
'MCO',
'MDLZ',
'MDT',
'MET',
'MHK',
'MKC',
'MLM',
'MMC',
'MMM',
'MNST',
'MO',
'MON',
'MOS',
'MPC',
'MRK',
'MRO',
'MS',
'MSFT',
'MSI',
'MTB',
'MU',
'MYL',
'NAVI',
'NBL',
'NDAQ',
'NEE',
'NEM',
'NFLX',
'NFX',
'NI',
'NKE',
'NLSN',
'NOC',
'NOV',
'NRG',
'NSC',
'NTAP',
'NTRS',
'NUE',
'NVDA',
'NWL',
'NWSA',
'OKE',
'OMC',
'ORCL',
'ORLY',
'OXY',
'PAYX',
'PBCT',
'PCAR',
'PCG',
'PCLN',
'PDCO',
'PEG',
'PEP',
'PFE',
'PFG',
'PG',
'PGR',
'PH',
'PHM',
'PKI',
'PLD',
'PM',
'PNC',
'PNR',
'PNW',
'PPG',
'PPL',
'PRGO',
'PRU',
'PSA',
'PSX',
'PVH',
'PWR',
'PX',
'PXD',
'QCOM',
'RCL',
'REGN',
'RF',
'RHI',
'RHT',
'RL',
'ROK',
'ROP',
'ROST',
'RRC',
'RSG',
'RTN',
'SBUX',
'SCG',
'SCHW',
'SEE',
'SHW',
'SJM',
'SLB',
'SNA',
'SNI',
'SO',
'SPG',
'SPGI',
'SPLS',
'SRCL',
'SRE',
'STI',
'STT',
'STX',
'STZ',
'SWK',
'SYK',
'SYMC',
'SYY',
'T',
'TAP',
'TEL',
'TGT',
'TIF',
'TJX',
'TMK',
'TMO',
'TRIP',
'TROW',
'TRV',
'TSCO',
'TSN',
'TSS',
'TWX',
'TXN',
'TXT',
'UA',
'UHS',
'UNH',
'UNM',
'UNP',
'UPS',
'URI',
'USB',
'UTX',
'V',
'VAR',
'VFC',
'VIAB',
'VLO',
'VMC',
'VNO',
'VRSN',
'VRTX',
'VTR',
'VZ',
'WAT',
'WBA',
'WDC',
'WEC',
'WFC',
'WHR',
'WM',
'WMB',
'WMT',
'WRK',
'WU',
'WY',
'WYN',
'WYNN',
'XEC',
'XEL',
'XL',
'XLNX',
'XOM',
'XRAY',
'XRX',
'XYL',
'YUM',
'ZBH',
'ZION',
'ZTS',
]

stocks = [
'WLTW',
'CHD',
'CSRA',
'ILMN',
'SYF',
'HPE',
'VRSK',
'FOX',
'NWS',
'CMCSA',
'UAL',
'ATVI',
'SIG',
'PYPL',
'AAP',
'KHC',
'JBHT',
'QRVO',
'O',
'AAL',
'SLG',
'HBI',
'EQIX',
'HSIC',
'SWKS',
]

# stocks = get_snp_tickers()

# 2015-10-30
# stocks = ['HPQ']
# stocks = ['BAC']

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


