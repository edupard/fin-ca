from portfolio.single_stock_data import download_px, preprocess_px
from portfolio.single_stock_train import train
from portfolio.net_turtle import NetTurtle
from portfolio.single_stock_config import get_config

net = NetTurtle()

stocks =[
    # 'ABT',
    # 'ARNC',
    # 'HON',
    # 'SHW',
    # 'CMI',
    # 'EMR',
    # 'SLB',
    # 'CSX',
    # 'CLX',
    # 'GIS',
    # 'NEM',
    # 'MCD',
    # 'LLY',
    # 'BAX',
    # 'BDX',
    # 'JNJ',
    # 'GPC',
    # 'HPQ',
    # 'WMB',
    # 'BCR',
    # 'JPM',
    # 'IFF',
    # 'AET',
    # 'AXP',
    # 'BAC',
    # 'CI',
    # 'DUK',
    # 'LNC',
    # 'TAP',
    # 'NEE',
    # 'DIS',

    # 'XRX',
    'IBM',
    #
    # 'WFC',
    # 'INTC',
    # 'TGT',
    # 'TXT',
    # 'VFC',
    # 'WBA',
    # 'AIG',
    # 'FLR',
    # 'FDX',
    # 'PCAR',
    # 'ADP',
    # 'GWW',
    # 'MAS',
    # 'ADM',
    # 'MAT',
    # 'WMT',
    # 'SNA',
    # 'SWK',
    # 'BF-B',
    # 'AAPL',
    # 'OXY',
    # 'CAG',
    # 'LB',
    # 'T',
    # 'VZ',
    # 'LOW',
    # 'PHM',
    # 'HES',
    # 'LMT',
    # 'HAS',
    # 'BLL',
    # 'APD',
    # 'NUE',
    # 'PKI',
    # 'NOC',
    # 'CNP',
    # 'TJX',
    # 'DOV',
    # 'PH',
    # 'ITW',
    # 'GPS',
    # 'JWN',
    # 'MDT',
    # 'HRB',
    # 'SYY',
    # 'CA',
    # 'MMC',
    # 'AVY',
    # 'HD',
    # 'PNC',
    # 'C',
    # 'STI',
    # 'NKE',
    # 'ECL',
    # 'NWL',
    # 'TMK',
    # 'ORCL',
    # 'ADSK',
    # 'MRO',
    # 'AEE',
    # 'AMGN',
    # 'PX',
    # 'IPG',
    # 'COST',
    # 'CSCO',
    # 'EMN',
    # 'KEY',
    # 'UNM',
    # 'MSFT',
    # 'LUV',
    # 'UNH',
    # 'CBS',
    # 'MU',
    # 'BSX',
    # 'ADBE',
    # 'EFX',
    # 'PGR',
    # 'YUM',
    # 'RF',
    # 'SPLS',
    # 'NTAP',
    # 'BBY',
    # 'VMC',
    # 'XLNX',
    # 'A',
    # 'TIF',
    # 'DVN',
    # 'EOG',
    # 'INTU',
    # 'RHI',
    # 'SYK'
]

for stock in stocks:
    get_config().TICKER = stock
    get_config().reload()
    try:
        # download_px()
        # preprocess_px()
        train(net)
    except:
        pass