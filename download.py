import datetime

from tickers import get_nyse_nasdaq_tickers, get_snp_tickers
from download_utils import download_data
from config import get_config

# BEG = get_config().HIST_BEG
# END = get_config().HIST_END

# tickers = get_snp_tickers()
# download_data(tickers, 'data/prices_snp.csv', BEG, END)

# tickers = get_nyse_nasdaq_tickers()
# download_data(tickers, 'data/prices.csv', BEG, END)


BEG = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d').date()
END = BEG

# tickers = [
# 'A',
# 'ABBV',
# 'ABT',
# 'ALK',
# 'ALL',
# 'AMD',
# 'AMT',
# 'APA',
# 'APC',
# 'BXP',
# 'CMA',
# 'COP',
# 'EXPE',
# 'FCX',
# 'GGP',
# 'GIS',
# 'GPS',
# 'GWW',
# 'HD',
# 'HRL',
# 'HRS',
# 'INCY',
# 'IP',
# 'JBHT',
# 'KLAC',
# 'LRCX',
# 'LUV',
# 'LVLT',
# 'MCHP',
# 'MS',
# 'NEM',
# 'OKE',
# 'PGR',
# 'PRGO',
# 'RMD',
# 'RRC',
# 'STI',
# 'SYMC',
# 'TAP',
# 'TRV',
# 'TSS',
# 'ULTA',
# 'URI',
# 'VAR',
# 'VIAB',
# 'XL']

tickers = [
'A',
'ABBV',
'ABT',
'ALK',
'ALL',
'AMD',
'AMT',
'APA',
'APC',
'BBY',
'CMA',
'COP',
'EXPE',
'FCX',
'GGP',
'GIS',
'GPS',
'GWW',
'HD',
'HRS',
'INCY',
'IP',
'JBHT',
'KLAC',
'LRCX',
'LUV',
'LVLT',
'MCHP',
'MYL',
'NEM',
'OKE',
'PGR',
'PRGO',
'RMD',
'RRC',
'SEE',
'SWKS',
'SYMC',
'TAP',
'TRV',
'TSS',
'ULTA',
'URI',
'VAR',
'XL',
'XYL'
]

download_data(tickers, 'data/port_mod_px.csv', BEG, END)
