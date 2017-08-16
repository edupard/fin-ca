import datetime

from enum import Enum

_5_USD = 5.0
_ZERO = 0
_10M = 10000000
YR_90 = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d').date()
YR_00 = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
YR_10 = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
YR_15 = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()

class SelectionAlgo(Enum):
    TOP = 0
    BOTTOM = 1
    MIDDLE = 2
    MIDDLE_ALT = 3


class SelectionType(Enum):
    PCT = 0
    FIXED = 1


class StopLossType(Enum):
    NO = 0
    EOD = 1
    LB = 2
    STOCK = 3


class Config(object):

    HIST_BEG = datetime.datetime.strptime('1990-01-01', '%Y-%m-%d').date()
    HIST_END = datetime.datetime.strptime('2017-08-06', '%Y-%m-%d').date()

    NUM_WEEKS = 12
    NUM_DAYS = 5

    ENT_ON_MON = False
    ENT_MON_OPEN = True
    EXIT_ON_MON = False
    EXIT_ON_MON_OPEN = True

    SLCT_PCT = 100
    SLCT_ALG = SelectionAlgo.TOP

    MIN_STOCKS_TRADABLE_PER_TRADING_DAY = 30

    # train period
    TRAIN_BEG = YR_00
    TRAIN_END = YR_10

    # filters
    MIN_SELECTION_STOCKS = None
    AVG_DAY_TO_LIMIT = _10M
    TOP_TRADABLE_STOCKS = None
    FILTER_BY_DAY_TO = False
    DAY_TO_LIMIT = _10M
    CLOSE_PX_FILTER = _5_USD

    # cv params
    CV_BEG = YR_10
    CV_END = HIST_END

    SLCT_TYPE = SelectionType.PCT
    SLCT_VAL = 1

    STOP_LOSS_HPR = -0.12
    STOP_LOSS_TYPE = StopLossType.NO

    GRID_SEARCH = True

    USE_DROP_OUT = True

    PRINT_PORTFOLIO = False

_config = Config()

def get_config() -> Config:
    return _config