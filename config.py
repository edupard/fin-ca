import datetime

from enum import Enum

_5_USD = 5.0
_ZERO = 0
_10M = 10000000
_5M = 5000000
YR_90 = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d').date()
YR_00 = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
YR_07 = datetime.datetime.strptime('2007-01-01', '%Y-%m-%d').date()
YR_10 = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
YR_15 = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()

class SelectionAlgo(Enum):
    CONFIRMED = 0
    NON_CONFIRMED = 1
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

    ENT_ON_MON = True
    ENT_MON_OPEN = False
    EXIT_ON_MON = False
    EXIT_ON_MON_OPEN = True

    SLCT_PCT = 50
    SLCT_ALG = SelectionAlgo.NON_CONFIRMED

    MIN_STOCKS_TRADABLE_PER_TRADING_DAY = 30

    # train period
    TRAIN_BEG = YR_00
    TRAIN_END = YR_07

    # train filters
    MIN_SELECTION_STOCKS = None
    AVG_DAY_TO_LIMIT = None
    TOP_TRADABLE_STOCKS = None
    DAY_TO_LIMIT = None
    CLOSE_PX_FILTER = None
    SNP_FILTER = True

    # cv filters
    MIN_SELECTION_STOCKS_CV = None
    AVG_DAY_TO_LIMIT_CV = None
    TOP_TRADABLE_STOCKS_CV = None
    DAY_TO_LIMIT_CV = None
    CLOSE_PX_FILTER_CV = None
    SNP_FILTER_CV = True

    # cv params
    CV_BEG = TRAIN_END
    CV_END = HIST_END

    SLCT_TYPE = SelectionType.FIXED
    SLCT_VAL = 15

    STOP_LOSS_HPR = -0.0
    STOP_LOSS_TYPE = StopLossType.NO

    GRID_SEARCH = False

    USE_DROP_OUT = False

    PRINT_PORTFOLIO = True

    LONG_LEG = False
    SHORT_LEG = True

_config = Config()

def get_config() -> Config:
    return _config