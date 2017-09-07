import datetime
from enum import Enum

class TradingFrequency(Enum):
    DAILY = 0
    WEEKLY = 1

class NetVersion(Enum):
    APPLE = 0
    BANANA = 1
    WORM = 2
    SNAKE = 3

class Mode(Enum):
    TRAIN = 0
    TEST = 1

YR_90 = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d').date()
YR_00 = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
YR_07 = datetime.datetime.strptime('2007-01-01', '%Y-%m-%d').date()
YR_10 = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
YR_15 = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()


class Config(object):

    HIST_BEG = datetime.datetime.strptime('1999-01-01', '%Y-%m-%d').date()
    HIST_END = datetime.datetime.strptime('2017-08-27', '%Y-%m-%d').date()

    DATA_FEATURES = ['o', 'h', 'l', 'c', 'v', 'a_o', 'a_h', 'a_l', 'a_c', 'a_v']

    OPEN_DATA_IDX = 0
    HIGH_DATA_IDX = 1
    LOW_DATA_IDX = 2
    CLOSE_DATA_IDX = 3
    VOLUME_DATA_IDX = 4
    ADJ_OPEN_DATA_IDX = 5
    ADJ_HIGH_DATA_IDX = 6
    ADJ_LOW_DATA_IDX = 7
    ADJ_CLOSE_DATA_IDX = 8
    ADJ_VOLUME_DATA_IDX = 9

    MIN_STOCKS_TRADABLE_PER_TRADING_DAY = 30

    LOG_RET = False
    RET_MUL = 1
    LOG_VOL_CHG = False
    VOL_CHG_MUL = 1

    RNN_HISTORY = datetime.timedelta(days=20)
    TRADING_PERIOD_DAYS = 1

    LSTM_LAYERS_SIZE = [5,5,5]
    FC_LAYERS_SIZE = [30]

    TRAIN_BEG = YR_00
    TRAIN_END = YR_07

    SHUFFLE = True

    TEST_BEG = YR_07
    TEST_END = HIST_END

    MIN_PARTITION_Z = 1e-6

    NET_VER = NetVersion.APPLE
    TRAIN_STAT_PATH = 'nets/portfolio/%s/train_stat.csv' % NET_VER.name
    WEIGHTS_PATH = 'nets/portfolio/%s/weights' % NET_VER.name

    PRINT_PREDICTION = True

    BATCH_NORM = True

    MODE = Mode.TEST
    EPOCH_WEIGHTS_TO_LOAD = 94

    COVARIANCE_LENGTH = 20


_config = Config()

def get_config() -> Config:
    return _config