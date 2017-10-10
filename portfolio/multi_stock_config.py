import datetime
from enum import Enum


class Mode(Enum):
    TRAIN = 0
    TEST = 1


YR_90 = datetime.datetime.strptime('1990-01-01', '%Y-%m-%d').date()
YR_00 = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
YR_07 = datetime.datetime.strptime('2007-01-01', '%Y-%m-%d').date()
YR_10 = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()
YR_15 = datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date()

TODAY = datetime.datetime.strptime('2017-10-08', '%Y-%m-%d').date()


class Config(object):
    TICKER = 'HPQ'

    HIST_BEG = YR_90
    HIST_END = TODAY

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

    # LSTM_LAYERS_SIZE = [50, 50, 50, 50, 50]
    # LSTM_LAYERS_SIZE = [15, 15, 15]
    LSTM_LAYERS_SIZE = [30, 30, 30, 30]
    # LSTM_LAYERS_SIZE = [5, 5, 5]
    FC_LAYERS_SIZE = [30]

    TRAIN_BEG = YR_07
    TRAIN_END = TODAY
    # TRAIN_END = YR_07

    TEST_BEG = TRAIN_END
    TEST_END = HIST_END

    # MODE = Mode.TEST
    # EPOCH_WEIGHTS_TO_LOAD = 1000
    MODE = Mode.TRAIN
    EPOCH_WEIGHTS_TO_LOAD = 0
    MAX_EPOCH = 600

    BPTT_STEPS = 1000
    PRED_HORIZON = 5
    REBALANCE_FRI = True
    REBALANCE_FREQ = 5
    FIT_FRI_PREDICTION_ONLY = True

    CAPM = False
    CAPM_USE_NET_PREDICTIONS = True
    COVARIANCE_LENGTH = 60

    TEST = False

    PREDICTION_MODE = False

    def __init__(self):
        self.reload()

    def reload(self):
        self.WEIGHTS_FOLDER_PATH = 'nets/portfolio/stocks/%s' % self.TICKER
        self.TRAIN_STAT_PATH = '%s/train_stat.csv' % self.WEIGHTS_FOLDER_PATH
        self.WEIGHTS_PATH = '%s/weights' % self.WEIGHTS_FOLDER_PATH

        if self.PREDICTION_MODE:
            self.DATA_FOLDER_PATH = 'data/stocks/temp/%s' % self.TICKER
        else:
            self.DATA_FOLDER_PATH = 'data/stocks/%s' % self.TICKER

        self.DATA_PATH = '%s/%s.csv' % (self.DATA_FOLDER_PATH, self.TICKER)
        self.DATA_NPZ_PATH = '%s/%s.npz' % (self.DATA_FOLDER_PATH, self.TICKER)

        self.TRAIN_FIG_PATH = 'data/stocks/%s/eq/train' % self.TICKER
        self.TEST_FIG_PATH = 'data/stocks/%s/eq/test' % self.TICKER

        self.TRAIN_EQ_PATH = 'data/stocks/%s/eq/train/eq.csv' % self.TICKER
        self.TEST_EQ_PATH = 'data/stocks/%s/eq/test/eq.csv' % self.TICKER

        if self.TICKER == 'SNP_IND':
            self.MIN_STOCKS_TRADABLE_PER_TRADING_DAY = 30
        else:
            self.MIN_STOCKS_TRADABLE_PER_TRADING_DAY = 1


_config = Config()


def get_config() -> Config:
    return _config
