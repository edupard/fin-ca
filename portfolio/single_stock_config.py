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
TODAY = datetime.datetime.strptime('2017-08-27', '%Y-%m-%d').date()


class Config(object):
    TICKER = 'MSFT'

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

    LSTM_LAYERS_SIZE = [15, 15, 15]
    # LSTM_LAYERS_SIZE = [5, 5, 5]
    FC_LAYERS_SIZE = [30]

    TRAIN_BEG = YR_00
    TRAIN_END = YR_07

    TEST_BEG = YR_07
    TEST_END = HIST_END

    MODE = Mode.TEST
    EPOCH_WEIGHTS_TO_LOAD = 410

    BPTT_STEPS = 20
    PRED_HORIZON = 5
    REBALANCE_FREQ = 5

    WEIGHTS_FOLDER_PATH = 'nets/portfolio/stocks/%s' % TICKER
    TRAIN_STAT_PATH = '%s/train_stat.csv' % WEIGHTS_FOLDER_PATH
    WEIGHTS_PATH = '%s/weights' % WEIGHTS_FOLDER_PATH

    DATA_FOLDER_PATH = 'data/stocks/%s' % TICKER
    DATA_PATH = '%s/%s.csv' % (DATA_FOLDER_PATH, TICKER)
    DATA_NPZ_PATH = '%s/%s.npz' % (DATA_FOLDER_PATH, TICKER)

    TRAIN_FIG_PATH = 'data/stocks/%s/eq/train' % TICKER
    TEST_FIG_PATH = 'data/stocks/%s/eq/test' % TICKER

    SAVE_EQ = True

    MAX_EPOCH = 500

    RESET_PRED_PX_EACH_N_DAYS = 200

_config = Config()


def get_config() -> Config:
    return _config
