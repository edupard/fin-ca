import os

from download_utils import download_data, preprocess_data
from portfolio.single_stock_config import get_config


def create_folders():
    if not os.path.exists(get_config().DATA_FOLDER_PATH):
        os.makedirs(get_config().DATA_FOLDER_PATH)

def download_px():
    create_folders()
    tickers = get_tickers()
    download_data(tickers, get_config().DATA_PATH, get_config().HIST_BEG, get_config().HIST_END)


def preprocess_px():
    create_folders()
    tickers = get_tickers()
    preprocess_data(tickers, get_config().DATA_PATH, get_config().HIST_BEG, get_config().HIST_END,
                    get_config().DATA_NPZ_PATH,
                    get_config().DATA_FEATURES)


def get_tickers():
    return [get_config().TICKER]
