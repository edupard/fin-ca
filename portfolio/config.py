import datetime

class Config(object):

    HIST_BEG = datetime.datetime.strptime('1999-01-01', '%Y-%m-%d').date()
    HIST_END = datetime.datetime.strptime('2017-08-27', '%Y-%m-%d').date()

_config = Config()

def get_config() -> Config:
    return _config