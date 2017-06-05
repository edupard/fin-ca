from download_utils import download_data
import datetime

START_DATE = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date()
END_DATE = datetime.datetime.strptime('2017-04-18', '%Y-%m-%d').date()

download_data('test.csv', START_DATE, END_DATE, 300)