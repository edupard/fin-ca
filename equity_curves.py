from portfolio.single_stock_data import download_px, preprocess_px
from portfolio.single_stock_train import train, plot_equity_curve
from portfolio.stat import get_draw_down, get_sharpe_ratio
from portfolio.net_turtle import NetTurtle
from portfolio.single_stock_config import get_config
from portfolio.graphs import show_plots, plot_two_equity_curves, draw_grid, format_time_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

DDMMMYY_FMT = matplotlib.dates.DateFormatter('%y %b %d')
YYYY_FMT = matplotlib.dates.DateFormatter('%Y')

curves = [
    # ('data/eq/eq_70_30_net.csv','%d.%m.%Y', '70x30 rbm net mon-fri','b'),
    # ('data/eq/eq_70_30_no_net.csv','%d.%m.%Y', '70x30 no net mon-fri','r'),

    # ('data/eq/eq_70_30_net_sl_5_bp.csv','%d.%m.%Y', '70x30 rbm net mon-fri','b'),
    # ('data/eq/eq_70_30_no_net_sl_5_bp.csv','%d.%m.%Y', '70x30 no net mon-fri','r'),

    # ('data/eq/eq_70_30_net_5d.csv','%Y-%m-%d', '70x30 l/s softmax net 1d-4d ','c'),
    # ('data/eq/eq_70_30_no_net_5d.csv','%Y-%m-%d', '70x30 no net 1d-5d','k'),
    # ('data/eq/eq_70_30_no_net_1d.csv','%Y-%m-%d', '70x30 no net 1d','c'),


    # ('data/eq/eq_long_softmax.csv','%Y-%m-%d', 'long softmax','g'),
    # ('data/eq/eq_short_softmax.csv','%Y-%m-%d', 'short softmax','r'),
    # ('data/eq/eq_50_50_long_short_softmax.csv','%d.%m.%Y', '50x50 long short softmax','c'),

    # ('data/eq/eq_long_softmax_09.csv','%Y-%m-%d', 'long softmax','g'),
    # ('data/eq/eq_short_softmax_09.csv','%Y-%m-%d', 'short softmax','r'),
    # ('data/eq/eq_50_50_long_short_softmax_09.csv','%d.%m.%Y', '50x50 long short softmax','c'),
    # ('data/eq/eq_snp_09.csv','%d.%m.%Y', 'snp','m'),


    # ('data/eq/eq_long_softmax_sel_23.csv','%Y-%m-%d', 'long softmax 23 sel','g'),
    # ('data/eq/eq_short_softmax_sel_23.csv','%Y-%m-%d', 'short softmax 23 sel','r'),
    # ('data/eq/eq_50_50_long_short_softmax_sel_23.csv','%d.%m.%Y', '50x50 long short softmax sel 23','c'),
    # ('data/eq/eq_70_30_long_short_softmax_sel_23.csv','%d.%m.%Y', '70x30 long short softmax sel 23','k'),

    ('data/eq/eq_snp.csv','%Y-%m-%d', 'snp','m'),


    ('data/eq/eq_avg_stocks.csv','%Y-%m-%d', 'avg stocks','c'),
          ]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
draw_grid(ax)
format_time_labels(ax, fmt=DDMMMYY_FMT)
ax.set_title("Equity curves")
handles = []

for path,fmt,name,clr in curves:
    df = pd.read_csv(path)
    df['date'] =  pd.to_datetime(df['date'], format=fmt).dt.date
    capital = np.array(df['capital'])

    years = (get_config().TEST_END - get_config().TEST_BEG).days / 365
    capital = np.array(df['capital'])
    dd = get_draw_down(capital, False)
    rets = capital[1:] - capital[:-1]
    sharpe = get_sharpe_ratio(rets, years)
    y_avg = (capital[-1] - capital[0]) / years
    print('%s dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (name, dd * 100, y_avg * 100, sharpe))
    line, = ax.plot_date(df['date'], capital, color=clr, fmt='-', label=name)
    handles.append(line)

plt.legend(handles=handles)
fig.savefig('data/equity_curves.png')

show_plots()