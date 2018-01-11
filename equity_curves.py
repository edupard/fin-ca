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
import datetime

DDMMMYY_FMT = matplotlib.dates.DateFormatter('%y %b %d')
YYYY_FMT = matplotlib.dates.DateFormatter('%Y')

curves = [
    # ('data/eq/eq_70_30_net.csv','%d.%m.%Y', '70x30 rbm net mon-fri','b'),
    # ('data/eq/eq_70_30_no_net.csv','%d.%m.%Y', '70x30 no net mon-fri','r'),

    # ('data/eq/eq_70_30_net_sl_5_bp.csv','%d.%m.%Y', '70x30 rbm net mon-fri','b'),
    # ('data/eq/eq_70_30_no_net_sl_5_bp.csv','%d.%m.%Y', '70x30 no net mon-fri','b'),

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

    # ('data/eq/50_25_70.csv', '%Y-%m-%d', '25 70 vs 30', 'k'),
    # ('data/eq/100_23_70.csv', '%Y-%m-%d', '23 70 vs 30 pure net', 'g'),
    # ('data/eq/0_23_70.csv', '%Y-%m-%d', '23 70 vs 30 pure decline', 'k'),

    # ('data/eq/test_50_25_70.csv', '%Y-%m-%d', '25 70 vs 30', 'k'),


    # ('data/eq/test_50_23_50.csv', '%Y-%m-%d', '23 50 vs 50', 'r'),
    # ('data/eq/test_50_23_60.csv', '%Y-%m-%d', '23 60 vs 40', 'g'),
    # ('data/eq/test_50_23_70.csv', '%Y-%m-%d', '23 70 vs 30', 'b'),
    # ('data/eq/test_50_23_80.csv', '%Y-%m-%d', '23 80 vs 20', 'k'),

    # ('data/eq/test_100_23_70.csv', '%Y-%m-%d', '23 70 vs 30 pure net', 'g'),
    # ('data/eq/test_0_23_70.csv', '%Y-%m-%d', '23 70 vs 30 pure decline', 'b'),
    # ('data/eq/test_50_23_70.csv', '%Y-%m-%d', '50 vs 50 23 70 vs 50', 'r'),


    # ('data/eq/universal_100_23_0.csv','%Y-%m-%d', 'universal 0','b'),
    # ('data/eq/universal_100_23_10.csv','%Y-%m-%d', 'universal 10','b'),
    # ('data/eq/universal_100_23_20.csv','%Y-%m-%d', 'universal 20','b'),
    # ('data/eq/universal_100_23_30.csv','%Y-%m-%d', 'universal 30','b'),
    # ('data/eq/universal_100_23_40.csv','%Y-%m-%d', 'universal 40','b'),
    # ('data/eq/universal_100_23_50.csv','%Y-%m-%d', 'universal 50','b'),
    # ('data/eq/universal_100_23_60.csv','%Y-%m-%d', 'universal 60','b'),
    # ('data/eq/universal_100_23_70.csv','%Y-%m-%d', 'universal 70','b'),
    # ('data/eq/universal_100_23_80.csv','%Y-%m-%d', 'universal 80','b'),
    # ('data/eq/universal_100_23_90.csv','%Y-%m-%d', 'universal 90','b'),
    # ('data/eq/universal_100_23_100.csv','%Y-%m-%d', 'universal 100','b'),


    # ('data/eq/rev_universal_100_23_0.csv','%Y-%m-%d', 'inv universal 0','r'),
    # ('data/eq/rev_universal_100_23_10.csv','%Y-%m-%d', 'inv universal 10','r'),
    # ('data/eq/rev_universal_100_23_20.csv','%Y-%m-%d', 'inv universal 20','r'),
    # ('data/eq/rev_universal_100_23_30.csv','%Y-%m-%d', 'inv universal 30','r'),
    # ('data/eq/rev_universal_100_23_40.csv','%Y-%m-%d', 'inv universal 40','r'),
    # ('data/eq/rev_universal_100_23_50.csv','%Y-%m-%d', 'inv universal 50','r'),
    # ('data/eq/rev_universal_100_23_60.csv','%Y-%m-%d', 'inv universal 60','r'),
    # ('data/eq/rev_universal_100_23_70.csv','%Y-%m-%d', 'inv universal 70','r'),
    # ('data/eq/rev_universal_100_23_80.csv','%Y-%m-%d', 'inv universal 80','r'),
    # ('data/eq/rev_universal_100_23_90.csv','%Y-%m-%d', 'inv universal 90','r'),
    # ('data/eq/rev_universal_100_23_100.csv','%Y-%m-%d', 'inv universal 100','r'),


    # ('data/eq/new_test_0_23_20.csv', '%Y-%m-%d', '23 20 vs 80', 'r'),
    # ('data/eq/new_test_0_23_30.csv', '%Y-%m-%d', '23 30 vs 70', 'r'),
    # ('data/eq/new_test_0_23_40.csv', '%Y-%m-%d', '23 40 vs 60', 'r'),
    # ('data/eq/new_test_0_23_50.csv', '%Y-%m-%d', '23 50 vs 50', 'g'),
    # ('data/eq/new_test_0_23_60.csv', '%Y-%m-%d', '23 60 vs 40', 'r'),
    # ('data/eq/new_test_0_23_70.csv', '%Y-%m-%d', '23 70 vs 30', 'r'),
    # ('data/eq/new_test_0_23_80.csv', '%Y-%m-%d', '23 80 vs 20', 'r'),
    # ('data/eq/new_test_0_23_90.csv', '%Y-%m-%d', '23 90 vs 10', 'r'),
    # ('data/eq/new_test_0_23_100.csv', '%Y-%m-%d', '23 100 vs 0', 'r'),

    # ('data/eq/decline_test_0_23_0.csv', '%Y-%m-%d', 'decline 23 0 vs 100', 'r'),
    # ('data/eq/decline_test_0_23_10.csv', '%Y-%m-%d', 'decline 23 10 vs 90', 'r'),
    # ('data/eq/decline_test_0_23_20.csv', '%Y-%m-%d', 'decline 23 20 vs 80', 'r'),
    # ('data/eq/decline_test_0_23_30.csv', '%Y-%m-%d', 'decline 23 30 vs 70', 'r'),
    # ('data/eq/decline_test_0_23_40.csv', '%Y-%m-%d', 'decline 23 40 vs 60', 'r'),

    # ('data/eq/decline_test_0_23_50.csv', '%Y-%m-%d', 'decline 23 50 vs 50', 'r'),
    # ('data/eq/decline_test_0_23_60.csv', '%Y-%m-%d', 'decline 23 60 vs 40', 'r'),
    # ('data/eq/eq_70_30_no_net_sl_5_bp.csv','%d.%m.%Y', '70x30 no net mon-fri','b'),
    # ('data/eq/decline_test_0_23_70.csv', '%Y-%m-%d', 'decline 23 70 vs 30', 'r'),
    # ('data/eq/decline_test_0_23_80.csv', '%Y-%m-%d', 'decline 23 80 vs 20', 'r'),
    # ('data/eq/decline_test_0_23_90.csv', '%Y-%m-%d', 'decline 23 90 vs 10', 'r'),
    # ('data/eq/decline_test_0_23_100.csv', '%Y-%m-%d', 'decline 23 100 vs 0', 'r'),

    # ('data/eq/ma_100_23_0.csv', '%Y-%m-%d', '23 0 vs 100', 'r'),
    # ('data/eq/ma_100_23_10.csv', '%Y-%m-%d', '23 10 vs 90', 'r'),
    # ('data/eq/ma_100_23_20.csv', '%Y-%m-%d', '23 20 vs 80', 'r'),
    # ('data/eq/ma_100_23_30.csv', '%Y-%m-%d', '23 30 vs 70', 'r'),
    # ('data/eq/ma_100_23_40.csv', '%Y-%m-%d', '23 40 vs 60', 'r'),
    # ('data/eq/ma_100_23_50.csv', '%Y-%m-%d', '23 50 vs 50', 'g'),
    # ('data/eq/ma_100_23_60.csv', '%Y-%m-%d', '23 60 vs 40', 'r'),
    # ('data/eq/ma_100_23_70.csv', '%Y-%m-%d', '23 70 vs 30', 'r'),
    # ('data/eq/ma_100_23_80.csv', '%Y-%m-%d', '23 80 vs 20', 'r'),
    # ('data/eq/ma_100_23_90.csv', '%Y-%m-%d', '23 90 vs 10', 'r'),
    # ('data/eq/ma_100_23_100.csv', '%Y-%m-%d', '23 100 vs 0', 'r'),

    # ('data/eq/ma_100_100_100_0.csv', '%Y-%m-%d', '23 0 vs 100', 'r'),
    # ('data/eq/ma_100_100_100_10.csv', '%Y-%m-%d', '23 10 vs 90', 'r'),
    # ('data/eq/ma_100_100_100_20.csv', '%Y-%m-%d', '23 20 vs 80', 'r'),
    # ('data/eq/ma_100_100_100_30.csv', '%Y-%m-%d', '23 30 vs 70', 'r'),
    # ('data/eq/ma_100_100_100_40.csv', '%Y-%m-%d', '23 40 vs 60', 'r'),
    # ('data/eq/ma_100_100_100_50.csv', '%Y-%m-%d', '23 50 vs 50', 'g'),
    # ('data/eq/ma_100_100_100_60.csv', '%Y-%m-%d', '23 60 vs 40', 'r'),
    # ('data/eq/ma_100_100_100_70.csv', '%Y-%m-%d', '23 70 vs 30', 'r'),
    # ('data/eq/ma_100_100_100_80.csv', '%Y-%m-%d', '23 80 vs 20', 'r'),
    # ('data/eq/ma_100_100_100_90.csv', '%Y-%m-%d', '23 90 vs 10', 'r'),
    # ('data/eq/ma_100_100_100_100.csv', '%Y-%m-%d', '23 100 vs 0', 'r'),

    ('data/eq/eq_snp.csv','%Y-%m-%d', 'snp','m'),
    # ('data/eq/eq_snp_inv.csv','%d.%m.%Y', 'snp inv','m'),
    # ('data/eq/eq_snp_test_sl_5bp.csv','%Y-%m-%d', 'snp 5bp','m'),
    # ('data/eq/eq_snp_inv_test_sl_5bp.csv','%Y-%m-%d', 'snp 5bp','m'),
    # ('data/eq/eq_snp_test.csv','%Y-%m-%d', 'snp 0bp','m'),
    # ('data/eq/eq_snp_inv_test.csv','%Y-%m-%d', 'snp 0bp','m'),



    # ('data/eq/eq_avg_stocks.csv','%Y-%m-%d', 'avg stocks','c'),
    # ('data/eq/eq_2007_tod_ms.csv','%Y-%m-%d', 'avg stocks','c'),
    # ('data/eq/o.csv','%Y-%m-%d', '*','k'),
          ]

# curves = []
#
# NET_FACTOR_GRID = [0.5]
# SELECTION_GRID = [35]
# # SELECTION_GRID = [5,15,25,35]
# LONG_PCT_GRID = [0.5,0.6,0.7,0.8]
#
# for NET_FACTOR in NET_FACTOR_GRID:
#     for SELECTION in SELECTION_GRID:
#         for LONG_PCT in LONG_PCT_GRID:
#             SHORT_PCT = 1 - LONG_PCT
#             csv_path = 'data/eq/%.0f_%d_%.0f.csv' % (NET_FACTOR * 100, SELECTION, LONG_PCT * 100)
#             curves.append((csv_path, '%Y-%m-%d', '%d %.0f vs %.0f' % (SELECTION, LONG_PCT * 100, SHORT_PCT * 100), 'c'))


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

    END = get_config().TEST_END
    BEG = get_config().TEST_BEG
    # END = datetime.datetime.strptime('2017-08-27', '%Y-%m-%d').date()
    # BEG = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d').date()

    selection = (df['date'] >= BEG) & (df['date'] <= END)
    df = df.loc[selection]

    years = (END - BEG).days / 365
    capital = np.array(df['capital'])
    dd = get_draw_down(capital, False)
    rets = capital[1:] - capital[:-1]
    sharpe = get_sharpe_ratio(rets, years)
    y_avg = (capital[-1] - capital[0]) / years
    print('%s dd: %.2f%% y_avg: %.2f%% sharpe: %.2f' % (name, dd * 100, y_avg * 100, sharpe))
    line, = ax.plot_date(df['date'], capital, fmt='-', label=name)
    # line, = ax.plot_date(df['date'], capital, color=clr, fmt='-', label=name)
    handles.append(line)

plt.legend(handles=handles)
fig.savefig('data/equity_curves.png')

show_plots()