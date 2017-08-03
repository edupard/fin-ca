import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt

df = read_csv('data/grid_search_bak.csv')

SELECTION = 4

mask = df['stop loss type'] == 'stock'
mask &= df['selection'] == SELECTION
mask &= df['type'] == 'fixed'
serie = df.loc[mask]
serie.sort_values('stop loss')

sl = serie['stop loss'] * 100.0
max_dd = serie['max dd'] * 100.0
w_dd = serie['w dd'] * 100.0
w_avg = serie['w avg'] * 100.0
w_best = serie['w best'] * 100.0
sharpe = serie['sharpe']
y_avg = serie['y avg'] * 100.0

fig = plt.figure()
fig.clear()
ax = fig.add_subplot(1, 1, 1)
ax.grid(True, linestyle='-', color='0.75')
ax.plot(sl, max_dd, 'o', label='%d' % SELECTION)
ax.set_title("max dd")

fig = plt.figure()
fig.clear()
ax = fig.add_subplot(1, 1, 1)
ax.grid(True, linestyle='-', color='0.75')
ax.plot(sl, w_dd, 'o', label='%d' % SELECTION)
ax.set_title("w dd")

fig = plt.figure()
fig.clear()
ax = fig.add_subplot(1, 1, 1)
ax.grid(True, linestyle='-', color='0.75')
ax.plot(sl, w_avg, 'o', label='%d' % SELECTION)
ax.set_title("w avg")

fig = plt.figure()
fig.clear()
ax = fig.add_subplot(1, 1, 1)
ax.grid(True, linestyle='-', color='0.75')
ax.plot(sl, sharpe, 'o', label='%d' % SELECTION)
ax.set_title("sharpe")

fig = plt.figure()
fig.clear()
ax = fig.add_subplot(1, 1, 1)
ax.grid(True, linestyle='-', color='0.75')
ax.plot(sl, y_avg, 'o', label='%d' % SELECTION)
ax.set_title("y avg")


plt.show(True)
