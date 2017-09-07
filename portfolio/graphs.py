import matplotlib.pyplot as plt
import matplotlib

DDMMMYY_FMT = matplotlib.dates.DateFormatter('%y %b %d')
YYYY_FMT = matplotlib.dates.DateFormatter('%Y')


def format_time_labels(ax, fmt):
    ax.xaxis.set_major_formatter(fmt)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)


def draw_grid(ax):
    ax.grid(True, linestyle='-', color='0.75')


def hide_time_labels(ax):
    plt.setp(ax.get_xticklabels(), visible=False)


def plot_equity_curve(caption, dt, capital):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=DDMMMYY_FMT)
    ax.set_title(caption)
    ax.plot_date(dt, capital, color='b', fmt='-')

def show_plots():
    plt.show(True)
