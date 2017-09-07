import math
import numpy as np


def get_capital(ret, recap):
    capital = np.ones((ret.shape[0] + 1))
    if recap:
        recap_ret = (ret) + 1.00
        capital[1:] = np.cumprod(recap_ret)
    else:
        capital[1:] = np.cumsum(ret) + 1.0
    return capital


def get_draw_down(c, recap):
    # c == capital in time array
    # recap == flag indicating recapitalization or fixed bet
    def generate_previous_max():
        max = c[0]
        for idx in range(len(c)):
            # update max
            if c[idx] > max:
                max = c[idx]
            yield max

    prev_max = np.fromiter(generate_previous_max(), dtype=np.float64)
    if recap:
        dd_a = (c - prev_max) / prev_max
    else:
        dd_a = c - prev_max

    return np.min(dd_a)


def get_sharpe_ratio(ret, years):
    return math.sqrt(ret.shape[0] / years) * np.mean(ret) / np.std(ret)


def get_avg_yeat_ret(ret, years):
    return np.sum(ret) / years


def print_alloc(pl, ent, tickers, stk_mask, weights):
    abs_weights = np.abs(weights)
    alloc = np.sum(abs_weights) * 100
    idxs = np.argsort(abs_weights)
    top_5_idxs = idxs[-5:]
    top_5_idxs = top_5_idxs[::-1]
    top_5_weights = weights[top_5_idxs] * 100
    all_stk_idxs = np.nonzero(stk_mask)[0]
    stk_idxs = all_stk_idxs[top_5_idxs]

    print("%s %.2f%%: %.2f%% W1: %.2f%% [%s], W2: %.2f%% [%s], W3: %.2f%% [%s], W4: %.2f%% [%s], W5: %.2f%% [%s]" % (
        ent.strftime('%Y.%m.%d'),
        alloc,
        pl * 100,
        top_5_weights[0],
        tickers[stk_idxs[0]],
        top_5_weights[1],
        tickers[stk_idxs[1]],
        top_5_weights[2],
        tickers[stk_idxs[2]],
        top_5_weights[3],
        tickers[stk_idxs[3]],
        top_5_weights[4],
        tickers[stk_idxs[4]]))
