import pandas as pd
import numpy as np
import datetime

RECAP = True

def calc_dd(c_b, c_e, recap):
    # c == capital in time array
    # recap == flag indicating recapitalization or fixed bet
    def generate_previous_max(c):
        max = c[0]
        for idx in range(len(c)):
            # update max
            if c[idx] > max:
                max = c[idx]
            yield max

    prev_max = np.fromiter(generate_previous_max(c_b), dtype=np.float64)
    if recap:
        dd_a = (c_e - prev_max) / prev_max
    else:
        dd_a = c_e - prev_max

    return np.min(dd_a)

def calc_yield(c_beg,c_end,recap):
    if recap:
        return (c_end - c_beg) / c_beg
    else:
        return c_end - c_beg

df = pd.read_csv('data/weekly_no_sl.csv')

ext_date_col = pd.to_datetime(df.end)

df['year'] = ext_date_col.map(lambda x: x.year)
df['month'] = ext_date_col.map(lambda x: x.month)

# progress = df.hpr
# wealth = np.cumsum(progress) + 1.0
#
# rc_progress = df.hpr + 1.00
# rc_wealth = np.cumprod(rc_progress)

columns = ['year', 'indicator']
for m in range(1,13):
    month = datetime.date(1900, m, 1).strftime('%b')
    columns.append(month)
cols = tuple(columns)

result = pd.DataFrame(columns=cols)

years = df.year.unique()
i = 0
for y in years:
    a_m_y = [y, 'mtd']
    a_y_y = [y, 'ytd']
    a_m_dd = [y, 'm dd']
    a_y_dd = [y, 'y dd']
    for m in range(1,13):
        ytd_p = df[(df.year == y) & (df.month <= m)]
        mtd_p = df[(df.year == y) & (df.month == m)]
        # or calc it using hpr
        mtd_c_beg = mtd_p['r cap beg'].values if RECAP else mtd_p['cap beg'].values
        mtd_c_end = mtd_p['r cap end'].values if RECAP else mtd_p['cap end'].values

        ytd_c_beg = ytd_p['r cap beg'].values if RECAP else ytd_p['cap beg'].values
        ytd_c_end = ytd_p['r cap end'].values if RECAP else ytd_p['cap end'].values

        if mtd_c_beg.shape[0] > 0:
            m_y = calc_yield(mtd_c_beg[0], mtd_c_end[-1], RECAP)
            y_y = calc_yield(ytd_c_beg[0], ytd_c_end[-1],RECAP)
            m_dd = calc_dd(mtd_c_beg, mtd_c_end, RECAP)
            y_dd = calc_dd(ytd_c_beg, ytd_c_end, RECAP)
        else:
            m_y = 0
            y_y = 0
            m_dd = 0
            y_dd = 0
        a_m_y.append(m_y)
        a_y_y.append(y_y)
        a_m_dd.append(m_dd)
        a_y_dd.append(y_dd)
        # result.loc[i] = []
    result.loc[i*4 + 0] = a_m_y
    result.loc[i * 4 + 1] = a_y_y
    result.loc[i * 4 + 2] = a_m_dd
    result.loc[i * 4 + 3] = a_y_dd
    i += 1
result.to_csv('data/montly_stat.csv', index=False)