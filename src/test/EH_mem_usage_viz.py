""" Theoretical use of memory in EH vs linear """
import altair as alt
import pandas as pd
import math


def mem_usage(eps=0.05, adjust=False):
    memLinear = np.linspace(1, 10000, 10000)

    memDf = pd.DataFrame()

    memDf['linear'] = memLinear
    memDf['EH (upper bound)'] = memLinear
    memDf['EH (lower bound)'] = memLinear
    memDf['window_size'] = memLinear

    # assimptotic bounds
    memDf['EH (upper bound)'] = memDf['EH (upper bound)'].apply(lambda N: (1 / (eps ** 2)) * math.log(N, 2))
    memDf['EH (lower bound)'] = memDf['EH (lower bound)'].apply(lambda N: (1 / eps) * math.log(N, 2))

    if adjust:
        # adjust bounds where memory usage of EH is higher (naive solution is O(N))
        memDf.loc[memDf['EH (upper bound)'] > memDf['linear'], 'EH (upper bound)'] = memDf['linear']
        memDf.loc[memDf['EH (lower bound)'] > memDf['linear'], 'EH (lower bound)'] = memDf['linear']

    memDf = pd.melt(memDf, id_vars=['window_size'],
                    value_vars=['EH (upper bound)', 'EH (lower bound)', 'linear'],
                    var_name='Structure', value_name='memory_usage')
    return memDf


alt.data_transformers.enable('default', max_rows=None)

memDf005 = mem_usage()
memDf01 = mem_usage(eps=0.1)

alt.hconcat(alt.Chart(memDf005, title='Memory usage, eps=0.05').mark_line().encode(
    x='window_size:Q',
    y='memory_usage:Q',
    color='Structure'
), alt.Chart(memDf01, title='Memory usage, eps=0.1').mark_line().encode(
    x='window_size:Q',
    y='memory_usage:Q',
    color='Structure'
))
