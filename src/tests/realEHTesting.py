from random import random
import pandas as pd
import altair as alt
from src.EHs.binaryCounterEH import SumEH, ExactWindow, MeanEH

if __name__ == "__main__":
    EHMean = MeanEH(100, 0.01, isReal=True, resolution=100)
    EHSum = SumEH(100, 0.01, isReal=True, resolution=100)
    exact = ExactWindow(100)
    resMean = []
    resSum = []
    for i in range(10000):
        value = random()
        EHMean.add(i, value)
        EHSum.add(i, value)
        exact.add(value)
        if i % 50 == 0:
            resMean.append(
                pd.DataFrame(data=[[i, EHMean.get_estimate(), exact.mean()]], columns=['index', 'EH', 'exact']))
            resSum.append(
                pd.DataFrame(data=[[i, EHSum.get_estimate(), exact.sum()]], columns=['index', 'EH', 'exact']))

    resMean = pd.concat(resMean, ignore_index=True)
    resMean = resMean.melt(id_vars='index')
    alt.data_transformers.disable_max_rows()
    alt.Chart(resMean, width=500).mark_line(opacity=0.7).encode(
        x='index',
        y='value',
        color='variable'
    ).show()

    resSum = pd.concat(resSum, ignore_index=True)
    resSum = resSum.melt(id_vars='index')
    alt.data_transformers.disable_max_rows()
    alt.Chart(resSum, width=500).mark_line(opacity=0.7).encode(
        x='index',
        y='value',
        color='variable'
    ).show()

