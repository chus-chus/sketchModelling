import pandas as pd

# todo apply means, target as onehot
from src.test.multiResolutionMeans import apply_increasing_mean_EH
from src.utils.csvToArff import pd_to_arff

if __name__ == "__main__":
    # Sudden drift clearly separated, NB, nwait 50
    elecDf = pd.read_csv('./data/rawStreams/electricity/electricity.csv')

    ####
    # EH MEANS

    # windowLengths = [2, 4, 8, 16, 32, 64]
    # # does not make sense to summarize date and day
    # data = apply_increasing_mean_EH(elecDf.drop(['date', 'day'], axis=1), windowLengths, eps=0.05, realNums=True, res=100)
    # data.insert(0, 'date', elecDf.loc[windowLengths[-1]:, 'date'].tolist())
    # data.insert(0, 'day', elecDf.loc[windowLengths[-1]:, 'day'].tolist())
    # pd_to_arff(data, 'electricityMeans', './data/processedStreams/', 'ORDINAL', ['UP', 'DOWN'])

    ####
    # APPENDING MEAN TO ORIGINAL

    windowLengths = [32, 64]
    data = apply_increasing_mean_EH(elecDf.drop(['date', 'day'], axis=1), windowLengths, eps=0.05, realNums=True,
                                    res=100)
    # delete rows for which we do not have an available mean
    elecDf = elecDf.loc[windowLengths[-1]:]
    data = data.drop(['class'], axis=1)
    for column in elecDf.columns:
        data[column] = elecDf[column].tolist()
    pd_to_arff(data, 'electricityAppendedMeans32_64', './data/processedStreams/', 'ORDINAL', ['UP', 'DOWN'])



