import pandas as pd

from src.tests.multiResolutionMeans import apply_increasing_EH
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
    # APPENDING MEAN AND VARIANCE TO ORIGINAL

    windowLengths = [[4], [4, 8], [4, 8, 16], [4, 8, 16, 32], [4, 8, 16, 32, 64],
                     [32], [32, 64], [32, 64, 128], [32, 64, 128, 256], [32, 64, 128, 256, 512],
                     [48]]
    eps = 0.05
    for windowLength in windowLengths:
        data = apply_increasing_EH(elecDf.drop(['date', 'day'], axis=1), windowLength, eps=eps, realNums=True, maxValue=1)
        # delete rows for which we do not have an available mean
        # elecDf = elecDf.loc[windowLengths[-1]:]
        data = data.drop(['class'], axis=1)
        for column in elecDf.columns:
            data[column] = elecDf[column].tolist()
        windowStr = ''
        first = True
        for window in windowLength:
            if first:
                windowStr += str(window)
                first = False
            else:
                windowStr += '_' + str(window)
        data.to_csv('./data/processedStreams/electricity/AppendedMeanVar' + windowStr + '.csv')
    # pd_to_arff(data, 'electricityAppendedMeansVar48', './data/processedStreams/', 'ORDINAL', ['UP', 'DOWN'])



