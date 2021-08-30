import pandas as pd
import numpy as np


# pivoting of sub DFs of windowLen size
from src.utils.csvToArff import pd_to_arff

if __name__ == "__main__":
    dataPath = './data/rawStreams/electricity/'
    dataWritePath = './data/processedStreams/'
    fileName = 'electricity'
    data = pd.read_csv(dataPath + fileName + '.csv')
    dataNoTarget = data.drop(data.columns[-1], axis=1)
    colnames = dataNoTarget.columns
    newDf = []
    # number of past elements to consider
    windowLen = 36
    for i, row in enumerate(data.itertuples()):
        if i >= windowLen:
            # Not including index nor target
            prevData = np.append(row[1:-1], np.array(dataNoTarget.loc[(i-1):i-windowLen:-1].to_numpy().flatten()))
            prevData = np.append(prevData, row[-1])
            newNames = np.array(np.append([col + '-' + str(i) for i in range(1, windowLen+1) for col in colnames], 'target'))
            newNames = np.array(np.append(colnames, newNames))
            newDf.append(pd.DataFrame([prevData], columns=list(newNames)))
    newDf = pd.concat(newDf, ignore_index=True)
    pd_to_arff(newDf, 'electricityWindow36', dataWritePath, 'ORDINAL', ['DOWN', 'UP'])
