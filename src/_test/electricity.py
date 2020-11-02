import skmultiflow as skm
import pandas as pd

# todo apply means, target as onehot
from src._test.multiResolutionMeans import apply_increasing_mean_EH

if __name__ == "__main__":
    # Sudden drift clearly separated, NB, nwait 50
    elecDf = pd.read_csv('./data/rawStreams/electricity.csv')
    elecDf.loc[:, 'class'] = elecDf.loc[:, 'class'].apply(lambda t: 1 if t == 'UP' else 0)
    windowLengths = [2, 4, 8, 16, 32, 64]
    # does not make sense to summarize date and day
    data = apply_increasing_mean_EH(elecDf.drop(['date', 'day'], axis=1), windowLengths, eps=0.01, realNums=True, res=100)
    data.insert(0, 'date', elecDf.loc[windowLengths[-1]:, 'date'])
    data.insert(0, 'day', elecDf.loc[windowLengths[-1]:, 'day'])
    data.to_csv('./data/processedStreams/electricityMeans.csv')
    #stream = skm.data.FileStream('./data/rawStreams/electricity.csv')
    #evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=False, pretrain_size=200, max_samples=20000,
    #                                               output_file='./logs/sine/NB_EHmeansSuddenDriftSeparatedResults.csv')
    #evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())