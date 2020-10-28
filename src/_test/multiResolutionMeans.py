""" Predicting with multiple increasing-size windows that compute the mean. For each point, generate a point with n
    features, where n is the number of windows. """

import skmultiflow as skm
import numpy as np
import random
import itertools
import altair as alt
import altair_viewer
import matplotlib.pyplot as plt
import pandas as pd

from src.sketches.EH import MeanEH, ExactWindow

# todo change save paths, add evaluation of models over exact windowed streams
# todo test positive integers

alt.data_transformers.disable_max_rows()


def concept1(value, offset):
    return int(10 * (np.sin(value / (2 * np.pi)) + offset))


def concept2(value, offset):
    return int(10 * (np.sin(value / (2 * np.pi)) + offset))


def sudden_drift(offset1, offset2):
    sine1 = [[concept1(value, offset1), 0] for value in np.linspace(0, 180, 10000)]
    sine2 = [[concept2(value, offset2), 1] for value in np.linspace(0, 180, 10000)]
    return sine1 + sine2


def incremental_drift(conceptChangeRange, concept1Offset, concept2Offset):
    incrementalDrift = []
    for i, x in enumerate(np.linspace(0, 360, 20000)):
        if i > conceptChangeRange[1]:
            p = 1
        elif i < conceptChangeRange[0]:
            p = 0
        else:
            p = 2 ** ((i - conceptChangeRange[1]) / 500)
        bernSample = np.random.binomial(1, p)
        if bernSample == 1:
            incrementalDrift.append([concept2(x, concept1Offset), 1])
        else:
            incrementalDrift.append([concept1(x, concept2Offset), 0])
    return incrementalDrift


def apply_increasing_mean_EH(data, windowLengthsList, eps):
    hists = [MeanEH(windowLength, eps) for windowLength in windowLengthsList]
    means = []
    for i in range(len(data)):
        for j in range(len(windowLengthsList)):
            hists[j].add(i, data[i][0])
        if i >= windowLengthsList[-1]:
            point = []
            for windowIndex in range(len(windowLengthsList)):
                if not hists[windowIndex].empty():
                    point.append(hists[windowIndex].get_estimate())
                else:
                    point.append(0)
            means.append(pd.DataFrame(data=[point],
                                      columns=['window_' + str(windowLengthsList[t])
                                               for t in range(len(windowLengthsList))]))
    meanDf = pd.concat(means, ignore_index=True)
    data = np.array(data)
    meanDf['target'] = data[windowLengthsList[-1]:, 1].astype(int)
    return meanDf


def apply_increasing_mean(data, windowLengthsList):
    windows = [ExactWindow(windowLength) for windowLength in windowLengthsList]
    means = []
    for i in range(len(data)):
        for j in range(len(windowLengthsList)):
            windows[j].add(data[i][0])
        if i >= windowLengthsList[-1]:
            point = []
            for windowIndex in range(len(windowLengthsList)):
                if not windows[windowIndex].empty():
                    point.append(windows[windowIndex].mean())
                else:
                    point.append(0)
            means.append(pd.DataFrame(data=[point],
                                      columns=['window_' + str(windowLengthsList[t])
                                               for t in range(len(windowLengthsList))]))
    meanDf = pd.concat(means, ignore_index=True)
    data = np.array(data)
    meanDf['target'] = data[windowLengthsList[-1]:, 1].astype(int)
    return meanDf


def model_with_eh_means(suddenDriftSeparated, suddenDriftMixed, reocurringConceptSeparated, reocurringConceptMixed,
                        incrementalDriftSep, incrementalDriftMixed, windowLengths):
    # Sudden drift clearly separated, NB, nwait 50
    approxMeansDf = apply_increasing_mean_EH(suddenDriftSeparated, windowLengths, 0.01)
    approxMeansDf.to_csv('./data/processedStreams/sine/EHmeansSuddenDriftSeparated.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/EHmeansSuddenDriftSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_EHmeansSuddenDriftSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Sudden drift halfway mixed, NB, nwait 50
    approxMeansDf = apply_increasing_mean_EH(suddenDriftMixed, windowLengths, 0.01)
    approxMeansDf.to_csv('./data/processedStreams/sine/EHmeansSuddenDriftMixed.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/EHmeansSuddenDriftMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_EHmeansSuddenDriftMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Reocurring concept clearly separated, NB, blocks of 1250, nwait 10
    approxMeansDf = apply_increasing_mean_EH(reocurringConceptSeparated, windowLengths, 0.01)
    approxMeansDf.to_csv('./data/processedStreams/sine/EHmeansReocurringSeparated.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/EHmeansReocurringSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_EHmeansReocurringSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Reocurring concept mixed, NB, blocks of 1250, nwait 10
    approxMeansDf = apply_increasing_mean_EH(reocurringConceptMixed, windowLengths, 0.01)
    approxMeansDf.to_csv('./data/processedStreams/sine/EHmeansReocurringMixed.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/EHmeansReocurringMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_EHmeansReocurringMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Incremental drift sep, NB, nwait 50
    approxMeansDf = apply_increasing_mean_EH(incrementalDriftSep, windowLengths, 0.01)
    approxMeansDf.to_csv('./data/processedStreams/sine/EHmeansIncrementalSeparated.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/EHmeansIncrementalSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_EHmeansIncrementalSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Incremental drift mixed, NB, nwait 50
    approxMeansDf = apply_increasing_mean_EH(incrementalDriftMixed, windowLengths, 0.01)
    approxMeansDf.to_csv('./data/processedStreams/sine/EHmeansIncrementalMixed.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/EHmeansIncrementalMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_EHmeansIncrementalMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())


def model_with_true_means(suddenDriftSeparated, suddenDriftMixed, reocurringConceptSeparated, reocurringConceptMixed,
                          incrementalDriftSep, incrementalDriftMixed, windowLengths):
    # Sudden drift clearly separated, NB, nwait 50
    MeansDf = apply_increasing_mean(suddenDriftSeparated, windowLengths)
    MeansDf.to_csv('./data/processedStreams/sine/meansSuddenDriftSeparated.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/meansSuddenDriftSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_meansSuddenDriftSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Sudden drift halfway mixed, NB, nwait 50
    MeansDf = apply_increasing_mean(suddenDriftMixed, windowLengths)
    MeansDf.to_csv('./data/processedStreams/sine/meansSuddenDriftMixed.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/meansSuddenDriftMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_meansSuddenDriftMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Reocurring concept clearly separated, NB, blocks of 1250, nwait 10
    MeansDf = apply_increasing_mean(reocurringConceptSeparated, windowLengths)
    MeansDf.to_csv('./data/processedStreams/sine/meansReocurringSeparated.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/meansReocurringSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_meansReocurringSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Reocurring concept mixed, NB, blocks of 1250, nwait 10
    MeansDf = apply_increasing_mean(reocurringConceptMixed, windowLengths)
    MeansDf.to_csv('./data/processedStreams/sine/meansReocurringMixed.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/meansReocurringMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_meansReocurringMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Incremental drift sep, NB, nwait 50
    MeansDf = apply_increasing_mean(incrementalDriftSep, windowLengths)
    MeansDf.to_csv('./data/processedStreams/sine/meansIncrementalSeparated.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/meansIncrementalSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_meansIncrementalSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Incremental drift mixed, NB, nwait 50
    MeansDf = apply_increasing_mean(incrementalDriftMixed, windowLengths)
    MeansDf.to_csv('./data/processedStreams/sine/meansIncrementalMixed.csv', index=False)
    stream = skm.data.FileStream('./data/processedStreams/sine/meansIncrementalMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_meansIncrementalMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())


def plot_results_means():
    csvNames = [['NB_EHmeansIncrementalMixedResults', 'NB_EHmeansIncrementalSeparatedResults',
                'NB_meansIncrementalMixedResults', 'NB_meansIncrementalSeparatedResults'],
                ['NB_EHmeansReocurringMixedResults', 'NB_EHmeansReocurringSeparatedResults',
                'NB_meansReocurringMixedResults', 'NB_meansReocurringSeparatedResults'],
                ['NB_EHmeansSuddenDriftMixedResults', 'NB_EHmeansSuddenDriftSeparatedResults',
                'NB_meansSuddenDriftMixedResults', 'NB_meansSuddenDriftSeparatedResults']]

    dfs = [pd.read_csv('./logs/sine/'+name+'.csv', skiprows=5).drop(['current_acc_[M0]', 'current_kappa_[M0]'], axis=1) for names in csvNames for name in names]

    colsNames = ['id', 'mean_acc', 'mean_kappa']

    for df in dfs:
        df.columns = colsNames

    dfs = [df.melt('id', var_name='metrics') for df in dfs]

    incrementals = dfs[:4]
    reocurrings = dfs[4:8]
    sudden = dfs[8:]

    dfs = [incrementals, reocurrings, sudden]

    for j, df in enumerate(dfs):
        for i in range(len(df)-2):
            alt.hconcat(alt.Chart(df[i], title=csvNames[j][i]).mark_line(opacity=0.7).encode(
                x='id',
                y='value',
                color='metrics',
            ), alt.Chart(df[i+2], title=csvNames[j][i+2]).mark_line(opacity=0.7).encode(
                x='id',
                y='value',
                color='metrics'
            )).show()



if __name__ == "__main__":
    random.seed(888)

    # Data generation
    # Sudden drift
    suddenDriftSeparated = np.array(sudden_drift(2, 4))
    suddenDriftMixed = np.array(sudden_drift(2, 3))
    suddenSepDf = pd.DataFrame(data={'value': suddenDriftSeparated[:, 0], 'target': suddenDriftSeparated[:, 1]})
    suddenMixedDf = pd.DataFrame(data={'value': suddenDriftMixed[:, 0], 'target': suddenDriftMixed[:, 1]})
    suddenSepDf.to_csv('./data/rawStreams/sine/SuddenDriftSeparated.csv', index=False)
    suddenMixedDf.to_csv('./data/rawStreams/sine/SuddenDriftMixed.csv', index=False)

    # Reocurring concept
    blockedData = [suddenDriftSeparated[x:x + 1250] for x in range(0, len(suddenDriftSeparated), 1250)]
    random.shuffle(blockedData)
    reocurringConceptSeparated = np.array(list(itertools.chain.from_iterable(blockedData)))
    reocurringSepDf = pd.DataFrame(data={'value': reocurringConceptSeparated[:, 0], 'target': reocurringConceptSeparated[:, 1]})
    reocurringSepDf.to_csv('./data/rawStreams/sine/ReocurringSeparated.csv', index=False)

    blockedData = [suddenDriftMixed[x:x + 1250] for x in range(0, len(suddenDriftMixed), 1250)]
    random.shuffle(blockedData)
    reocurringConceptMixed = np.array(list(itertools.chain.from_iterable(blockedData)))
    reocurringMixedDf = pd.DataFrame(data={'value': reocurringConceptMixed[:, 0], 'target': reocurringConceptMixed[:, 1]})
    reocurringMixedDf.to_csv('./data/rawStreams/sine/ReocurringMixed.csv', index=False)

    # Incremental drift
    incrementalDriftSep = np.array(incremental_drift([8000, 12000], 2, 4))
    incrementalSepDf = pd.DataFrame(data={'value': incrementalDriftSep[:, 0], 'target': incrementalDriftSep[:, 1]})
    incrementalSepDf.to_csv('./data/rawStreams/sine/IncrementalSeparated.csv', index=False)

    incrementalDriftMixed = np.array(incremental_drift([8000, 12000], 2, 3))
    incrementalMixedDf = pd.DataFrame(data={'value': incrementalDriftMixed[:, 0], 'target': incrementalDriftMixed[:, 1]})
    incrementalMixedDf.to_csv('./data/rawStreams/sine/IncrementalMixed.csv', index=False)

    # Modelling

    windowLengths = [10, 100, 200, 400, 800, 1600, 3200]

    # model_with_eh_means(suddenDriftSeparated, suddenDriftMixed, reocurringConceptSeparated, reocurringConceptMixed,
    #                    incrementalDriftSep, incrementalDriftMixed, windowLengths)

    # model_with_true_means(suddenDriftSeparated, suddenDriftMixed, reocurringConceptSeparated, reocurringConceptMixed,
    #                       incrementalDriftSep, incrementalDriftMixed, windowLengths)

    plot_results_means()



