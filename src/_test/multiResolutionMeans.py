""" Predicting with multiple increasing-size windows that compute the mean. For each point, generate a point with n
    features, where n is the number of windows. """

import skmultiflow as skm
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd

from src.sketches.EH import MeanEH, MeanExactWindow

# todo change save paths, add evaluation of movels over exact windowed streams


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


def apply_increasing_mean_exact_window(data, windowLengthsList):
    windows = [MeanExactWindow(windowLength) for windowLength in windowLengthsList]
    means = []
    for i in range(len(data)):
        for j in range(len(windowLengthsList)):
            windows[j].add(data[i][0])
        if i >= windowLengthsList[-1]:
            point = []
            for windowIndex in range(len(windowLengthsList)):
                if not windows[windowIndex].empty():
                    point.append(windows[windowIndex].query())
                else:
                    point.append(0)
            means.append(pd.DataFrame(data=[point],
                                      columns=['window_' + str(windowLengthsList[t])
                                               for t in range(len(windowLengthsList))]))
    meanDf = pd.concat(means, ignore_index=True)
    data = np.array(data)
    meanDf['target'] = data[windowLengthsList[-1]:, 1].astype(int)
    return meanDf


if __name__ == "__main__":
    random.seed(888)

    windowLengths = [10, 100, 200, 400, 800, 1600, 3200]

    # Sudden drift clearly separated, NB, nwait 50
    suddenDriftSeparated = sudden_drift(2, 4)
    approxMeansDf = apply_increasing_mean_EH(suddenDriftSeparated, windowLengths, 0.01)
    meansDf = apply_increasing_mean_exact_window(suddenDriftSeparated, windowLengths)
    meansDf.to_csv('./data/meansSuddenDriftSeparated.csv', index=False)
    approxMeansDf.to_csv('./logs/meansSuddenDriftSeparated.csv', index=False)
    stream = skm.data.FileStream('../../data/meansSuddenDriftSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=True, pretrain_size=200, max_samples=20000,
                                                   output_file='../../data/NBmeansSuddenDriftSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Sudden drift halfway mixed, NB, nwait 50
    suddenDriftMixed = sudden_drift(2, 3)
    approxMeansDf = apply_increasing_mean_EH(suddenDriftMixed, windowLengths, 0.01)
    approxMeansDf.to_csv('./logs/meansSuddenDriftMixed.csv', index=False)
    stream = skm.data.FileStream('../../data/meansSuddenDriftMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=True, pretrain_size=200, max_samples=20000,
                                                   output_file='../../data/NBmeansSuddenDriftMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Reocurring concept clearly separated, NB, blocks of 1250, nwait 10
    blockedData = [suddenDriftSeparated[x:x + 1250] for x in range(0, len(suddenDriftSeparated), 1250)]
    random.shuffle(blockedData)
    reocurringConceptSeparated = list(itertools.chain.from_iterable(blockedData))
    approxMeansDf = apply_increasing_mean_EH(reocurringConceptSeparated, windowLengths, 0.01)
    approxMeansDf.to_csv('./logs/meansReocurringSeparated.csv', index=False)
    stream = skm.data.FileStream('../../data/meansReocurringSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=True, pretrain_size=200, max_samples=20000,
                                                   output_file='../../data/NBmeansReocurringSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Reocurring concept mixed, NB, blocks of 1250, nwait 10
    blockedData = [suddenDriftMixed[x:x + 1250] for x in range(0, len(suddenDriftMixed), 1250)]
    random.shuffle(blockedData)
    reocurringConceptMixed = list(itertools.chain.from_iterable(blockedData))
    approxMeansDf = apply_increasing_mean_EH(reocurringConceptMixed, windowLengths, 0.01)
    approxMeansDf.to_csv('./logs/meansReocurringMixed.csv', index=False)
    stream = skm.data.FileStream('../../data/meansReocurringMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=True, pretrain_size=200, max_samples=20000,
                                                   output_file='../../data/NBmeansReocurringMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Incremental drift sep, NB, nwait 50
    incrementalDriftSep = incremental_drift([8000, 12000], 2, 4)
    approxMeansDf = apply_increasing_mean_EH(incrementalDriftSep, windowLengths, 0.01)
    approxMeansDf.to_csv('./logs/meansIncrementalSeparated.csv', index=False)
    stream = skm.data.FileStream('../../data/meansIncrementalSeparated.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=True, pretrain_size=200, max_samples=20000,
                                                   output_file='../../data/NBmeansIncrementalSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    # Incremental drift mixed, NB, nwait 50
    incrementalDriftMixed = incremental_drift([8000, 12000], 2, 3)
    approxMeansDf = apply_increasing_mean_EH(incrementalDriftMixed, windowLengths, 0.01)
    approxMeansDf.to_csv('./logs/meansIncrementalMixed.csv', index=False)
    stream = skm.data.FileStream('../../data/meansIncrementalMixed.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=10, show_plot=True, pretrain_size=200, max_samples=20000,
                                                   output_file='../../data/NBmeansIncrementalMixedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())

    plt.plot(suddenDriftSeparated)
    plt.show()
    plt.plot(suddenDriftMixed)
    plt.show()
    plt.plot(incrementalDriftSep)
    plt.show()
    plt.plot(incrementalDriftMixed)
    plt.show()
    plt.plot(reocurringConceptMixed)
    plt.show()
    plt.plot(reocurringConceptSeparated)
    plt.show()

