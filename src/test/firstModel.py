import skmultiflow as skm
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

from src.sketches.EH import MeanEH, SumEH


def concept1(value):
    return int(10 * (np.sin(value / (2 * np.pi)) + 2))


def concept2(value):
    return int(10 * (np.sin(value / (2 * np.pi)) + 4))


if __name__ == "__main__":
    random.seed(888)
    incrementalDrift = []
    for i, x in enumerate(np.linspace(0, 360, 20000)):
        if i > 12000:
            p = 1
        elif i < 8000:
            p = 0
        else:
            p = 2 ** ((i - 12000) / 500)
        bernSample = np.random.binomial(1, p)
        if bernSample == 1:
            incrementalDrift.append([concept2(x), 1])
        else:
            incrementalDrift.append([concept1(x), 0])

    sine1 = [[concept1(value), 0] for value in np.linspace(0, 180, 10000)]
    sine2 = [[concept2(value), 1] for value in np.linspace(0, 180, 10000)]

    suddenDrift = sine1 + sine2

    blockedData = [suddenDrift[x:x + 1250] for x in range(0, len(suddenDrift), 1250)]
    random.shuffle(blockedData)
    reocurringConcept = list(itertools.chain.from_iterable(blockedData))

    windowLength = 100

    suddenDriftMean = []
    suddenDriftSum = []

    step = int(windowLength / 2)
    nHists = 5
    eps = 0.01
    hists = [IntMeanEH(windowLength, eps) for _ in range(nHists)]
    means = np.array([])

    for i in range(len(suddenDrift[:1000])):
        if i >= step*(nHists-1):
            for j in range(nHists):
                hists[j].add(i, suddenDrift[i-step*j][0])

        if i >= (windowLength + (nHists - 1)*step):
            means = np.concatenate((means, [hists[j].get_estimate() for j in range(nHists)]))

    print(means)

    plt.plot(means)
    plt.show()