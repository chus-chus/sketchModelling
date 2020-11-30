from src.sketches.EH import VarEH, ExactWindow
import random


def mean(lst):
    return sum(lst) / len(lst)


def var(lst):
    mu = mean(lst)
    return sum([((x - mu) ** 2) for x in lst])


def diff(set1, set2):

    newset = set1.copy()

    for x in set2:
        newset.remove(x)

    ni = len(set1)
    nj = len(set2)
    nij = ni - nj

    muij = sum(set1 + set2) / len(set1 + set2)
    mui = sum(set1) / len(set1)
    muj = sum(set2) / len(set2)

    vi = var(set1)
    vj = var(set2)

    di = mui - muij
    dj = muj - muij

    realv = var(newset)

    newNElems = len(set1) - len(set2)

    mu = (mean(set1) * len(set1) - mean(set2) * len(set2)) / newNElems

    v = var(set1) - var(set2) - ((newNElems * len(set2)) / len(set1)) * ((mu - mean(set2))**2)

    # v = sum((x - mui + di)**2 for x in set1) + sum((x - muj + dj)**2 for x in set2)

    # v = vi + vj + (di**2)*ni + (dj**2)*nj + 2*di*(sum(set1) - ni*mui) + 2*dj*(sum(set2) - nj*muj)

    # v = vi - vj + (di ** 2) * ni - (dj ** 2) * nj

    # v = sum((x - mui + di)**2 for x in set1) + sum((x - muj + dj)**2 for x in set2)

    return mu, v, realv


def merge(lst1, lst2):
    ni = len(lst1)
    nj = len(lst2)

    realv = var(lst1+lst2)

    v = var(lst1) + var(lst2) + \
        (((ni * nj) / (ni + nj)) *
         ((mean(lst1) - mean(lst2)) ** 2))

    return realv, v


if __name__ == "__main__":
    random.seed(888)

    winLen = 100
    eps = 0.01

    mod = VarEH(winLen, eps, maxValue=10)
    w = ExactWindow(winLen)

    totalVarPreds = 0
    totalMeanPreds = 0

    sumVarRelativeError = 0
    sumMeanRelativeError = 0

    maxVarRelativeError = 0
    maxMeanRelativeError = 0

    numVarViolations = 0
    numMeanViolations = 0
    for i in range(100000):
        elem = random.uniform(-10, 2)
        w.add(elem)
        mod.add(elem)

        realMean = w.mean()
        realVar = w.variance()
        estMean = mod.get_mean_estimate()
        estVar = mod.get_var_estimate()

        # avoid division by 0
        if realVar > 0 and i > winLen:
            totalVarPreds += 1
            relativeVarError = abs(realVar - estVar) / realVar
            if relativeVarError > eps:
                numVarViolations += 1
            maxVarRelativeError = max(maxVarRelativeError, relativeVarError)
            sumVarRelativeError += relativeVarError

        if realMean != 0 and i > winLen:
            totalMeanPreds += 1
            relativeMeanError = abs(realMean - estMean) / abs(realMean)
            if relativeMeanError > eps:
                numMeanViolations += 1
            maxMeanRelativeError = max(maxMeanRelativeError, relativeMeanError)
            sumMeanRelativeError += relativeMeanError

    print('VARIANCE:')
    print('Average relative error:', sumVarRelativeError / totalVarPreds)
    print('Max relative error:', maxVarRelativeError)
    print('Relative error bound:', eps)
    print('Violation of epsilon:', maxVarRelativeError > eps)
    print('Violation proportion: ', numVarViolations / totalVarPreds)
    print('-----------------------')
    print('MEAN:')
    print('Average relative error:', sumMeanRelativeError / totalMeanPreds)
    print('Max relative error:', maxMeanRelativeError)
    print('Relative error bound:', eps)
    print('Violation of epsilon:', maxMeanRelativeError > eps)
    print('Violation proportion: ', numMeanViolations / totalMeanPreds)

