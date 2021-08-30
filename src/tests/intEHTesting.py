import random

from src.EHs.binaryCounterEH import MeanEH, SumEH, ExactWindow

random.seed(888)

if __name__ == "__main__":
    # (num. of timeticks to monitor, eps)
    # histSum = NaturalSumEH(100, 0.01)
    eps = 0.05
    winLength = 500

    # histNum = BinaryCounterEH(winLength, eps)
    histNum = SumEH(winLength, eps)
    # histNum = MeanEH(winLength, eps)
    trueWindow = ExactWindow(winLength)

    nSumViolations = 0
    nMeanViolations = 0
    nNumViolations = 0

    meanMeanViolation = 0
    sumRelativeError = 0
    maxSumRelativeError = 0
    meanNumViolation = 0

    nPred = 0

    for i in range(10000):
        item = random.randint(1, 10)
        #histMean.add(i, item)
        #histSum.add(i, item)
        trueWindow.add(item)
        histNum.add(i, item)

        if i >= winLength:

            nPred += 1
            n = histNum.get_estimate()
            trueN = trueWindow.sum()

            # avoid division by 0
            if trueN > 0:
                relativeError = abs(trueN - n) / float(trueN)
                sumRelativeError += relativeError
                maxSumRelativeError = max(maxSumRelativeError, relativeError)

            if n != trueN:
                if relativeError > 0.01:
                    a = 1

    print('Average relative error: ', sumRelativeError / nPred)
    print('Max relative error: ', maxSumRelativeError)
    print('Relative error violation: ', maxSumRelativeError > eps)

