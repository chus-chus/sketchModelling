import random
from src.sketches.EH import BinaryExactWindow, BinaryCountEH

random.seed(888)

if __name__ == "__main__":

    nElems = 100000
    windowLen = 10000
    eps = 0.01
    hist = BinaryCountEH(windowLen, eps)

    exactWindow = BinaryExactWindow(windowLen)
    sumRelativeError = 0
    maxRelativeError = 0
    sumBucketCount = 0
    maxBucketCount = 0

    for eventTimestamp in range(nElems):
        event = random.randint(1, 10) > 5
        exactWindow.add(event)
        hist.add(eventTimestamp, event)
        if eventTimestamp >= windowLen:
            if eventTimestamp % windowLen == 0:
                exactCount = exactWindow.query()
                approxCount = hist.get_estimate()

                if exactCount != 0:
                    relativeError = abs(exactCount - approxCount) / float(exactCount)
                    sumRelativeError += relativeError
                    maxRelativeError = max(maxRelativeError, relativeError)

                sumBucketCount += hist.buckets_count()
                maxBucketCount = max(maxBucketCount, hist.buckets_count())

                print('Estimated: ' + str(approxCount) + ', real: ' + str(exactCount))

    print('Average relative error    =  ' + str(sumRelativeError / nElems))
    print('Maximum relative error    =  ' + str(maxRelativeError))
    print('Relative error violation  =  ' + str(maxRelativeError > eps))
    print('Average bucket count      =  ' + str(sumBucketCount / (nElems - windowLen)))
    print('Size relative to window   =  ' + str((sumBucketCount / nElems) / windowLen))
    print('Maximum bucket count      =  ' + str(maxBucketCount))
    print('Size relative to window   =  ' + str(maxBucketCount / windowLen))