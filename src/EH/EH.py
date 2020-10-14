
from collections import deque
import math
import random

random.seed(888)
# Older buckets are to the right. The most recent ones are to the left:
# b2^0 b2^1 b2^2 b2^3 ...


class EH(object):

    class Bucket(object):

        def __init__(self, timestamp, count):
            self.timestamp = timestamp
            self.count = count

    def __init__(self, n, eps):
        self.total = 0
        self.n = n
        self.eps = eps
        self.buckets = deque()
        self.k = math.ceil(1/self.eps)
        self.threshold = math.ceil(0.5*self.k)+2

    def buckets_count(self):
        return len(self.buckets)

    def add(self, timestamp, event):
        self.remove_expired_buckets(timestamp)
        if not event:
            return
        self.add_new_bucket(timestamp)
        self.merge_buckets()

    def remove_expired_buckets(self, timestamp):
        # check if empty
        if not self.buckets:
            return
        while len(self.buckets) != 0:
            if self.buckets[-1].timestamp <= timestamp - self.n:
                self.total -= self.buckets[-1].count
                self.buckets.pop()
            else:
                break

    def add_new_bucket(self, timestamp):
        self.buckets.appendleft(self.Bucket(timestamp, 1))
        self.total += 1

    def merge_buckets(self):
        numberOfSameCount = 0
        formerBucket = self.buckets[0]
        for index in range(1, self.buckets_count()-1):
            try:
                latterBucket = self.buckets[index]
            except IndexError:
                break
            if formerBucket.count == latterBucket.count:
                numberOfSameCount += 1
            else:
                numberOfSameCount = 1
            # merge
            if numberOfSameCount == self.threshold:
                formerBucket.count += latterBucket.count
                del self.buckets[index]
                # to account for bucket deletion
                index -= 1
            else:
                formerBucket = latterBucket

    def get_estimate(self):
        if self.buckets_count() == 0:
            return 0
        else:
            return int(self.total - self.buckets[-1].count / 2)


class Window:
    def __init__(self, size):
        self.nElems = 0
        self.buffer = deque()
        self.maxElems = size

    def update(self, element):
        self.buffer.appendleft(element)
        if len(self.buffer) > self.maxElems:
            elemRemoved = self.buffer.pop()
            if elemRemoved:
                self.nElems -= 1
        if element:
            self.nElems += 1

    def query(self):
        return self.nElems


if __name__ == "__main__":
    nElems = 1000000
    windowLen = 100000
    eps = 0.01
    hist = EH(windowLen, eps)
    exactWindow = Window(windowLen)
    sumRelativeError = 0
    maxRelativeError = 0
    sumBucketCount = 0
    maxBucketCount = 0

    for eventTimestamp in range(nElems):
        event = random.randint(1, 10) > 5
        exactWindow.update(event)
        hist.add(eventTimestamp, event)
        if eventTimestamp >= windowLen:
            if eventTimestamp % 100000 == 0:
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

