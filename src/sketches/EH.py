from collections import deque
import math

# Older buckets are to the left. The most recent ones are to the right:
# ... b2^3 b2^2 b2^1 b2^0


class Bucket(object):

    def __init__(self, timestamp, count):
        self.timestamp = timestamp
        self.count = count


class BinaryCountEH(object):

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
            if self.buckets[0].timestamp <= timestamp - self.n:
                self.total -= self.buckets[0].count
                self.buckets.popleft()
            else:
                break

    def add_new_bucket(self, timestamp):
        self.buckets.append(Bucket(timestamp, 1))
        self.total += 1

    def merge_buckets(self):
        numberOfSameCount = 0
        formerBucket = self.buckets[-1]
        for index in reversed(range(0, self.buckets_count()-2)):
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
                index += 1
            else:
                formerBucket = latterBucket

    def get_estimate(self):
        if self.buckets_count() == 0:
            return 0
        else:
            return int(self.total - self.buckets[0].count / 2)


class BinaryExactWindow(object):
    def __init__(self, size):
        self.nElems = 0
        self.buffer = deque()
        self.maxElems = size

    def add(self, element):
        self.buffer.appendleft(element)
        if len(self.buffer) > self.maxElems:
            elemRemoved = self.buffer.pop()
            if elemRemoved:
                self.nElems -= 1
        if element:
            self.nElems += 1

    def query(self):
        return self.nElems


class IntCountEH(BinaryCountEH):

    def __init__(self, n, eps):
        super().__init__(n, eps)
        self.maxBufferSize = math.log2(n)
        self.buffer = list()
        self.bufferSum = 0

    def add(self, timestamp, number):
        # insert number bucket in buffer (bucket only contains ints)
        self.buffer.append(Bucket(timestamp, number))
        self.bufferSum += number
        if len(self.buffer) == self.maxBufferSize:
            # remove expired buckets
            self.remove_expired_buckets(timestamp)
            # compute l-canonical
            self.l_canonical(self.total + self.bufferSum)
            # create new EH, empty buffer
            pass

    def l_canonical(self, totalSum):
        raise NotImplementedError

    def get_estimate(self):
        if self.buckets_count() == 0:
            return self.bufferSum
        else:
            return int(self.total - self.buckets[0].count / 2) + self.bufferSum

