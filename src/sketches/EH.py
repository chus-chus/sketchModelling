from collections import deque
import math


# Older buckets are to the left. The most recent ones are to the right:
# ... b2^3 b2^2 b2^1 b2^0

# todo update struct for buckets to allow for constant delete time
# todo document

class Bucket(object):

    def __init__(self, timestamp, count):
        self.timestamp = timestamp
        self.count = count


class BinaryCounterEH(object):

    def __init__(self, n, eps):
        self.total = 0
        self.n = n
        self.eps = eps
        self.buckets = deque()
        self.k = math.ceil(1 / self.eps)
        self.bucketThreshold = math.ceil(0.5 * self.k) + 2

    def buckets_count(self):
        return len(self.buckets)

    def add(self, timestamp, event):
        self.remove_expired_buckets(timestamp)
        if not event:
            return
        self.add_new_bucket(timestamp)
        self.merge_buckets()

    def remove_expired_buckets(self, timestamp):
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
        # account for the first bucket
        numberOfSameCount = 1
        bucketIndex = len(self.buckets) - 1
        formerBucket = self.buckets[bucketIndex]
        bucketIndex -= 1
        while bucketIndex > 0:
            try:
                # will break if the oldest bucket is merged
                latterBucket = self.buckets[bucketIndex]
            except IndexError:
                break
            if formerBucket.count == latterBucket.count:
                numberOfSameCount += 1
            else:
                numberOfSameCount = 1
            if numberOfSameCount == self.bucketThreshold:
                formerBucket.count += latterBucket.count
                del self.buckets[bucketIndex]
                # if buckets are merged, next element of deque moves into
                # the position of the deleted bucket. So, index does not change,
                # as it now points to a latter bucket and formerBucket is updated
                # from the merge operation.
            else:
                formerBucket = latterBucket
                bucketIndex -= 1

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
        self.buffer.append(element)
        if len(self.buffer) > self.maxElems:
            elemRemoved = self.buffer.pop()
            if elemRemoved:
                self.nElems -= 1
        if element:
            self.nElems += 1

    def query(self):
        return self.nElems


class NaturalCounterEH(BinaryCounterEH):
    """ An EH devised for counting where elements added are naturals. The idea is to treat each new element
        (elem, timestamp) as a series of 1s with the same timestamp. In order to maintain the EH with small amortized
        time, a buffer is used to keep some elements. When the buffer is full, a new EH is created from the bucket
        and the previous EH using what's called the 'l-canonical' representation. """

    def __init__(self, n, eps):
        super().__init__(n, eps)
        self.maxBufferSize = math.floor(math.log2(n))
        self.buffer = deque()
        self.bufferSum = 0

    def add(self, timestamp, number):
        if not number:
            return
        # bucket appended (most recent items to the right) contains an int and its timestamp.
        self.buffer.appendleft(Bucket(timestamp, number))
        self.bufferSum += number
        if len(self.buffer) == self.maxBufferSize:
            self.remove_expired_buckets(timestamp)
            self.total += self.bufferSum
            self.buckets_from_lcanonical(self.l_canonical(self.total))
            self.buffer.clear()
            self.bufferSum = 0

    def l_canonical(self, totalSum):
        """ Returns the l-canonical representation of a positive integer 'totalSum'. The representation is equivalent
            to knowing the number k_i of buckets of size 2^i, i=0,1,...,n given the number of elements in the EH.
            That is, [k_0, k_1, ..., k_n]. l + 1 stands for the maximum number of buckets of each size. See the README
            file for a basic understanding of the intuition behind the algorithm.

            This implementation uses some clever manipulations from Twitter's Algebird. Thank you! """

        def little_endian_bit(value, n):
            """ Returns the n-th bit (left to right) of the little-endian binary rep. of 'value', 'value' >= 0. """
            return (value >> n) & 1

        if totalSum == 0:
            return []
        num = totalSum + self.bucketThreshold
        den = 1 + self.bucketThreshold
        j = int(math.log2(num / den))
        posInGroup = num - (den * (2**j))
        posMod = posInGroup % (2**j)
        lCanonical = [self.bucketThreshold + little_endian_bit(posMod, nBit) for nBit in range(j)]
        lCanonical.append(math.floor(posInGroup / (2**j)) + 1)
        return lCanonical

    def buckets_from_lcanonical(self, lCanonical):
        """ Overwrites self.buckets with a valid EH histogram taking into account its l-canonical representation
            and the timestamps of the buckets in both the previous EH and the buffer. """
        if not lCanonical:
            return deque()
        else:
            bucketsAndBuffer = self.buckets + self.buffer
            self.buckets.clear()
            for i, nBuckets in enumerate(lCanonical):
                for _ in range(nBuckets):
                    self.buckets.appendleft(self.extract_bucket(bucketsAndBuffer, 2 ** i))
            return

    def extract_bucket(self, bucketsAndBuffer, bucketSize):
        timestamp = bucketsAndBuffer[-1].timestamp
        self.deque_popper(bucketsAndBuffer, bucketSize)
        return Bucket(timestamp, bucketSize)

    def deque_popper(self, dequeObj, nElems):
        """ If the sum of the values of dequeObj is n, deque_popper returns a deque that sums n - nElems. It performs
        substractions and / or pops on the most right hand-side elements of the deque. For example:
            deque_popper(deque([1, 2]), 1) = deque([1, 1])
            deque_popper(deque([1, 2]), 2) = deque([1])
            deque_popper(deque([1, 2]), 3) = deque([])

        The method will never pop an empty deque: the sum of the elements in 'dequeObj' is equal to sum(k_i * 2^i),
        i = [0, ..., j], k_i is each entry of the l-canonical representation. Thus, from the way it's called,
        exactly sum(k_i * 2^i) elements will be substracted from 'dequeObj' (nElems is 2^i). """

        if not dequeObj:
            raise Exception('Empty deque: invalid l-canonical representation.')
        if dequeObj[-1].count == nElems:
            dequeObj.pop()
        elif dequeObj[-1].count > nElems:
            dequeObj[-1].count -= nElems
        else:
            nElems -= dequeObj[-1].count
            dequeObj.pop()
            self.deque_popper(dequeObj, nElems)
        return

    def get_estimate(self):
        if self.buckets_count() == 0:
            return self.bufferSum
        else:
            return int(self.total - self.buckets[0].count / 2) + self.bufferSum
