from collections import deque
import math


# Older buckets are to the left. The most recent ones are to the right:
# ... b2^3 b2^2 b2^1 b2^0

# todo update struct for buckets to allow for constant delete time


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
    """ An EH devised for counting where elements added are integers. The idea is to treat each new element
        (elem, timestamp) as a series of 1s with the same timestamp. In order to maintain the EH with small amortized
        time, a buffer is used to keep some elements. When the buffer is full, a new EH is created from the bucket
        and the previous EH using what's called the 'l-canonical' representation. """

    def __init__(self, n, eps):
        super().__init__(n, eps)
        self.maxBufferSize = math.log2(n)
        self.buffer = list()
        self.bufferSum = 0

    def add(self, timestamp, number):
        # insert number bucket in buffer (bucket only contains an int)
        self.buffer.append(Bucket(timestamp, number))
        self.bufferSum += number
        if len(self.buffer) == self.maxBufferSize:
            # remove expired buckets
            self.remove_expired_buckets(timestamp)
            # compute l-canonical
            lCanonical = self.l_canonical(self.total + self.bufferSum)
            # create new EH, empty buffer
            pass

    def l_canonical(self, totalSum):
        """ Returns the l-canonical representation of a positive integer 'totalSum'. The representation is equivalent
            to knowing the number k_i of buckets of size 2^i, i=0,1,...,n given the number of elements in the EH.
            That is, [k_0, k_1, ..., k_n]. The 'l' stands for the maximum number of buckets of each size.

            The reasoning behind the algorithm is as follows: the l-canonical representation groups natural numbers
            into differently sized groups determined by the length of the representation itself ([k_0, k_1, ..., k_n]).
            So, we first need to find in what group our natural number S (the size of the new EH) lies into: that
            is what 'j' is for.
            Because we know the sizes of each group, we know that S's group is of size
                (l + 1) * 2^j
            So, given
            this size and 'l', we can extract the natural number in which 'our' group starts:
                startOfGroup = ((l + 1) * 2^j) - l
            Then, we can tell the position of our S in the group. In our code, this position 'posInGroup' is
            computed from an expression derived from
                posInGroup = S - startOfGroup
            But, what's the use? Actually, it turns out that the first n - 1 components of an l-canonical
            representation can computed from the bits of the little endian rep. of a value given by
            'posInGroup':
                modValue = posInGroup mod 2^j
            Note that the sequence generated by 'modValue' is recurrent: [0, 1, 0, 1, 0, 1]. Then, the y-th (y < n)
            component of the l-canonical rep. of S is
                yThComponent = littleEldianBit(modValue, y)
            That is, the bit in the y-th position of the little eldian representation of 'modValue' plus l.
            The value of the n-th component is finally given by
                nTh = ⌊posInGroup / 2^j⌋ + 1

            This implementation uses some clever manipulations from Twitter's Algebird. Thank you! A visual explanation
            of the presented procedure can be found here:
            https://twitter.github.io/algebird/datatypes/approx/exponential_histogram.html """

        def little_endian_bit(value, n):
            """ Returns the n-th bit (from the right) of the little-endian
                representation of 'value', 'value' >= 0. """
            return (value >> n) & 1

        if totalSum == 0:
            return []
        num = totalSum + self.bucketThreshold
        den = 1 + self.bucketThreshold
        # j s.t. 2^j <= num / den
        j = int(math.log2(num / den))
        # note: (1 << j) is equivalent to (1 * (2^j))

        # position in the group from 0 to (l + 1)2^j - 1)
        posInGroup = num - (den << j)
        # we take advantage of the fact that n mod (2^j) = n & (2^j - 1)
        posMod = posInGroup & ((1 << j) - 1)
        lCanonical = [self.bucketThreshold + little_endian_bit(posMod, nBit) for nBit in range(j)]
        lCanonical.append(math.floor(posInGroup / (1 << j)) + 1)
        return lCanonical

    def get_estimate(self):
        if self.buckets_count() == 0:
            return self.bufferSum
        else:
            return int(self.total - self.buckets[0].count / 2) + self.bufferSum
