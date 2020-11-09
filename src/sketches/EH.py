# author: Jesus Antonanzas

from collections import deque
import math


# Older buckets are to the left. The most recent ones are to the right:
# ... b2^3 b2^2 b2^1 b2^0

# todo update struct for buckets to allow for constant delete time (mid of struct)
# todo document
# todo merge sumEH and meanEH
# todo assert isinstace of values passed
# todo recycle timestamps


class Bucket(object):
    """ Simple structure with a timestamp and a var. representing number the of elements in it. """

    def __init__(self, timestamp, nElems):
        self.timestamp = timestamp
        self.nElems = nElems


class BinaryCounterEH(object):
    """ Solves the Basic Counting problem, counting the number of elements in a window of length n with a relative error
        eps. """

    def __init__(self, n, eps):
        self.total = 0
        self.n = n
        self.buckets = deque()
        self.k = math.ceil(1 / eps)
        self.bucketThreshold = math.ceil(self.k / 2) + 2

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
                self.total -= self.buckets[0].nElems
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
            if formerBucket.nElems == latterBucket.nElems:
                numberOfSameCount += 1
            else:
                numberOfSameCount = 1
            if numberOfSameCount == self.bucketThreshold:
                formerBucket.nElems += latterBucket.nElems
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
            return int(self.total - self.buckets[0].nElems / 2)

    def empty(self):
        return True if len(self.buckets) == 0 else False


class SumEH(BinaryCounterEH):
    """ An EH devised for counting where elements added are positive within a relative error eps.
        The idea is to treat each new element (elem, timestamp) as a series of 1s with the same timestamp.
        In order to maintain the EH with small amortized time, a buffer is used to keep some elements.
        When the buffer is full, a new EH is created from the bucket and the previous EH using what's called
        the 'l-canonical' representation. """

    def __init__(self, n, eps, isReal=False, resolution=100):
        super().__init__(n, eps)
        self.isReal = isReal
        self.resolution = resolution
        self.lVal = math.ceil(self.k / 2)
        self.maxBufferSize = math.floor(math.log2(n))
        self.buffer = deque()
        self.bufferSum = 0

    def add(self, timestamp, number):
        """ Add a positive integer to the window """
        self.remove_expired_buckets(timestamp)
        if not number:
            return
        if self.isReal:
            number = int(number * self.resolution)
        # bucket appended (most recent items to the right) contains an int and its timestamp.
        self.buffer.appendleft(Bucket(timestamp, number))
        self.bufferSum += number
        # todo remove expired buckets every n events even if numbers arriving are 0
        # todo check if maxBufferSize refers to sum or nElems
        # todo check problems if a lot of elems are 0 (keeps the mean until a non-zero elem arrives)
        if len(self.buffer) == self.maxBufferSize:
            self.total += self.bufferSum
            self.rebucket_from_lcanonical(self.l_canonical(self.total))

    def l_canonical(self, totalSum):
        """ Returns the l-canonical representation of a positive integer 'totalSum'. The representation is equivalent
            to knowing the number k_i of buckets of size 2^i, i=0,1,...,n given the number of elements in the EH.
            That is, [k_0, k_1, ..., k_n]. l + 1 stands for the maximum number of buckets allowed of each size. See the
            README file for a basic understanding of the intuition behind the algorithm.

            This implementation uses some clever manipulations from Twitter's Algebird. Thank you! """

        def little_endian_bit(value, n):
            """ Returns the n-th bit (left to right) of the little-endian binary rep. of 'value', 'value' >= 0. """
            return (value >> n) & 1

        if totalSum == 0:
            return []
        num = totalSum + self.lVal
        den = 1 + self.lVal
        j = int(math.log2(num / den))
        posInGroup = num - (den * (2 ** j))
        posMod = posInGroup % (2 ** j)
        lCanonical = [self.lVal + little_endian_bit(posMod, nBit) for nBit in range(j)]
        lCanonical.append(math.floor(posInGroup / (2 ** j)) + 1)
        return lCanonical

    def rebucket_from_lcanonical(self, lCanonical):
        """ Overwrites self.buckets with a valid EH histogram taking into account its l-canonical representation
            and the timestamps of the buckets in both the previous EH and the buffer. Empties buffer."""
        if not lCanonical:
            return deque()
        else:
            # decreasingly ordered by timestamp
            bucketsAndBuffer = self.buckets + self.buffer
            self.buckets.clear()
            self.buffer.clear()
            self.bufferSum = 0
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
        if dequeObj[-1].nElems == nElems:
            dequeObj.pop()
        elif dequeObj[-1].nElems > nElems:
            dequeObj[-1].nElems -= nElems
        else:
            nElems -= dequeObj[-1].nElems
            dequeObj.pop()
            self.deque_popper(dequeObj, nElems)
        return

    def get_estimate(self):
        if self.isReal:
            if self.buckets_count() == 0:
                return self.bufferSum / self.resolution
            else:
                return (int(self.total - self.buckets[0].nElems / 2) + self.bufferSum) / self.resolution
        else:
            if self.buckets_count() == 0:
                return self.bufferSum
            else:
                return int(self.total - self.buckets[0].nElems / 2) + self.bufferSum

    def empty(self):
        return True if len(self.buckets) == 0 and len(self.buffer) == 0 else False


class MeanEH(object):
    """ Keeps track of the mean of the elements (positive integers) in a window of size n with a relative error very
        close to eps. """

    def __init__(self, n, eps, isReal=False, resolution=100):
        self.sumEH = SumEH(n, eps, isReal, resolution)
        self.nElemsEH = BinaryCounterEH(n, eps)

    def add(self, timestamp, number):
        if not number:
            return
        self.sumEH.add(timestamp, number)
        self.nElemsEH.add(timestamp, 1)

    def get_estimate(self):
        nItems = self.nElemsEH.get_estimate()
        return 0 if not nItems else self.sumEH.get_estimate() / nItems

    def empty(self):
        return self.sumEH.empty()


class ExactWindow(object):
    """ Keeps track of exact statistics in a window of size n. """

    def __init__(self, n):
        self.nElems = 0
        self.buffer = deque()
        self.maxElems = n

    def add(self, element):
        self.buffer.appendleft(element)
        if len(self.buffer) > self.maxElems:
            elemRemoved = self.buffer.pop()
            if elemRemoved:
                self.nElems -= 1
        if element:
            self.nElems += 1

    def n_elems(self):
        return self.nElems

    def sum(self):
        return sum(self.buffer)

    def mean(self):
        return sum(self.buffer) / self.nElems

    def variance(self):
        variance = 0
        mean = self.mean()
        for i in range(len(self.buffer)):
            variance += (self.buffer[i] - mean) ** 2
        return variance

    def empty(self):
        return True if self.nElems == 0 else False


class VarBucket(Bucket):

    """ An extension of the basic bucket that also contains its mean and its variance. """

    def __init__(self, timestamp, value):
        super().__init__(timestamp, 1)
        self.bucketMean = value
        self.var = 0


class VarEH(object):

    def __init__(self, n, eps):
        self.n = n
        self.k = 9 / (eps ** 2)
        self.buckets = deque([])
        self.lastSuffix = VarBucket(0, 0)
        self.interSuffix = None

    def add(self, timestamp, value):
        """ Todo overview of procedure """
        # New element does not affect statistics
        if value == self.buckets[-1].bucketMean:
            self.buckets[-1].nElems += 1
        else:
            self.buckets.append(VarBucket(timestamp, value))
            self.insert_into_last_suffix()

        # Delete expired bucket
        if self.buckets[0].timestamp <= timestamp - self.n:
            # update suffix
            self.pop_from_last_suffix()
            self.buckets.popleft()

        # Merge buckets
        if len(self.buckets) > 2:
            self.interSuffix = VarBucket(0, 0)
            i = len(self.buckets) - 3
            j = i + 1
            newNElems = self.buckets[i].nElems + self.buckets[j].nElems
            newVar = self.compute_new_variance(self.buckets[i], self.buckets[j], newNElems)
            self.update_inter_suffix(len(self.buckets) - 1)
            while i > 0 and self.k * newVar <= self.interSuffix.var:
                # Merge buckets with combination rule
                self.buckets[i].bucketMean = self.compute_new_mean(self.buckets[i], self.buckets[j], newNElems)
                self.buckets[i].nElems = newNElems
                self.buckets[i].var = newVar
                self.buckets[i].timestamp = self.buckets[j].timestamp
                # Update suffix bucket before deleting
                self.update_inter_suffix(j)
                del self.buckets[j]
                # Prepare for conditional check
                i -= 1
                j = i + 1
                newNElems = self.buckets[i].nElems + self.buckets[j].nElems
                newVar = self.compute_new_variance(self.buckets[i], self.buckets[j], newNElems)

    @staticmethod
    def compute_new_mean(bucket1, bucket2, nElems):
        """ Computes the mean of the combination of two buckets. """
        return (bucket1.bucketMean * bucket1.nElems + bucket2.bucketMean * bucket2.nElems) / nElems

    @staticmethod
    def compute_new_variance(bucket1, bucket2, nElems):
        """ Computes the variance of the combination of two buckets. """
        return (bucket1.var + bucket2.var + ((bucket1.nElems * bucket2.nElems) / nElems) *
                ((bucket1.bucketMean - bucket2.bucketMean) ** 2))

    def update_inter_suffix(self, index):
        """ Updates the suffix bucket B_index* (see reference -> insert procedure -> step 3) such that it now also
        contains the statistics of elements from bucket in position 'index'. That is, B_index* now
        represents B_(index+1)*. When called, it assumes that the new elements to include have not been taken into
        account yet. """
        # Reminder: old buckets are in low indexes.
        newNElems = self.interSuffix.nElems + self.buckets[index].nElems
        self.interSuffix.bucketMean = self.compute_new_mean(self.buckets[index], self.interSuffix, newNElems)
        self.interSuffix.var = self.compute_new_variance(self.buckets[index], self.interSuffix, newNElems)
        self.interSuffix.nElems = newNElems

    def insert_into_last_suffix(self):
        """ Updates the statistics of the suffix bucket B_m* (in reference) such that it now takes
        into account the elements in the most recently created bucket. """
        newNElems = self.lastSuffix.nElems + self.buckets[-1].nElems
        self.lastSuffix.bucketMean = self.compute_new_mean(self.lastSuffix, self.buckets[-1], newNElems)
        self.lastSuffix.var = self.compute_new_variance(self.lastSuffix, self.buckets[-1], newNElems)
        self.lastSuffix.nElems = newNElems

    def pop_from_last_suffix(self):
        """ Updates the statistics of the suffix bucket B_m* (in reference) such that it does not take
         into account the oldest bucket anymore: it now represents B_(m-1)* """
        # todo delete statistics from last bucket
        pass

    def get_estimate(self):
        pass

