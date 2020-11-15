# author: Jesus Antonanzas

from collections import deque
from math import ceil, floor, log2

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
        self.k = ceil(1 / eps)
        self.bucketThreshold = ceil(self.k / 2) + 2

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
                # todo revise, indexes do change
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
        self.lVal = ceil(self.k / 2)
        self.maxBufferSize = floor(log2(n))
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
        j = int(log2(num / den))
        posInGroup = num - (den * (2 ** j))
        posMod = posInGroup % (2 ** j)
        lCanonical = [self.lVal + little_endian_bit(posMod, nBit) for nBit in range(j)]
        lCanonical.append(floor(posInGroup / (2 ** j)) + 1)
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
        # Indicate value=None if the Bucket is to be initialized empty
        if value is None:
            super().__init__(timestamp, 0)
            self.bucketMean = 0
        else:
            super().__init__(timestamp, 1)
            self.bucketMean = value

        self.var = 0


class Counter(object):

    """ A simple wraparound counter. """

    def __init__(self, upperLimit):
        self.step = 0
        self.upperLimit = upperLimit

    def increment(self):
        if self.step < self.upperLimit:
            self.step += 1
        else:
            # reset
            self.step = 1

    def dist_between_ticks(self, tick1, tick2):

        """ Returns the number of steps that have taken place between 'tick1' and 'tick2' assuming that at most 1
        wraparound has occured. """

        if tick1 <= tick2:
            return tick2 - tick1
        else:
            return self.upperLimit - tick1 + tick2


class VarEH(object):

    """ A variation of the original EH structure that keeps track of the variance (k-medians with k = 1) to some eps
    of relative error and mean in a window of length n. Although less space-efficient than the original, it still is
    sublinear to window length and provides approximations in constant time. Moreover, it is more flexible in terms of
    the functions it supports and can work with real numbers. Amortized update time is O(1) given that the |max. value|
    of the data is known a priori.

    Note that although the guarantees for the maximum relative error for the estimate of the mean are not presented in
    the paper, but almost always hold experimentally. # todo test thoroughly"""

    def num_elements(self):

        """ Returns the number of elements (expired and not expired) contained within the structure. """

        nElems = 0
        for bucket in self.buckets:
            nElems += bucket.nElems
        return nElems

    def print_eh(self):

        """ Prints buckets and their information from most recent to oldest (0). """

        for i in range(len(self.buckets)-1, -1, -1):
            print('Bucket ', i, '. nElems = ', self.buckets[i].nElems, ', mean = ', self.buckets[i].bucketMean,
                  ', var = ', self.buckets[i].var, ', timestamp = ', self.buckets[i].timestamp)

    def __init__(self, n, eps, maxValue=None):
        self.n = n
        self.k = 9 / (eps ** 2)
        self.buckets = deque([])
        self.lastSuffix = VarBucket(0, None)
        self.interSuffix = None
        self.timeCounter = Counter(2*n)

        # If resolution is not specified, then amortized running time per element will not be O(1)
        self.stepsBetweenMerges = int(round((1 / eps) * log2(n * (maxValue ** 2)))) if maxValue is not None else 1
        # Elements processed since last merge
        self.stepsSinceLastMerge = 0

    def add(self, value):

        # todo make max value of wraparaound timestamp consistent.
        """ Process a new element arrival, updating the statistics of the structure. If the EH is empty, just insert it.
            If there's at least one element:
                1. Update B_m* (lastSuffix)
                2. Insert the value (either in the previous or in a new Bucket)
                3. If the last bucket is expired, delete it and update B_m* so that it represents B_(m-1)*
                4. If 'stepsBetweenMerges' elements have arrived after the last bucket merge, merge buckets again. """

        self.timeCounter.increment()
        self.stepsSinceLastMerge += 1

        if self.buckets:
            # Maintain B_m*
            self.insert_into_last_suffix(value)
            # New element does not affect statistics
            if value == self.buckets[-1].bucketMean:
                self.buckets[-1].nElems += 1
                self.buckets[-1].timestamp = self.timeCounter.step
            else:
                self.buckets.append(VarBucket(self.timeCounter.step, value))
        else:
            self.buckets.append(VarBucket(self.timeCounter.step, value))
            return

        # Delete expired bucket, check on counter's wraparound property
        if self.get_position(self.buckets[0].timestamp) > self.n:
            self.pop_from_last_suffix()
            self.buckets.popleft()

        # Merge every self.stepsBetweenMerge steps to ensure amortized time O(1) (only if maxValue has been specified)
        if self.stepsSinceLastMerge == self.stepsBetweenMerges:
            self.merge_buckets()
            self.stepsSinceLastMerge = 0

    @staticmethod
    def compute_new_mean(bucket1, bucket2, nElems):

        """ Computes the mean of the combination of two buckets. """

        return (bucket1.bucketMean * bucket1.nElems + bucket2.bucketMean * bucket2.nElems) / nElems

    @staticmethod
    def compute_new_variance(bucket1, bucket2, nElems):

        """ Computes the variance of the combination of two buckets. """

        return (bucket1.var + bucket2.var + ((bucket1.nElems * bucket2.nElems) / nElems) *
                ((bucket1.bucketMean - bucket2.bucketMean) ** 2))

    def insert_into_last_suffix(self, element):
        """ Updates the statistics of the suffix bucket B_m* (in reference) such that it now takes
        into account another element. """
        # order of operations crucial!
        newNElems = self.lastSuffix.nElems + 1
        self.lastSuffix.var = (self.lastSuffix.var + (self.lastSuffix.nElems / newNElems) *
                               ((self.lastSuffix.bucketMean - element) ** 2))
        self.lastSuffix.bucketMean = (self.lastSuffix.bucketMean * self.lastSuffix.nElems + element) / newNElems
        self.lastSuffix.nElems = newNElems

    def pop_from_last_suffix(self):

        """ Updates the statistics of the suffix bucket B_m* (in reference) such that it does not take
         into account the oldest bucket anymore: it now represents B_(m-1)* """

        # order of operations crucial!
        newNElems = self.lastSuffix.nElems - self.buckets[1].nElems
        self.lastSuffix.bucketMean = (self.lastSuffix.bucketMean * self.lastSuffix.nElems -
                                      self.buckets[1].bucketMean * self.buckets[1].nElems) / newNElems
        self.lastSuffix.var = (self.lastSuffix.var - self.buckets[1].var -
                               ((newNElems*self.buckets[1].nElems)/self.lastSuffix.nElems) *
                               ((self.lastSuffix.bucketMean - self.buckets[1].bucketMean) ** 2))
        self.lastSuffix.nElems = newNElems

    def get_var_estimate(self):

        """ Returns an estimate of the variance within the window. """

        numEst = self.n + 1 - self.get_position(self.buckets[0].timestamp)
        return (self.buckets[0].var / 2 + self.lastSuffix.var +
                ((numEst * self.lastSuffix.nElems)/(numEst + self.lastSuffix.nElems)) *
                ((self.buckets[0].bucketMean - self.lastSuffix.bucketMean)**2))

    def get_mean_estimate(self):

        """ Returns an estimate of the mean within the window. """

        numEst = self.n + 1 - self.get_position(self.buckets[0].timestamp)
        return (((numEst * self.buckets[0].bucketMean) +
                 (self.lastSuffix.nElems * self.lastSuffix.bucketMean)) /
                (numEst + self.lastSuffix.nElems))

    def get_position(self, timestamp):

        """ Gets position of an element in the EH (from 1 to n) based on its timestamp. """

        return self.timeCounter.dist_between_ticks(timestamp, self.buckets[-1].timestamp) + 1

    def merge_buckets(self):
        # Merge buckets
        if len(self.buckets) > 2:
            self.interSuffix = VarBucket(0, None)
            i = len(self.buckets) - 3
            j = i + 1
            newNElems = self.buckets[i].nElems + self.buckets[j].nElems
            newVar = self.compute_new_variance(self.buckets[i], self.buckets[j], newNElems)
            self.update_inter_suffix(len(self.buckets) - 1)
            # here
            while i >= 0:
                if self.k * newVar <= self.interSuffix.var:
                    self.buckets[i].bucketMean = self.compute_new_mean(self.buckets[i], self.buckets[j], newNElems)
                    self.buckets[i].nElems = newNElems
                    self.buckets[i].var = newVar
                    self.buckets[i].timestamp = self.buckets[j].timestamp
                    # Update intermediate suffix bucket before deleting the bucket
                    del self.buckets[j]
                    # with bucket deleted, j represents buckets[j+1] before deletion
                else:
                    self.update_inter_suffix(j)
                # Prepare for conditional check
                j = i
                i -= 1
                newNElems = self.buckets[i].nElems + self.buckets[j].nElems
                newVar = self.compute_new_variance(self.buckets[i], self.buckets[j], newNElems)

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
