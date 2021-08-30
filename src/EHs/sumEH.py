from collections import deque
from math import ceil, log2, floor

from src.EHs.baseStructures import Bucket
from src.EHs.binaryCounterEH import BinaryCounterEH


class SumEH(BinaryCounterEH):
    """ An EH devised for counting where elements added are positive within a relative error eps. From

        "M. Datar et al. (2002). Maintaining Stream Statistics over Sliding Windows. Society for Industrial and Applied
        Mathematics, 31(6), 1794-1813."

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

        if totalSum == 0:
            return []
        num = totalSum + self.lVal
        den = 1 + self.lVal
        j = int(log2(num / den))
        posInGroup = num - (den * (2 ** j))
        posMod = posInGroup % (2 ** j)
        lCanonical = [self.lVal + self.little_endian_bit(posMod, nBit) for nBit in range(j)]
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

    @staticmethod
    def little_endian_bit(value, n):
        """ Returns the n-th bit (left to right) of the little-endian binary rep. of 'value', 'value' >= 0. """
        return (value >> n) & 1

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
                return (int(self.total - self.buckets[0].nElems / 2.) + self.bufferSum) / self.resolution
        else:
            if self.buckets_count() == 0:
                return self.bufferSum
            else:
                return ceil(self.total - self.buckets[0].nElems / 2.) + self.bufferSum

    def empty(self):
        return True if len(self.buckets) == 0 and len(self.buffer) == 0 else False
