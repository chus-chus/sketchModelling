# author: Jesus Antonanzas

from collections import deque
from math import ceil
from src.EHs.baseStructures import Bucket

# Older buckets are to the left. The most recent ones are to the right:
# ... b2^3 b2^2 b2^1 b2^0

# todo update struct for buckets to allow for constant delete time (mid of struct)


class BinaryCounterEH(object):
    """ Solves the Basic Counting problem, counting the number of elements in a window of length n with a relative error
        eps. From "M. Datar et al. (2002). Maintaining Stream Statistics over Sliding Windows. Society for Industrial
        and Applied Mathematics, 31(6), 1794-1813." """

    # todo recycle timestamps
    # todo skips relative error bound when window is small and eps is low (eps=0.01, w=100)

    def print_eh(self):
        print("Buckets:", [bucket.nElems for bucket in self.buckets])
        print("timestamps:", [bucket.timestamp for bucket in self.buckets])

    def __init__(self, n, eps):
        self.total = 0
        self.n = n
        self.buckets = deque()
        self.k = ceil(1 / eps)
        self.bucketThreshold = ceil(self.k * 0.5) + 2

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
        while len(self.buckets) > 0:
            if self.buckets[0].timestamp <= timestamp - self.n:
                self.total -= self.buckets[0].nElems
                self.buckets.popleft()
            else:
                break

    def add_new_bucket(self, timestamp):
        self.buckets.append(Bucket(timestamp, 1))
        self.total += 1

    def merge_buckets(self):
        """ Traverse buckets comparing current count with the latter buckets until they are different
            while counting the number of buckets that are of the same capacity and merging the oldest if necessary. """
        bucketIndex = len(self.buckets) - 1
        # most recent bucket with a specific number of elements inside
        currentCount = self.buckets[bucketIndex].nElems
        while bucketIndex >= 0:
            nBucketsWithSameCount = 0
            # count buckets of the same capacity until either end of deque or new capacity found
            while bucketIndex >= 0 and currentCount == self.buckets[bucketIndex].nElems:
                nBucketsWithSameCount += 1
                bucketIndex -= 1

            # either:
            # 1) end of deque and not over threshold: stop
            # 2) discrepancy between counts and not over threshold: stop because latter elements won't need to be merged
            # if discrepancy between counts and over threshold merge until either 1) or 2)

            # if discrepancy bucketIndex points to the most recent bucket of new (bigger) capacity
            if nBucketsWithSameCount >= self.bucketThreshold:
                # merge
                while nBucketsWithSameCount >= self.bucketThreshold:
                    # no illegal accesses thanks to how the threshold is computed
                    self.buckets[bucketIndex + 1].nElems += self.buckets[bucketIndex + 2].nElems
                    self.buckets[bucketIndex + 1].timestamp = self.buckets[bucketIndex + 2].timestamp
                    del self.buckets[bucketIndex + 2]
                    nBucketsWithSameCount -= 1
                    bucketIndex += 1
                # update capacity to compare to
                currentCount = self.buckets[bucketIndex].nElems
            # no discrepancy, stop merging
            else:
                break

    def get_estimate(self):
        if self.buckets_count() == 0:
            return 0
        else:
            return ceil(self.total - self.buckets[0].nElems / 2.)

    def empty(self):
        return True if len(self.buckets) == 0 else False
