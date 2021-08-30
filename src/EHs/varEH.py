from collections import deque
from math import log2

from src.EHs.baseStructures import VarBucket, Counter


class VarEH(object):

    # todo add reset function, eh summary function, robustness when returning estimates

    """ A variation of the original EH structure that keeps track of the variance (k-medians with k = 1) to some eps
    of relative error from

    "B. Babcock et al. (2003). Maintaining Variance and k-Medians over Data Stream Windows.
    Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems, 22, 234-243."

    Although less space-efficient than the original, it still is sublinear to window length and
    provides approximations in constant time. Moreover, it is more flexible in terms of the functions it supports and
    can work with real numbers. Amortized update time is O(1) given that the |max. value| of the data is known a priori.

    A consequence of this structure is the ability to also keep track of the mean. Note that although the guarantees for
    the maximum relative error for this mean estimate are not presented in the paper, they hold experimentally
    in the majority of cases.

    In this documentation we adopt the notation of the paper, where buckets are refered to as B_i, 1 <= i <= m,
    B_1 being the most recent one and B_m being the oldest one. A suffix bucket B_i* contains the statistics of all
    elements arrived after the most recent element of the bucket B_i. """

    def __init__(self, n, eps, maxValue=None):
        self.n = n
        self.k = 9 / (eps ** 2)
        self.buckets = deque([])
        self.lastSuffix = VarBucket(0, None)
        self.interSuffix = None
        # timestamps up to n + 1 excluded
        self.timeCounter = Counter(n+1)

        self.stepsBetweenMerges = int(round((1 / eps) * log2(n * (maxValue ** 2)))) if maxValue is not None else 1
        # elements processed since last merge
        self.stepsSinceLastMerge = 0

    def add(self, value):

        """ Process a new element arrival, updating the statistics of the structure. If the EH is empty, just insert it.
            If there's at least one element:
                1. Update B_m* (lastSuffix)
                2. Insert the value (either in the previous or in a new Bucket)
                3. If the last bucket is expired, delete it and update B_m* so that it represents B_(m-1)*
                4. If 'stepsBetweenMerges' elements have arrived after the last bucket merge, merge buckets again. """

        self.timeCounter.increment()
        self.stepsSinceLastMerge += 1

        if self.buckets:
            # maintain B_m*
            self.insert_into_last_suffix(value)
            # new element does not affect statistics
            if value == self.buckets[-1].bucketMean:
                self.buckets[-1].nElems += 1
                self.buckets[-1].timestamp = self.timeCounter.step
            else:
                self.buckets.append(VarBucket(self.timeCounter.step, value))
        else:
            self.buckets.append(VarBucket(self.timeCounter.step, value))
            return

        # delete expired bucket, check on counter's wraparound property
        if self.get_timestamp_position(self.buckets[0].timestamp) > self.n:
            self.pop_from_last_suffix()
            self.buckets.popleft()

        # merge every self.stepsBetweenMerge steps to ensure amortized time O(1) (only if maxValue has been specified)
        if self.stepsSinceLastMerge == self.stepsBetweenMerges:
            self.merge_buckets()
            self.stepsSinceLastMerge = 0

    def insert_into_last_suffix(self, element):
        """ Updates the statistics of the suffix bucket B_m* (in reference) such that it now takes
        into account another element. """

        # order of operations crucial!
        newNElems = self.lastSuffix.nElems + 1

        self.lastSuffix.var += self.lastSuffix.nElems * ((self.lastSuffix.bucketMean - element) ** 2) / newNElems

        self.lastSuffix.bucketMean = (self.lastSuffix.bucketMean * self.lastSuffix.nElems + element) / newNElems

        self.lastSuffix.nElems = newNElems

    def pop_from_last_suffix(self):

        """ Updates the statistics of the suffix bucket B_m* (in reference) such that it does not take
         into account the oldest bucket anymore: it now represents B_(m-1)* """

        newNElems = self.lastSuffix.nElems - self.buckets[1].nElems

        if newNElems == 0:
            self.lastSuffix.bucketMean = 0
            self.lastSuffix.var = 0
            self.lastSuffix.nElems = 0
        else:
            # order of operations crucial!
            self.lastSuffix.bucketMean = (self.lastSuffix.bucketMean * self.lastSuffix.nElems -
                                          self.buckets[1].bucketMean * self.buckets[1].nElems) / newNElems

            self.lastSuffix.var = (self.lastSuffix.var - self.buckets[1].var -
                                   ((newNElems*self.buckets[1].nElems)/self.lastSuffix.nElems) *
                                   ((self.lastSuffix.bucketMean - self.buckets[1].bucketMean) ** 2))
            self.lastSuffix.nElems = newNElems

    def merge_buckets(self):

        """ Merges buckets following the procedure specified in the paper. Given V_(i,i-1) the variance of the
        combination of buckets B_i and B_(i-1), V_(i-1)* the variance of the suffix bucket B_(i-1)* and k=9 * (1/eps^2):

            while there exists i > 2:
                find the smallest i that satisfies k * V_(i,i-1) <= V_(i-1)*
                merge buckets B_i and B_(i-1)

        Note that V_(i-1)* is computed incrementally. """

        if len(self.buckets) > 2:
            self.interSuffix = VarBucket(0, None)
            # this implementation has the most recent buckets to the end of the structure self.buckets, hence i is
            # traversed decreasingly.
            i = len(self.buckets) - 3
            j = i + 1
            newNElems = self.buckets[i].nElems + self.buckets[j].nElems
            newVar = self.compute_new_variance(self.buckets[i], self.buckets[j], newNElems)
            self.update_inter_suffix(len(self.buckets) - 1)
            while i >= 0:
                if self.k * newVar <= self.interSuffix.var:
                    if i == 0:
                        # merging last two buckets, so need to update prefix B_m*
                        self.pop_from_last_suffix()
                    self.buckets[i].bucketMean = self.compute_new_mean(self.buckets[i], self.buckets[j], newNElems)
                    self.buckets[i].nElems = newNElems
                    self.buckets[i].var = newVar
                    self.buckets[i].timestamp = self.buckets[j].timestamp
                    del self.buckets[j]
                    # with bucket deleted, j represents buckets[j+1] before deletion
                self.update_inter_suffix(j)
                # prepare for next conditional check
                j = i
                i -= 1
                newNElems = self.buckets[i].nElems + self.buckets[j].nElems
                newVar = self.compute_new_variance(self.buckets[i], self.buckets[j], newNElems)

    def update_inter_suffix(self, index):

        """ Updates the suffix bucket B_index* (see reference -> insert procedure -> step 3) such that it now also
        contains the statistics of elements from bucket in position 'index'. That is, B_index* now
        represents B_(index+1)*. When called, it assumes that the new elements to include have not been taken into
        account yet. """

        newNElems = self.interSuffix.nElems + self.buckets[index].nElems
        self.interSuffix.var = self.compute_new_variance(self.buckets[index], self.interSuffix, newNElems)

        self.interSuffix.bucketMean = self.compute_new_mean(self.buckets[index], self.interSuffix, newNElems)
        self.interSuffix.nElems = newNElems

    @staticmethod
    def compute_new_mean(bucket1, bucket2, nElems):

        """ Computes the mean of the combination of two buckets. """

        return (bucket1.bucketMean * bucket1.nElems + bucket2.bucketMean * bucket2.nElems) / nElems

    @staticmethod
    def compute_new_variance(bucket1, bucket2, nElems):

        """ Computes the variance of the combination of two buckets. """

        return (bucket1.var + bucket2.var + ((bucket1.nElems * bucket2.nElems) / nElems) *
                ((bucket1.bucketMean - bucket2.bucketMean) ** 2))

    def get_timestamp_position(self, timestamp):

        """ Gets position of an element in the EH (from 1 to n) based on its timestamp. """

        return self.timeCounter.dist_between_ticks(timestamp, self.buckets[-1].timestamp) + 1

    def get_var_estimate(self):

        """ Returns an estimate of the variance within the window. """

        numEst = self.n + 1 - self.get_timestamp_position(self.buckets[0].timestamp)
        return (self.buckets[0].var / 2 + self.lastSuffix.var +
                ((numEst * self.lastSuffix.nElems)/(numEst + self.lastSuffix.nElems)) *
                ((self.buckets[0].bucketMean - self.lastSuffix.bucketMean)**2)) / (numEst + self.lastSuffix.nElems - 1)

    def get_mean_estimate(self):

        """ Returns an estimate of the mean within the window. """

        numEst = self.n + 1 - self.get_timestamp_position(self.buckets[0].timestamp)
        return (((numEst * self.buckets[0].bucketMean) +
                 (self.lastSuffix.nElems * self.lastSuffix.bucketMean)) /
                (numEst + self.lastSuffix.nElems))

    def empty(self):

        """ Tells if there are no buckets in the sketch. """

        return False if self.buckets else True
