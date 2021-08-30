from collections import deque


class Bucket(object):
    """ Simple structure with a timestamp and a var. representing number the of elements in it. """

    def __init__(self, timestamp, nElems):
        self.timestamp = timestamp
        self.nElems = nElems


class VarBucket(Bucket):

    """ An extension of the basic bucket that also contains its mean and variance. """

    def __init__(self, timestamp, value):
        # indicate value=None if the Bucket is to be initialized as empty
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
        wraparound has occured. 'tick2' is older than or the same as 'tick1'."""

        if tick1 <= tick2:
            return tick2 - tick1
        else:
            return self.upperLimit - tick1 + tick2


class ExactWindow(object):
    """ Keeps track of exact statistics in a window of size n. """

    def __init__(self, n):
        self.nElems = 0
        self.buffer = deque()
        self.maxElems = n

    def add(self, element):
        self.buffer.appendleft(element)
        self.nElems += 1
        if len(self.buffer) > self.maxElems:
            self.buffer.pop()
            self.nElems -= 1

    def n_elems(self):
        return self.nElems

    def sum(self):
        return sum(self.buffer)

    def mean(self):
        return sum(self.buffer) / self.nElems

    def variance(self):
        if len(self.buffer) <= 1:
            return 0
        variance = 0
        mean = self.mean()
        for i in range(len(self.buffer)):
            variance += (self.buffer[i] - mean) ** 2
        return variance / float(len(self.buffer) - 1)

    def empty(self):
        return True if self.nElems == 0 else False