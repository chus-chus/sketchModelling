from src.EHs.binaryCounterEH import BinaryCounterEH
from src.EHs.sumEH import SumEH


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