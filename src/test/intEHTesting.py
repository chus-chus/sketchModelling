import random

from src.sketches.EH import IntSumEH, IntMeanEH

random.seed(888)

if __name__ == "__main__":
    # (num. of timeticks to monitor, eps)
    # histSum = NaturalSumEH(100, 0.01)
    histMean = IntMeanEH(100, 0.01)

    for i in range(10000):
        item = random.randint(0, 10)
        histMean.add(i, item)
        if i % 100 == 0:
            print(histMean.get_estimate())
