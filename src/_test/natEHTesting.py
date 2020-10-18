import random

from src.sketches.EH import NaturalCounterEH

random.seed(888)

if __name__ == "__main__":

    hist = NaturalCounterEH(1000, 0.01)

    for i in range(10000):
        item = random.randint(0, 1)
        hist.add(i, item)
        if i % 100 == 0:
            print(hist.get_estimate())