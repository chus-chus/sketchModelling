import random

from src.sketches.EH import IntCountEH

random.seed(888)

if __name__ == "__main__":

    hist = IntCountEH(10, 0.01)

    for i in range(10000):
        item = random.randint(0, 10)
        hist.add(i, item)
        if i % 100 == 0:
            print(hist.get_estimate())