import random

from src.sketches.EH import IntCountEH

random.seed(888)

if __name__ == "__main__":

    hist = IntCountEH(10000, 0.01)
    print(hist.k)