import numpy as np


def check_array(X, squeeze=False):
    try:
        return np.asscalar(X)
    except:
        if squeeze:
            return np.squeeze(X)
        return X


class Interval:
    def __init__(self, low, high, left_inclusive=True, right_inclusive=True):
        if low > high:
            raise ValueError("Interval low must be lesser than high")
        if low == high:
            raise ValueError("Interval low and high cannot be the same value")

        self.low = low
        self.high = high
        self.left_inclusive = left_inclusive
        self.right_inclusive = right_inclusive

    def __lt__(self, val):
        if self.left_inclusive:
            return val < self.low
        return val <= self.low

    def __le__(self, val):
        return val < self or val in self

    def __gt__(self, val):
        if self.right_inclusive:
            return val > self.low
        return val >= self.low

    def __ge__(self, val):
        return val > self or val in self

    def __contains__(self, val):
        if self.left_inclusive and self.right_inclusive:
            return val >= self.low and val <= self.high
        elif self.left_inclusive and not self.right_inclusive:
            return val >= self.low and val < self.high
        elif not self.left_inclusive and self.right_inclusive:
            return val > self.low and val <= self.high
        elif not self.left_inclusive and not self.right_inclusive:
            return val > self.low and val < self.high

