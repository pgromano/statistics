import numpy as np
import scipy.special as sc
from .distribution import Distribution


class Uniform(Distribution):
    def __init__(self, low=0, high=1, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed
        self._state = np.random.RandomState(seed)

    def sample(self, *size, dtype=np.float):
        return self._state.uniform(self.low, self.high, size=size).astype(dtype)

    def probability(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)

        # Boolean conditions for inside and outside distribution range
        a_bool = np.logical_and(X >= self.low, X < self.high)
        b_bool = np.logical_or(X < self.low, X > self.high)

        # Distribution values by range
        a = 1 / (self.high - self.low)
        b = 0

        # return probability
        return np.piecewise(X, [a_bool, b_bool], [a, b])

    def log_probability(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)

        # Boolean conditions for inside and outside distribution range
        a_bool = np.logical_and(X >= self.low, X < self.high)
        b_bool = np.logical_or(X < self.low, X >= self.high)

        # Distribution values by range
        a = self.low.le(value).type_as(self.low)
        b = self.high.gt(value).type_as(self.low)
        return np.piecewise(X, [a_bool, b_bool], [-np.log(self.high - self.low), -np.inf])

    def cumulative(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)

        # Boolean conditions for below, inside, and above distribution range
        a_bool = X < self.low
        b_bool = np.logical_and(X >= self.low, X < self.high)
        c_bool = X >= self.high

        # values for distribution range
        a = 0
        def b(x): return (x - self.low) / (self.high - self.low) 
        c = 1

        # return cumulative density
        return np.piecewise(X, [a_bool, b_bool, c_bool], [a, b, c])

    def percentile(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        
        # Boolean conditions for outside and inside distribution range
        a_bool = np.logical_or(X < self.low, X >= self.high)
        b_bool = np.logical_and(X >= self.low, X < self.high)
        
        # values for distribution range
        a = np.inf
        def b(x): return x * (self.high - self.low) + self.low

        return np.piecewise(X, [a_bool, b_bool], [a, b])

    def survival(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return 1 - self.cumulative(X)

    @property
    def mean(self):
        return 0.5 * (self.low + self.high)

    @property
    def median(self):
        return 0.5 * (self.low + self.high)

    @property
    def mode(self):
        return self.low, self.high

    @property
    def scale(self):
        return (self.high - self.low) / 12 ** 0.5

    @property
    def variance(self):
        return (self.high - self.low) ** 2 / 12

    @property
    def skewness(self):
        return 0

    @property
    def kurtosis(self):
        return -6 / 5

    @property
    def entropy(self):
        return np.log(self.high - self.low)

    @property
    def perplexity(self):
        return np.exp(self.entropy)
