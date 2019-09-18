import numpy as np
import scipy.special as sc
from .distribution import Distribution


class Uniform(Distribution):
    """ Continous Uniform Distribution

    The continuous uniform distribution or rectangular distribution is a family 
    of symmetric probability distributions such that for each member of the 
    family, all intervals of the same length on the distribution's support are 
    equally probable.

    Arguments
    ---------
    low : float, default=0.0
        The lower bound of the uniform distribution.
    high : float, default=1.0
        The upper bounds of the uniform distribution. Currently, this 
        implementation treats the upper bound exclusively. This will deviate 
        from scipy.stats which inclusively treats the upper bound.
    seed : int, default=None
        The seed to initialize the random number generator.
    """

    def __init__(self, low=0.0, high=1.0, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        """ Reset random number generator """
        if seed is None:
            seed = self.seed
        self._state = np.random.RandomState(seed)

    def sample(self, *size):
        """ Sample from distribution """
        return self._state.uniform(self.low, self.high, size=size)

    def probability(self, *X):
        """ Return the probability density for a given value """ 
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
        """ Return the log probability density for a given value """
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
        """ Return the cumulative density for a given value """
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
        """ Return values for the given percentiles """
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
        """ Return the likelihood of a value or greater """
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
