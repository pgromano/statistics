import numpy as np
import scipy.special as sc
from .distribution import Distribution


class Normal(Distribution):
    """ Normal Distribution

    The normal (or Gaussian or Gauss or Laplace–Gauss) distribution is a very 
    common continuous probability distribution. Normal distributions are 
    important in statistics and are often used in the natural and social 
    sciences to represent real-valued random variables whose distributions are 
    not known.

    Arguments
    ---------
    loc : float, default=0.0
        The center, or mean, of the normal distribution.
    high : float, default=1.0
        The full width at half max, or standard deviation, of the normal 
        distribution.
    seed : int, default=None
        The seed to initialize the random number generator.
    """
    def __init__(self, loc=0.0, scale=1.0, seed=None):
        self.loc = loc
        self.scale = scale
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        """ Reset random number generator """
        if seed is None:
            seed = self.seed
        self._state = np.random.RandomState(seed)

    def sample(self, *size):
        """ Sample from distribution """
        return self._state.normal(self.loc, self.scale, size=size)

    def probability(self, *X):
        """ Return the probability density for a given value """
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return np.exp(-X ** 2 / 2.0) / np.sqrt(2 * np.pi)

    def log_probability(self, *X):
        """ Return the log probability density for a given value """
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return -X ** 2 / 2.0 - np.log(np.sqrt(2 * np.pi))

    def cumulative(self, *X):
        """ Return the cumulative density for a given value """
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        num = X - self.loc
        denom = self.scale * np.sqrt(2)
        return 0.5 * (1 + sc.erf(num / denom))

    def percentile(self, *X):
        """ Return values for the given percentiles """
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return self.loc + self.scale * sc.erfinv(2 * X - 1) * np.sqrt(2)

    def survival(self, *X):
        """ Return the likelihood of a value or greater """
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return 1 - self.cumulative(X)

    @property
    def mean(self):
        return self.loc

    @property
    def median(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2

    @property
    def skewness(self):
        return 0

    @property
    def kurtosis(self):
        return 0

    @property
    def entropy(self):
        return 0.5 + 0.5 * np.log(2 * np.pi) + np.log(self.scale)

    @property
    def perplexity(self):
        return np.exp(self.entropy)
