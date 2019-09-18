import numpy as np
import scipy.special as sc
from .distribution import Distribution


class Normal(Distribution):
    def __init__(self, loc=0, scale=1, seed=None):
        self.loc = loc
        self.scale = scale
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed
        self._state = np.random.RandomState(seed)

    def sample(self, *size, dtype=np.float):
        return self._state.normal(self.loc, self.scale, size=size).astype(dtype)

    def probability(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return np.exp(-X ** 2 / 2.0) / np.sqrt(2 * np.pi)

    def log_probability(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return -X ** 2 / 2.0 - np.log(np.sqrt(2 * np.pi))

    def cumulative(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        num = X - self.loc
        denom = self.scale * np.sqrt(2)
        return 0.5 * (1 + sc.erf(num / denom))

    def percentile(self, *X):
        if not isinstance(X, np.ndarray):
            X = np.squeeze(X)
        return self.loc + self.scale * sc.erfinv(2 * X - 1) * np.sqrt(2)

    def survival(self, *X):
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
