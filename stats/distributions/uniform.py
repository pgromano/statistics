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

    def probability(self, *X):
        pass

    def log_probability(self, *X):
        pass

    def cumulative(self, *X):
        pass

    def percentile(self, *X):
        pass

    def survival(self, *X):
        pass

    @property
    def variance(self):
        pass

    @property
    def entropy(self):
        pass

    @property
    def perplexity(self):
        pass

    def sample(self, *size, dtype=np.float):
        return self._state.uniform(self.low, self.high, size=size).astype(dtype)
