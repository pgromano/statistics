import numpy as np


class Distribution:
    """ Generic Distribution Class

    This method handles several python methods as well as operator overloading
    to facillitate operations with numpy ndarrays.
    """

    def __lt__(self, X):
        return self.cumulative(X)

    def __rlt__(self, X):
        return self.__gt__(X)

    def __gt__(self, X):
        return self.survival(X)

    def __rgt__(self, X):
        return self.__lt__(X)

    def __add__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return sample + X

    def __radd__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X.__radd__
        sample = self.sample(*X.shape, dtype=X.dtype)
        return X + sample

    def __sub__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return sample - X

    def __rsub__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return X - sample

    def __mul__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return sample * X

    def __rmul__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return X * sample

    def __truediv__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return sample / X

    def __rtruediv__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return X / sample

    def __floordiv__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return sample // X

    def __rfloordiv__(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        sample = self.sample(*X.shape, dtype=X.dtype)
        return X // sample

    #TODO: lt/e, gt/e as statistical test?

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if ufunc == np.add:
            return self.__radd__(args[0])
        elif ufunc == np.subtract:
            return self.__rsub__(args[0])
        elif ufunc == np.multiply:
            return self.__rmul__(args[0])
        elif ufunc == np.true_divide:
            return self.__rtruediv__(args[0])
        elif ufunc == np.floor_divide:
            return self.__rtruediv__(args[0])
        elif ufunc == np.less:
            return self.__rlt__(args[0])
        elif ufunc == np.greater:
            return self.__rgt__(args[0])
        raise ValueError("{} not supported with {} distribution".format(ufunc, self.__name__))

    def __call__(self, *size, dtype=np.float):
        return self.sample(*size, dtype=dtype)

    @property
    def __name__(self):
        return self.__class__.__name__

    def __str__(self):
        return "{}({})".format(
            self.__name__,
            ', '.join(
                f"{key}={val}"
                for key, val in self.__dict__.items()
                if not key.startswith("_")
                and key != 'seed'
            )
        )

    def __repr__(self):
        return self.__str__()
