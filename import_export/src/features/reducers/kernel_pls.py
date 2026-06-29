import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer


class KernelPLS(BaseEstimator, TransformerMixin):
    """Kernel PLS via centered Gram matrix + PLSRegression (dual-form KPLS)."""

    def __init__(
        self,
        n_components=2,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
    ):
        self.n_components = int(n_components)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _pairwise(self, X, Y=None):
        return pairwise_kernels(
            X,
            Y,
            metric=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            filter_params=True,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        self.X_train_ = X
        kernel_matrix = self._pairwise(X)
        self._centerer = KernelCenterer()
        kernel_centered = self._centerer.fit_transform(kernel_matrix)

        self.pls_ = PLSRegression(
            n_components=self.n_components,
            scale=False,
        )
        self.pls_.fit(kernel_centered, y)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        kernel_matrix = self._pairwise(X, self.X_train_)
        kernel_centered = self._centerer.transform(kernel_matrix)
        return self.pls_.transform(kernel_centered)
