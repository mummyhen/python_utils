import numpy as np
import scipy.sparse as sp
from scipy import linalg
from sklearn.base import RegressorMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model.base import LinearModel, _preprocess_data, _rescale_data
from sklearn.utils import check_X_y

from ..utils import check_constraints, optim_fun


class nnLinearRegression(LinearModel, RegressorMixin):
    """
    least squares Linear Regression with optional constraints on coefficients.

    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
        an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, optional, default 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. This will only provide speedup for
        n_targets > 1 and sufficient large problems.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : array
        Independent term in the linear model.

    Notes
    -----
    From the implementation point of view, this is just Ordinary
    Least Squares (scipy.linalg.lstsq) if no constraint is given
    otherwise Non Negative Least Squares (scipy.optimize.nnls) wrapped
    as a predictor object.

    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None, constraints=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample

        constraints : dictionary with attributes :
            "sign" : numpy array of shape [n_features] -1 for a maximum,
                +1 for a minimum
            "value" : numpy array of shape [n_features] with coefficient
                constraint value

        Returns
        -------
        self : returns an instance of self.
        """

        n_jobs_ = self.n_jobs
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)

        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        # constraints
        is_constr = (constraints is not None)
        if is_constr:
            constraints = check_constraints(X, constraints)
            cs = constraints["sign"]
            cv = constraints["value"]
            y_offset_c = np.outer(np.dot(X, cv), np.ones(y.shape[1]))
            X = X * cs
            y = y - y_offset_c

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)

        if is_constr:
            X_scale = X_scale * cs

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if sp.issparse(X) or is_constr:
            if y.ndim < 2 or y.shape[1] == 1:
                self.coef_, self._residues = optim_fun(X, y, is_constr=is_constr)
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(optim_fun)(X, y[:, j], is_constr)
                    for j in range(y.shape[1]))
                self.coef_ = np.vstack(out[0] for out in outs)
                self._residues = np.vstack(out[1] for out in outs)
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = \
                linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)

        self._set_intercept(X_offset, y_offset, X_scale)
        if is_constr:
            self.coef_ = self.coef_ * cs
            self.intercept_ = np.outer(self.intercept_, np.ones(y.shape[1])) - y_offset_c
        return self


