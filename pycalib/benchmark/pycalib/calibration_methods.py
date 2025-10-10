"""Calibration Methods."""

# Standard library imports
import os
import numpy as np
import numpy.matlib
import warnings
import matplotlib.pyplot as plt

# SciPy imports
import scipy.stats
import scipy.optimize
import scipy.special
import scipy.cluster.vq
from scipy.spatial.distance import pdist

# Scikit learn imports
import sklearn
import sklearn.multiclass
import sklearn.utils
from sklearn.base import clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed

# Calibration models
import sklearn.isotonic
import sklearn.linear_model
import betacal

# Gaussian Processes and TensorFlow
import gpflow
from gpflow.utilities import set_trainable
from gpflow.mean_functions import MeanFunction
import tensorflow as tf
import tensorflow_probability as tfp
os.environ["TF_USE_LEGACY_KERAS"] = "1"

#from pycalib import gp_classes
from pycalib.gp_pac import Softmax_PAC, SVGP_PAC

# Turn off tensorflow deprecation warnings
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

# Kernel ridge/logistic regression based calibration
import scipy.special
import warnings
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
import torch
import torch.optim as optim


# Plotting
#import pycalib.texfig

# Ignore binned_statistic FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def select_sigma_from_data(X, subsample_size=1000):
        """
        Select sigma as the median of pairwise Euclidean distances from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, d)
            Each row represents a d-dimensional data point (typically d=2).
        subsample_size : int, default=1000
            Number of samples for subsampling. If exceeded, choose a random subset.
            
        Returns
        -------
        sigma : float
            Selected kernel width (median).
        """
        n = X.shape[0]
        if n > subsample_size:
            idx = np.random.permutation(n)[:subsample_size]
            X_sample = X[idx]
        else:
            X_sample = X

        # Compute pairwise Euclidean distances using pdist
        distances = pdist(X_sample, metric='euclidean')
        median_val = np.median(distances)
        return median_val



class LogMeanFunction(MeanFunction):
    def __call__(self, X):
        X = tf.cast(X, dtype=tf.float64) 
        return tf.math.log(X + 1e-6)

class BoundedPositiveBijector(tfp.bijectors.Bijector):
    def __init__(self, lower, upper, validate_args=False, name="bounded_positive"):
        super().__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)
        self.lower = tf.cast(lower, tf.float64)
        self.upper = tf.cast(upper, tf.float64)

    def _forward(self, x):
        # scaling to [0, 1]
        scaled_x = tf.sigmoid(x)
        # fix [lower, upper]
        return self.lower + (self.upper - self.lower) * scaled_x

    def _inverse(self, y):
        # y-->[0, 1]
        scaled_y = (y - self.lower) / (self.upper - self.lower)
        # Inverse of sigmoid --> logit.
        return tf.math.log(scaled_y / (1 - scaled_y))

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., dtype=x.dtype)

class CalibrationMethod(sklearn.base.BaseEstimator):
    """
    A generic class for probability calibration

    A calibration method takes a set of posterior class probabilities and transform them into calibrated posterior
    probabilities. Calibrated in this sense means that the empirical frequency of a correct class prediction matches its
    predicted posterior probability.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict(self, X):
        """
        Predict the class of new samples after scaling. Predictions are identical to the ones from the uncalibrated
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted classes.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def plot(self, filename, xlim=[0, 1], **kwargs):
        """
        Plot the calibration map.

        Parameters
        ----------
        xlim : array-like
            Range of inputs of the calibration map to be plotted.

        **kwargs :
            Additional arguments passed on to :func:`matplotlib.plot`.
        """
        # TODO: Fix this plotting function

        # Generate data and transform
        x = np.linspace(0, 1, 10000)
        y = self.predict_proba(np.column_stack([1 - x, x]))[:, 1]

        # Plot and label
        plt.plot(x, y, **kwargs)
        plt.xlim(xlim)
        plt.xlabel("p(y=1|x)")
        plt.ylabel("f(p(y=1|x))")


class NoCalibration(CalibrationMethod):
    """
    A class that performs no calibration.

    This class can be used as a baseline for benchmarking.

    logits : bool, default=False
        Are the inputs for calibration logits (e.g. from a neural network)?
    """

    def __init__(self, logits=False):
        self.logits = logits

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        if self.logits:
            return scipy.special.softmax(X, axis=1)
        else:
            return X


class TemperatureScaling(CalibrationMethod):
    """
    Probability calibration using temperature scaling

    Temperature scaling [1]_ is a one parameter multi-class scaling method. Output confidence scores are calibrated,
    meaning they match empirical frequencies of the associated class prediction. Temperature scaling does not change the
    class predictions of the underlying model.

    Parameters
    ----------
    T_init : float
        Initial temperature parameter used for scaling. This parameter is optimized in order to calibrate output
        probabilities.
    verbose : bool
        Print information on optimization procedure.

    References
    ----------
    .. [1] On calibration of modern neural networks, C. Guo, G. Pleiss, Y. Sun, K. Weinberger, ICML 2017
    """

    def __init__(self, T_init=1, verbose=False, alpha=None):
        super().__init__()
        if T_init <= 0:
            raise ValueError("Temperature not greater than 0.")
        self.T_init = T_init
        self.verbose = verbose
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities or logits X and ground truth
        labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities or logits of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        # Define objective function (NLL / cross entropy)
        def objective(T):
            # Calibrate with given T
            P = scipy.special.softmax(X / T, axis=1)

            # Compute negative log-likelihood
            P_y = P[np.array(np.arange(0, X.shape[0])), y]
            tiny = np.finfo(float).tiny  # to avoid division by 0 warning
            NLL = - np.sum(np.log(P_y + tiny))
            return NLL

        # Derivative of the objective with respect to the temperature T
        def gradient(T):
            # Exponential terms
            E = np.exp(X / T)

            # Gradient
            dT_i = (np.sum(E * (X - X[np.array(np.arange(0, X.shape[0])), y].reshape(-1, 1)), axis=1)) \
                   / np.sum(E, axis=1)
            if self.alpha == None:
                grad = - dT_i.sum() / T ** 2
            else:
                grad = - dT_i.sum() / T ** 2 + 2*self.alpha*T
            return grad

        # Optimize
        self.T = scipy.optimize.fmin_bfgs(f=objective, x0=self.T_init,
                                          fprime=gradient, gtol=1e-06, disp=self.verbose)[0]

        # Check for T > 0
        if self.T <= 0:
            raise ValueError("Temperature not greater than 0.")

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        # Check is fitted
        check_is_fitted(self, "T")

        # Transform with scaled softmax
        return scipy.special.softmax(X / self.T, axis=1)

    def latent(self, z):
        """
        Evaluate the latent function Tz of temperature scaling.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence for which to evaluate the latent function.

        Returns
        -------
        f : array-like, shape=(n_evaluations,)
            Values of the latent function at z.
        """
        check_is_fitted(self, "T")
        return self.T * z

    def plot_latent(self, z, filename, **kwargs):
        """
        Plot the latent function of the calibration method.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence to plot latent function for.
        filename :
            Filename / -path where to save output.
        kwargs
            Additional arguments passed on to matplotlib.pyplot.subplots.

        Returns
        -------

        """
        check_is_fitted(self, "T")

        # Plot latent function
        fig, axes = pycalib.texfig.subplots(nrows=1, ncols=1, sharex=True, **kwargs)
        axes.plot(z, self.T * z, label="latent function")
        axes.set_ylabel("$T\\bm{z}$")
        axes.set_xlabel("$\\bm{z}_k$")
        fig.align_labels()

        # Save plot to file
        pycalib.texfig.savefig(filename)
        plt.close()


class PlattScaling(CalibrationMethod):
    """
    Probability calibration using Platt scaling

    Platt scaling [1]_ [2]_ is a parametric method designed to output calibrated posterior probabilities for (non-probabilistic)
    binary classifiers. It was originally introduced in the context of SVMs. It works by fitting a logistic
    regression model to the model output using the negative log-likelihood as a loss function.

    Parameters
    ----------
    regularization : float, default=10^(-12)
        Regularization constant, determining degree of regularization in logistic regression.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling the data.
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` instance, `random_state` is the random number generator;
        If `None`, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    .. [1] Platt, J. C. Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood
           Methods in Advances in Large-Margin Classifiers (MIT Press, 1999)
    .. [2] Lin, H.-T., Lin, C.-J. & Weng, R. C. A note on Platt’s probabilistic outputs for support vector machines.
           Machine learning 68, 267–276 (2007)
    """

    def __init__(self, regularization=10 ** -12, random_state=None):
        super().__init__()
        self.regularization = regularization
        self.random_state = sklearn.utils.check_random_state(random_state)

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.logistic_regressor_ = sklearn.linear_model.LogisticRegression(C=1 / self.regularization,
                                                                               solver='lbfgs',
                                                                               random_state=self.random_state)
            self.logistic_regressor_.fit(X[:, 1].reshape(-1, 1), y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, "logistic_regressor_")
            return self.logistic_regressor_.predict_proba(X[:, 1].reshape(-1, 1))
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class IsotonicRegression(CalibrationMethod):
    """
    Probability calibration using Isotonic Regression

    Isotonic regression [1]_ [2]_ is a non-parametric approach to mapping (non-probabilistic) classifier scores to
    probabilities. It assumes an isotonic (non-decreasing) relationship between classifier scores and probabilities.

    Parameters
    ----------
    out_of_bounds : string, optional, default: "clip"
        The ``out_of_bounds`` parameter handles how x-values outside of the
        training domain are handled.  When set to "nan", predicted y-values
        will be NaN.  When set to "clip", predicted y-values will be
        set to the value corresponding to the nearest train interval endpoint.
        When set to "raise", allow ``interp1d`` to throw ValueError.

    References
    ----------
    .. [1] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)
    .. [2] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """

    def __init__(self, out_of_bounds="clip"):
        super().__init__()
        self.out_of_bounds = out_of_bounds

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.isotonic_regressor_ = sklearn.isotonic.IsotonicRegression(increasing=True,
                                                                           out_of_bounds=self.out_of_bounds)
            self.isotonic_regressor_.fit(X[:, 1], y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, "isotonic_regressor_")
            p1 = self.isotonic_regressor_.predict(X[:, 1])
            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class BetaCalibration(CalibrationMethod):
    """
    Probability calibration using Beta calibration

    Beta calibration [1]_ [2]_ is a parametric approach to calibration, specifically designed for probabilistic
    classifiers with output range [0,1]. Here, a calibration map family is defined based on the likelihood ratio between
    two Beta distributions. This parametric assumption is appropriate if the marginal class distributions follow Beta
    distributions. The beta calibration map has three parameters, two shape parameters `a` and `b` and one location
    parameter `m`.

    Parameters
    ----------
        params : str, default="abm"
            Defines which parameters to fit and which to hold constant. One of ['abm', 'ab', 'am'].

    References
    ----------
    .. [1] Kull, M., Silva Filho, T. M., Flach, P., et al. Beyond sigmoids: How to obtain well-calibrated probabilities
           from binary classifiers with beta calibration. Electronic Journal of Statistics 11, 5052–5080 (2017)
    .. [2] Kull, M., Filho, T. S. & Flach, P. Beta calibration: a well-founded and easily implemented improvement on
           logistic calibration for binary classifiers in Proceedings of the 20th International Conference on Artificial
           Intelligence and Statistics (AISTATS)
    """

    def __init__(self, params="abm"):
        super().__init__()
        self.params = params

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.beta_calibrator_ = betacal.BetaCalibration(self.params)
            self.beta_calibrator_.fit(X[:, 1].reshape(-1, 1), y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, "beta_calibrator_")
            p1 = self.beta_calibrator_.predict(X[:, 1].reshape(-1, 1))
            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class HistogramBinning(CalibrationMethod):
    """
    Probability calibration using histogram binning

    Histogram binning [1]_ is a nonparametric approach to probability calibration. Classifier scores are binned into a given
    number of bins either based on fixed width or frequency. Classifier scores are then computed based on the empirical
    frequency of class 1 in each bin.

    Parameters
    ----------
        mode : str, default='equal_width'
            Binning mode used. One of ['equal_width', 'equal_freq'].
        n_bins : int, default=20
            Number of bins to bin classifier scores into.
        input_range : list, shape (2,), default=[0, 1]
            Range of the classifier scores.

    .. [1] Zadrozny, B. & Elkan, C. Obtaining calibrated probability estimates from decision trees and naive Bayesian
           classifiers in Proceedings of the 18th International Conference on Machine Learning (ICML, 2001), 609–616.
    """

    def __init__(self, mode='equal_freq', n_bins=20, input_range=[0, 1]):
        super().__init__()
        if mode in ['equal_width', 'equal_freq']:
            self.mode = mode
        else:
            raise ValueError("Mode not recognized. Choose on of 'equal_width', or 'equal_freq'.")
        self.n_bins = n_bins
        self.input_range = input_range

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            return self._fit_binary(X, y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def _fit_binary(self, X, y):
        if self.mode == 'equal_width':
            # Compute probability of class 1 in each equal width bin
            binned_stat = scipy.stats.binned_statistic(x=X[:, 1], values=np.equal(1, y), statistic='mean',
                                                       bins=self.n_bins, range=self.input_range)
            self.prob_class_1 = binned_stat.statistic
            self.binning = binned_stat.bin_edges
        elif self.mode == 'equal_freq':
            # Find binning based on equal frequency
            self.binning = np.quantile(X[:, 1],
                                       q=np.linspace(self.input_range[0], self.input_range[1], self.n_bins + 1))

            # Compute probability of class 1 in equal frequency bins
            digitized = np.digitize(X[:, 1], bins=self.binning)
            digitized[digitized == len(self.binning)] = len(self.binning) - 1  # include rightmost edge in partition
            self.prob_class_1 = [y[digitized == i].mean() for i in range(1, len(self.binning))]

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, ["binning", "prob_class_1"])
            # Find bin of predictions
            digitized = np.digitize(X[:, 1], bins=self.binning)
            digitized[digitized == len(self.binning)] = len(self.binning) - 1  # include rightmost edge in partition
            # Transform to empirical frequency of class 1 in each bin
            p1 = np.array([self.prob_class_1[j] for j in (digitized - 1)])
            # If empirical frequency is NaN, do not change prediction
            p1 = np.where(np.isfinite(p1), p1, X[:, 1])
            assert np.all(np.isfinite(p1)), "Predictions are not all finite."

            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class BayesianBinningQuantiles(CalibrationMethod):
    """
    Probability calibration using Bayesian binning into quantiles

    Bayesian binning into quantiles [1]_ considers multiple equal frequency binning models and combines them through
    Bayesian model averaging. Each binning model :math:`M` is scored according to
    :math:`\\text{Score}(M) = P(M) \\cdot P(D | M),` where a uniform prior :math:`P(M)` is assumed. The marginal likelihood
    :math:`P(D | M)` has a closed form solution under the assumption of independent binomial class distributions in each
    bin with beta priors.

    Parameters
    ----------
        C : int, default = 10
            Constant controlling the number of binning models.
        input_range : list, shape (2,), default=[0, 1]
            Range of the scores to calibrate.

    .. [1] Naeini, M. P., Cooper, G. F. & Hauskrecht, M. Obtaining Well Calibrated Probabilities Using Bayesian Binning
           in Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, Austin, Texas, USA.
    """

    def __init__(self, C=10, input_range=[0, 1]):
        super().__init__()
        self.C = C
        self.input_range = input_range

    def _binning_model_logscore(self, probs, y, partition, N_prime=2):
        """
        Compute the log score of a binning model

        Each binning model :math:`M` is scored according to :math:`Score(M) = P(M) \\cdot P(D | M),` where a uniform prior
        :math:`P(M)` is assumed and the marginal likelihood :math:`P(D | M)` has a closed form solution
        under the assumption of a binomial class distribution in each bin with beta priors.

        Parameters
        ----------
        probs : array-like, shape (n_samples, )
            Predicted posterior probabilities.
        y : array-like, shape (n_samples, )
            Target classes.
        partition : array-like, shape (n_bins + 1, )
            Interval partition defining a binning.
        N_prime : int, default=2
            Equivalent sample size expressing the strength of the belief in the prior distribution.

        Returns
        -------
        log_score : float
            Log of Bayesian score for a given binning model
        """
        # Setup
        B = len(partition) - 1
        p = (partition[1:] - partition[:-1]) / 2 + partition[:-1]

        # Compute positive and negative samples in given bins
        N = np.histogram(probs, bins=partition)[0]

        digitized = np.digitize(probs, bins=partition)
        digitized[digitized == len(partition)] = len(partition) - 1  # include rightmost edge in partition
        m = [y[digitized == i].sum() for i in range(1, len(partition))]
        n = N - m

        # Compute the parameters of the Beta priors
        tiny = np.finfo(float).tiny  # Avoid scipy.special.gammaln(0), which can arise if bin has zero width
        alpha = N_prime / B * p
        alpha[alpha == 0] = tiny
        beta = N_prime / B * (1 - p)
        beta[beta == 0] = tiny

        # Prior for a given binning model (uniform)
        log_prior = - np.log(self.T)

        # Compute the marginal log-likelihood for the given binning model
        log_likelihood = np.sum(
            scipy.special.gammaln(N_prime / B) + scipy.special.gammaln(m + alpha) + scipy.special.gammaln(n + beta) - (
                    scipy.special.gammaln(N + N_prime / B) + scipy.special.gammaln(alpha) + scipy.special.gammaln(
                beta)))

        # Compute score for the given binning model
        log_score = log_prior + log_likelihood
        return log_score

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError("Calibration training data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            self.binnings = []
            self.log_scores = []
            self.prob_class_1 = []
            self.T = 0
            return self._fit_binary(X, y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(calibrator=clone(self), n_jobs=n_jobs)
            self.onevsrest_calibrator_.fit(X, y)
            return self

    def _fit_binary(self, X, y):
        # Determine number of bins
        N = len(y)
        min_bins = int(max(1, np.floor(N ** (1 / 3) / self.C)))
        max_bins = int(min(np.ceil(N / 5), np.ceil(self.C * N ** (1 / 3))))
        self.T = max_bins - min_bins + 1

        # Define (equal frequency) binning models and compute scores
        self.binnings = []
        self.log_scores = []
        self.prob_class_1 = []
        for i, n_bins in enumerate(range(min_bins, max_bins + 1)):
            # Compute binning from data and set outer edges to range
            binning_tmp = np.quantile(X[:, 1], q=np.linspace(self.input_range[0], self.input_range[1], n_bins + 1))
            binning_tmp[0] = self.input_range[0]
            binning_tmp[-1] = self.input_range[1]
            # Enforce monotonicity of binning (np.quantile does not guarantee monotonicity)
            self.binnings.append(np.maximum.accumulate(binning_tmp))
            # Compute score
            self.log_scores.append(self._binning_model_logscore(probs=X[:, 1], y=y, partition=self.binnings[i]))

            # Compute empirical accuracy for all bins
            digitized = np.digitize(X[:, 1], bins=self.binnings[i])
            # include rightmost edge in partition
            digitized[digitized == len(self.binnings[i])] = len(self.binnings[i]) - 1

            def empty_safe_bin_mean(a, empty_value):
                """
                Assign the bin mean to an empty bin. Corresponds to prior assumption of the underlying classifier
                being calibrated.
                """
                if a.size == 0:
                    return empty_value
                else:
                    return a.mean()

            self.prob_class_1.append(
                [empty_safe_bin_mean(y[digitized == k], empty_value=(self.binnings[i][k] + self.binnings[i][k - 1]) / 2)
                 for k in range(1, len(self.binnings[i]))])

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, ["binnings", "log_scores", "prob_class_1", "T"])

            # Find bin for all binnings and the associated empirical accuracy
            posterior_prob_binnings = np.zeros(shape=[np.shape(X)[0], len(self.binnings)])
            for i, binning in enumerate(self.binnings):
                bin_ids = np.searchsorted(binning, X[:, 1])
                bin_ids = np.clip(bin_ids, a_min=0, a_max=len(binning) - 1)  # necessary if X is out of range
                posterior_prob_binnings[:, i] = [self.prob_class_1[i][j] for j in (bin_ids - 1)]

            # Computed score-weighted average
            norm_weights = np.exp(np.array(self.log_scores) - scipy.special.logsumexp(self.log_scores))
            posterior_prob = np.sum(posterior_prob_binnings * norm_weights, axis=1)

            # Compute probability for other class
            return np.column_stack([1 - posterior_prob, posterior_prob])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)



class GPCalibration(CalibrationMethod):
    """
    This is the modified implementation of probability calibration using a latent Gaussian process.

    Gaussian process calibration [1]_ is a non-parametric approach to calibrate posterior probabilities from an arbitrary
    classifier based on a hold-out data set. Inference is performed using a sparse variational Gaussian process
    (SVGP) [2]_ implemented in `gpflow` [3]_.
    However, tensorflow is no longer support 1.x version. 
    So, we implement the algorithm of [1]_ via tensorflow==2.x and gpflow==2.x.
    
    Parameters
    ----------
    n_classes : int
        Number of classes in calibration data.
    logits : bool, default=False
        Are the inputs for calibration logits (e.g. from a neural network)?
    mean_function : GPflow object
        Mean function of the latent GP.
    kernel : GPflow object
        Kernel function of the latent GP.
    likelihood : GPflow object
        Likelihood giving a prior on the class prediction.
    n_inducing_points : int, default=100
        Number of inducing points for the variational approximation.
    maxiter : int, default=1000
        Maximum number of iterations for the likelihood optimization procedure.
    n_monte_carlo : int, default=100
        Number of Monte Carlo samples for the inference procedure.
    max_samples_monte_carlo : int, default=10**7
        Maximum number of Monte Carlo samples to draw in one batch when predicting. Setting this value too large can
        cause memory issues.
    inf_mean_approx : bool, default=False
        If True, when inferring calibrated probabilities, only the mean of the latent Gaussian process is taken into
        account, not its covariance.
    session : tf.Session, default=None
        `tensorflow` session to use.
    random_state : int, default=0
        Random seed for reproducibility. Needed for Monte-Carlo sampling routine.
    verbose : bool
        Print information on optimization routine.

    References
    ----------
    .. [1] Wenger, J., Kjellström H. & Triebel, R. Non-Parametric Calibration for Classification in
           Proceedings of AISTATS (2020)
    .. [2] Hensman, J., Matthews, A. G. d. G. & Ghahramani, Z. Scalable Variational Gaussian Process Classification in
           Proceedings of AISTATS (2015)
    .. [3] Matthews, A. G. d. G., van der Wilk, M., et al. GPflow: A Gaussian process library using TensorFlow. Journal
           of Machine Learning Research 18, 1–6 (Apr. 2017)
    """

    def __init__(self, 
                 n_classes,
                 logits=False,
                 mean_function=None,
                 kernel=None,
                 likelihood=None,
                 model_type='SVGP',
                 pac = False,
                 likelihood_type=None,
                 loss_type=None, 
                 opt_method=None,
                 learning_rate=0.01,
                 num_inducing=10, 
                 maxiter=1000,
                 alpha=1.,
                 n_monte_carlo=100,
                 max_samples_monte_carlo=10 ** 7,
                 inf_mean_approx=False,
                 random_state=1,
                 verbose=False
                 ):
        
        super().__init__()
        
        self.n_classes = n_classes
        self.verbose = verbose
        self.model_type = model_type
        self.likelihood_type = likelihood_type
        self.pac = pac
        self.loss_type = loss_type
        self.num_inducing = num_inducing
        self.n_monte_carlo = n_monte_carlo
        self.max_samples_monte_carlo = max_samples_monte_carlo
        self.logits = logits
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.model = None
        self.maxiter = maxiter
        self.inf_mean_approx = inf_mean_approx
        self.random_state = random_state
        self.opt_method = opt_method
        np.random.seed(self.random_state)

        # Set likelihood
        if likelihood is None:
            if self.pac:
                self.likelihood = Softmax_PAC(self.n_classes, self.likelihood_type)
            else:
                self.likelihood = gpflow.likelihoods.Softmax(self.n_classes)
        else:
            self.likelihood = likelihood

        # Set mean function
        if mean_function is None:
            if logits:
                self.mean_function = gpflow.mean_functions.Identity()
            else:
                self.mean_function = LogMeanFunction()
        else:
            self.mean_function = mean_function
        
        # Set kernel
        if kernel is None:
            k_white = gpflow.kernels.White(variance=0.01)
            if logits:
                kernel_lengthscale = 10
                self.kernel = gpflow.kernels.RBF(lengthscales=kernel_lengthscale, variance=1)
            else:
                kernel_lengthscale = 0.5
                k_rbf = gpflow.kernels.RBF(lengthscales=kernel_lengthscale, variance=1)
                
                # Place constraints [a,b] on kernel parameters
                transform_lengthscale = BoundedPositiveBijector(.001, 10)
                transform_variance = BoundedPositiveBijector(0.01, 5)
                
                k_rbf.lengthscales = gpflow.Parameter(kernel_lengthscale, transform=transform_lengthscale)
                k_rbf.variance = gpflow.Parameter(1, transform=transform_variance)
                self.kernel = k_rbf + k_white
        else:
            self.kernel = kernel

    def fit(self, X, y):
        
        # Check for correct dimensions
        if X.ndim == 1 or np.shape(X)[1] != self.n_classes:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")

        # Fit GP in TF session
        self._fit_multiclass(X, y)
        
        return self
    
    def _fit_multiclass(self, X, y):
        
        y = y.reshape(-1, 1)
        if self.model_type == 'SVGP':
            inducing_variable = scipy.cluster.vq.kmeans(obs=X,
                                    k_or_guess=min(X.shape[0] * X.shape[1], self.num_inducing, ))[0]
            
            if self.pac:
                self.model = SVGP_PAC(kernel=self.kernel, likelihood=self.likelihood, inducing_variable=inducing_variable, 
                                            mean_function=self.mean_function, num_latent_gps=self.n_classes, whiten=True, q_diag=True, type=self.loss_type, alpha=self.alpha)
            else:
                self.model = gpflow.models.SVGP(kernel=self.kernel, likelihood=self.likelihood, inducing_variable=inducing_variable, 
                                            mean_function=self.mean_function, num_latent_gps=self.n_classes, whiten=True, q_diag=True)
            
            
            data = (X, y)
            training_loss = self.model.training_loss_closure(data, compile=False)

            # Optimization
            if self.opt_method == None:
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(training_loss, variables=self.model.trainable_variables, options=dict(maxiter=self.maxiter))
                
            else: 
                if self.opt_method == 'sgd':
                    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate) ## If you are Mac user
                    #optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
                elif self.opt_method == 'adam':
                    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate) ## If you are Mac user
                    #optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                optimizer.minimize(training_loss, self.model.trainable_variables)
        else:
            raise ValueError(f"Unexpected recalibratin method: {self.model_type}.")
        
        return self
    
    def predict_proba(self, X, mean_approx=False):
        
        check_is_fitted(self, "model")
        
        if mean_approx or self.inf_mean_approx:
            # Evaluate latent GP
            f, _ = self.model.predict_f(X)
            latent = f.numpy().reshape(np.shape(X))

            # Return softargmax of fitted GP at input
            return scipy.special.softmax(latent, axis=1)
        else:
            # Seed for Monte_Carlo
            tf.random.set_seed(self.random_state)
            if X.ndim == 1 or np.shape(X)[1] != self.n_classes:
                raise ValueError("Calibration data must have shape (n_samples, n_classes).")
            
            else:
                # Predict in batches to keep memory usage in Monte-Carlo sampling low
                n_data = np.shape(X)[0]
                samples_monte_carlo = self.n_classes * self.n_monte_carlo * n_data
                if samples_monte_carlo >= self.max_samples_monte_carlo:
                    n_pred_batches = np.divmod(samples_monte_carlo, self.max_samples_monte_carlo)[0]
                else:
                    n_pred_batches = 1

                p_pred_list = []
                for i in range(n_pred_batches):
                    if self.verbose:
                        print("Predicting batch {}/{}.".format(i + 1, n_pred_batches))
                    ind_range = np.arange(start=self.max_samples_monte_carlo * i,
                                              stop=np.minimum(self.max_samples_monte_carlo * (i + 1), n_data))
                    p_pred_list.append(tf.exp(self.predict_full_density(X[ind_range, :])).numpy())
                    #mean, var = self.model.predict_y(X[ind_range, :])
                    #p_pred_list.append(mean.numpy())

                return np.concatenate(p_pred_list, axis=0)
    
    def predict_full_density(self, X):
        
        mu, var = self.model.predict_f(X)
        N = tf.shape(mu)[0]
        epsilon = tf.random.normal((self.n_monte_carlo, N, self.n_classes), dtype=tf.float64)
        f_star = mu[None, :, :] + tf.sqrt(var[None, :, :]) * epsilon  # S x N x K
        p_y_f_star = tf.nn.softmax(f_star, axis=2)
        return tf.math.log(tf.reduce_mean(p_y_f_star, axis=0))
    
    def kl_divergence(self):
        return self.model.prior_kl().numpy()

    def latent(self, z):
        """
        Evaluate the latent function f(z) of the GP calibration method.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence for which to evaluate the latent function.

        Returns
        -------
        f : array-like, shape=(n_evaluations,)
            Values of the latent function at z.
        f_var : array-like, shape=(n_evaluations,)
            Variance of the latent function at z.
        """
        # Evaluate latent GP
        f, var = self.model.predict_f(z.reshape(-1, 1))
        latent = f.eval().flatten()
        latent_var = var.eval().flatten()

        return latent, latent_var

    def plot_latent(self, z, filename, plot_classes=True, **kwargs):
        """
        Plot the latent function of the calibration method.

        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence to plot latent function for.
        filename :
            Filename / -path where to save output.
        plot_classes : bool, default=True
            Should classes also be plotted?
        kwargs
            Additional arguments passed on to matplotlib.pyplot.subplots.

        Returns
        -------

        """
        # Evaluate latent GP
        f, var = self.model.predict_f(z.reshape(-1, 1))
        latent = f.eval().flatten()
        latent_var = var.eval().flatten()
        Z = self.model.X.value

        # Plot latent GP
        if plot_classes:
            fig, axes = pycalib.texfig.subplots(nrows=2, ncols=1, sharex=True, **kwargs)
            axes[0].plot(z, latent, label="GP mean")
            axes[0].fill_between(z, latent - 2 * np.sqrt(latent_var), latent + 2 * np.sqrt(latent_var), alpha=.2)
            axes[0].set_ylabel("GP $g(\\textnormal{z}_k)$")
            axes[1].plot(Z.reshape((np.size(Z),)),
                         np.matlib.repmat(np.arange(0, self.n_classes), np.shape(Z)[0], 1).reshape((np.size(Z),)), 'kx',
                         markersize=5)
            axes[1].set_ylabel("class $k$")
            axes[1].set_xlabel("confidence $\\textnormal{z}_k$")
            fig.align_labels()
        else:
            fig, axes = pycalib.texfig.subplots(nrows=1, ncols=1, sharex=True, **kwargs)
            axes.plot(z, latent, label="GP mean")
            axes.fill_between(z, latent - 2 * np.sqrt(latent_var), latent + 2 * np.sqrt(latent_var), alpha=.2)
            axes.set_xlabel("GP $g(\\textnormal{z}_k)$")
            axes.set_ylabel("confidence $\\textnormal{z}_k$")

        # Save plot to file
        pycalib.texfig.savefig(filename)
        plt.close()

###############################################################################
# Kernel Ridge Regression with selectable kernel (Gaussian or Laplace)
###############################################################################
class KernelRidgeCalibration(CalibrationMethod):
    """
    Calibration using Kernel Ridge Regression (squared loss)
    
    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel to use. Either a string ("rbf" for Gaussian kernel or "laplace" for Laplace kernel) or a callable.
    alpha : float, default=1.0
        Regularization parameter.
    gamma : float or None, default=None
        Gamma parameter used by the kernel function.
    **kwargs : dict
        Additional keyword arguments passed to KernelRidge.
    """
    def __init__(self, kernel='rbf', alpha=0.01, gamma=1.0, **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.kwargs = kwargs
        
        # If kernel is specified as a string and is 'laplace', define a custom kernel function
        if isinstance(kernel, str) and kernel.lower() == 'laplace':
            # Laplace kernel: k(x,z) = exp(-gamma * |x - z|)
            if gamma is None:
                raise ValueError("For Laplace kernel, gamma must be specified.")
            def laplace_kernel(X, Y):
                # Assume each input is 1D here (shape: (n_samples, 1))
                diff = np.abs(X - Y.T)
                return np.exp(-gamma * diff)
            self.kernel = laplace_kernel
        else:
            # Otherwise, leave as-is (e.g., 'rbf' handled internally by KernelRidge)
            self.kernel = kernel

    def fit(self, X, y):
        """
        Fit the model on binary calibration data X (n_samples, 2) and labels y.
        Uses X[:,1] (uncalibrated probability for class 1) as a 1D input feature.
        """
        if X.ndim == 1 or np.shape(X)[1] != 2:
            raise ValueError("KernelRidgeCalibration: Calibration training data must have shape (n_samples, 2).")
        X_feature = X[:, 1].reshape(-1, 1)
        self.gamma = select_sigma_from_data(X_feature)
        self.kr_model_ = KernelRidge(kernel=self.kernel, alpha=self.alpha, gamma=self.gamma, **self.kwargs)
        self.kr_model_.fit(X_feature, y)
        return self

    def predict_proba(self, X):
        """
        Compute calibrated probabilities. As in training, uses X[:,1] as input
        and returns results in the form [1-p, p].
        """
        if X.ndim == 1 or np.shape(X)[1] != 2:
            raise ValueError("KernelRidgeCalibration: Calibration data must have shape (n_samples, 2).")
        check_is_fitted(self, "kr_model_")
        X_feature = X[:, 1].reshape(-1, 1)
        pred = self.kr_model_.predict(X_feature)
        pred = np.clip(pred, 0, 1)
        return np.column_stack([1 - pred, pred])


###############################################################################
# Kernel Logistic Regression using PyTorch (with selectable kernel)
###############################################################################

class KernelLogisticRegressionCalibration(CalibrationMethod):
    """
    Calibration via Kernel Logistic Regression implemented in PyTorch.
    
    Input: Treat the uncalibrated binary probability X[:,1] as a 1D feature.
    The learned model has the form:
    
        f(x) = ∑ᵢ αᵢ · k(x, xᵢ),
        p(x) = σ(f(x)) = 1 / (1 + exp(-f(x)))
    
    Here, k(x, xᵢ) is computed using the chosen kernel (Gaussian or Laplace).
    Optimization uses PyTorch-based gradient descent (Adam) to learn α and b.
    
    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type: 'rbf' or 'laplace'.
    gamma : float, default=1.0
        Parameter used by the kernel function.
    alpha : float, default=0.01
        Regularization parameter.
    lr : float, default=0.01
        Learning rate.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence threshold for change in loss.
    verbose : bool, default=False
        Whether to print training progress.
    random_state : int, default=42
        Random seed.
    """
    def __init__(self, kernel='rbf', gamma=1.0, alpha=0.01, lr=0.01, max_iter=1000, tol=1e-6, verbose=False, random_state=42):
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        torch.manual_seed(self.random_state)
    
    def _kernel_torch(self, X, Y):
        """
        Compute the kernel matrix for PyTorch tensors.
        
        Parameters
        ----------
        X : torch.Tensor, shape (n_samples_X, 1)
        Y : torch.Tensor, shape (n_samples_Y, 1)
        
        Returns
        -------
        K : torch.Tensor, shape (n_samples_X, n_samples_Y)
        """
        diff = X - Y.t()  # broadcasted difference: shape (n, m)
        self.gamma = select_sigma_from_data(X)
        #if self.kernel.lower() == 'rbf':
        if self.kernel == 'rbf':
            K = torch.exp(-self.gamma * (diff ** 2))
        #elif self.kernel.lower() == 'laplace':
        elif self.kernel == 'laplace':
            K = torch.exp(-self.gamma * torch.abs(diff))
        else:
            raise ValueError("Unknown kernel type. Please choose 'rbf' or 'laplace'.")
        return K

    def fit(self, X, y):
        """
        Train the model using training data X (n_samples, 2) and labels y.
        Uses X[:,1] as a 1D feature.
        """
        if X.ndim == 1 or np.shape(X)[1] != 2:
            raise ValueError("KernelLogisticRegressionCalibrationPT: Calibration training data must have shape (n_samples, 2).")
        X_feature = X[:, 1].reshape(-1, 1)
        n_samples = X_feature.shape[0]
        
        # Convert to PyTorch tensors (float32)
        self.X_train_torch = torch.tensor(X_feature, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        
        # Compute kernel matrix on training data
        K = self._kernel_torch(self.X_train_torch, self.X_train_torch)  # shape: (n_samples, n_samples)
        
        if isinstance(K, np.ndarray):
            K_np = K.astype(np.float32)
            K_tensor = torch.from_numpy(K_np)
        else:
            K_tensor = K.float()
            K_np = K_tensor.detach().cpu().numpy().astype(np.float32)
        
        approx_threshold = 3000
        if n_samples >= approx_threshold:
            lambda_max = self.estimate_lambda_max_randomized(K_np, l=10, q=2)
            print("Using approximate lambda_max estimation (Randomized SVD).")
        else:
            lambda_max = self.estimate_lambda_max(K_np)
            print("Using exact lambda_max estimation (Power Method).")
        
        L = (0.25 + self.alpha) * lambda_max
        self.lr = 0.5 / L if L > 0 else self.lr
        print(f"Estimated lambda_max: {lambda_max:.4f}, L: {L:.4f}, using learning rate: {self.lr:.6f}")
        
        # Initialize parameters to learn: α (n_samples×1) and b (scalar)
        self.alpha_pt = torch.zeros((n_samples, 1), dtype=torch.float32, requires_grad=True)
        #self.b_pt = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        
        #optimizer = optim.SGD([self.alpha_pt, self.b_pt], lr=self.lr)
        optimizer = optim.SGD([self.alpha_pt], lr=self.lr)
        prev_loss = np.inf
        
        # Gradient descent loop
        for it in range(self.max_iter):
            optimizer.zero_grad()
            #f = torch.matmul(K, self.alpha_pt) + self.b_pt  # shape: (n_samples, 1)
            f = torch.matmul(K_tensor, self.alpha_pt)  # shape: (n_samples, 1)
            p = torch.sigmoid(f)
            # Negative log likelihood (binary cross entropy)
            loss = - torch.sum(y_tensor * torch.log(p + 1e-8) + (1 - y_tensor) * torch.log(1 - p + 1e-8))
            # Regularization term: (1/2) * λ * α^T K α
            #reg_term = 0.5 * self.alpha * torch.matmul(self.alpha_pt.t(), torch.matmul(K, self.alpha_pt))
            reg_term = self.alpha * torch.matmul(self.alpha_pt.t(), torch.matmul(K_tensor, self.alpha_pt))
            loss = (loss + reg_term.squeeze()) / n_samples
            
            loss_val = loss.item()
            if self.verbose and it % 100 == 0:
                print(f"Iteration {it}: loss = {loss_val:.6f}")
            if np.abs(prev_loss - loss_val) < self.tol:
                if self.verbose:
                    print(f"Convergence reached at iteration {it}: loss = {loss_val:.6f}")
                break
            prev_loss = loss_val
            
            loss.backward()
            optimizer.step()
        
        # Detach learned parameters for storage
        self.alpha_pt = self.alpha_pt.detach()
        #self.b_pt = self.b_pt.detach()
        return self

    def predict_proba(self, X):
        """
        For uncalibrated data X (n_samples, 2), compute calibrated probabilities and
        return them in the form [1-p, p].
        """
        if X.ndim == 1 or np.shape(X)[1] != 2:
            raise ValueError("KernelLogisticRegressionCalibrationPT: Calibration data must have shape (n_samples, 2).")
        #check_is_fitted(self, ["alpha_pt", "b_pt", "X_train_torch"])
        check_is_fitted(self, ["alpha_pt", "X_train_torch"])
        X_feature = X[:, 1].reshape(-1, 1)
        X_feature_torch = torch.tensor(X_feature, dtype=torch.float32)
        # Compute new kernel matrix between X and training inputs
        K_new = self._kernel_torch(X_feature_torch, self.X_train_torch)  # shape: (n_samples_new, n_train)
        #f_new = torch.matmul(K_new, self.alpha_pt) + self.b_pt
        f_new = torch.matmul(K_new, self.alpha_pt)
        p_new = torch.sigmoid(f_new)
        p_new_np = p_new.numpy().ravel()
        return np.column_stack([1 - p_new_np, p_new_np])
    
    # ------------------------------
    # Lipschitz constant estimation
    # ------------------------------

    def estimate_lambda_max(self, K, num_iter=100):
        K = np.asarray(K, dtype=np.float32)
        n = K.shape[0]
        b = np.random.rand(n).astype(np.float32)
        b /= np.linalg.norm(b)
        for _ in range(num_iter):
            b = K @ b
            b_norm = np.linalg.norm(b)
            if b_norm == 0:
                break
            b /= b_norm
        lambda_max = (b.T @ (K @ b)).item()
        return lambda_max

    def estimate_lambda_max_randomized(self, K, l=10, q=2):
        '''
        Approximate lambda_max estimation via randomized SVD.
        '''
        n = K.shape[0]
        Omega = np.random.randn(n, l).astype(np.float32)
        Y = K @ Omega
        for _ in range(q):
            Y = K @ Y
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ K @ Q
        eigenvalues = np.linalg.eigvalsh(B)
        lambda_max = eigenvalues[-1]
        return lambda_max



class OneVsRestCalibrator(sklearn.base.BaseEstimator):
    """One-vs-the-rest (OvR) multiclass strategy
    Also known as one-vs-all, this strategy consists in fitting one calibrator
    per class. The probabilities to be calibrated of the other classes are summed.
    For each calibrator, the class is fitted against all the other classes.

    Parameters
    ----------
    calibrator : CalibrationMethod object
        A CalibrationMethod object implementing `fit` and `predict_proba`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        for more details.

    Attributes
    ----------
    calibrators_ : list of `n_classes` estimators
        Estimators used for predictions.
    classes_ : array, shape = [`n_classes`]
        Class labels.
    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.
    """

    def __init__(self, calibrator, n_jobs=None):
        self.calibrator = calibrator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Calibration data.
        y : (sparse) array-like, shape = [n_samples, ]
            Multi-class labels.
        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.calibrators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(OneVsRestCalibrator._fit_binary)(self.calibrator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i], self.label_binarizer_.classes_[i]]) for i, column in
            enumerate(columns))
        return self

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : (sparse) array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self, ["classes_", "calibrators_"])

        # Y[i, j] gives the probability that sample i has the label j.
        Y = np.array([c.predict_proba(
            np.column_stack([np.sum(np.delete(X, obj=i, axis=1), axis=1), X[:, self.classes_[i]]]))[:, 1] for i, c in
                      enumerate(self.calibrators_)]).T

        if len(self.calibrators_) == 1:
            # Only one estimator, but we still want to return probabilities for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        # Pad with zeros for classes not in training data
        if np.shape(Y)[1] != np.shape(X)[1]:
            p_pred = np.zeros(np.shape(X))
            p_pred[:, self.classes_] = Y
            Y = p_pred

        # Normalize probabilities to 1.
        Y = sklearn.preprocessing.normalize(Y, norm='l1', axis=1, copy=True, return_norm=False)
        return np.clip(Y, a_min=0, a_max=1)

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def _first_calibrator(self):
        return self.calibrators_[0]

    @staticmethod
    def _fit_binary(calibrator, X, y, classes=None):
        """
        Fit a single binary calibrator.

        Parameters
        ----------
        calibrator
        X
        y
        classes

        Returns
        -------

        """
        # Sum probabilities of combined classes in calibration training data X
        cl = classes[1]
        X = np.column_stack([np.sum(np.delete(X, cl, axis=1), axis=1), X[:, cl]])

        # Check whether only one label is present in training data
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            if classes is not None:
                if y[0] == -1:
                    c = 0
                else:
                    c = y[0]
                warnings.warn("Label %s is present in all training examples." %
                              str(classes[c]))
            calibrator = _ConstantCalibrator().fit(X, unique_y)
        else:
            calibrator = clone(calibrator)
            calibrator.fit(X, y)
        return calibrator


class _ConstantCalibrator(CalibrationMethod):

    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat([np.hstack([1 - self.y_, self.y_])], X.shape[0], axis=0)