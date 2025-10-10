"""Scoring functions and metrics for classification models."""

import time

import numpy as np
import cvxpy as cp

import scipy.stats
import sklearn.metrics
import sklearn.utils.validation
import torch
from torchmetrics.classification import MulticlassCalibrationError


def accuracy(y, p_pred):
    """
    Computes the accuracy.

    Parameters
    ----------
    y : array-like
        Ground truth labels.
    p_pred : array-like
        Array of confidence estimates.

    Returns
    -------
    accuracy : float
    """
    return sklearn.metrics.accuracy_score(y_true=y, y_pred=np.argmax(p_pred, axis=1))


def error(y, p_pred):
    """
    Computes the classification error.

    Parameters
    ----------
    y : array-like
        Ground truth labels.
    p_pred : array-like
        Array of confidence estimates.

    Returns
    -------
    error : float
    """
    return 1 - accuracy(y=y, p_pred=p_pred)


def odds_correctness(y, p_pred):
    """
    Computes the odds of making a correct prediction.

    Parameters
    ----------
    y : array-like
        Ground truth labels.
    p_pred : array-like
        Array of confidence estimates.

    Returns
    -------
    odds : float
    """
    return accuracy(y=y, p_pred=p_pred) / error(y=y, p_pred=p_pred)

def ece_ours(y, p_pred, n_bins=100, n_classes=None):
    y = sklearn.utils.validation.column_or_1d(y)
    y_pred = np.argmax(p_pred, axis=1)
    y_pred = sklearn.utils.validation.column_or_1d(y_pred)
    n_bins = int(n_bins)
    if n_classes is None:
        n_classes = np.unique(np.concatenate([y, y_pred])).shape[0]
    
    ece = MulticlassCalibrationError(num_classes=n_classes, n_bins=n_bins, norm='l1')
    return ece(torch.tensor(p_pred), torch.tensor(y).long()).numpy()


def expected_calibration_error(y, p_pred, n_bins=100, n_classes=None, p=1):
    """
    Computes the expected calibration error ECE_p.

    Computes the empirical p-expected calibration error for a vector of confidence
    estimates by binning.

    Parameters
    ----------
    y : array-like
        Ground truth labels.
    p_pred : array-like
        Array of confidence estimates.
    n_bins : int, default=15
        Number of bins of :math:`[\\frac{1}{n_{\\text{classes}},1]` for the confidence estimates.
    n_classes : int default=None
        Number of classes. Estimated from `y` and `y_pred` if not given.
    p : int, default=1
        Power of the calibration error, :math:`1 \\leq p \\leq \\infty`.

    Returns
    -------
    float
        Expected calibration error
    """
    # Check input
    y = sklearn.utils.validation.column_or_1d(y)
    y_pred = np.argmax(p_pred, axis=1)
    y_pred = sklearn.utils.validation.column_or_1d(y_pred)
    n_bins = int(n_bins)
    if n_classes is None:
        n_classes = np.unique(np.concatenate([y, y_pred])).shape[0]

    # Compute bin means
    #bin_range = [1 / n_classes, 1]
    bin_range = [0, 1]
    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

    # Find prediction confidence
    p_max = np.max(p_pred, axis=1)

    # Compute empirical accuracy
    empirical_acc = scipy.stats.binned_statistic(p_max, (y_pred == y).astype(int),
                                                 bins=n_bins, range=bin_range)[0]
    nanindices = np.where(np.logical_not(np.isnan(empirical_acc)))[0]

    # Perfect calibration
    calibrated_acc = np.linspace(bin_range[0] + bin_range[1] / (2 * n_bins), bin_range[1] - bin_range[1] / (2 * n_bins), n_bins)

    # Expected calibration error
    weights_ece = np.histogram(p_max, bins)[0][nanindices]
    if p < np.inf:
        ece = np.average(abs(empirical_acc[nanindices] - calibrated_acc[nanindices]) ** p,
                         weights=weights_ece)
    elif np.isinf(p):
        ece = np.max(abs(empirical_acc[nanindices] - calibrated_acc[nanindices]))

    return ece


def sharpness(y, p_pred, ddof=1):
    """
    Computes the empirical sharpness of a classifier.

    Computes the empirical sharpness of a classifier by computing the sample variance of a
    vector of confidence estimates.

    Parameters
    ----------
    y : array-like
        Ground truth labels. Dummy argument for consistent cross validation.
    p_pred : array-like
        Array of confidence estimates
    ddof : int, optional, default=1
        Degrees of freedom for the variance estimator.

    Returns
    -------
    float
        Sharpness
    """
    # Number of classes
    n_classes = np.shape(p_pred)[1]

    # Find prediction confidence
    p_max = np.max(p_pred, axis=1)

    # Compute sharpness
    sharp = np.var(p_max, ddof=ddof) * 4 * n_classes ** 2 / (n_classes - 1) ** 2

    return sharp


def overconfidence(y, p_pred):
    """
    Computes the overconfidence of a classifier.

    Computes the empirical overconfidence of a classifier on a test sample by evaluating
    the average confidence on the false predictions.

    Parameters
    ----------
    y : array-like
        Ground truth labels
    p_pred : array-like
        Array of confidence estimates

    Returns
    -------
    float
        Overconfidence
    """
    # Find prediction and confidence
    y_pred = np.argmax(p_pred, axis=1)
    p_max = np.max(p_pred, axis=1)

    return np.average(p_max[y_pred != y])


def underconfidence(y, p_pred):
    """
    Computes the underconfidence of a classifier.

    Computes the empirical underconfidence of a classifier on a test sample by evaluating
    the average uncertainty on the correct predictions.

    Parameters
    ----------
    y : array-like
        Ground truth labels
    p_pred : array-like
        Array of confidence estimates

    Returns
    -------
    float
        Underconfidence
    """
    # Find prediction and confidence
    y_pred = np.argmax(p_pred, axis=1)
    p_max = np.max(p_pred, axis=1)

    return np.average(1 - p_max[y_pred == y])


def ratio_over_underconfidence(y, p_pred):
    """
    Computes the ratio of over- and underconfidence of a classifier.

    Computes the empirical ratio of over- and underconfidence of a classifier on a test sample.

    Parameters
    ----------
    y : array-like
        Ground truth labels
    p_pred : array-like
        Array of confidence estimates

    Returns
    -------
    float
        Ratio of over- and underconfidence
    """
    return overconfidence(y=y, p_pred=p_pred) / underconfidence(y = y, p_pred=p_pred)


def average_confidence(y, p_pred):
    """
    Computes the average confidence in the prediction

    Parameters
    ----------
    y : array-like
        Ground truth labels. Here a dummy variable for cross validation.
    p_pred : array-like
        Array of confidence estimates.

    Returns
    -------
    avg_conf:float
        Average confidence in prediction.
    """
    return np.mean(np.max(p_pred, axis=1))


def weighted_abs_conf_difference(y, p_pred):
    """
    Computes the weighted absolute difference between over and underconfidence.

    Parameters
    ----------
    y : array-like
        Ground truth labels. Here a dummy variable for cross validation.
    p_pred : array-like
        Array of confidence estimates.

    Returns
    -------
    weighted_abs_diff: float
        Accuracy weighted absolute difference between over and underconfidence.
    """
    y_pred = np.argmax(p_pred, axis=1)
    of = overconfidence(y, p_pred)
    uf = underconfidence(y, p_pred)

    return abs((1 - np.average(y == y_pred)) * of - np.average(y == y_pred) * uf)


def brier_score(y, p_pred):
    """
    Compute the Brier score.

    The smaller the Brier score, the better, hence the naming with "loss".
    Across all items in a set N predictions, the Brier score measures the
    mean squared difference between (1) the predicted probability assigned
    to the possible outcomes for item i, and (2) the actual outcome.
    Therefore, the lower the Brier score is for a set of predictions, the
    better the predictions are calibrated. Note that the Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). The Brier loss is composed of refinement loss and
    calibration loss.

    Note: We interface the `sklearn.metrics.brier_score_loss` method here to provide a consistent method signature.

    Parameters
    ----------
    y : array-like
        Ground truth labels. Here a dummy variable for cross validation.
    p_pred : array-like
        Array of confidence estimates.

    Returns
    -------
    score : float
        Brier score
    """
    p = np.clip(p_pred[:, 1], a_min=0, a_max=1)
    return sklearn.metrics.brier_score_loss(y, p)


def precision(y, p_pred, **kwargs):
    """
    Computes the precision.

    Parameters
    ----------
    y
    p_pred

    Returns
    -------

    """
    y_pred = np.argmax(p_pred, axis=1)
    return sklearn.metrics.precision_score(y_true=y, y_pred=y_pred, **kwargs)


def recall(y, p_pred, **kwargs):
    """
    Computes the recall.

    Parameters
    ----------
    y
    p_pred

    Returns
    -------

    """
    y_pred = np.argmax(p_pred, axis=1)
    return sklearn.metrics.recall_score(y_true=y, y_pred=y_pred, **kwargs)

def to_one_hot(labels, num_classes=None):
    labels = np.array(labels)
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]

def kernel_calibration_error(y, p_pred, sigma=0.1, kernel_type='gaussian'):
    
    num_classes = len(np.unique(y))
    y = to_one_hot(y, num_classes)
    
    n = y.shape[0]
    r = y - p_pred
    Yp = p_pred.reshape(-1, 1)
    diff = Yp - Yp.T
    
    if kernel_type == 'gaussian':
        K_mat = np.exp(- (diff**2) / (2 * sigma**2))
    elif kernel_type == 'laplace':
        K_mat = np.exp(- np.abs(diff) / sigma)
    else:
        raise ValueError("kernel_type must be 'gaussian' or 'laplace'")
    
    kce_sq = (r.reshape(-1, 1) * r.reshape(1, -1) * K_mat).sum() / (n**2)
    return np.sqrt(np.maximum(kce_sq, 0.0))

def compute_LinECE(y, p_pred):
    num_classes = len(np.unique(y))
    y = to_one_hot(y, num_classes)
    
    v = p_pred.flatten()
    y = y.flatten()
    n = len(v)
    sorted_indices = np.argsort(v)
    v_sorted = v[sorted_indices]
    y_sorted = y[sorted_indices]
    z = cp.Variable(n)
    objective = cp.Maximize((1/n) * cp.sum((y_sorted - v_sorted) * z))
    constraints = [z >= -1, z <= 1]
    v_diff = np.abs(np.diff(v_sorted))
    z_diff = cp.abs(z[:-1] - z[1:])
    constraints.append(z_diff <= v_diff)
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError as e:
        print("ECOS failed, trying SCS instead...")
        prob.solve(solver=cp.SCS, verbose=False)
    return prob.value

def select_sigma_from_vector(v, subsample_size=1000, min_sigma=1e-3):
    """
    Select sigma as the median of absolute pairwise differences among elements of vector v.
    
    Parameters
    ----------
    v : array-like of shape (n_samples,)
        One-dimensional numeric vector.
    subsample_size : int, default=1000
        Number of samples for subsampling. If exceeded, a random subset is used.
    min_sigma : float, default=1e-3
        Lower bound on sigma. If the median is smaller, this value is returned.
        
    Returns
    -------
    sigma : float
        Selected kernel width.
    """
    v = np.asarray(v, dtype=float).flatten()
    n = v.shape[0]
    if n > subsample_size:
        idx = np.random.permutation(n)[:subsample_size]
        v_sample = v[idx]
    else:
        v_sample = v

    # Compute absolute pairwise differences using broadcasting
    diff = np.abs(v_sample[:, np.newaxis] - v_sample[np.newaxis, :])
    median_val = np.median(diff)
    return max(median_val, min_sigma)

class MultiScorer:
    """
    Use this class to encapsulate and/or aggregate multiple scoring functions so that it can be passed as an argument
    for scoring in scikit's cross_val_score function. Instances of this class are also callables, with signature as
    needed by `cross_val_score`. Evaluating multiple scoring function in this way versus scikit learns native way in the
    `cross_validate` function avoids the unnecessary overhead of predicting anew for each scorer. This class is slightly
    adapted from Kyriakos Stylianopoulos's implementation [1]_.

    .. [1] https://github.com/StKyr/multiscorer
    """

    def __init__(self, metrics, plots=None, gp=False):
        """
        Create a new instance of MultiScorer.

        Parameters
        ----------
        metrics: dict
            The metrics to be used by the scorer.
            The dictionary must have as key a name (str) for the metric and as value a tuple containing the metric
            function itself and a dict literal of the additional named arguments to be passed to the function. The
            metric function should be one of the `sklearn.metrics` function or any other callable with the same
            signature: `metric(y_true, p_pred, **kwargs)`.
        plots: dict
            Plots to be generated for each CV run.
        """
        self.metrics = metrics
        self.plots = plots
        self.gp = gp
        self.results = {}
        self._called = False
        self.n_folds = 0

        for metric in metrics.keys():
            self.results[metric] = []
        self.results["cal_time"] = []
        if self.gp:
            self.results["KL"] = []

    def __call__(self, estimator, X, y):
        """
        To be called by for evaluation from sklearn's GridSearchCV or cross_val_score. Parameters are as they are
        defined in the respective documentation.

        Returns
        -------
        dummy: float
            A dummy value of 0.5 just for compatibility reasons.
        """
        self.n_folds += 1

        # Predict probabilities
        start_time = time.time()
        p_pred = estimator.predict_proba(X)
        cal_time = time.time() - start_time

        # Compute metrics
        for key in self.metrics.keys():
            # Evaluate metric and save
            metric, kwargs = self.metrics[key]
            self.results[key].append(metric(y, p_pred, **kwargs))
        if self.gp:
            kl = estimator.kl_divergence()
            self.results["KL"].append(kl)
            print("KL:", kl)
        self.results["cal_time"].append(cal_time)

        # Generate plots
        #for key in self.plots.keys():
            # Evaluate plots and save
        #    plot_fun, kwargs = self.plots[key]
            # Plots in CV runs
            # TODO: make this safe for no filename argument
        #    kwargs_copy = copy.deepcopy(kwargs)
        #    kwargs_copy["filename"] = kwargs.get("filename", "") + "_" + str(self.n_folds)
        #    plot_fun(y=y, p_pred=p_pred, **kwargs_copy)
        #plt.close("all")

        # Set evaluation to true
        self._called = True

        # Return dummy value
        return 0.5

    def get_metric_names(self):
        """
        Get all the metric names as given when initialized.

        Returns
        -------
        metric_names: list
            A list containing the given names (str) of the metrics
        """
        return self.metrics.keys()

    def get_results(self, metric=None, fold='all'):
        """
        Get the results of a specific or all the metrics.

        This method should be called after the object itself has been called so that the metrics are applied.

        Parameters
        ----------
        metric: str or None (default)
            The given name of a metric to return its result(s). If omitted the results of all metrics will be returned.
        fold: int in range [1, number_of_folds] or 'all' (Default)
            Get the metric(s) results for the specific fold.
            The number of folds corresponds to the number of times the instance is called.
            If its value is a number, either the score of a single metric for that fold or a dictionary of the (single)
            scores for that fold will be returned, depending on the value of `metric` parameter. If its value is 'all',
            either a list of a single metric or a dictionary containing the lists of scores for all folds will be
            returned, depending on the value of `metric` parameter.
        Returns
        -------
        metric_result_for_one_fold
            The result of the designated metric function for the specific fold, if `metric` parameter was not omitted
            and an integer value was given to `fold` parameter. If  the value of `metric` does not correspond to a
            metric name, `None` will be returned.
        all_metric_results_for_one_fold: dict
            A dict having as keys the names of the metrics and as values their results for the specific fold.
            This will be returned only if `metric` parameter was omitted and an integer value was given to `fold`
            parameter.
        metric_results_for_all_folds: list
            A list of length number_of_folds containing the results of all folds for the specific metric, if `metric`
            parameter was not omitted and value 'all' was given to `fold`. If  the value of `metric` does not correspond
            to a metric name, `None` will be returned.
        all_metric_results_for_all_folds: dict of lists
            A dict having as keys the names of the metrics and as values lists (of length number_of_folds) of their
            results for all folds. This will be returned only if `metric` parameter was omitted and 'all' value was
            given to `fold` parameter.
        Raises
        ------
        UserWarning
            If this method is called before the instance is called for evaluation.
        ValueError
            If the value for `fold` parameter is not appropriate.
        """
        if not self._called:
            raise UserWarning('Evaluation has not been performed yet.')
        if isinstance(fold, str) and fold == 'all':
            if metric is None:
                return self.results
            else:
                return self.results[metric]
        elif isinstance(fold, int):
            if fold not in range(1, self.n_folds + 1):
                raise ValueError('Invalid fold index: ' + str(fold))
            if metric is None:
                res = dict()
                for key in self.results.keys():
                    res[key] = self.results[key][fold - 1]
                return res
            else:
                return self.results[metric][fold - 1]
        else:
            raise ValueError('Unexpected fold value: %s' % (str(fold)))
