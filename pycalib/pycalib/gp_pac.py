from typing import Optional, Any

import numpy as np
import tensorflow as tf
from check_shapes import check_shapes, inherit_check_shapes

import gpflow
from gpflow import kullback_leiblers, posteriors
from gpflow.base import AnyNDArray, InputData, MeanAndVariance, Parameter, RegressionData, TensorType
from gpflow.conditionals import conditional
from gpflow.config import default_float
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import positive, triangular
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.models.util import InducingVariablesLike, inducingpoint_wrapper

class Softmax_PAC(gpflow.likelihoods.MonteCarloLikelihood):
    """
    The soft-max multi-class likelihood for PAC-Bayes calibration.  It can only provide a stochastic
    Monte-Carlo estimate of the variational expectations term, but this
    added variance tends to be small compared to that due to mini-batching
    (when using the SVGP model).
    """

    def __init__(self, num_classes: int, type: str = None, **kwargs: Any) -> None:
        super().__init__(input_dim=None, latent_dim=num_classes, observation_dim=None, **kwargs)
        self.num_classes = self.latent_dim
        self.type = type

    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        y_onehot = tf.cast(tf.one_hot(Y[:, 0], F.shape[1]), dtype=default_float())
        if self.type == 'total':
            return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=Y[:, 0]) + tf.reduce_sum(tf.math.square(y_onehot - F), 1)
        else:
            return tf.reduce_sum(tf.math.square(y_onehot - F),1)

    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return tf.nn.softmax(F)

    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        p = self.conditional_mean(X, F)
        return p - p ** 2


class SVGP_PAC_deprecated(GPModel, ExternalDataTrainingLossMixin):
    """
    This is the Sparse Variational GP (SVGP).

    The key reference is :cite:t:`hensman2014scalable`.

    For a use example see
    :doc:`../../../../notebooks/getting_started/classification_and_other_data_distributions`.
    """

    @check_shapes(
        "q_mu: [M, P]",
        "q_sqrt: [M, P] if q_diag",
        "q_sqrt: [P, M, M] if (not q_diag)",
    )
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariablesLike,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu: Optional[tf.Tensor] = None,
        q_sqrt: Optional[tf.Tensor] = None,
        whiten: bool = True,
        num_data: Optional[tf.Tensor] = None,
        type: str = None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.whiten = whiten
        self.inducing_variable: InducingVariables = inducingpoint_wrapper(inducing_variable)
        self.type = type

        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    @check_shapes(
        "q_mu: [M, P]",
        "q_sqrt: [M, P] if q_diag",
        "q_sqrt: [P, M, M] if (not q_diag)",
    )
    def _init_variational_parameters(
        self,
        num_inducing: int,
        q_mu: Optional[AnyNDArray],
        q_sqrt: Optional[AnyNDArray],
        q_diag: bool,
    ) -> None:
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the
        routine initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with
        P, number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing:
            Number of inducing variables, typically refered to as M.
        :param q_mu:
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt:
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag:
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if q_diag:
                ones: AnyNDArray = np.ones(
                    (num_inducing, self.num_latent_gps), dtype=default_float()
                )
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                np_q_sqrt: AnyNDArray = np.array(
                    [
                        np.eye(num_inducing, dtype=default_float())
                        for _ in range(self.num_latent_gps)
                    ]
                )
                self.q_sqrt = Parameter(np_q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    @check_shapes(
        "return: []",
    )
    def prior_kl(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:  # type: ignore[override]
        return self.elbo(data)

    @check_shapes(
        "return: []",
    )
    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            num_data = X.shape[0]
            scale = tf.cast(1.0, kl.dtype)
        if self.type == 'root':
            return (tf.reduce_sum(var_exp) * scale / num_data) + 3 * tf.math.square(2 * kl / num_data)
            #return (tf.reduce_sum(var_exp) * scale) + 3 * tf.math.square(2 * num_data * kl)
        else:
            #return tf.reduce_sum(var_exp) * scale + kl
            return (tf.reduce_sum(var_exp) * scale + kl)/num_data
        #return tf.reduce_sum(var_exp) * scale - kl

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var


class SVGP_PAC_with_posterior(SVGP_PAC_deprecated):
    """
    This is the Sparse Variational GP (SVGP).

    The key reference is :cite:t:`hensman2014scalable`.

    This class provides a posterior() method that enables caching for faster subsequent predictions.
    """

    def posterior(
        self,
        precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
    ) -> posteriors.BasePosterior:
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """
        return posteriors.create_posterior(
            self.kernel,
            self.inducing_variable,
            self.q_mu,
            self.q_sqrt,
            whiten=self.whiten,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, SVGP's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.

        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


class SVGP_PAC(SVGP_PAC_with_posterior):
    # subclassed to ensure __class__ == "SVGP"

    __doc__ = SVGP_PAC_deprecated.__doc__  # Use documentation from SVGP_deprecated.