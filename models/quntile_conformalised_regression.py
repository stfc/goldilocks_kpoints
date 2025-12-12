# Code is adapted from https://github.com/scikit-learn-contrib/MAPIE to let it work with RandomForestQuantileRegressor
# from sklearn_quantile and ALIGNN pre-trained with quantile loss

from typing import Iterable, List, Optional, Tuple, Union, cast, Any

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_y, _num_samples, check_is_fitted, indexable

from numpy.typing import ArrayLike, NDArray
from mapie.utils import (
    _check_alpha_and_n_samples,
    _check_defined_variables_predict_cqr,
    _check_estimator_fit_predict,
    _check_lower_upper_bounds,
    _check_null_weight,
    _fit_estimator,
)

from mapie.regression.regression import _MapieRegressor
from mapie.utils import (
    _cast_predictions_to_ndarray_tuple,
    _prepare_params,
    _prepare_fit_params_and_sample_weight,
    _raise_error_if_previous_method_not_called,
    _raise_error_if_method_already_called,
    _raise_error_if_fit_called_in_prefit_mode,
    _transform_confidence_level_to_alpha,
)

class MapieQuantileRegressorKpoints(_MapieRegressor):
    """
    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``QuantileRegressor`` instance.

        By default ``"None"``.

    method: str
        Method to choose for prediction, in this case, the only valid method
        is the ``"quantile"`` method.

        By default ``"quantile"``.
    cv: Optional[str]
        The cross-validation strategy for computing conformity scores.
        In theory a split method is implemented as it is needed to provide
        both a training and calibration set.

        By default ``None``.

    alpha: float
    
        Between ``0.0`` and ``1.0``, represents the risk level of the
        confidence interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default ``0.1``.
    Attributes
    ----------
    valid_methods_: List[str]
        List of all valid methods.

    single_estimator_: RegressorMixin
        Estimator fitted on the whole training set.

    estimators_: List[RegressorMixin]
        - [0]: Estimator with quantile value of alpha/2
        - [1]: Estimator with quantile value of 1 - alpha/2
        - [2]: Estimator with quantile value of 0.5

    conformity_scores_: NDArray of shape (n_samples_train, 3)
        Conformity scores between ``y_calib`` and ``y_pred``.

        - [:, 0]: for ``y_calib`` coming from prediction estimator
          with quantile of alpha/2
        - [:, 1]: for ``y_calib`` coming from prediction estimator
          with quantile of 1 - alpha/2
        - [:, 2]: maximum of those first two scores

    n_calib_samples: int
        Number of samples in the calibration dataset.

    References
    ----------
    Yaniv Romano, Evan Patterson and Emmanuel J. Candès.
    "Conformalized Quantile Regression"
    Advances in neural information processing systems 32 (2019)."""

    valid_methods_ = ["quantile"]
    fit_attributes = [
        "estimators_",
        "conformity_scores_",
        "n_calib_samples",
    ]

    quantile_estimator_params = {
        "GradientBoostingRegressor": {"loss_name": "loss", "alpha_name": "alpha"},
        "QuantileRegressor": {"loss_name": "quantile", "alpha_name": "quantile"},
        "HistGradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "quantile",
        },
        "LGBMRegressor": {"loss_name": "objective", "alpha_name": "alpha"},
        "RandomForestQuantileRegressor": {"loss_name": None, "alpha_name": "q", "multi_quantile": True}
    }

    def __init__(
        self,
        estimator: Optional[
            Union[RegressorMixin, Pipeline, List[Union[RegressorMixin, Pipeline]]]
        ] = None,
        method: str = "quantile",
        cv: Optional[str] = None,
        alpha: float = 0.1,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
        )
        self.cv = cv
        self.alpha = alpha

    def _check_alpha(self, alpha: float = 0.1) -> NDArray:
        if isinstance(alpha, float):
            if np.any(np.logical_or(alpha <= 0, alpha >= 1.0)):
                raise ValueError(
                    "Invalid confidence_level. Allowed values are between 0.0 and 1.0."
                )
            else:
                alpha_np = np.array([alpha / 2, 1 - alpha / 2, 0.5])
        else:
            raise ValueError("Invalid confidence_level. Allowed values are float.")
        return alpha_np

    def _check_cv(self, cv: Optional[str] = None) -> str:
            if cv is None:
                return "split"
            if cv in ("split", "prefit"):
                return cv
            else:
                raise ValueError("Invalid cv method, only valid method is ``split``.")