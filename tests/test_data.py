import numpy as np
import pytest
from numpy.testing import assert_allclose

from robusthw.experiments.data import (
    bayes_error,
    calibrated_parameter,
    draw_population,
    original_anchor_split,
    standardize_split,
    synthetic_split,
)


@pytest.mark.parametrize("model", ["mean", "variance", "correlation", "mixture"])
@pytest.mark.parametrize("s", [2, 4, 8])
def test_exact_bayes_calibration(model, s):
    parameter = calibrated_parameter(model, s)
    assert bayes_error(model, s, parameter) == pytest.approx(0.2, abs=1e-11)


def test_report_parameter_values():
    expected = {"mean": 0.420811, "variance": 1.871151, "correlation": 0.6, "mixture": 0.857215}
    for model, value in expected.items():
        assert calibrated_parameter(model, 4) == pytest.approx(value, abs=5e-7)


@pytest.mark.parametrize("model", ["mean", "variance", "correlation", "mixture"])
def test_independent_named_streams_shared_permutation_and_train_only_scaling(model):
    a = synthetic_split(model, 10, 8, 4, 17, test_per_class=12)
    b = synthetic_split(model, 10, 8, 4, 17, test_per_class=200)
    for name in ("train0", "train1", "val0", "val1"):
        assert_allclose(getattr(a, name), getattr(b, name))
    assert_allclose(np.vstack(a.train).mean(0), 0, atol=1e-14)
    assert_allclose(np.vstack(a.train).std(0, ddof=1), 1, atol=1e-14)
    assert {a.feature_ids[j] for j in a.signal_indices} == {0, 1, 2, 3}
    assert not np.array_equal(a.train0, a.val0[:8])
    assert not a.train0.flags.writeable


def test_constant_columns_are_removed_and_original_identity_preserved():
    pair = (
        np.array([[1.0, 1.0, 3.0], [1.0, 3.0, 4.0]]),
        np.array([[1.0, 2.0, 5.0], [1.0, 4.0, 6.0]]),
    )
    result = standardize_split([pair, pair, pair], [9, 7, 3], signal_ids=[7])
    assert result.feature_ids == (7, 3) and result.signal_indices == (0,)
    assert result.metadata["preprocessing"]["dropped_feature_ids"] == [9]


def test_raw_pair_correlation_and_noise_generation():
    rho = calibrated_parameter("correlation", 4)
    for label in (0, 1):
        x = draw_population(
            "correlation", 6, 4, rho, np.random.default_rng(22 + label), 30000, label
        )
        correlation = np.corrcoef(x.T)
        assert correlation[0, 1] == pytest.approx((1 - 2 * label) * rho, abs=0.02)
        assert abs(correlation[0, 2]) < 0.02
        assert abs(correlation[0, 4]) < 0.02
        assert_allclose(x.mean(0), 0, atol=0.025)


def test_anchor_is_separate_uncalibrated_full_variable_prediction():
    split = original_anchor_split(9)
    assert split.train0.shape == (10, 100)
    assert split.val0.shape == (20, 100) and split.test0.shape == (1000, 100)
    assert split.metadata["parameter"] == 0.4 and split.metadata["target_bayes_error"] is None
    assert split.signal_indices == () and not split.metadata["recovery_defined"]


def test_odd_interaction_signal_size_is_rejected():
    with pytest.raises(ValueError, match="even"):
        synthetic_split("mixture", 10, 5, 3, 1)
