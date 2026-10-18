import numpy as np
import pytest

from robusthw.experiments.metrics import (
    classification_metrics,
    paired_bootstrap_ci,
    recovery_metrics,
    subject_metrics,
)


def test_score_orientation_randomized_loss_factor_and_zero_tie():
    result = classification_metrics(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert result["ber_prob"] == 0.25
    assert result["ber_hard"] == 0.25
    assert result["error0_hard"] == 0 and result["error1_hard"] == 0.5
    assert result["half_hinge_risk"] == 2 * result["ber_prob"]
    assert classification_metrics([0], [0])["ber_prob"] == 0.5


def test_recovery_distinguishes_signal_count_and_budget():
    result = recovery_metrics((0, 1, 4, 5), (0, 1), 4)
    assert result == {"signal_hits": 2, "precision": 0.5, "recall": 1.0}
    assert recovery_metrics((0,), (), 1) == {}
    with pytest.raises(ValueError):
        recovery_metrics((0,), (0, 1), 2)


def test_pairing_is_preserved_before_resampling():
    assert paired_bootstrap_ci([2, 2, 2]) == (2.0, 2.0)
    assert paired_bootstrap_ci([-0.5, 0, 0.5], seed=4) == paired_bootstrap_ci(
        [-0.5, 0, 0.5], seed=4
    )


def test_subject_average_has_equal_person_weight():
    result = subject_metrics([1, 1, 1, -1], [-1, -1, -1, 1], [1, 1, 1, 2], [1, 1, 1, 2])
    assert np.mean([value["ber_prob"] for value in result.values()]) == 0.5
    assert classification_metrics([1, 1, 1, -1], [-1, -1, -1, 1])["ber_prob"] == 0.25
