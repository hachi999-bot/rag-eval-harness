import math

import pytest

from harness.stats import bootstrap_ci, cohens_d_paired, paired_permutation_test


# --- normal cases ---------------------------------------------------------


def test_bootstrap_ci_contains_the_sample_mean():
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    lower, upper = bootstrap_ci(values, confidence=0.95, n_resamples=2000, seed=1)
    sample_mean = sum(values) / len(values)
    assert lower <= sample_mean <= upper


def test_bootstrap_ci_is_deterministic_given_a_seed():
    values = [0.2, 0.4, 0.6, 0.8]
    result_a = bootstrap_ci(values, n_resamples=1000, seed=7)
    result_b = bootstrap_ci(values, n_resamples=1000, seed=7)
    assert result_a == result_b


def test_paired_permutation_test_significant_for_large_consistent_improvement():
    before = [0.1] * 20
    after = [0.9] * 20
    p_value = paired_permutation_test(before, after, n_permutations=2000, seed=1)
    assert p_value < 0.01


def test_paired_permutation_test_not_significant_for_noise():
    before = [0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4]
    after = [0.4, 0.5, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5]
    p_value = paired_permutation_test(before, after, n_permutations=2000, seed=1)
    assert p_value > 0.1


def test_cohens_d_paired_positive_for_improvement():
    before = [0.2, 0.3, 0.25, 0.35]
    after = [0.6, 0.7, 0.65, 0.75]
    d = cohens_d_paired(before, after)
    assert d > 0


# --- boundary cases --------------------------------------------------------


def test_bootstrap_ci_single_value_collapses_to_point():
    lower, upper = bootstrap_ci([0.42], n_resamples=100, seed=1)
    assert lower == upper == 0.42


def test_paired_permutation_test_identical_sequences_returns_one():
    values = [0.3, 0.4, 0.5]
    assert paired_permutation_test(values, values, n_permutations=500, seed=1) == 1.0


def test_cohens_d_paired_zero_for_identical_sequences():
    values = [0.3, 0.4, 0.5, 0.6]
    assert cohens_d_paired(values, values) == 0.0


def test_cohens_d_paired_constant_nonzero_difference_is_infinite():
    before = [0.1, 0.1, 0.1]
    after = [0.2, 0.2, 0.2]
    assert cohens_d_paired(before, after) == math.inf


# --- abnormal cases ---------------------------------------------------------


def test_bootstrap_ci_rejects_empty_values():
    with pytest.raises(ValueError):
        bootstrap_ci([])


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_ci_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2, 0.3], confidence=confidence)


def test_paired_permutation_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_permutation_test([0.1, 0.2], [0.1], n_permutations=10)


def test_paired_permutation_test_rejects_empty_input():
    with pytest.raises(ValueError):
        paired_permutation_test([], [], n_permutations=10)


def test_cohens_d_paired_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohens_d_paired([0.1, 0.2], [0.1])


def test_cohens_d_paired_rejects_single_observation():
    with pytest.raises(ValueError):
        cohens_d_paired([0.1], [0.2])
