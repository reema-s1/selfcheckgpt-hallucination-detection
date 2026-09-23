import numpy as np
import pytest

from selfcheck.metrics import auc_pr, evaluate

A, MI, MA = "accurate", "minor_inaccurate", "major_inaccurate"

ANNOTATIONS = {
    1: [A, MI, MA],
    2: [MA, MA],  # fully major-inaccurate: excluded from NonFact*
    3: [A, A, MI],
}


def test_perfect_scores_give_perfect_metrics():
    # score = label value, so ranking is perfect
    values = {A: 0.0, MI: 0.5, MA: 1.0}
    scores = {i: [values[a] for a in labels] for i, labels in ANNOTATIONS.items()}
    m = evaluate(ANNOTATIONS, scores)
    assert m["nonfact_auc_pr"] == pytest.approx(100)
    assert m["nonfact_star_auc_pr"] == pytest.approx(100)
    assert m["factual_auc_pr"] == pytest.approx(100)
    assert m["passage_pearson"] == pytest.approx(100)
    assert m["num_passages"] == 3
    assert m["num_sentences"] == 8
    assert m["nonfact_random_baseline"] == pytest.approx(100 * 5 / 8)


def test_minor_inaccurate_counts_as_nonfactual_but_not_as_major():
    # only the minor-inaccurate sentences score high
    scores = {1: [0, 1, 0], 2: [0, 0], 3: [0, 0, 1]}
    m = evaluate(ANNOTATIONS, scores)
    # NonFact positives = {MI, MA}: the MI sentences are ranked first
    assert m["nonfact_auc_pr"] > m["nonfact_random_baseline"]
    # NonFact* positives are MA only (passage 2 excluded), so high MI scores hurt
    assert m["nonfact_star_auc_pr"] < 50


def test_infinite_scores_are_ranked_highest():
    scores = {1: [0.1, 2.0, np.inf], 2: [np.inf, np.inf], 3: [0.1, 0.2, 1.0]}
    m = evaluate(ANNOTATIONS, scores)
    assert np.isfinite(m["nonfact_auc_pr"])
    assert np.isfinite(m["passage_pearson"])


def test_missing_or_misaligned_scores_raise():
    with pytest.raises(ValueError, match="missing scores"):
        evaluate(ANNOTATIONS, {1: [0, 0, 0]})
    with pytest.raises(ValueError, match="labels vs"):
        evaluate(ANNOTATIONS, {1: [0, 0], 2: [0, 0], 3: [0, 0, 0]})


def test_auc_pr_is_trapezoidal_area():
    assert auc_pr([0, 1], [0.0, 1.0]) == pytest.approx(1.0)
    assert auc_pr([1, 0], [0.0, 1.0]) < 1.0
