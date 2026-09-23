import math

import numpy as np
import pytest

from selfcheck.ngram import UnigramConsistencyScorer


def _scorer(**kwargs):
    return UnigramConsistencyScorer(
        tokenize=str.split,
        split_sentences=lambda text: [s for s in text.split(".") if s.strip()],
        **kwargs,
    )


def test_rare_tokens_get_higher_scores():
    passage = "Paris is a city. Zorbex founded it."
    samples = ["Paris is a city.", "Paris is a big city.", "It is a city."]
    scores = _scorer().score(["Paris is a city", "Zorbex founded it"], passage, samples)
    # "Zorbex" appears once across passage + samples, every token of sentence 1 repeats
    assert scores[1] > scores[0]


def test_score_is_max_token_surprisal():
    scores = _scorer().score(["a b"], "a a a b.", [])
    total = 4
    assert scores[0] == pytest.approx(-math.log(1 / total))


def test_lowercasing_merges_counts():
    passage = "Apple apple APPLE pear."
    lower = _scorer().score(["Apple"], passage, [])
    cased = _scorer(lowercase=False).score(["Apple"], passage, [])
    assert lower[0] < cased[0]


def test_unseen_token_is_infinite_and_empty_sentence_is_zero():
    scores = _scorer().score(["unseen", ""], "a b c.", [])
    assert np.isposinf(scores[0])
    assert scores[1] == 0.0
