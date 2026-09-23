import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import (
    BatchEncoding,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from selfcheck.nli import NLIConsistencyScorer

LABELS = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
WORDS = [
    "the",
    "cat",
    "sat",
    "on",
    "a",
    "mat",
    "dog",
    "ran",
    "far",
    "away",
    "he",
    "was",
    "born",
    "in",
    "paris",
    "london",
]


@pytest.fixture(scope="module")
def tiny_nli(tmp_path_factory):
    """A small randomly initialised BERT classifier with a real tokenizer."""
    vocab = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(WORDS) + 5,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
        # large init so outputs differ clearly between inputs (default is ~constant)
        initializer_range=1.0,
    )
    return BertForSequenceClassification(config), tokenizer


SENTENCES = ["he was born in paris", "the cat sat", "a dog ran far away on the mat"]
SAMPLES = [
    "he was born in london",
    "the dog sat on a mat the cat ran",
    "cat",
    "he ran far away from paris and was born in london a dog sat",
]


def test_batched_scores_match_one_pair_per_forward_pass(tiny_nli):
    model, tokenizer = tiny_nli
    unbatched = NLIConsistencyScorer(model, tokenizer, batch_size=1)
    for batch_size in (2, 5, 64):
        batched = NLIConsistencyScorer(model, tokenizer, batch_size=batch_size)
        np.testing.assert_allclose(
            batched.score(SENTENCES, SAMPLES),
            unbatched.score(SENTENCES, SAMPLES),
            atol=1e-5,
        )


def test_pair_probabilities_are_returned_in_input_order(tiny_nli):
    model, tokenizer = tiny_nli
    scorer = NLIConsistencyScorer(model, tokenizer, batch_size=3)
    pairs = [(s, x) for s in SENTENCES for x in SAMPLES]
    together = scorer.contradiction_probs(pairs)
    one_by_one = [scorer.contradiction_probs([p])[0] for p in pairs]
    np.testing.assert_allclose(together, one_by_one, atol=1e-5)


def test_scores_are_per_sentence_probabilities(tiny_nli):
    model, tokenizer = tiny_nli
    scores = NLIConsistencyScorer(model, tokenizer).score(SENTENCES, SAMPLES)
    assert scores.shape == (len(SENTENCES),)
    assert np.all((scores >= 0) & (scores <= 1))


class _FixedLogitsModel(torch.nn.Module):
    """Returns preset logits for pair k, where the sentence text is 'k'."""

    def __init__(self, logits_per_pair):
        super().__init__()
        self._logits = torch.tensor(logits_per_pair, dtype=torch.float)
        self.config = SimpleNamespace(id2label=LABELS)

    def forward(self, pair_index):
        return SimpleNamespace(logits=self._logits[pair_index])


def _fixed_tokenizer(first, second, **kwargs):
    return BatchEncoding({"pair_index": torch.tensor([int(s) for s in first])})


def test_neutral_class_is_dropped_and_probabilities_renormalised():
    # logits: entailment, neutral, contradiction
    logits = [[0.0, 50.0, math.log(3.0)], [2.0, -1.0, 2.0]]
    scorer = NLIConsistencyScorer(_FixedLogitsModel(logits), _fixed_tokenizer)
    probs = scorer.contradiction_probs([("0", "x"), ("1", "x")])
    # softmax over (entailment, contradiction) only: 3/(1+3) and 1/2
    np.testing.assert_allclose(probs, [0.75, 0.5], atol=1e-6)


def test_sentence_score_is_mean_over_samples():
    logits = [[0.0, 0.0, 0.0], [0.0, 0.0, 50.0], [0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]
    scorer = NLIConsistencyScorer(_FixedLogitsModel(logits), _fixed_tokenizer)
    # pairs are (sentence, sample) in row-major order, so the "sentence" text
    # carries the pair index directly
    probs = scorer.contradiction_probs([(str(i), "x") for i in range(4)])
    np.testing.assert_allclose(probs, [0.5, 1.0, 0.5, 0.0], atol=1e-6)
    # sentence "0" -> pairs use row 0 twice (0.5, 0.5); sentence "1" -> row 1 (1.0, 1.0)
    np.testing.assert_allclose(scorer.score(["0", "1"], ["a", "b"]), [0.5, 1.0])


def test_empty_sentences_and_missing_samples():
    scorer = NLIConsistencyScorer(_FixedLogitsModel([[0, 0, 0]]), _fixed_tokenizer)
    assert scorer.score([], ["sample"]).size == 0
    with pytest.raises(ValueError, match="sampled passage"):
        scorer.score(["0"], [])


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        NLIConsistencyScorer(
            _FixedLogitsModel([[0, 0, 0]]), _fixed_tokenizer, batch_size=0
        )
    model = _FixedLogitsModel([[0, 0, 0]])
    model.config.id2label = {0: "yes", 1: "no"}
    with pytest.raises(ValueError, match="entailment"):
        NLIConsistencyScorer(model, _fixed_tokenizer)


def test_explicit_label_indices_for_unnamed_labels():
    # 2-way model with generic label names, as in the default checkpoint
    model = _FixedLogitsModel([[math.log(3.0), 0.0]])
    model.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    with pytest.raises(ValueError, match="entailment"):
        NLIConsistencyScorer(model, _fixed_tokenizer)
    scorer = NLIConsistencyScorer(model, _fixed_tokenizer, label_indices=(0, 1))
    np.testing.assert_allclose(scorer.contradiction_probs([("0", "x")]), [0.25])
