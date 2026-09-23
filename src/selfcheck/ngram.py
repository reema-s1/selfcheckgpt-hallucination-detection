"""SelfCheckGPT-Unigram: a sample-trained unigram LM as a hallucination proxy.

A unigram model is fit on the response plus its N sampled passages (no
smoothing). Tokens that the model rarely repeats across samples get low
probability, so a sentence's score is its maximum token surprisal:

    score(i) = max_j -log p(token_ij)

Scores are unbounded (not in [0, 1]); only their ranking matters.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

Tokenize = Callable[[str], list[str]]
SplitSentences = Callable[[str], list[str]]


class UnigramConsistencyScorer:
    def __init__(
        self,
        tokenize: Tokenize,
        split_sentences: SplitSentences,
        *,
        lowercase: bool = True,
    ):
        self._tokenize = tokenize
        self._split_sentences = split_sentences
        self._lowercase = lowercase

    @classmethod
    def with_spacy(cls, model: str = "en_core_web_sm") -> UnigramConsistencyScorer:
        import spacy  # noqa: PLC0415

        nlp = spacy.load(model)
        return cls(
            tokenize=lambda text: [t.text for t in nlp.tokenizer(text)],
            split_sentences=lambda text: [s.text.strip() for s in nlp(text).sents],
        )

    def _tokens(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        return [t.lower() for t in tokens] if self._lowercase else tokens

    def score(
        self,
        sentences: Sequence[str],
        passage: str,
        samples: Sequence[str],
    ) -> np.ndarray:
        """Max negative log-probability per sentence; higher = likelier hallucinated."""
        counts: Counter[str] = Counter()
        for text in [passage, *samples]:
            for sentence in self._split_sentences(text):
                counts.update(self._tokens(sentence))
        total = sum(counts.values())

        scores = []
        for sentence in sentences:
            tokens = self._tokens(sentence)
            if not tokens:
                scores.append(0.0)
                continue
            # a token never seen in passage or samples has probability 0
            scores.append(
                max(
                    -math.log(counts[t] / total) if counts[t] else math.inf
                    for t in tokens
                )
            )
        return np.array(scores)
