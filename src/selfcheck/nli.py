"""SelfCheckGPT-NLI: sentence-level hallucination scores from sample consistency.

For a response split into sentences s_1..s_I and N stochastic re-samples
S^1..S^N of the same prompt, the score of sentence i is

    score(i) = 1/N * sum_n P(contradiction | s_i, S^n)

where the NLI probability is renormalised over {entailment, contradiction}
only (the neutral class is dropped). A sentence the model keeps contradicting
across its own samples is likely to be hallucinated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

DEFAULT_NLI_MODEL = "potsawee/deberta-v3-large-mnli"

# Checkpoints whose config uses generic names (LABEL_0, ...). For the default
# 2-way model, index 0 = entailment and 1 = contradiction, which was checked on
# unambiguous pairs (identical sentences -> 1.000 entailment; "black" vs
# "white" -> 1.000 contradiction).
KNOWN_LABEL_INDICES: dict[str, tuple[int, int]] = {DEFAULT_NLI_MODEL: (0, 1)}


def _label_index(id2label: dict[int, str], name: str) -> int:
    for idx, label in id2label.items():
        if label.lower() == name:
            return int(idx)
    msg = f"NLI model has no '{name}' label; labels are {id2label}"
    raise ValueError(msg)


class NLIConsistencyScorer:
    """Scores sentences by how often sampled passages contradict them.

    All (sentence, sample) pairs of a passage are scored in length-sorted
    mini-batches instead of one forward pass per pair. Sorting by length keeps
    padding (wasted compute) low; the attention mask makes the result
    independent of how pairs are grouped into batches.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        device: str | torch.device = "cpu",
        label_indices: tuple[int, int] | None = None,
    ):
        """label_indices = (entailment, contradiction) logit indices. If None,
        they are looked up by name in `model.config.id2label`."""
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)

        self._tokenizer = tokenizer
        self._model = model.to(device).eval()
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length

        if label_indices is None:
            id2label = model.config.id2label
            label_indices = (
                _label_index(id2label, "entailment"),
                _label_index(id2label, "contradiction"),
            )
        self._entailment_idx, self._contradiction_idx = label_indices

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_NLI_MODEL,
        **kwargs: Any,
    ) -> NLIConsistencyScorer:
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        kwargs.setdefault("label_indices", KNOWN_LABEL_INDICES.get(model_name))
        return cls(model, tokenizer, **kwargs)

    def score(self, sentences: Sequence[str], samples: Sequence[str]) -> np.ndarray:
        """One score in [0, 1] per sentence; higher = more likely hallucinated."""
        if not sentences:
            return np.zeros(0)
        if not samples:
            msg = "at least one sampled passage is required"
            raise ValueError(msg)

        pairs = [(sentence, sample) for sentence in sentences for sample in samples]
        probs = self.contradiction_probs(pairs)
        return probs.reshape(len(sentences), len(samples)).mean(axis=1)

    @torch.inference_mode()
    def contradiction_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """P(contradiction) for each (sentence, sample) pair, in input order.

        The sentence is the first text of the pair and the sample the second,
        the same ordering as the SelfCheckGPT paper's setup.
        """
        order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][0] + pairs[i][1]))
        probs = np.empty(len(pairs))

        for start in range(0, len(order), self._batch_size):
            batch_idx = order[start : start + self._batch_size]
            encoded = self._tokenizer(
                [pairs[i][0] for i in batch_idx],
                [pairs[i][1] for i in batch_idx],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            logits = self._model(**encoded).logits
            two_way = logits[:, [self._entailment_idx, self._contradiction_idx]]
            probs[batch_idx] = torch.softmax(two_way, dim=-1)[:, 1].cpu().numpy()

        return probs
