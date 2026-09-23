"""Sample-consistency hallucination detection (SelfCheckGPT)."""

from .metrics import evaluate
from .ngram import UnigramConsistencyScorer
from .nli import NLIConsistencyScorer

__all__ = ["NLIConsistencyScorer", "UnigramConsistencyScorer", "evaluate"]
