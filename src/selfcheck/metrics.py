"""Evaluation protocol of the SelfCheckGPT paper on WikiBio-GPT3.

Human labels per sentence: accurate = 0, minor_inaccurate = 0.5,
major_inaccurate = 1.

- NonFact:  positives are minor or major inaccurate sentences (label > 0.499).
- NonFact*: positives are major inaccurate sentences (label > 0.99), restricted
  to passages whose mean label is < 0.99 (not entirely major-inaccurate).
- Factual:  positives are accurate sentences; the score is negated.
- Passage ranking: correlation between the mean sentence score and the mean
  human label of each passage.

AUC-PR is the trapezoidal area under the precision-recall curve, the same
quantity the paper reports (not sklearn's step-wise average precision).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve

LABEL_VALUES = {"accurate": 0.0, "minor_inaccurate": 0.5, "major_inaccurate": 1.0}
_INACCURATE = 0.499
_MAJOR = 0.99


def auc_pr(labels: Sequence[int], scores: Sequence[float]) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def _finite(scores: np.ndarray) -> np.ndarray:
    """Replaces +inf (unsmoothed n-gram scores) with a value above the max."""
    finite = scores[np.isfinite(scores)]
    ceiling = finite.max() + 1.0 if finite.size else 1.0
    return np.where(np.isposinf(scores), ceiling, scores)


def evaluate(
    annotations: Mapping[int, Sequence[str]],
    scores: Mapping[int, Sequence[float]],
) -> dict[str, float]:
    """Computes all paper metrics. Both mappings are keyed by passage id."""
    ids = sorted(annotations)
    missing = [i for i in ids if i not in scores]
    if missing:
        msg = f"missing scores for {len(missing)} passages, e.g. {missing[:5]}"
        raise ValueError(msg)

    labels = {i: np.array([LABEL_VALUES[a] for a in annotations[i]]) for i in ids}
    preds = {i: _finite(np.asarray(scores[i], dtype=float)) for i in ids}
    for i in ids:
        if len(labels[i]) != len(preds[i]):
            msg = f"passage {i}: {len(labels[i])} labels vs {len(preds[i])} scores"
            raise ValueError(msg)

    all_labels = np.concatenate([labels[i] for i in ids])
    all_preds = np.concatenate([preds[i] for i in ids])

    hard_ids = [i for i in ids if labels[i].mean() < _MAJOR]
    hard_labels = np.concatenate([labels[i] for i in hard_ids])
    hard_preds = np.concatenate([preds[i] for i in hard_ids])

    passage_pred = [preds[i].mean() for i in ids]
    passage_label = [labels[i].mean() for i in ids]

    return {
        "nonfact_auc_pr": 100 * auc_pr(all_labels > _INACCURATE, all_preds),
        "nonfact_star_auc_pr": 100 * auc_pr(hard_labels > _MAJOR, hard_preds),
        "factual_auc_pr": 100 * auc_pr(all_labels < _INACCURATE, -all_preds),
        "passage_pearson": 100 * float(pearsonr(passage_pred, passage_label)[0]),
        "passage_spearman": 100 * float(spearmanr(passage_pred, passage_label)[0]),
        "num_passages": len(ids),
        "num_sentences": int(all_labels.size),
        "nonfact_random_baseline": 100 * float((all_labels > _INACCURATE).mean()),
    }
