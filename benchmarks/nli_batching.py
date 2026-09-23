"""Throughput of SelfCheck-NLI scoring: one pair per forward pass vs batched.

batch_size=1 is the unbatched baseline (one DeBERTa forward pass per
(sentence, sample) pair). Larger sizes use length-sorted mini-batches. The
benchmark also reports the largest per-sentence score difference against the
unbatched run, to show batching does not change results.

    python benchmarks/nli_batching.py --passages 2 --batch-sizes 1 8 16 32
"""

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from selfcheck.data import load_wikibio
from selfcheck.nli import (
    DEFAULT_NLI_MODEL,
    KNOWN_LABEL_INDICES,
    NLIConsistencyScorer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=int, default=2)
    parser.add_argument(
        "--samples", type=int, default=None, help="use first N samples per passage"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--out", default="results/nli_batching_benchmark.json")
    args = parser.parse_args()

    passages = load_wikibio(args.passages)
    if args.samples is not None:
        passages = [replace(p, samples=p.samples[: args.samples]) for p in passages]
    num_pairs = sum(len(p.sentences) * len(p.samples) for p in passages)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_NLI_MODEL, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_NLI_MODEL)
    threads = torch.get_num_threads()
    print(f"passages={len(passages)} pairs={num_pairs} torch_threads={threads}")

    rows, reference = [], None
    for batch_size in args.batch_sizes:
        scorer = NLIConsistencyScorer(
            model,
            tokenizer,
            batch_size=batch_size,
            label_indices=KNOWN_LABEL_INDICES[DEFAULT_NLI_MODEL],
        )
        start = time.perf_counter()
        scores = np.concatenate(
            [scorer.score(p.sentences, p.samples) for p in passages]
        )
        seconds = time.perf_counter() - start
        if reference is None:
            reference, baseline_seconds = scores, seconds
        rows.append(
            {
                "batch_size": batch_size,
                "seconds": round(seconds, 1),
                "pairs_per_second": round(num_pairs / seconds, 2),
                "speedup": round(baseline_seconds / seconds, 2),
                "max_abs_score_diff": float(np.abs(scores - reference).max()),
            }
        )
        print(rows[-1])

    with open(args.out, "w", encoding="utf-8") as f:
        summary = {
            "passages": len(passages),
            "samples_per_passage": args.samples or "all",
            "pairs": num_pairs,
            "torch_threads": threads,
            "runs": rows,
        }
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
