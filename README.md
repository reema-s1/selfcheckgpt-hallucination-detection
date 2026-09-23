# SelfCheck: Hallucination Detection from Sample Consistency

My implementation of
[*SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*](https://arxiv.org/abs/2303.08896),
reproduced against the paper's WikiBio-GPT3 benchmark, with a tested,
resumable evaluation pipeline.

**What this project covers**

- **Paper reproduction:** I implemented SelfCheck-Unigram and SelfCheck-NLI
  from the paper. SelfCheck-Unigram matches the paper's reported metrics to
  within 0.05 points on the full benchmark (238 passages, 1,908 sentences).
- **Factual consistency and hallucination detection:** sentence-level scores
  from black-box sample consistency, with no log-probabilities, retrieval or
  labels needed.
- **Automated evaluation:** the paper's protocol, implemented as code:
  AUC-PR for NonFact, NonFact\* and Factual, plus passage-level
  Pearson/Spearman correlation.
- **Production-oriented engineering:** batched and length-sorted NLI
  inference that is verified to leave scores unchanged, resumable JSONL
  checkpoints, injectable models, 17 tests that need no model downloads,
  and CI.

---

## The problem

LLMs state false facts fluently. Most ways of detecting this need something
you often don't have: token log-probabilities (not exposed by many APIs), an
external knowledge base, or labelled data. The paper asks whether
hallucinations can be detected **zero-resource** and **black-box**, using
nothing but the model's own text outputs.

## Core mechanism

When a model *knows* a fact, independent samples of the same prompt tend to
agree on it. When it is *hallucinating*, samples diverge and contradict each
other. So:

1. Take the response `R`, split into sentences `r_1 … r_I`.
2. Draw `N` stochastic samples `S^1 … S^N` from the same prompt.
3. Score each sentence by how inconsistent the samples are with it. A higher
   score means the sentence is more likely hallucinated.

Two scoring variants are implemented here:

| Variant | Sentence score |
|---|---|
| **NLI** | `mean_n P(contradiction | r_i, S^n)` from a DeBERTa-v3-large MNLI classifier. The softmax is taken over {entailment, contradiction} only; neutral is dropped. |
| **Unigram** | `max_j −log p(token_ij)` under a unigram LM fitted on `R` plus all samples (no smoothing). Facts the model doesn't repeat get low probability. |

## Implementation

```
src/selfcheck/
  nli.py        NLIConsistencyScorer: batched (sentence, sample) NLI scoring
  ngram.py      UnigramConsistencyScorer: sample-trained unigram surprisal
  metrics.py    the paper's evaluation protocol (AUC-PR, passage correlation)
  data.py       WikiBio-GPT3 loader (238 passages, 1,908 annotated sentences)
  evaluate.py   CLI: score passages, write scores + metrics, resumable
benchmarks/
  nli_batching.py   unbatched vs batched NLI throughput + score equivalence
tests/          17 unit tests, no model downloads needed
results/        committed scores and metrics from the runs below
```

- **Components are injectable.** `NLIConsistencyScorer` takes any Hugging
  Face sequence-classification model and tokenizer. It reads the
  entailment/contradiction label indices from the model config, or takes them
  explicitly. The paper's checkpoint names its labels only `LABEL_0` and
  `LABEL_1`, so I checked its mapping on unambiguous pairs: identical sentences
  gave entailment 1.000, and "black" vs "white" gave contradiction 1.000. That
  mapping is registered in `KNOWN_LABEL_INDICES`.
  `UnigramConsistencyScorer` takes its own tokenizer and sentence splitter
  (spaCy by default). This is what lets the tests run on tiny models.
- **The paper's protocol is followed exactly.** Labels are mapped as
  accurate = 0, minor = 0.5, major = 1:
  - NonFact: positives are sentences with label > 0.499.
  - NonFact\*: positives are major-inaccurate sentences, only in passages
    that aren't entirely major-inaccurate.
  - Factual: positives are accurate sentences, with the score negated.
  - Passage ranking: Pearson and Spearman correlation between the mean
    sentence score and the mean human label.
  - AUC-PR is the trapezoidal area under the PR curve, as reported in the
    paper.
- **Evaluation is reproducible and resumable.** Scores are appended to JSONL
  one passage at a time, so a long CPU run that is interrupted continues where
  it stopped. Metrics are recomputed from the saved scores.

### Engineering work

- **Batched NLI scoring.** One passage needs `sentences × samples` NLI calls,
  about 160 pairs for 8 sentences and 20 samples, or about 38k pairs for the
  whole benchmark. `contradiction_probs`:
  - scores the pairs in length-sorted mini-batches, so each batch pads to a
    similar length;
  - puts the results back in input order;
  - relies on the attention mask, so batching does not change any score;
  - runs under `torch.inference_mode()`.

  Scores are verified identical to one pair per forward pass. The measured
  speedup on CPU is small (see the benchmark below).
- **A label-mapping check.** The paper's checkpoint exposes unnamed labels, so
  a name-based lookup would fail or silently guess. The mapping was verified
  empirically and made explicit (see above).
- **Resumable, offline-capable runs.** Per-passage JSONL checkpoints mean an
  interrupted run resumes where it stopped. The pipeline also runs fully from
  the local model and dataset cache.

## Evaluation and results

### SelfCheck-Unigram: full benchmark (238 passages, 1,908 sentences)

| Metric | This implementation | Paper |
|---|---:|---:|
| NonFact AUC-PR | **85.62** | 85.63 |
| NonFact\* AUC-PR | 41.03 | – |
| Factual AUC-PR | **58.43** | 58.47 |
| Passage Pearson | **64.72** | 64.71 |
| Passage Spearman | 64.91 | – |
| Random baseline (NonFact) | 72.96 | 72.96 |

The reproduction matches the paper to within 0.05 points on every reported
metric. Command: `python -m selfcheck.evaluate --method unigram`, which took
about 3.5 minutes on CPU. "–" means the paper does not report that metric for
this method. Output: `results/unigram_metrics.json`.

### SelfCheck-NLI: end-to-end run on a 3-passage subset

Command: `python -m selfcheck.evaluate --method nli --limit 3`. This uses the
paper's DeBERTa-v3-large MNLI checkpoint and all 20 samples per passage: 25
sentences, 500 NLI pairs, about 25 minutes on CPU. Output:
`results/nli_metrics.json`.

| Metric | Value |
|---|---:|
| NonFact AUC-PR | 96.18 |
| Random baseline (NonFact) | 96.00 |

**This run is not a reproduction of the paper's NLI numbers.** The paper
reports 92.50 NonFact AUC-PR on all 238 passages. In these first 3 passages,
24 of the 25 sentences are labelled inaccurate, so the random baseline is
already 96%. Factual AUC-PR and passage correlation rest on a single accurate
sentence and three passages, so they are not meaningful here. What this run
does show is that the full NLI path works end to end on the real model and
data: loading, batched scoring, resumable checkpointing and metrics. The run
was also interrupted after 2 passages and resumed correctly.

A full NLI run (238 passages, about 38k pairs) is the natural next step on a
GPU. At the measured about 0.4 pairs/s, it is impractical on this CPU.

### Batching benchmark

Command: `python benchmarks/nli_batching.py --passages 1 --samples 5 --batch-sizes 1 16`.
This scores 45 (sentence, sample) pairs with the full DeBERTa-v3-large NLI
model on CPU (14 torch threads). Output: `results/nli_batching_benchmark.json`.

| batch_size | seconds | pairs/s | speedup | max abs score diff vs batch 1 |
|---:|---:|---:|---:|---:|
| 1 (one pair per pass) | 114.1 | 0.39 | 1.00x | 0 |
| 16 | 109.0 | 0.41 | 1.05x | 1.1e-7 |

**Batching changes no scores** (difference 1.1e-7, floating-point noise), but
**on this CPU it gives only a 1.05x speedup.** A 435M-parameter model on CPU
is limited by compute: a single forward pass already keeps all threads busy,
so grouping pairs saves little. The machine was also running another
CPU-heavy workload, so absolute times are pessimistic. Batching is expected
to matter more on a GPU, but that was not measured here.

### Tests

`pytest` gives **17 passed** in about 20 s on CPU, with no model downloads.

- **NLI.** Batched scores match one-pair-per-pass scores at batch sizes 2, 5
  and 64, using a small random BERT with a real tokenizer so padding and
  attention masks are really exercised. Other tests check:
  - pair order is preserved;
  - the neutral-dropping renormalisation, against hand-computed values;
  - the mean over samples;
  - explicit label indices for unnamed labels;
  - edge cases and invalid configurations.
- **Unigram.** Surprisal values, lowercasing, and unseen or empty inputs.
- **Metrics.** Perfect-ranking sanity checks, the minor vs major label
  semantics, `inf` handling, and error handling.

Two checks confirm the tests guard real properties:

- **Mutation check.** Writing batch results back in sorted order instead of
  input order makes the order test fail.
- **Padding check.** Padding every input to the maximum length leaves every
  score unchanged, which shows the attention mask makes batching safe.

## Tech stack

Python 3.10+ · PyTorch · Hugging Face Transformers (DeBERTa-v3) · Hugging Face
Datasets · spaCy · NumPy / SciPy / scikit-learn · pytest · ruff · GitHub Actions

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

pytest -v                                               # unit tests
python -m selfcheck.evaluate --method unigram            # full benchmark, CPU minutes
python -m selfcheck.evaluate --method nli --limit 3      # NLI on the first 3 passages
python benchmarks/nli_batching.py --passages 1 --samples 5 --batch-sizes 1 16
```

The first NLI run downloads the DeBERTa-v3-large MNLI checkpoint (~1.7 GB) and
the benchmark dataset from the Hugging Face Hub. After that, set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` to run fully from the local cache
with no network calls. On CPU, NLI scoring is the
slow part. See the throughput numbers above to estimate the run time for a
given `--limit`.

Library use:

```python
from selfcheck import NLIConsistencyScorer

scorer = NLIConsistencyScorer.from_pretrained(batch_size=16)
scores = scorer.score(sentences=response_sentences, samples=sampled_responses)
# scores[i] in [0, 1]: probability-like hallucination score for sentence i
```

## License

MIT; see [LICENSE](LICENSE). The evaluation data is downloaded at runtime from
the Hugging Face Hub and is not redistributed here. Only per-sentence scores
are committed in `results/`.
