"""Score WikiBio-GPT3 passages with a SelfCheck method and report paper metrics.

    python -m selfcheck.evaluate --method unigram
    python -m selfcheck.evaluate --method nli --limit 20 --batch-size 16

Scores are appended to <out-dir>/<method>_scores.jsonl one passage at a time,
so an interrupted run resumes where it stopped. Metrics are computed over the
passages scored so far and written to <out-dir>/<method>_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from .data import Passage, load_wikibio
from .metrics import evaluate


def _load_scores(path: Path) -> dict[int, list[float]]:
    if not path.exists():
        return {}
    scores = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            scores[record["id"]] = record["scores"]
    return scores


def _make_scorer(method: str, batch_size: int):
    if method == "nli":
        from .nli import NLIConsistencyScorer  # noqa: PLC0415

        nli = NLIConsistencyScorer.from_pretrained(batch_size=batch_size)
        return lambda p: nli.score(p.sentences, p.samples)
    if method == "unigram":
        from .ngram import UnigramConsistencyScorer  # noqa: PLC0415

        unigram = UnigramConsistencyScorer.with_spacy()
        return lambda p: unigram.score(p.sentences, p.text, p.samples)
    msg = f"unknown method {method!r}"
    raise ValueError(msg)


def run(method: str, limit: int | None, batch_size: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"{method}_scores.jsonl"

    passages: list[Passage] = load_wikibio(limit)
    scores = _load_scores(scores_path)
    todo = [p for p in passages if p.id not in scores]

    start = time.perf_counter()
    if todo:
        score_fn = _make_scorer(method, batch_size)
        with scores_path.open("a", encoding="utf-8") as f:
            for passage in tqdm(todo, desc=f"SelfCheck-{method}"):
                passage_scores = [float(s) for s in score_fn(passage)]
                scores[passage.id] = passage_scores
                f.write(json.dumps({"id": passage.id, "scores": passage_scores}) + "\n")
                f.flush()
    elapsed = time.perf_counter() - start

    metrics = evaluate(
        {p.id: p.annotations for p in passages},
        {p.id: scores[p.id] for p in passages},
    )
    metrics.update(
        method=method,
        limit=limit,
        batch_size=batch_size if method == "nli" else None,
        seconds_this_run=round(elapsed, 1),
        passages_scored_this_run=len(todo),
    )
    (out_dir / f"{method}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["nli", "unigram"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N passages")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    metrics = run(args.method, args.limit, args.batch_size, args.out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
