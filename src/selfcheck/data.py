"""WikiBio-GPT3 hallucination benchmark (238 annotated GPT-3 passages)."""

from __future__ import annotations

from dataclasses import dataclass

DATASET_NAME = "potsawee/wiki_bio_gpt3_hallucination"


@dataclass(frozen=True)
class Passage:
    id: int
    text: str  # the GPT-3 response being checked
    sentences: list[str]  # the response split into sentences
    samples: list[str]  # stochastic re-samples of the same prompt
    annotations: list[str]  # per-sentence human label


def load_wikibio(limit: int | None = None) -> list[Passage]:
    from datasets import load_dataset  # noqa: PLC0415

    rows = load_dataset(DATASET_NAME)["evaluation"]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [
        Passage(
            id=int(r["wiki_bio_test_idx"]),
            text=r["gpt3_text"],
            sentences=list(r["gpt3_sentences"]),
            samples=list(r["gpt3_text_samples"]),
            annotations=list(r["annotation"]),
        )
        for r in rows
    ]
