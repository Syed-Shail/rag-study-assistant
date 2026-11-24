"""Verify and repair inline citations after generation.

Language models, small local ones especially, often state the right fact but cite the
wrong passage, or forget to cite. This module re-attaches each answer sentence to the
retrieved passage that best supports it, using IDF-weighted term overlap, and replaces
whatever citation the model wrote. Sentences with no supporting passage are left uncited
rather than given a misleading source.
"""

from __future__ import annotations

from collections import Counter
import math
import re

from app.retrieval import tokenize

CITATION = re.compile(r"\s*\[Source:[^\]]*\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"“])")
ABSTAIN = ("couldn’t find", "couldn't find", "does not say", "does not mention", "doesn't say", "not stated",
           "not explicitly stated", "does not contain", "no information")


def _support(sentence_terms: list[str], passage_terms: Counter, idf: dict[str, float]) -> float:
    if not sentence_terms:
        return 0.0
    shared = sum(idf.get(t, 0.0) for t in set(sentence_terms) if t in passage_terms)
    total = sum(idf.get(t, 0.0) for t in set(sentence_terms))
    return shared / total if total else 0.0


def repair_citations(answer: str, sources: list, min_support: float = 0.35) -> str:
    """Return the answer with each sentence cited to its best-supporting source.

    ``sources`` need ``filename``, ``page`` and ``text`` attributes.
    """
    if not sources or any(marker in answer.lower() for marker in ABSTAIN):
        return answer
    passages = [Counter(tokenize(source.text)) for source in sources]
    n = len(passages)
    frequency = Counter(term for passage in passages for term in passage)
    idf = {term: math.log(1 + n / df) for term, df in frequency.items()}
    repaired = []
    for sentence in SENTENCE.split(CITATION.sub("", answer).strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        terms = tokenize(sentence)
        # Terms absent from every passage still count against support, so pure padding is never cited.
        weights = {t: idf.get(t, math.log(1 + n)) for t in terms}
        scores = [_support(terms, passage, weights) for passage in passages]
        best = max(range(n), key=scores.__getitem__)
        if scores[best] >= min_support:
            source = sources[best]
            body = sentence.rstrip()
            end = body[-1] if body[-1] in ".!?" else ""
            body = body[:-1] if end else body
            sentence = f"{body} [Source: {source.filename}, p. {source.page}]{end}"
        repaired.append(sentence)
    return " ".join(repaired)
