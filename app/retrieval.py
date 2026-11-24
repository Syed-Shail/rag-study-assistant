"""Hybrid retrieval: dense vector search fused with BM25 keyword search.

Dense embeddings capture meaning, but study questions often hinge on exact terms and
numbers ("28.4 BLEU", "β2 = 0.98") that embeddings blur. BM25 catches those. The two
ranked lists are combined with reciprocal rank fusion (RRF), which needs no score
calibration between the two methods.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = frozenset(
    "a an and are as at be by does did do for from how in is it of on or that the this to was what when where "
    "which who why with use used uses using".split()
)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return [token for token in TOKEN.findall(text) if token not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        results = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            results.append(score)
        return results


@dataclass(frozen=True)
class Hit:
    text: str
    metadata: dict
    score: float


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    best_distance: float  # closest dense match; used to decide whether the library can answer at all


_INDEX_CACHE: dict[tuple, tuple[list[str], list[str], list[dict], BM25]] = {}


def _keyword_index(collection):
    """Build (or reuse) the BM25 index for the collection's current contents."""
    key = (collection.name, collection.count())
    cached = _INDEX_CACHE.get(key)
    if cached is None:
        records = collection.get(include=["documents", "metadatas"])
        ids, docs, metas = records["ids"], records["documents"], records["metadatas"]
        cached = (ids, docs, metas, BM25(docs))
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = cached
    return cached


def retrieve(collection, query_embedding: list[float], question: str, k: int, *, mode: str = "hybrid",
             max_distance: float = float("inf"), candidates: int = 20, rrf_k: int = 60) -> Retrieval:
    """Return the top-k passages for a question using dense, keyword or hybrid ranking.

    Dense matches further than ``max_distance`` are treated as unrelated and never returned.
    """
    count = collection.count()
    if count == 0:
        return Retrieval([], float("inf"))
    dense = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidates if mode == "hybrid" else k, count),
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense["ids"][0]
    distances = dense["distances"][0]
    best_distance = distances[0] if distances else float("inf")
    by_id = {i: (doc, meta) for i, doc, meta in zip(dense_ids, dense["documents"][0], dense["metadatas"][0])}
    close = [(i, d) for i, d in zip(dense_ids, distances) if d <= max_distance]
    dense_ids = [i for i, _ in close]
    distances = [d for _, d in close]

    if mode == "dense":
        hits = [Hit(by_id[i][0], by_id[i][1], 1 - d) for i, d in zip(dense_ids[:k], distances[:k])]
        return Retrieval(hits, best_distance)

    ids, docs, metas, bm25 = _keyword_index(collection)
    keyword_scores = bm25.scores(question)
    keyword_ranked = [ids[i] for i in sorted(range(len(ids)), key=keyword_scores.__getitem__, reverse=True)
                      if keyword_scores[i] > 0][:candidates]
    for position, record_id in enumerate(ids):
        by_id.setdefault(record_id, (docs[position], metas[position]))

    if mode == "keyword":
        score_of = {record_id: keyword_scores[ids.index(record_id)] for record_id in keyword_ranked}
    else:
        score_of = {}
        for ranking in (dense_ids, keyword_ranked):
            for rank, record_id in enumerate(ranking, start=1):
                score_of[record_id] = score_of.get(record_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(score_of, key=score_of.__getitem__, reverse=True)
    hits = [Hit(by_id[i][0], by_id[i][1], score_of[i]) for i in ranked[:k]]
    return Retrieval(hits, best_distance)
