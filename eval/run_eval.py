"""Benchmark the study assistant: retrieval, abstention, answers, citations and latency.

The benchmark indexes four arXiv papers (58 pages) into a fresh, temporary Chroma
collection and asks 37 questions with known evidence plus 10 questions the papers
cannot answer. Questions are split into a development half (used to calibrate the
"can the library answer this?" cutoff) and a held-out test half (used to report it).

    python -m eval.run_eval                 # retrieval + abstention (about a minute)
    python -m eval.run_eval --generate      # also generate and grade full answers

Results are written to eval/results/<provider>.json and printed as a summary.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import statistics
import tempfile
import time
import unicodedata
import urllib.request

from app.config import get_settings
from app.ingestion import StudyEmbeddings, get_collection, ingest_document, parse_document
from app.qa import answer_question
from app.retrieval import retrieve

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
RESULTS_DIR = EVAL_DIR / "results"
MODES = ("dense", "keyword", "hybrid")
CITATION = re.compile(r"\[Source:\s*([^,\]]+?)\s*,\s*p\.?\s*(\d+)", re.IGNORECASE)
ABSTAIN_MARKERS = ("couldn’t find", "couldn't find", "does not say", "does not mention", "does not contain",
                   "doesn't say", "doesn't mention", "not mentioned", "no information", "not provided",
                   "does not provide", "not in the provided", "cannot answer", "can't answer")


def normalise(text: str) -> str:
    """Fold PDF ligatures and whitespace so evidence phrases match extracted text."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def compact(text: str) -> str:
    return re.sub(r"[\s,]", "", normalise(text))


def ensure_corpus(corpus: list[dict]) -> None:
    CORPUS_DIR.mkdir(exist_ok=True)
    for item in corpus:
        path = CORPUS_DIR / item["file"]
        if not path.exists():
            print(f"Downloading {item['title']} …")
            request = urllib.request.Request(item["url"], headers={"User-Agent": "rag-study-assistant-eval"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())


def evidence_pages(spec: dict) -> dict[str, set[int]]:
    """Map each answerable question to the pages containing its evidence, failing loudly on typos."""
    pages = {item["file"]: parse_document(item["file"], (CORPUS_DIR / item["file"]).read_bytes()) for item in spec["corpus"]}
    found: dict[str, set[int]] = {}
    for q in spec["answerable"]:
        phrases = [normalise(p) for p in q["evidence"]]
        hits = {page.page for page in pages[q["file"]] if any(p in normalise(page.text) for p in phrases)}
        if not hits:
            raise SystemExit(f"Evidence for {q['id']} was not found in {q['file']}; fix eval/questions.json.")
        found[q["id"]] = hits
    return found


def is_relevant(question: dict, metadata: dict, text: str) -> bool:
    body = normalise(text)
    return metadata.get("source") == question["file"] and any(normalise(p) in body for p in question["evidence"])


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))]


def ranking_metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "hit_at_1": round(sum(r == 1 for r in ranks) / n, 4),
        "hit_at_3": round(sum(r is not None and r <= 3 for r in ranks) / n, 4),
        "hit_at_5": round(sum(r is not None and r <= 5 for r in ranks) / n, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
    }


def calibrate_cutoff(answerable: list[float], unanswerable: list[float]) -> float:
    """Choose the distance cutoff that best separates the development questions (balanced accuracy)."""
    candidates = sorted(set(answerable + unanswerable))
    best, best_score = candidates[0], -1.0
    for low, high in zip(candidates, candidates[1:] + [candidates[-1] + 0.05]):
        cut = (low + high) / 2  # midpoint between neighbouring distances leaves a margin on both sides
        score = (sum(d <= cut for d in answerable) / len(answerable) + sum(d > cut for d in unanswerable) / len(unanswerable)) / 2
        if score > best_score:
            best, best_score = cut, score
    return round(best, 4)


def gate_metrics(cut: float, answerable: list[float], unanswerable: list[float]) -> dict:
    return {
        "unanswerable_correctly_abstained": round(sum(d > cut for d in unanswerable) / len(unanswerable), 4),
        "answerable_wrongly_blocked": round(sum(d > cut for d in answerable) / len(answerable), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true", help="also generate full answers and grade them")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per question (default 5)")
    args = parser.parse_args()

    spec = json.loads((EVAL_DIR / "questions.json").read_text())
    ensure_corpus(spec["corpus"])
    pages_with_evidence = evidence_pages(spec)
    answerable, unanswerable = spec["answerable"], spec["unanswerable"]
    dev = {q["id"] for q in answerable[0::2] + unanswerable[0::2]}

    with tempfile.TemporaryDirectory() as tmp:
        base = get_settings()
        settings = replace(base, chroma_dir=Path(tmp), collection_name=f"eval_{base.provider}")
        print(f"Provider: {settings.provider} | chat: {settings.chat_model} | embeddings: {settings.embedding_model}")

        start = time.perf_counter()
        pages = chunks = 0
        for item in spec["corpus"]:
            result = ingest_document(item["file"], (CORPUS_DIR / item["file"]).read_bytes(), settings)
            pages += result.pages_read
            chunks += result.chunks_added
        ingest_seconds = time.perf_counter() - start

        collection = get_collection(settings)
        embedder = StudyEmbeddings(settings.embedding_model, settings.ollama_base_url, provider=settings.provider)
        query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in answerable + unanswerable}

        # Retrieval quality for each ranking method (ranking only; the abstention gate is measured separately).
        comparison, latencies, best_distance, per_question = {}, [], {}, []
        for mode in MODES:
            ranks = []
            for q in answerable:
                t0 = time.perf_counter()
                found = retrieve(collection, query_vectors[q["id"]], q["question"], args.k, mode=mode)
                if mode == "hybrid":
                    latencies.append(time.perf_counter() - t0)
                rank = next((i + 1 for i, hit in enumerate(found.hits) if is_relevant(q, hit.metadata, hit.text)), None)
                ranks.append(rank)
                best_distance[q["id"]] = found.best_distance
                if mode == "hybrid":
                    per_question.append({"id": q["id"], "rank": rank, "best_distance": round(found.best_distance, 4)})
            comparison[mode] = ranking_metrics(ranks)
        for q in unanswerable:
            best_distance[q["id"]] = retrieve(collection, query_vectors[q["id"]], q["question"], 1, mode="dense").best_distance

        # Abstention gate: calibrate on the development half, report on the held-out half.
        split = lambda items, in_dev: [best_distance[q["id"]] for q in items if (q["id"] in dev) == in_dev]
        cutoff = calibrate_cutoff(split(answerable, True), split(unanswerable, True))
        report = {
            "provider": settings.provider,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "k": args.k,
            "corpus": {"documents": len(spec["corpus"]), "pages": pages, "chunks": chunks,
                       "ingest_seconds": round(ingest_seconds, 2), "pages_per_second": round(pages / ingest_seconds, 2)},
            "retrieval": {
                "questions": len(answerable),
                "by_method": comparison,
                "hybrid_median_latency_ms": round(statistics.median(latencies) * 1000, 1),
                "hybrid_p95_latency_ms": round(percentile(latencies, 95) * 1000, 1),
            },
            "abstention_gate": {
                "calibrated_cutoff": cutoff,
                "previous_default_cutoff": 0.72,
                "test_split": gate_metrics(cutoff, split(answerable, False), split(unanswerable, False)),
                "dev_split": gate_metrics(cutoff, split(answerable, True), split(unanswerable, True)),
                "with_previous_default": gate_metrics(0.72, [best_distance[q["id"]] for q in answerable],
                                                      [best_distance[q["id"]] for q in unanswerable]),
            },
            "per_question": per_question,
        }

        if args.generate:
            settings = replace(settings, relevance_cutoff=cutoff, retrieval_mode="hybrid")
            graded, cited, correct_cites, answer_latency = [], 0, 0, []
            for q in answerable:
                t0 = time.perf_counter()
                answer = answer_question(q["question"], [], settings, k=args.k)
                answer_latency.append(time.perf_counter() - t0)
                text = compact(answer.text)
                is_correct = any(compact(a) in text for a in q["answer"]) if q["answer"] else None
                cites = CITATION.findall(answer.text)
                good = [c for c in cites if c[0].strip() == q["file"] and int(c[1]) in pages_with_evidence[q["id"]]]
                cited += bool(cites)
                correct_cites += bool(good)
                graded.append({"id": q["id"], "correct": is_correct, "cited": bool(cites),
                               "citation_correct": bool(good), "answer": answer.text})
            abstained = 0
            for q in unanswerable:
                t0 = time.perf_counter()
                answer = answer_question(q["question"], [], settings, k=args.k)
                answer_latency.append(time.perf_counter() - t0)
                did_abstain = (not answer.sources) or any(m in normalise(answer.text) for m in ABSTAIN_MARKERS)
                abstained += did_abstain
                graded.append({"id": q["id"], "abstained": did_abstain, "answer": answer.text})
            scored = [g for g in graded if g.get("correct") is not None]
            report["answers"] = {
                "graded_questions": len(scored),
                "accuracy": round(sum(g["correct"] for g in scored) / len(scored), 4),
                "answers_with_citations": round(cited / len(answerable), 4),
                "citations_pointing_to_evidence_page": round(correct_cites / len(answerable), 4),
                "unanswerable_abstained": round(abstained / len(unanswerable), 4),
                "median_answer_seconds": round(statistics.median(answer_latency), 2),
                "p95_answer_seconds": round(percentile(answer_latency, 95), 2),
            }
            report["answer_log"] = graded

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{report['provider']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    summary = {key: value for key, value in report.items() if key not in ("per_question", "answer_log")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
