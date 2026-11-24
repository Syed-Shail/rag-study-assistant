from dataclasses import dataclass
import tempfile
import unittest
from pathlib import Path

from app.citations import repair_citations
from app.config import Settings
from app.ingestion import get_collection
from app.retrieval import BM25, retrieve, tokenize


@dataclass
class Src:
    filename: str
    page: int
    text: str


SOURCES = [
    Src("transformer.pdf", 3, "The encoder is composed of a stack of N = 6 identical layers."),
    Src("transformer.pdf", 7, "We used the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and warmup_steps = 4000."),
]


class CitationRepairTests(unittest.TestCase):
    def test_wrong_citation_is_moved_to_the_supporting_passage(self):
        answer = "The encoder has a stack of 6 identical layers [Source: transformer.pdf, p. 7]."
        self.assertEqual(
            repair_citations(answer, SOURCES),
            "The encoder has a stack of 6 identical layers [Source: transformer.pdf, p. 3].",
        )

    def test_missing_citations_are_added_per_sentence(self):
        answer = "The encoder stacks 6 identical layers. Training used Adam with beta2 = 0.98."
        repaired = repair_citations(answer, SOURCES)
        self.assertIn("identical layers [Source: transformer.pdf, p. 3].", repaired)
        self.assertIn("0.98 [Source: transformer.pdf, p. 7].", repaired)

    def test_unsupported_sentences_are_not_given_a_source(self):
        answer = "Bananas are an excellent source of potassium."
        self.assertEqual(repair_citations(answer, SOURCES), answer)

    def test_abstentions_are_left_untouched(self):
        answer = "The provided material does not say what mitochondria produce."
        self.assertEqual(repair_citations(answer, SOURCES), answer)


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.collection = get_collection(Settings(chroma_dir=Path(self.tmp.name), collection_name="hybrid_test"))
        self.collection.add(
            ids=["semantic", "exact", "unrelated"],
            documents=[
                "Attention lets the model relate distant positions in a sequence.",
                "The big model reached a BLEU score of 28.4 on WMT 2014.",
                "Sourdough needs a long, cool fermentation.",
            ],
            embeddings=[[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
            metadatas=[{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 8}, {"source": "b.txt", "page": 1}],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_tokenizer_keeps_decimal_numbers_and_drops_stopwords(self):
        self.assertEqual(tokenize("What is the BLEU score of 28.4?"), ["bleu", "score", "28.4"])

    def test_bm25_ranks_exact_term_matches_first(self):
        scores = BM25(["bleu score 28.4", "attention positions"]).scores("28.4 bleu")
        self.assertGreater(scores[0], scores[1])

    def test_hybrid_promotes_an_exact_number_match_that_dense_ranks_lower(self):
        dense = retrieve(self.collection, [1.0, 0.0], "Which BLEU score of 28.4 was reported?", 1, mode="dense")
        hybrid = retrieve(self.collection, [1.0, 0.0], "Which BLEU score of 28.4 was reported?", 2, mode="hybrid",
                          max_distance=0.5)
        self.assertEqual(dense.hits[0].metadata["page"], 1)
        self.assertIn(8, [hit.metadata["page"] for hit in hybrid.hits])

    def test_distant_passages_are_excluded_from_hybrid_results(self):
        found = retrieve(self.collection, [1.0, 0.0], "attention positions", 3, mode="hybrid", max_distance=0.5)
        self.assertNotIn("b.txt", [hit.metadata["source"] for hit in found.hits])
        self.assertAlmostEqual(found.best_distance, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
