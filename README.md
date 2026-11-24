# RAG Study Assistant

Ask questions about your own study material (PDF, DOCX or TXT) and get short answers that cite the exact file and page they came from. The assistant runs on the **OpenAI API** or fully offline on **local Ollama models**, combines semantic and keyword search, refuses to answer when your documents don't contain the answer, and checks every citation before showing it.

It ships with a reproducible benchmark, so its quality is measured rather than assumed.

## Results

Benchmark: four arXiv papers (Transformer, BERT, ResNet, Adam; 58 pages, 288 chunks), **37 questions with known evidence** and **10 questions the papers cannot answer**. Measured with local models (`phi3` 3.8B for answers, `nomic-embed-text` for embeddings) on an Apple M5 laptop. Full output: [`eval/results/ollama.json`](eval/results/ollama.json).

**Retrieval** (is the passage containing the answer in the top results?)

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Dense vectors only | 40.5% | 62.2% | 78.4% | 0.547 |
| **Hybrid: dense + BM25 (default)** | **51.4%** | **78.4%** | **86.5%** | **0.639** |

**Answers**

| Metric | Result |
|---|---|
| Answer accuracy (35 fact questions) | **85.7%** |
| Answers that include a citation | **89.2%** |
| Citations pointing to the page that holds the evidence | **70.3%** (46.0% before citation verification) |
| Unanswerable questions correctly refused | **10 / 10** |
| Answerable questions wrongly refused (held-out half) | **0 / 18** |
| Median answer time / p95 | 3.6 s / 7.0 s |
| Hybrid retrieval latency (median, after query embedding) | 1.9 ms |
| Indexing speed | 7.2 pages/s (58 pages in 8 s) |

The refusal cutoff was calibrated on a development half of the questions and reported on the held-out half. The previous fixed cutoff (0.72) refused **none** of the off-topic questions with this embedding model.

## How it works

```
upload ─► parse (PDF/DOCX/TXT, keeps page numbers) ─► chunk (900 chars, 150 overlap)
       ─► embed in batches of 64 ─► Chroma (persistent, cosine)

question ─► embed ─► dense top-20 ─┐
         └─► BM25 top-20 ──────────┴─► reciprocal rank fusion ─► top-5 passages
         ─► refuse if the closest passage is beyond the calibrated cutoff
         ─► grounded prompt (few-shot, temperature 0, chat history) ─► answer
         ─► citation verification: each sentence re-linked to the passage that supports it
```

- **Two providers.** `LLM_PROVIDER=auto` uses OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) when `OPENAI_API_KEY` is set and local Ollama (`phi3`, `nomic-embed-text`) otherwise. Each provider gets its own Chroma collection, because their vectors have different sizes.
- **Hybrid retrieval** ([`app/retrieval.py`](app/retrieval.py)). Embeddings capture meaning but blur exact terms and numbers such as "28.4 BLEU" or "β2 = 0.98". An Okapi BM25 index over the same chunks catches those, and reciprocal rank fusion merges the two rankings without needing to calibrate their scores against each other.
- **Calibrated refusal.** If even the closest passage is too far from the question, the assistant says the material doesn't cover it instead of guessing. The cutoff is set per embedding model from the benchmark.
- **Citation verification** ([`app/citations.py`](app/citations.py)). Models often state the right fact but cite the wrong passage. After generation, each sentence is matched to the retrieved passage with the highest IDF-weighted term overlap and its citation is rewritten to point there; unsupported sentences are left uncited rather than given a misleading source.
- **Large documents.** Embedding runs in bounded batches of 64, tested with a 1,024-page file. Re-uploading a file replaces its old chunks, and a failed upload never deletes the existing copy.

## Run locally

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # add OPENAI_API_KEY, or leave it empty to use Ollama
streamlit run app/main.py
```

For offline mode, install [Ollama](https://ollama.com) and pull the models first:

```bash
ollama pull phi3:latest
ollama pull nomic-embed-text
```

### Docker

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:8501`. Indexed data persists in `data/chroma/`. In offline mode the container reaches Ollama on the host through `host.docker.internal`.

## Reproduce the benchmark

```bash
python -m eval.run_eval              # retrieval and refusal metrics (about a minute)
python -m eval.run_eval --generate   # also generate and grade every answer
```

The script downloads the four papers from arXiv on first run, builds a fresh temporary index, verifies that every evidence phrase in [`eval/questions.json`](eval/questions.json) exists in the extracted text, and writes `eval/results/<provider>.json`. Run it after changing models to recalibrate the refusal cutoff.

## Tests

```bash
python -m unittest discover -s tests -v
```

36 tests with a temporary Chroma database and mocked model calls, so no API key or model download is needed. They cover parsing, 1,024-page batched indexing, collection replacement and failure recovery, provider selection, OpenAI and Ollama model construction, BM25 and hybrid ranking, distance filtering, refusal, citation repair, chat history, the full RAG flow and Streamlit start-up.

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example)): provider and model names, `RETRIEVAL_MODE` (`hybrid`, `dense` or `keyword`), `RETRIEVAL_K` (1–10), `RELEVANCE_CUTOFF` (empty uses the calibrated default), `REPAIR_CITATIONS`, chunk size and overlap, Chroma location and the upload size limit.

## Limitations

- The benchmark questions reuse the papers' terminology, which favours keyword search; keyword-only ranking scored slightly higher than hybrid here (Hit@5 89.2%). Hybrid stays the default because it is more robust to questions phrased in your own words.
- 47 questions is a small benchmark; treat differences of a few points as indicative.
- Scanned PDFs need OCR first. DOCX files have no reliable page boundaries, so their citations point to page 1.
- The local index contains your study text. Keep `data/chroma/` and `.env` private; both are git-ignored.

## Project structure

```
app/
  main.py         Streamlit interface: upload, library, chat
  config.py       settings, provider selection, calibrated defaults
  ingestion.py    parsing, chunking, batched embedding, Chroma storage
  retrieval.py    BM25, dense and hybrid (RRF) retrieval
  qa.py           refusal gate, grounded prompt, answer generation
  citations.py    post-generation citation verification
eval/
  questions.json  benchmark questions, evidence phrases and answers
  run_eval.py     benchmark runner
  results/        saved benchmark results
tests/            unit and integration tests
```
