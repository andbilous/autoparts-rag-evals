# Autoparts RAG: support assistant with an eval pipeline and a CI gate

A retrieval-augmented support assistant for an online auto parts store, plus the evaluation
setup that decides whether a new version is allowed to ship. A customer asks a technical
question in Ukrainian ("is a wear sensor included?", "what torque for the caliper bolts?"),
and the assistant answers only from manufacturer PDFs, citing the document and page. When
the documents have no answer, or the question is about price, delivery or orders, it says so
instead of guessing.

The project is the practical example for lesson 9 ("Evaluations for AI systems") of an AI
Engineering course: every metric from the lesson is implemented in code, shown on a
dashboard, and enforced by a GitHub Actions gate against a saved baseline.

> All manufacturers, part numbers and values in `data/pdfs` are fictional. They are generated
> by `scripts/make_synthetic_pdfs.py` and deliberately include page headers, a table split across
> two pages, a scanned PDF without a text layer, and two documents that contradict each other.

## What it does

- **Ingestion:** PDF parsing (PyMuPDF), OCR for scans (Tesseract, Ukrainian), chunking, and
  embeddings (`bge-m3`) stored in Postgres with pgvector.
- **Retrieval:** hybrid search. Lemma-based full-text search, exact part-number matching and
  vector search are merged with RRF and reranked by `bge-reranker-v2-m3`. A score threshold
  decides "not in the documents".
- **Answering:** Claude Haiku analyses the query, and Claude Sonnet streams the answer with
  `[n]` citations.
  - Product-card context (article and manufacturer of the page the user is on) narrows the
    search.
  - Safety-related questions get a warning.
  - Every number in an answer is checked against the cited sources. An answer with an
    unsupported number is retracted.
- **Chat UI** (`/`): streaming answers (SSE), citations that open the PDF on the right page,
  "was this helpful?" feedback, and sample questions taken from the testset.
- **Eval dashboard** (`/evals`) shows:
  - the latest report against the baseline;
  - stability across repeats;
  - judge calibration;
  - a threshold sweep;
  - per-case details;
  - online metrics from real traffic (latency percentiles, cost, feedback).

## Metrics

Each metric is tagged with who computes it: code (deterministic), an LLM-as-a-judge, or a
measurement. Definitions and formulas are in [`eval/METRICS.md`](eval/METRICS.md).

| Group | Metric | Computed by |
|---|---|---|
| Correctness | numbers from the reference found in the answer; LLM-as-a-judge correctness against a reference answer | code; judge |
| Format | answers citing a `[n]` label that was not in the context | code |
| Stability | pass@1, passed in every repeat, flaky cases | code |
| Abstain | correct refusal when the answer is not in the documents or out of scope; false refusal | code |
| Cost and speed | tokens and $ per question, first-token and total latency p50/p95 | measurement |
| Retrieval | recall@20 before the reranker, recall@5 after it | code |
| Context | `in_context`: the right page actually reached the prompt | code |
| Citations | citation precision: cited pages are the expected ones | code |
| Groundedness | faithfulness: every claim is supported by the cited chunks; unsupported numbers | judge; code |
| Judge quality | agreement, false pass and false fail against human labels (`rag calibrate`) | code over labels |
| Testset hygiene | lint: documents and pages exist, expected numbers appear on the source pages | code |

The testset (`eval/questions.jsonl`) has 53 cases covering:
- product-card questions;
- part numbers in free format;
- surzhyk (mixed Ukrainian and Russian);
- OCR'd scans;
- conflicting sources;
- follow-ups;
- safety-critical questions;
- no-answer and out-of-scope questions;
- prompt injection.

## Gate

`rag gate` compares a report with `eval/baseline.json` and exits with code 1 when:

- a rate drops by more than **0.05** (for "lower is better" rates such as false refusal: rises by more than 0.05);
- latency, cost or tokens grow by more than **25%**;
- any count that must stay at zero (unsupported numbers, invalid citations) becomes non-zero;
- a case that passed in the baseline now fails (a per-case regression).

The tolerance exists because one case out of 53 moves a rate by about 0.02. The gate ignores
one-case noise and catches real drops.

## Results

Current baseline: local run `20261109-113120-v1`, 53 questions × 3 repeats.

| pass | pass@1 | recall@5 | in_context | citation precision | judge correctness | faithfulness | false refusal | $ / question |
|---|---|---|---|---|---|---|---|---|
| 0.981 | 0.975 | 0.974 | 1.00 | 0.974 | 0.974 | 0.96 | 0.026 | 0.00235 |

The whole 3× run cost $0.37. `q04` (a surzhyk phrasing) is flaky, and `q39` is a false refusal.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, Tesseract with the
Ukrainian language pack, an Anthropic API key.

```bash
cp .env.example .env            # put ANTHROPIC_API_KEY here
docker compose up -d            # Postgres + pgvector on localhost:5444
uv sync
uv run rag migrate
uv run rag ingest               # parses data/pdfs; downloads bge-m3 and the reranker on first run
uv run uvicorn autoparts_rag.api:app --port 8765
```

Open http://localhost:8765 for the chat and http://localhost:8765/evals for the dashboard.

## CLI

| Command | What it does |
|---|---|
| `rag ask "..." [--article ...]` | one question from the terminal |
| `rag lint` | check the testset without calling a model |
| `rag eval [--repeat 3] [--gate] [--save-baseline]` | run the testset, write `eval/reports/<timestamp>.json` and `.md` |
| `rag gate [REPORT] [--skip latency]` | compare a report with the baseline; exit 1 on regression |
| `rag summary [REPORT]` | Markdown summary used for the GitHub Actions job summary |
| `rag calibrate [--dump]` | compare the groundedness judge with human labels in `eval/labels.jsonl` |

Unit tests: `uv run --group dev pytest -q` (no database or model needed).

## CI

[`.github/workflows/evals.yml`](.github/workflows/evals.yml) runs on every pull request that
touches code, the testset, the baseline, PDFs or dependencies. It can also be started manually.

1. **checks** (no API key, free): unit tests, Postgres service, OCR, building the index from
   `data/pdfs`, testset lint.
2. **evals** (needs the `ANTHROPIC_API_KEY` repository secret):
   - a full eval with the LLM-as-a-judge;
   - the gate against `eval/baseline.json`.

   Latency is excluded from the gate (`--skip latency`): the baseline was recorded on a
   different machine, and GitHub runners embed on CPU. Output:
   - a readable report with metrics, the gate result and failing cases in the job summary;
   - the full JSON report as the `eval-report` artifact.

Pull requests run one pass, about 25 minutes on a GitHub runner. For pass@1 and flaky
detection, start the workflow manually with `repeat: 3`.

To accept a new version, run `rag eval --repeat 3 --save-baseline` locally and commit
`eval/baseline.json`. The baseline change is then reviewed in the same diff as the change
that caused it.

## Layout

```
src/autoparts_rag/
  ingest/          PDF parsing, OCR, chunking, indexing
  models/          embedder, reranker, Claude client and judges
  retrieval.py     hybrid search + reranking
  pipeline.py      query analysis, context, answer, citations, retraction
  verify.py        number verification against sources
  evaluation/      dataset + lint, per-case metrics, aggregation, stability, gate, reports
  calibration.py   judge vs human labels
  web/             chat and eval dashboard (static HTML + JSON API)
  api.py, cli.py
eval/              questions.jsonl, baseline.json, METRICS.md, reports/ (git-ignored)
data/pdfs/         synthetic manufacturer documents + manifest.json
migrations/        database schema
TZ.md              product spec (in Ukrainian)
architecture.html  architecture diagram
```

Most of the content, including the UI, the testset, prompts and the spec, is in Ukrainian,
because the assistant serves Ukrainian-speaking customers.
