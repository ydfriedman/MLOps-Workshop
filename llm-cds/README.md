# LLM clinical decision support (CDS): a guideline assistant, and how to evaluate it

> **Status: ready, pending clinical review.** The three notebooks run live on hosted open-weight models
> (OpenRouter, the shared key); only the LlamaParse Markdown is prepared in advance. `cases.csv`,
> `eval_questions.csv` and `prescriptions.csv` are SIMULATED or drafted from the guideline text and still need a
> clinician's review. The notebooks use only the **questions** in `eval_questions.csv`, not its draft reference
> answers.

A health worker describes a patient or asks a question; the assistant finds passages in six Ethiopian public health
guidelines and an LLM answers **from those passages only**, citing pages (retrieval-augmented generation, RAG).

**The one idea:** an LLM application is a set of versioned parts (documents, parser, chunking, search, prompt,
model); change one part at a time and judge every version the same way: with a reference-free LLM judge (Ragas)
for each change, and with people, against whom the judge is checked, for the decision.

```
guideline PDFs ─► parser (pypdf | LlamaParse → Markdown) ─► chunks (fixed | by structure) ─► search (keywords | embeddings)
                                                                                                    │ k passages
question / case ───────────────────────────────────────────────────────────────────────────► LLM (prompt version, model)
                                                                                                    │
        answer + citations ──► Ragas judge: context relevance · groundedness · answer relevancy ──► MLflow (runs, traces)
```

## Setup

From the repo root, environment active:
```bash
python scripts/download.py guidelines        # the six PDFs (26 MB)
cd llm-cds
export OPENROUTER_API_KEY=...                # the shared workshop key
python rag.py "cholera alert" --docs markdown --chunking structure --retriever embed --search-only
```
The last line needs no key: it downloads the 67 MB English embedding model (`BAAI/bge-small-en-v1.5`, on
onnxruntime via `fastembed`) and embeds every passage once (a few minutes on a laptop; cached in `index/`).
Then open the notebooks in VSCode (kernel `.venv`), and in a terminal in `llm-cds/`:
```bash
python chat_app.py                           # the chat page: http://127.0.0.1:8000
mlflow ui                                    # http://127.0.0.1:5000, experiment "guideline-assistant"
```

## The guidelines

| Document | Publisher | Pages |
|---|---|---|
| National Guideline for Cholera Surveillance and Outbreak Response (2022) | EPHI (PHEM) | 124 |
| Emergency Nutrition Guideline (2022) | EPHI (PHEM) | 92 |
| Climate-Sensitive Diseases Surveillance Manual (2022) | EPHI (PHEM) | 110 |
| Community and Event-Based Surveillance Implementation Manual (2021) | EPHI (PHEM) | 50 |
| National Guideline for the Management of Acute Malnutrition (2019) | Ministry of Health | 156 |
| National Guidelines for TB, DR-TB and Leprosy, 6th ed. (2017) | Ministry of Health | 217 |

The PDFs are downloaded from the publishers (URLs and checksums: [`scripts/downloads.json`](../scripts/downloads.json)).
Their Markdown versions in `guidelines_md/` were made once by the facilitators with LlamaParse (tables as rows,
running heads and page numbers removed, PDF page markers kept); participants never need LlamaParse or its key.
Parameters, licence note and quality checks: [`data/SOURCES.md`](data/SOURCES.md).

## The session

| Notebook | What |
|---|---|
| [`1_build_rag.ipynb`](1_build_rag.ipynb) | why LLM systems need their own evaluation; parse, chunk, index, search, generate, step by step; the chat page |
| [`2_evaluate_rag.ipynb`](2_evaluate_rag.ipynb) | evaluating without reference answers: context relevance, groundedness, answer relevancy (Ragas); 24 questions; MLflow traces |
| [`3_improve_rag.ipynb`](3_improve_rag.ipynb) | Markdown vs PDF text, chunking by headings, embeddings vs keywords: each scored on the same questions and compared |

[`LAB_GUIDE.md`](LAB_GUIDE.md) has the timings and PAUSEs, and extras for time left over (rubrics and grader
agreement, a cheaper model through a promotion rule, validating an LLM tool in Ethiopia) that use the facilitators'
saved runs in `results/`.

## Files

| File | What |
|---|---|
| `1_build_rag.ipynb`, `2_evaluate_rag.ipynb`, `3_improve_rag.ipynb` | the session |
| `rag_eval.py` | reference-free evaluation of one setup (Ragas: context relevance, groundedness, answer relevancy; judge `anthropic/claude-haiku-5.5`), logged to MLflow as one run with a trace per question |
| `chat_app.py` | the chat page: answer, passages, prompt; two setups side by side |
| `rag.py` | passages (`--docs pdf|markdown`, `--chunking fixed|structure`), search (keywords or embeddings), answer |
| `evaluate.py` | the earlier, reference-based evaluation (right page, context recall, factual correctness); kept for the saved runs |
| `explain_score.py` | one Ragas score claim by claim |
| `eval_questions.csv` | 20 answerable + 4 out-of-scope questions with references and pages (draft, needs review) |
| `cases.py`, `cases.csv` | 20 SIMULATED vignettes: answer, grading sheet, agreement (Cohen, Fleiss), LLM judge, cost |
| `grades/` | grading sheets; `facilitator_draft.csv` is a fallback draft, not clinical |
| `check.py`, `evaluate_check.py`, `prescriptions.csv` | take-home: the prescription check (44 simulated records) |
| `guidelines_md/`, `data/SOURCES.md` | the guidelines as Markdown; sources, licence, parse parameters, comparisons |
| `facilitator/` | `parse_guidelines.py` (LlamaParse, facilitator only), `check_quotes.py` |
| `results/` | saved answers, verdicts, judge grades and scores (reused on re-runs; the offline fallback) |

## Helper functions in this lesson

| Helper | Where | What it does |
|---|---|---|
| `Settings` | `rag.py` | everything that defines one version of the assistant (documents, chunking, search, prompt, model); logged with every run |
| `load_pages`, `make_passages` | `rag.py` | PDF pages cut into overlapping passages (never across a page) |
| `markdown_blocks`, `merge_continued_tables`, `make_structure_passages` | `rag.py` | the structure-aware route: Markdown split into blocks, a table carried over a page break rejoined, passages that follow headings |
| `Retriever`, `EmbeddingRetriever`, `build` | `rag.py` | keyword search or embedding search over the passages |
| `answer(question, retriever, settings)` | `rag.py` | retrieve, then generate; returns the answer and everything needed to evaluate it |
| `fingerprint`, `source_ids` | `rag.py` | a short hash of the indexed text (changes when a document or the chunking changes) and `doc:p28` style source labels |
| `evaluate_setup(settings, name)`, `compare(results)` | `rag_eval.py` | answer the 24 questions with one setup, judge each answer three ways, log it all to MLflow; one row per setup to compare |
| `retrieval_checks`, `rule_checks`, `evaluate` | `evaluate.py` | right page in the top k; simple rules; Ragas scores; everything logged to MLflow |
| `sheet`, `agreement`, `fleiss_kappa`, `judge` | `cases.py` | the grading sheet, agreement between graders (Cohen, Fleiss), the LLM judge, checked against people |
| `call`, `cost` | `cases.py` | one LLM call with tokens, cost and seconds as OpenRouter reports them; total spend |

## Open items

- Clinical review of `cases.csv` (cases to check first in the facilitator notes), `eval_questions.csv`,
  `prescriptions.csv`; Yoni regrades `grades/facilitator_draft.csv`.
- Licence: the Markdown in `guidelines_md/` is a full-text copy of the guidelines; confirm before publishing.
- EFDA's current requirements for software as a medical device: not checked.
- Real patient data never goes to a hosted service; `LLM_BASE_URL` points everything at a self-hosted server.
