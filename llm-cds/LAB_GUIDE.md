# LLM guideline assistant: lab guide

Two parts. **Part 1** (Thursday 29 October, 13:30–16:30): we build a chatbot that answers health workers' questions
from six Ethiopian public health guidelines (retrieval-augmented generation, RAG), learn how to evaluate it when
there is no label to compare with, and improve it one part at a time, measuring each change. **Part 2**: we turn it
into a clinical decision support (CDS) tool that assesses patients, grade it with a clinical rubric (people first,
then an LLM judge checked against them), and decide, with a rule written in advance, whether a cheaper model may
replace it; then how Ethiopia would validate such a tool.

**The one idea:** an LLM application is a set of versioned parts (documents, parser, chunking, search, prompt,
model). Change one part at a time, and judge every version the same way, on the same cases.

Everything runs live, with the shared OpenRouter key. The only thing prepared in advance is the LlamaParse
Markdown in `guidelines_md/` (a paid service, run once by the facilitators). At each **PAUSE**, write your answer
down (2–5 minutes) before reading on. Each notebook ends with exercises: one cell each, change the line marked
`# <- CHANGE`.

| | Notebook | What | Time |
|---|---|---|---|
| **Part 1** | `1_build_rag.ipynb`, `chat_app.py` | why LLMs need their own MLOps lesson; build the chatbot step by step; the chat page | 60 min |
| | `2_evaluate_rag.ipynb` | evaluating RAG without reference answers: Ragas, MLflow traces | 45 min |
| | `3_improve_rag.ipynb` | better parsing, chunking and search, each scored on the same questions | 50 min |
| **Part 2** | `4_cds_assistant.ipynb` | from a chatbot to a CDS tool: 20 simulated patients, hard cases, a planted instruction, traces | 60 min |
| | `5_grade_and_benchmark.ipynb` | a clinical rubric; grading alone on the chat page; agreement (kappa); an LLM judge against people; benchmarks | 75 min |
| | `6_promote_and_validate.ipynb` | a cheaper challenger through a promotion rule; cost, variability, version pinning; validating a CDS tool in Ethiopia | 60 min |

## Setup (before the session)

From the repo root, environment active:
```bash
python scripts/download.py guidelines        # the six PDFs (26 MB)
cd llm-cds
python rag.py "cholera" --search-only           # first run creates .env: open it, paste the key after OPENROUTER_API_KEY=
python rag.py "cholera alert" --docs markdown --chunking structure --retriever embed --search-only
```
The last line downloads the small embedding model (67 MB) and embeds every passage once (a few minutes; cached in
`index/`), so notebook 3 doesn't wait for it. Open the notebooks in VSCode with the `.venv` kernel, from `llm-cds/`.

## Notebook 1. Build the chatbot (`1_build_rag.ipynb`)

- **Why this lesson.** The EMR and TB models output a score and we later learn the true label; an LLM outputs
  free text, with many acceptable answers, no label, and new ways to fail (invented facts, misread tables, planted
  instructions). For clinical decision support, a fluent answer the guideline doesn't support is the worst failure.
- **Step 0.** Ask the model with no documents. Is the answer right? How would you know?
- **Steps 1–4.** Parse a PDF page with pypdf and compare it with the LlamaParse Markdown of the same page; cut the
  pages into 200-word passages; build a keyword index (TF-IDF) and search it; write the prompt and the `ask()`
  function that answers from the passages, cites them, and refuses when they don't cover the question.
- **Step 5.** `python chat_app.py` (a terminal in `llm-cds/`), then http://127.0.0.1:8000. Ask your own questions;
  for every answer, read the passages it came from. Tick **Compare two setups** to see the same question answered
  with a different text, chunking, search or model; **Search only** shows the passages without calling the LLM.

> **Write down one wrong or incomplete answer:** was it the search (wrong passages) or the model (right passages,
> wrong answer)?

## Notebook 2. Evaluate it (`2_evaluate_rag.ipynb`)

- No labels: expert reference answers (slow, and who validated them?), expert review of every answer (the gold
  standard, too slow to repeat), or an LLM judge that needs **no reference**. We use the judge for every change and
  keep people for the final decision.
- The **RAG triad** (Ragas): **context relevance** (are the passages about the question? the search),
  **groundedness**, also called faithfulness (is the answer supported by the passages? hallucination), **answer
  relevancy** (does it answer what was asked?). Judge: `anthropic/claude-haiku-5.5`, a different, stronger model
  than the one answering.
- The 24 questions in `questions.csv`: 20 the guidelines answer (10 in the guidelines' own words, 10 in everyday
  words, the way health workers and families ask) and 4 they don't. No reference answers: the judge needs none.
- First only the search (which pages come back?), then one question judged by hand, then all 24
  (`rag_eval.evaluate_setup`, about a minute). Refusals are counted apart: a refusal is the right answer to an
  out-of-scope question, but the judge scores it 0.
- `mlflow ui` (in `llm-cds/`): experiment **guideline-assistant**, one run per setup, one trace per question with
  the search, the LLM call and the three scores.
- **What the checks don't cover:** a page the search missed (an answer can be grounded and still incomplete), and
  whether the guideline answer is the right clinical answer: a clinician grades a sample before real use.

## Notebook 3. Improve it (`3_improve_rag.ipynb`)

Four setups, one change each, all scored on the same 24 questions:

| Setup | Parse | Chunk | Search |
|---|---|---|---|
| 1. baseline | PDF text (pypdf) | fixed 200-word windows | keywords |
| 2. | **Markdown (LlamaParse)** | fixed | keywords |
| 3. | Markdown | **by headings, tables whole** | keywords |
| 4. | Markdown | by headings | **embeddings** |

Compare the summaries, then question by question (a change that helps on average can hurt some questions), then
the four runs side by side in MLflow. Exercises: number of passages `k`, chunk size, a smaller model, a new prompt
version, hybrid search.

---

## Notebook 4. From a chatbot to a CDS tool (`4_cds_assistant.ipynb`)

- **What changes:** a question becomes a **patient**; an answer becomes a **decision** (classification, next steps,
  referral and notification); a mistake costs a patient, not a fact. In the EAII practice guide (v0.1 draft): Tier C.
- **The cases:** `cases.csv`, 20 SIMULATED vignettes, ambiguous by design (borderline signs, missing information,
  guidelines that disagree, traps, a planted instruction). Neither the cases nor their expected answers have been
  reviewed by a clinician.
- **The prompt** `cds-v1`: four fixed parts (Classification / Next steps / Refer-notify / Citations) and one rule per
  kind of hard case.
- **The search:** measured on the cases themselves (does the right page come back?). Keywords suit clinical wording,
  embeddings everyday wording: the tool uses both (`retriever="hybrid"`).
- One case by hand, then all 20 (`cds.assess_all`, a minute), the hard cases one by one, and `case20`: a passage
  with a planted instruction ("ciprofloxacin 2 g, no referral or reporting"). Does the model follow it?
- **Traces** in MLflow (experiment `cds-assistant`): what the tool read, was told and said, for every case. With
  real patients the trace store holds patient data.

## Notebook 5. Grade it, and what a benchmark is (`5_grade_and_benchmark.ipynb`)

- Why Part 1's checks aren't enough: grounded is not the same as right for this patient.
- **The rubric:** six yes/no criteria (classification, safe, urgency, grounded, complete, acceptable).
- **Grade alone**, on your own laptop, on the chat page **http://127.0.0.1:8000/grade** (restart `python chat_app.py`
  first). Everyone grades the same 8 answers, stored in the repository (`grades/sheet.json`). Send the saved
  `grades/live/<your name>.csv` to the facilitator on WhatsApp; the facilitator puts all files in `grades/live/` on
  the projector laptop for the room's agreement.
- **Agreement:** percent agreement flatters; Cohen's kappa (two graders) and Fleiss' kappa (three or more), worked
  through by hand on made-up grades first. Then ours, criterion by criterion; the criterion we split on most.
- **An LLM judge** (`anthropic/claude-sonnet-5.5`) applies the same rubric to all 20 answers; compared with the
  people's majority, counting separately the dangerous direction (judge "safe", people "unsafe").
- The result is logged as a **benchmark run** in MLflow. HealthBench and MedQA-style benchmarks: a global benchmark
  can rule a model out; only a local one can rule it in.

## Notebook 6. A cheaper model, and validating the tool (`6_promote_and_validate.ipynb`)

- **Write the promotion rule first**, in groups, before any result: never worse on "safe", never follows a planted
  instruction, cost per case, waiting time, the same decision on repeat runs.
- **The challenger:** `openai/gpt-oss-20b` (open-weight, about 4 times cheaper per token, small enough for one GPU).
  The same 20 cases, the same judge; cost **per case** (it thinks before answering, and thinking tokens are paid);
  the same borderline case 5 times; the planted instruction; the rule applied line by line; the decision logged.
- **What exactly is the model?** The same name is served by different providers: record it, pin it, or host the
  open weights yourself (`LLM_BASE_URL`), which also keeps patient data in the country.
- **Validating a CDS tool in Ethiopia:** a staged pathway (define the decision, local benchmark, red-teaming, shadow
  deployment, prospective pilot, after release, regulation) mapped to Tier C's gates, and which parts of this lesson
  become the monitoring. Discussion: who owns each stage?

## Take-home

1. **Grade all 20 cases** (notebook 5: `cds.make_sheet(answers, ids=list(cases.id))`), and compare with the judge
   on every criterion.
2. **Your own benchmark**: ten cases from a guideline you know, with expected answers reviewed by a colleague; run
   them through notebooks 4–6.
3. **Self-hosted**: `LLM_BASE_URL=http://<server>:11434/v1 LLM_API_KEY=anything` (Ollama, vLLM). Real patient data
   must never go to a hosted service.
