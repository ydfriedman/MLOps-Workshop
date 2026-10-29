"""Evaluate one setup of the guideline assistant WITHOUT reference answers, and show it in MLflow.

Used by 2_evaluate_rag.ipynb and 3_improve_rag.ipynb:

    import rag, rag_eval
    result = rag_eval.evaluate_setup(rag.Settings(), "baseline: PDF text, fixed chunks, keywords")

For each question: retrieve, answer, then three checks by an LLM judge (Ragas), none of which needs a
"right answer" written in advance (the RAG triad):

    context relevance  are the retrieved passages about the question?        question + passages
    groundedness       is the answer supported by those passages?             answer + passages
    answer relevancy   does the answer address the question?                  question + answer

Groundedness (Ragas: ResponseGroundedness) is also called faithfulness. Ragas' own Faithfulness metric first splits
the answer into separate claims, which breaks the guidelines' compound case definitions ("can suck in the first
two days AND then stiff or convulsions") into fragments that are each "unsupported"; groundedness judges the
answer as a whole.

Everything runs through mlflow.genai.evaluate: each question becomes a trace (the search, the LLM call) with
the three scores attached, and each setup becomes one evaluation run that the MLflow
UI can compare with the others.
"""
import asyncio
import importlib.util
import logging
import os
import sys
import warnings

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "8")      # questions judged at the same time
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)   # "event loop is closed" noise from finished judge calls
if importlib.util.find_spec("litellm") is None:
    # MLflow looks for this optional package on every call; when it is missing, 8 threads searching at once can
    # deadlock the evaluation. Marking it absent makes the check fail at once.
    sys.modules["litellm"] = None

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.entities import AssessmentSource, Document, Feedback, SpanType  # noqa: E402
from mlflow.genai.scorers import scorer  # noqa: E402

import rag  # noqa: E402

QUESTIONS = rag.HERE / "questions.csv"
EXPERIMENT = "guideline-assistant"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-haiku-5.5")   # a strong model, fast and cheap
JUDGE_SOURCE = AssessmentSource(source_type="LLM_JUDGE", source_id=JUDGE_MODEL)
METRICS = ["context_relevance", "groundedness", "answer_relevancy"]


def load_questions():
    """The 24 test questions: 20 the guidelines answer (10 in the guidelines' own words, 10 in everyday words) and
    4 they don't. No reference answers: the judge needs none."""
    return pd.read_csv(QUESTIONS, dtype=str).fillna("")[["id", "type", "wording", "question"]]


# ---------------------------------------------------------------- the judge


def judge(client=None):
    """The Ragas judge: a different, stronger model than the one that answers, at temperature 0."""
    from ragas.llms import llm_factory

    return llm_factory(JUDGE_MODEL, client=client or rag.llm_client(async_=True), temperature=0.0, max_tokens=4096)


def embeddings():
    """Answer relevancy compares meanings, so it needs an embedding model: the same small one search can use."""
    from ragas.embeddings import BaseRagasEmbedding

    class FastEmbed(BaseRagasEmbedding):
        def __init__(self):
            super().__init__()
            from fastembed import TextEmbedding
            self.model = TextEmbedding(rag.EMBED_MODEL, cache_dir=str(rag.INDEX_CACHE / "models"))

        def embed_text(self, text, **kwargs):
            return [float(v) for v in next(iter(self.model.query_embed([text])))]

        async def aembed_text(self, text, **kwargs):
            return self.embed_text(text)

    return FastEmbed()


def _ragas(metric_name, **inputs):
    """Run one Ragas metric and turn its result into an MLflow Feedback (a score, or the reason there is none)."""
    from ragas.metrics.collections import AnswerRelevancy, ContextRelevance, ResponseGroundedness

    async def run():
        llm = judge()
        metric = {"context_relevance": lambda: ContextRelevance(llm=llm),
                  "groundedness": lambda: ResponseGroundedness(llm=llm),
                  "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=_EMBEDDINGS)}[metric_name]()
        args = {k: v for k, v in inputs.items() if not (metric_name == "groundedness" and k == "user_input")}
        return await metric.ascore(**args)

    try:
        result = asyncio.run(run())
    except Exception as e:  # noqa: BLE001 - a judge can fail to produce valid output: record it, don't crash
        return _record(inputs["user_input"], metric_name, None, f"{type(e).__name__}: {e}"[:300])
    value = result.value
    if value is None or value != value:                          # None or NaN: the judge gave no usable score
        return _record(inputs["user_input"], metric_name, None, "the judge returned no score")
    return _record(inputs["user_input"], metric_name, float(value), getattr(result, "reason", None))


_EMBEDDINGS = None
_ANSWERS, _SCORES, _NOTES = {}, {}, {}     # filled while an evaluation runs, keyed by question (and metric)


def _record(question, metric_name, value, note):
    _SCORES[(question, metric_name)] = value
    if value is None:
        _NOTES[(question, metric_name)] = note
        return Feedback(error=note or "no score", source=JUDGE_SOURCE)
    return Feedback(value=value, source=JUDGE_SOURCE, rationale=note)


@scorer
def context_relevance(inputs, outputs):
    return _ragas("context_relevance", user_input=inputs["question"], retrieved_contexts=outputs["passages"])


@scorer
def groundedness(inputs, outputs):
    return _ragas("groundedness", user_input=inputs["question"], response=outputs["answer"],
                  retrieved_contexts=outputs["passages"])


@scorer
def answer_relevancy(inputs, outputs):
    return _ragas("answer_relevancy", user_input=inputs["question"], response=outputs["answer"])


# ---------------------------------------------------------------- the assistant, traced


def traced_assistant(retriever, settings):
    """The assistant as two traced steps (search, then the LLM), so MLflow shows what each step did."""
    client = rag.llm_client()

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="search")
    def search(question):
        hits = retriever.search(question, settings.k)
        return [Document(page_content=t, metadata={"doc_uri": s, "score": round(float(sc), 3)})
                for t, s, sc in zip(hits.text, rag.source_ids(hits), hits.score)]

    @mlflow.trace(span_type=SpanType.CHAT_MODEL, name="llm")
    def generate(question, docs):
        passages = "\n\n".join(f"[{i}] ({d.metadata['doc_uri']}) {d.page_content}" for i, d in enumerate(docs, 1))
        response = client.chat.completions.create(
            model=settings.model, temperature=settings.temperature, **rag.thinking_switch(settings.thinking),
            messages=[{"role": "system", "content": rag.PROMPTS[settings.prompt_version]},
                      {"role": "user", "content": f"Passages:\n\n{passages}\n\nQuestion: {question}"}])
        return (response.choices[0].message.content or "").strip()

    @mlflow.trace(name="assistant")
    def assistant(question):
        docs = search(question)
        out = {"answer": generate(question, docs), "sources": [d.metadata["doc_uri"] for d in docs],
               "passages": [d.page_content for d in docs]}
        _ANSWERS[question] = out
        return out

    return assistant


# ---------------------------------------------------------------- one setup, end to end


def summarise(per_question):
    """The three scores averaged over the 20 questions the guidelines DO answer, plus refusals counted separately.

    - context relevance: the judge's score for every answerable question, answered or not. A wrong refusal usually
      means the search brought the wrong passages; that must pull the average down, not drop out of it.
    - groundedness and answer relevancy: 0 for a wrong refusal (the health worker got no answer), the judge's
      score otherwise.
    - the 4 out-of-scope questions are not averaged in: refusing them is right, so they are counted instead
      (refused_out_of_scope should be 4).
    """
    answerable = per_question[per_question.type != "out_of_scope"]
    scores = {m: pd.to_numeric(answerable[m], errors="coerce") for m in METRICS}
    summary = {"context_relevance": scores["context_relevance"].mean()}
    for m in ("groundedness", "answer_relevancy"):
        summary[m] = scores[m].where(~answerable.refused, 0.0).mean()
    in_scope = per_question.type != "out_of_scope"
    summary["refused_in_scope"] = int((per_question.refused & in_scope).sum())          # should be 0
    summary["refused_out_of_scope"] = int((per_question.refused & ~in_scope).sum())     # should be 4
    return summary


def use_local_store():
    """Log to mlflow.db in this folder (what `mlflow ui`, started here, opens), unless a server is set."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{rag.HERE / 'mlflow.db'}")


def evaluate_setup(settings, name, questions=None, retriever=None):
    """Answer every question with this setup, judge each answer, log it all as one MLflow evaluation run.

    Returns a table with one row per question: sources and passages retrieved, answer, the three scores, and
    a note for any check the judge could not score. `retriever` replaces the one the settings would build
    (for a search of your own, e.g. a hybrid of keywords and embeddings).
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = embeddings()
    questions = load_questions() if questions is None else questions
    for store in (_ANSWERS, _SCORES, _NOTES):
        store.clear()
    retriever = retriever or rag.build(settings)
    use_local_store()
    mlflow.autolog(disable=True)      # our own spans (search, llm) are the trace; automatic OpenAI tracing would
                                      # also trace every judge call, and its lazy imports can deadlock the 8 threads
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(rag.run_params(settings) | {"judge_model": JUDGE_MODEL, "n_questions": len(questions),
                                                      "index_fingerprint": rag.fingerprint(retriever.passages)})
        mlflow.log_text(rag.PROMPTS[settings.prompt_version], "prompt.txt")
        mlflow.genai.evaluate(
            data=[{"inputs": {"question": q}} for q in questions.question],
            predict_fn=traced_assistant(retriever, settings),
            scorers=[context_relevance, groundedness, answer_relevancy])
    rows = [{"question": q, "sources": "; ".join(_ANSWERS.get(q, {}).get("sources", [])),
             "answer": _ANSWERS.get(q, {}).get("answer", ""), **{m: _SCORES.get((q, m)) for m in METRICS},
             "judge_notes": "; ".join(f"{m}: {_NOTES[(q, m)]}" for m in METRICS if (q, m) in _NOTES),
             "passages": _ANSWERS.get(q, {}).get("passages", [])}
            for q in questions.question]
    per_question = questions.merge(pd.DataFrame(rows), on="question", how="left")
    # the prompt's refusal sentence, also when the model words it slightly differently ("The provided guidelines...")
    per_question["refused"] = (per_question.answer.str.lower().str.contains("do not cover this", regex=False)
                               & (per_question.answer.str.len() < 100))
    summary = summarise(per_question)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics({k: float(v) for k, v in summary.items()})
        mlflow.log_table(per_question.drop(columns=["passages"]).astype(str), "per_question.json")
    print(f"{name}: " + ", ".join(f"{k} {v:.2f}" if v % 1 else f"{k} {v:.0f}" for k, v in summary.items())
          + f"   (MLflow run {run.info.run_id[:8]})")
    return per_question


def compare(results):
    """{setup name: per-question table} -> one row per setup with the summary numbers."""
    return pd.DataFrame({name: summarise(t) for name, t in results.items()}).T.round(2)


def compare_wording(results):
    """The answerable questions split by wording (the guidelines' own words vs everyday words), for every setup:
    how relevant the passages were, and how many questions the assistant wrongly refused."""
    rows = {}
    for name, t in results.items():
        answerable = t[t.type == "answerable"]
        for wording, part in answerable.groupby("wording"):
            rows[(name, wording)] = {
                "context_relevance": pd.to_numeric(part.context_relevance, errors="coerce").mean(),
                "wrong_refusals": f"{int(part.refused.sum())} of {len(part)}"}
    table = pd.DataFrame(rows).T
    table["context_relevance"] = table.context_relevance.astype(float).round(2)
    return table.unstack()
