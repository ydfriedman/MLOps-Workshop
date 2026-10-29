"""LLM team: log every evaluated configuration of the guideline assistant, register the best as champion,
then run two weekly checks.

    python llm_pipeline.py

No API key or internet needed: it reads the scores of the four setups in the LLM lesson's notebook 3
(llm_results/setups.csv): the reference-free Ragas checks (context relevance, groundedness, answer relevancy,
judged by anthropic/claude-haiku-5.5) on the 24 questions in ../llm-cds/questions.csv, and the refusals.
What gets versioned here is a **configuration** (documents, chunking, search, language model), because that
is what defines a version of an LLM application.

The weekly checks:
    week 1  the guideline documents are the ones the champion was evaluated on -> gate passes
    week 2  EPHI publishes a revised cholera guideline (simulated)             -> gate blocks: re-evaluate first
"""
import hashlib
from pathlib import Path

import pandas as pd

import hub

import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402

HERE = Path(__file__).resolve().parent
LLM = HERE.parent / "llm-cds"
REPORTS = HERE / "data" / "reports" / "llm"
CONFIG = ["docs", "chunking", "retriever", "model"]
FIXED = {"k": 5, "prompt_version": "v1", "temperature": 0.0, "judge_model": "anthropic/claude-haiku-5.5"}
RULE = ("highest answer_rate (share of the 20 answerable questions not wrongly refused) among configurations with "
        "groundedness >= 0.9 that refuse all 4 out-of-scope questions")
EVERYDAY_ALERT = 0.6     # drift alert if more than 60% of questions use everyday words (the champion was chosen on 50%)


def guidelines_md5(revised: dict | None = None) -> str:
    """Fingerprint of the guideline documents; `revised` = {file name: new text} simulates an update."""
    h = hashlib.md5()
    for f in sorted((LLM / "guidelines_md").glob("*.md")):
        text = (revised or {}).get(f.name, f.read_text(encoding="utf-8"))
        h.update(f.name.encode() + text.encode("utf-8"))
    return h.hexdigest()[:12]


def main():
    hub.connect()
    tags = hub.use_project("llm-cds", "guideline-assistant", owner="Clinical decision support team", risk_tier="C")
    documents = guidelines_md5()

    # ---- 1. every configuration evaluated in the LLM lesson (notebook 3), as one evaluation run each
    table = pd.read_csv(HERE / "llm_results" / "setups.csv")
    metrics = [c for c in table.columns if c not in CONFIG]
    runs = []
    for _, row in table.iterrows():
        config = {k: row[k] for k in CONFIG}
        name = f"{row['model'].split('/')[-1]}, {row['docs']}, {row['chunking']}, {row['retriever']}"
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(tags | {"kind": "evaluation", "guidelines_md5": documents})
            mlflow.log_params(config | FIXED)
            mlflow.log_metrics({m: float(row[m]) for m in metrics})
        runs.append({"run_id": run.info.run_id, "config": config, **{m: row[m] for m in metrics}})

    # ---- 2. the written promotion rule picks the champion configuration
    eligible = [r for r in runs if r["groundedness"] >= 0.9 and r["refused_out_of_scope"] == 4]
    best = max(eligible, key=lambda r: (r["answer_rate"], r["groundedness"]))
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("promotion_rule", RULE)
        mlflow.log_artifact(str(HERE / "llm_results" / "setups.csv"), artifact_path="reports")
        info = mlflow.pyfunc.log_model(name="assistant-config", python_model=str(HERE / "assistant_config_model.py"),
                                       model_config=best["config"] | FIXED | {"guidelines_md5": documents})
    version = hub.promote(info.model_uri, "llm-cds.guideline-assistant", tags, headline_metric="answer_rate",
                          description="Answers health workers' questions from six national guidelines (RAG). "
                                      "Promotion rule: " + RULE + ".")
    print(f"{len(runs)} configurations logged; champion llm-cds.guideline-assistant v{version}: "
          f"{best['config']['model']}, {best['config']['docs']}/{best['config']['chunking']}/"
          f"{best['config']['retriever']}, answer_rate {best['answer_rate']:.2f}")

    # ---- 3. weekly checks. Gate: are the documents the ones the champion was evaluated on?
    #      Drift: are people asking in everyday words more than in the questions the champion was chosen on?
    #      (Notebook 3: keyword search fails on everyday wording.) The lesson's test questions stand in for a week
    #      of real questions.
    champion = mlflow.pyfunc.load_model("models:/llm-cds.guideline-assistant@champion")   # read from the registry
    evaluated_on = champion.model_config["guidelines_md5"]
    questions = pd.read_csv(LLM / "questions.csv")
    questions = questions[questions["type"] == "answerable"]
    everyday = (questions["wording"] == "everyday").mean()
    cholera = "ephi_cholera_2022.md"
    weeks = {"week 1": guidelines_md5(),
             "week 2": guidelines_md5({cholera: (LLM / "guidelines_md" / cholera).read_text(encoding="utf-8")
                                       + "\n\nRevised edition, 2026.\n"})}       # simulated update
    for week, current in weeks.items():
        same = current == evaluated_on
        hub.log_check(tags, week, same, everyday > EVERYDAY_ALERT,
                      {"questions": len(questions), "everyday_wording_share": everyday},
                      notes="" if same else "guideline documents changed since the champion was evaluated "
                                            f"({cholera}): re-run the evaluation before using the new text")
        print(f"  {week}: documents {'unchanged' if same else 'CHANGED'}, {everyday:.0%} of questions in everyday words")


if __name__ == "__main__":
    main()
