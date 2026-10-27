"""LLM team: log every evaluated configuration of the guideline assistant, register the best as champion,
then run two weekly checks.

    python llm_pipeline.py

No API key or internet needed: it reads the evaluation results the LLM lesson saved (../llm-cds/results/).
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
FIXED = {"k": 5, "prompt_version": "v1", "temperature": 0.0}    # the same for every configuration evaluated
RULE = ("highest factual_correctness among configurations with retrieval_hit_rate >= 0.9 "
        "and faithfulness >= 0.8")
AMHARIC_ALERT = 0.2      # drift alert if more than 20% of questions are in Amharic (the champion was tested in English)


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

    # ---- 1. every configuration evaluated in the LLM lesson, as one evaluation run each
    table = pd.read_csv(LLM / "results" / "search_and_ragas.csv")
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
    eligible = [r for r in runs if r["retrieval_hit_rate"] >= 0.9 and r["faithfulness"] >= 0.8]
    best = max(eligible, key=lambda r: r["factual_correctness"])
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("promotion_rule", RULE)
        mlflow.log_artifact(str(LLM / "results" / "search_and_ragas.csv"), artifact_path="reports")
        info = mlflow.pyfunc.log_model(name="assistant-config", python_model=str(HERE / "assistant_config_model.py"),
                                       model_config=best["config"] | FIXED | {"guidelines_md5": documents})
    version = hub.promote(info.model_uri, "llm-cds.guideline-assistant", tags, headline_metric="factual_correctness",
                          description="Answers health workers' questions from six national guidelines (RAG). "
                                      "Promotion rule: " + RULE + ".")
    print(f"{len(runs)} configurations logged; champion llm-cds.guideline-assistant v{version}: "
          f"{best['config']['model']}, {best['config']['docs']}/{best['config']['chunking']}/"
          f"{best['config']['retriever']}, factual_correctness {best['factual_correctness']:.3f}")

    # ---- 3. weekly checks. Gate: are the documents the ones the champion was evaluated on?
    #      Drift: are people asking in a language the champion was not evaluated in?
    #      (The lesson's test questions stand in for a week of real questions.)
    champion = mlflow.pyfunc.load_model("models:/llm-cds.guideline-assistant@champion")   # read from the registry
    evaluated_on = champion.model_config["guidelines_md5"]
    questions = pd.read_csv(LLM / "results" / "per_question_baseline.csv")
    amharic = (questions["type"] == "amharic").mean()
    cholera = "ephi_cholera_2022.md"
    weeks = {"week 1": guidelines_md5(),
             "week 2": guidelines_md5({cholera: (LLM / "guidelines_md" / cholera).read_text(encoding="utf-8")
                                       + "\n\nRevised edition, 2026.\n"})}       # simulated update
    for week, current in weeks.items():
        same = current == evaluated_on
        hub.log_check(tags, week, same, amharic > AMHARIC_ALERT,
                      {"questions": len(questions), "amharic_share": amharic},
                      notes="" if same else "guideline documents changed since the champion was evaluated "
                                            f"({cholera}): re-run the evaluation before using the new text")
        print(f"  {week}: documents {'unchanged' if same else 'CHANGED'}, {amharic:.0%} of questions in Amharic")


if __name__ == "__main__":
    main()
