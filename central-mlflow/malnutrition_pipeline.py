"""Malnutrition team: train the GAM risk model, make it the champion, then run seven months of checks.

    python malnutrition_pipeline.py

It reuses the malnutrition lessons' code: the simulator and model helpers (../malnutrition/03-mlflow),
the Great Expectations suite (../malnutrition/02-great-expectations/build_suite.py) and the Evidently
drift tests (lessons 05 and 06). The difference: everything goes to the central MLflow server.

The seven months follow the Prefect lesson's story:
    2025-01  normal
    2025-02  a messy export        -> the data gate blocks it
    2025-03  rollout to new regions -> drift alert
    2025-04+ food distribution stops: the inputs look normal, outcomes get worse
    2025-07  outcomes for April arrive -> performance alert, three months after the change
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import checks
import hub

HERE = Path(__file__).resolve().parent
LESSONS = HERE.parent / "malnutrition"
sys.path.insert(0, str(LESSONS / "03-mlflow"))
sys.path.insert(0, str(LESSONS / "02-great-expectations"))
import build_suite  # noqa: E402  (lesson 02: the EMR suite, one source of truth)
import emr_model as em  # noqa: E402

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.metrics import MissingValueCount, ValueDrift  # noqa: E402
from evidently.tests import lte  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

DATA = HERE / "data" / "malnutrition"
REPORTS = HERE / "data" / "reports" / "malnutrition"
MONTHS = {"2025-01": "baseline", "2025-02": "messy_export", "2025-03": "new_regions",
          "2025-04": "concept_shift", "2025-05": "concept_shift", "2025-06": "concept_shift",
          "2025-07": "concept_shift"}
OUTCOME_DELAY = 3                # months until we know who developed GAM
CALIBRATION_OK = (0.67, 1.5)     # observed / predicted GAM rate outside this -> performance alert


def receive_month(month: str, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The month's EMR export: as received (what the data gate checks) and prepared for the model."""
    folder = DATA / scenario
    em.simulate(month, 1, scenario=scenario, out=folder)       # stand-in for the month's EMR export
    return pd.read_csv(folder / f"emr_{month}.csv"), em.load_months(month, month, raw=folder)


def drift_report(model, reference: pd.DataFrame, current: pd.DataFrame, month: str):
    """Lesson 06's Evidently gate: the new month vs the same season of 2024."""
    lean = int(month[5:7]) in {6, 7, 8, 9}
    ref = reference[reference["lean_season"] == int(lean)].copy()
    cur = current.copy()
    for d in (ref, cur):
        d["risk"] = model.predict_proba(d[em.FEATURES])[:, 1]
    numeric = [c for c in em.NUMERIC if c != "lean_season"] + ["risk"]
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=em.CATEGORICAL)
    columns = numeric + em.CATEGORICAL
    return Report([
        ValueDrift(column="risk", tests=[lte(0.1)]),
        ValueDrift(column="region", tests=[lte(0.1)]),
        ValueDrift(column="muac_cm", tests=[lte(0.1)]),
        MissingValueCount(column="weight_kg", share_tests=[lte(0.05)]),
    ]).run(Dataset.from_pandas(cur[columns], data_definition=definition),
           Dataset.from_pandas(ref[columns], data_definition=definition))


def main():
    hub.connect()
    tags = hub.use_project("malnutrition", "gam-risk", owner="Nutrition programme data team", risk_tier="C")

    # ---- 1. train and promote the champion (as in the MLflow lesson)
    em.simulate("2024-01", 12, out=DATA / "history")
    history = em.load_months("2024-01", "2024-12", raw=DATA / "history")
    train, test = history[history["month"] <= "2024-10"], history[history["month"] >= "2024-11"]
    model = em.build_model("logistic").fit(train[em.FEATURES], train[em.LABEL])
    with mlflow.start_run(run_name="train logistic, Jan-Oct 2024"):
        mlflow.set_tags(tags | {"kind": "training", "data_md5": em.fingerprint(history), "git_commit": em.git_commit()})
        mlflow.log_params({"model": "logistic", "C": 1.0, "train": "2024-01..2024-10", "test": "2024-11..2024-12",
                           "alert_rate": 0.2})
        overall, by_group = em.evaluate(model, test)
        mlflow.log_metrics({k: v for k, v in overall.items() if k != "n"} | em.flat_group_metrics(by_group))
        info = mlflow.sklearn.log_model(
            model, name="gam-risk",
            signature=infer_signature(test[em.FEATURES], model.predict_proba(test[em.FEATURES])),
            pyfunc_predict_fn="predict_proba", skops_trusted_types=["numpy.dtype"])
    version = hub.promote(info.model_uri, "malnutrition.gam-risk", tags, headline_metric="roc_auc",
                          description="Risk of acute malnutrition (GAM) within 90 days, from routine EMR visits. "
                                      "The top 20% of visits by risk are followed up.")
    print(f"champion malnutrition.gam-risk v{version}: ROC-AUC {overall['roc_auc']:.3f}")

    # ---- 2. one monitoring run per month
    scored = {}                                                 # month -> (outcomes, predicted risk)
    for month, scenario in MONTHS.items():
        received, df = receive_month(month, scenario)
        gx_file = REPORTS / f"gx_{month}.json"
        gate = checks.run_gx(received, build_suite.expectations(), "emr_monthly_extract", gx_file)
        metrics = {"visits": len(df), "gx_expectations": gate["evaluated"],
                   "gx_critical_failures": len(gate["critical"]), "gx_warnings": len(gate["warnings"])}
        reports, drift, notes = [gx_file], False, ""
        if gate["passed"]:                                     # only data that passed the gate is scored
            risk = model.predict_proba(df[em.FEATURES])[:, 1]
            scored[month] = (df[em.LABEL].to_numpy(), risk)
            snapshot = drift_report(model, history, df, month)
            reports.append(checks.save_report(snapshot, REPORTS / f"evidently_{month}.html"))
            failed = checks.failed_tests(snapshot)
            drift = bool(failed)
            metrics |= {"mean_risk": risk.mean(), "drift_tests_failed": len(failed)}
            notes = "drift: " + "; ".join(failed) if failed else ""
        else:
            notes = "blocked by the data gate: " + ", ".join(gate["critical"][:3])

        # outcomes for the month scored OUTCOME_DELAY months ago are known now
        earlier = str(pd.Period(month) - OUTCOME_DELAY)
        performance_alert = None
        if earlier in MONTHS and earlier not in scored:
            notes = (notes + "; " if notes else "") + f"no outcomes to check: {earlier} was blocked, never scored"
        if earlier in scored:
            y, risk = scored[earlier]
            ratio = y.mean() / risk.mean()                      # observed GAM rate / predicted GAM rate
            metrics |= {"observed_over_predicted": ratio}
            performance_alert = not CALIBRATION_OK[0] <= ratio <= CALIBRATION_OK[1]
            if performance_alert:
                notes = (notes + "; " if notes else "") + (f"outcomes for {earlier}: observed/predicted {ratio:.2f}, "
                                                           f"outside {CALIBRATION_OK[0]}-{CALIBRATION_OK[1]}")
        hub.log_check(tags, month, gate["passed"], drift, metrics, performance_alert, notes, reports)
        print(f"  {month}: data gate {'passed' if gate['passed'] else 'BLOCKED'}"
              + (f", drift {'ALERT' if drift else 'ok'}" if gate["passed"] else "")
              + (f", outcomes for {earlier}: obs/pred {metrics['observed_over_predicted']:.2f}"
                 if performance_alert is not None else ""))


if __name__ == "__main__":
    main()
