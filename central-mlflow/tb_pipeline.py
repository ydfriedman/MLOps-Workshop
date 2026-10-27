"""TB team: train the chest X-ray screening model at Kanyama, make it the champion, then check each new site.

    python tb_pipeline.py

It reuses the TB lesson's helpers and data (../tb-chest-xray). It needs the study table: run
`python scripts/download.py zambia` from the repo root once (see the setup guide).

The checks follow the TB lesson's rollout:
    Chawama               same X-ray machine, different patients  -> gate passes, performance lower
    Chainda South         a different X-ray machine                -> the image headers fail the data gate
    Kanyama, HIV-positive same machine, different patients         -> gate passes, performance drops
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import checks
import hub

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tb-chest-xray"))
import tb_helpers as tb  # noqa: E402

import great_expectations.expectations as gxe  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPORTS = HERE / "data" / "reports" / "tb"
HEADER_FIELDS = ["Manufacturer", "PhotometricInterpretation", "PixelSpacing", "Columns"]
TARGET_SENSITIVITY = 0.90
SENSITIVITY_FLOOR = 0.80      # performance alert if a site's sensitivity falls below this ...
AUC_FLOOR = 0.75              # ... or its ROC-AUC (the ranking) falls below this (0.80 at Kanyama)
DRIFT_AUC = 0.8               # drift alert if a classifier can tell the site's images from Kanyama's this well
                              # (the TB lesson: set alerts on the SIZE of a difference, not on a drift-test count)


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000))


def header_expectations(reference: pd.DataFrame):
    """The data gate: each DICOM header field must take a value that was common (>= 5%) at Kanyama."""
    for field in HEADER_FIELDS:
        shares = reference[field].value_counts(normalize=True)
        yield gxe.ExpectColumnValuesToBeInSet(column=field, value_set=list(shares[shares >= 0.05].index), mostly=0.95)


def main():
    hub.connect()
    tags = hub.use_project("tb", "cxr-screening", owner="TB programme imaging team", risk_tier="C")

    people, X = tb.load_data()
    y = people.tb.to_numpy()

    def site(clinic, hiv):
        return np.flatnonzero((people.clinic == clinic).to_numpy() & (people.hiv == hiv).to_numpy())

    kanyama = site("Kanyama", "HIV-negative")
    train, test = train_test_split(kanyama, test_size=0.3, stratify=y[kanyama], random_state=0)

    # ---- 1. train and promote the champion (as in the TB lesson, step 1)
    p_cv = cross_val_predict(new_model(), X[train], y[train], cv=5, method="predict_proba")[:, 1]
    threshold = tb.threshold_for_sensitivity(y[train], p_cv, TARGET_SENSITIVITY)
    model = new_model().fit(X[train], y[train])
    with mlflow.start_run(run_name="train Kanyama, HIV-negative"):
        mlflow.set_tags(tags | {"kind": "training"})
        mlflow.log_params({"model": "logistic", "C": 0.01, "features": "DenseNet-121 image features (1,024)",
                           "train": "Kanyama, HIV-negative (70%)", "threshold": round(float(threshold), 4),
                           "target_sensitivity": TARGET_SENSITIVITY})
        held_out = tb.screening_metrics(y[test], model.predict_proba(X[test])[:, 1], threshold)
        mlflow.log_metrics({k.replace(" ", "_"): float(v) for k, v in held_out.items()})
        info = mlflow.sklearn.log_model(model, name="cxr-screening",
                                        signature=infer_signature(X[test], model.predict_proba(X[test])),
                                        pyfunc_predict_fn="predict_proba", skops_trusted_types=["numpy.dtype"])
    version = hub.promote(info.model_uri, "tb.cxr-screening", tags, headline_metric="roc_auc",
                          description="TB screening from chest X-ray image features; threshold set for 90% "
                                      "sensitivity at Kanyama. People flagged get a lab test.")
    print(f"champion tb.cxr-screening v{version}: ROC-AUC {held_out['roc_auc']:.3f}")

    # ---- 2. one monitoring run per new site
    reference = people.iloc[train]
    pca = PCA(n_components=10, random_state=0).fit(X[train])
    columns = [f"pc{i + 1}" for i in range(10)]

    def summaries(idx):
        return Dataset.from_pandas(pd.DataFrame(pca.transform(X[idx]), columns=columns),
                                   data_definition=DataDefinition(numerical_columns=columns))

    rollout = {"site: Chawama": site("Chawama", "HIV-negative"),
               "site: Chainda South": site("Chainda South", "HIV-negative"),
               "site: Kanyama, HIV-positive": site("Kanyama", "HIV-positive")}
    for period, idx in rollout.items():
        name = period.split(": ")[1].replace(", ", "_").replace(" ", "_")
        gx_file = REPORTS / f"gx_{name}.json"
        gate = checks.run_gx(people.iloc[idx][HEADER_FIELDS], header_expectations(reference), "dicom_headers", gx_file)

        snapshot = Report([DataDriftPreset()]).run(summaries(idx), summaries(train))
        evidently_file = checks.save_report(snapshot, REPORTS / f"evidently_{name}.html")
        drifted = next(m for m in snapshot.dict()["metrics"] if "DriftedColumnsCount" in m["metric_name"])["value"]

        both, which = np.r_[train, idx], np.r_[np.zeros(len(train)), np.ones(len(idx))]
        drift_auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
                                    X[both], which, cv=5, scoring="roc_auc").mean()

        # Performance from lab results. In the study everyone was lab-tested, so it is known at once; in routine
        # use it arrives later, and only for the people who were tested.
        perf = tb.screening_metrics(y[idx], model.predict_proba(X[idx])[:, 1], threshold)
        metrics = {"people": len(idx), "gx_critical_failures": len(gate["critical"]),
                   "feature_drift_auc": drift_auc, "evidently_drifted_share": drifted["share"]}
        metrics |= {k.replace(" ", "_"): float(v) for k, v in perf.items()}
        drift = drift_auc > DRIFT_AUC
        notes = ("image headers differ from Kanyama: " + ", ".join(gate["critical"])) if not gate["passed"] else ""
        if perf["sensitivity"] < SENSITIVITY_FLOOR or perf["roc_auc"] < AUC_FLOOR:
            notes = (notes + "; " if notes else "") + (f"lab results: sensitivity {perf['sensitivity']:.2f}, "
                                                       f"ROC-AUC {perf['roc_auc']:.2f}")
        hub.log_check(tags, period, gate["passed"], drift, metrics,
                      performance_alert=perf["sensitivity"] < SENSITIVITY_FLOOR or perf["roc_auc"] < AUC_FLOOR,
                      notes=notes,
                      artifacts=(gx_file, evidently_file))
        print(f"  {period}: data gate {'passed' if gate['passed'] else 'BLOCKED'}, "
              f"drift {'ALERT' if drift else 'ok'} (classifier AUC {drift_auc:.2f}), "
              f"sensitivity {perf['sensitivity']:.2f}, ROC-AUC {perf['roc_auc']:.2f}")


if __name__ == "__main__":
    main()
