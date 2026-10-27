"""The shared conventions every project follows when it reports to the central MLflow server.

Every project's pipeline imports this one file, so that runs from different teams can be found,
compared and summarised in one place. In an organisation this would be a small internal package
that every team installs.

The conventions:
- one **experiment per project**, named "<project>/<use case>";
- every experiment, run and registered model carries the same **tags**: project, use_case, owner, risk_tier;
- registered models are named "<project>.<model>", and the version in use has the alias **@champion**;
- each registered model names the metric that matters most for it (tag `headline_metric`), so a
  portfolio view can show one comparable number per project;
- every run has a `kind`: training, evaluation or monitoring. A monitoring run always carries
  `period`, `data_gate` (passed / blocked) and `drift` (ok / alert), whatever tools the project used
  to decide them, plus the tools' own reports as artifacts.
"""
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")   # no anonymous usage data
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")     # no MLflow tip on import
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")   # no "View run at ..." per run
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")    # no progress bars for uploads
warnings.filterwarnings("ignore", category=UserWarning)

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

# The "remote". On your laptop: the server started in the README (Terminal 1).
# In an organisation: its real address, e.g. https://mlflow.eaii.example
# Set MLFLOW_TRACKING_URI to change it; nothing else in any project's code changes.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
TAGS = ("project", "use_case", "owner", "risk_tier")


def connect() -> MlflowClient:
    """Point MLflow at the central server, and stop with a clear message if it isn't running."""
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        MlflowClient().search_experiments(max_results=1)
    except Exception as e:  # noqa: BLE001 - every connection problem gets the same advice
        raise SystemExit(f"Cannot reach the MLflow server at {TRACKING_URI} ({type(e).__name__}).\n"
                         "Start it first: see central-mlflow/README.md, Terminal 1.") from None
    return MlflowClient()


def use_project(project: str, use_case: str, owner: str, risk_tier: str) -> dict:
    """Select (or create) the project's experiment and tag it. Returns the tags every run should carry."""
    tags = {"project": project, "use_case": use_case, "owner": owner, "risk_tier": risk_tier}
    experiment = mlflow.set_experiment(f"{project}/{use_case}")
    for key, value in tags.items():
        MlflowClient().set_experiment_tag(experiment.experiment_id, key, value)
    return tags


def promote(model_uri: str, name: str, tags: dict, headline_metric: str, description: str) -> str:
    """Register a model version and make it the @champion. Returns the new version number.

    In a real organisation this step is a decision: a named person signs it off (the Gate 2 record),
    and the pipeline only carries it out. Here the pipelines promote straight away, to fill the server.
    """
    client = MlflowClient()
    version = mlflow.register_model(model_uri, name).version
    client.update_registered_model(name, description=description)
    for key, value in tags.items():
        client.set_registered_model_tag(name, key, value)
    client.set_registered_model_tag(name, "headline_metric", headline_metric)
    client.set_registered_model_alias(name, "champion", version)
    return version


def log_check(tags: dict, period: str, data_gate: bool, drift_alert: bool, metrics: dict,
              performance_alert: bool | None = None, notes: str = "", artifacts: tuple = ()) -> str:
    """Log one monitoring run in the standard shape. Returns its run id.

    data_gate         - did the new data pass the project's data checks (e.g. a Great Expectations suite)?
    drift_alert       - did the project's drift checks (e.g. an Evidently report) raise an alert?
    performance_alert - did performance on cases whose outcome is now known fall outside the agreed
                        range? None = no outcomes known yet ("pending")
    metrics           - anything else the project measures (numbers only)
    artifacts         - files to keep with the run: the tools' own reports (GX result, Evidently HTML, ...)
    """
    performance = "pending" if performance_alert is None else ("alert" if performance_alert else "ok")
    with mlflow.start_run(run_name=f"monitoring {period}") as run:
        mlflow.set_tags(tags | {"kind": "monitoring", "period": period,
                                "data_gate": "passed" if data_gate else "blocked",
                                "drift": "alert" if drift_alert else "ok",
                                "performance": performance})
        if notes:
            mlflow.set_tag("notes", notes)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path="reports")
    return run.info.run_id
