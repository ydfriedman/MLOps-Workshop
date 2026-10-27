"""Small wrappers that run Great Expectations and Evidently and write their reports to files.

The projects decide *what* to check; these functions only run the check and save the evidence,
so that each monitoring run on the central server carries the tools' own reports.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")  # no GX usage data

import great_expectations as gx  # noqa: E402
import pandas as pd  # noqa: E402
from great_expectations.data_context.types.base import ProgressBarsConfig  # noqa: E402


def run_gx(df: pd.DataFrame, expectations, suite_name: str, out: Path) -> dict:
    """Validate a dataframe against a list of GX expectations; save the full result as JSON.

    Uses a temporary ("ephemeral") GX context: nothing is written to a gx/ folder. The full result goes
    to `out` (which the project then logs to the central server), so every validation from every project
    can be read in one place.

    Returns {"passed": bool, "critical": [...], "warnings": [...], "evaluated": n}. Like lesson 02, only
    expectations marked critical block the data; warnings are reported.
    """
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False, metric_calculations=False)  # quiet output
    batch = (context.data_sources.add_pandas("source").add_dataframe_asset("batch")
             .add_batch_definition_whole_dataframe("whole"))
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
    for expectation in expectations:
        suite.add_expectation(expectation)
    validation = context.validation_definitions.add(gx.ValidationDefinition(name=suite_name, data=batch, suite=suite))
    result = validation.run(batch_parameters={"dataframe": df}, result_format="SUMMARY")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_json_dict(), indent=2, default=str))
    failed = [(r.expectation_config.severity.value,
               f"{r.expectation_config.type}({r.expectation_config.kwargs.get('column', 'table')})")
              for r in result.results if not r.success]
    critical = [name for severity, name in failed if severity == "critical"]
    return {"passed": not critical, "critical": critical,
            "warnings": [name for severity, name in failed if severity != "critical"],
            "evaluated": len(result.results)}


def save_report(snapshot, out: Path) -> Path:
    """Save an Evidently report (snapshot) as one self-contained HTML file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))
    return out


def failed_tests(snapshot) -> list:
    """Names of the Evidently tests that failed in a report."""
    return [t["name"].split(":")[0] for t in snapshot.dict()["tests"] if str(t["status"]).endswith("FAIL")]
