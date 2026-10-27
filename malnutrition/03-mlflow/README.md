# 03 – MLflow: tracking experiments and managing models

**Where we are.** DVC versions the data and pipeline; Great Expectations stops bad
data. But comparing models was clumsy (`dvc metrics diff`, one `metrics.json` per
commit), and nothing said *which model is approved for use*. MLflow fills both gaps:

| Question | MLflow component | Notebook |
|---|---|---|
| "What did we try, and how did each do?" | **Tracking**: runs with params, metrics, tags, artifacts, models | `1_tracking.ipynb` |
| "Which model is in use, who approved it, what was it trained on?" | **Model Registry**: versions + aliases (`@champion`) | `2_registry.ipynb` |
| "How does the EMR's software get a prediction?" | **Serving**: the registered model as an HTTP API | `2_registry.ipynb`, section 5 |

**DVC vs MLflow:** they overlap a little and complement a lot.

| | DVC | MLflow |
|---|---|---|
| Versions data | **yes** (core job) | records *which* data (a hash or dataset summary) |
| Pipeline definition / rerun what changed | **yes** | no |
| Experiment comparison | basic (`dvc exp show`, no UI) | **yes**, with UI and search |
| "Which model is in production" | no | **yes** (registry aliases) |
| Serving | no | **yes** (`mlflow models serve`) |

A common split, and the one the capstone uses: DVC owns data + pipeline, MLflow owns
runs + models, and each MLflow run records the DVC data hash.

---

Every MLflow call and command used here, explained: [`MLFLOW_CHEATSHEET.md`](MLFLOW_CHEATSHEET.md).

## Setup

From the repo root, activate the environment and move into this folder (MLflow
creates `mlflow.db` and `mlruns/` in whatever folder you start it from):

```bash
source .venv/bin/activate           # Windows: .venv\Scripts\activate
cd malnutrition/03-mlflow
export MLFLOW_DISABLE_TELEMETRY=true   # optional: MLflow 3 sends anonymous usage data by
                                       #   default (like DVC's analytics); this turns it off
```
To make the opt-out permanent, add the `export` line to your shell profile (`~/.zshrc`
on macOS). The notebooks turn it off themselves in their first cell.

Then open the notebooks in VSCode (kernel: `.venv`) and run them in order (the first
notebook starts the MLflow UI and tells you what to click):
1. **`1_tracking.ipynb`**: log one run by hand, then a grid of runs; compare them in
   the UI and with `mlflow.search_runs`; see what autolog does and doesn't capture.
2. **`2_registry.ipynb`**: register the best run as `gam-risk` v1 (`@champion`);
   score new visits by alias; the world changes in 2025, so train a challenger, compare
   both on the **same** benchmark months, apply a written promotion rule, promote,
   roll back; follow the audit trail; serve the model over HTTP.
3. **`3_ensemble.ipynb`** *(optional)*: track an ensemble (logistic + boosted trees + random forest, averaged)
   as one **parent run** with a **child run per member**, all on the same test months; log what each member adds
   (`roc_auc_without_<member>`); register the ensemble. Here it turns out *worse* than logistic alone (0.808 vs 0.812).

Each notebook takes about a minute to run. The data (generated with the shared
simulator) goes in `03-mlflow/data/`.

## Helper functions in this lesson

Not everything a run records comes from MLflow. These are our own helpers, so nothing is a surprise:

| Helper | Where | What it does |
|---|---|---|
| `em.fingerprint(df)` | `emr_model.py` | MD5 of the training data; logged as the `data_md5` tag. **Not automatic**: MLflow does not know which data you used |
| `em.git_commit()` | `emr_model.py` | current git commit; logged as the `git_commit` tag. MLflow records git itself only for runs started from a *script* (`mlflow run` / `python train.py`), not from a notebook, so we set it explicitly |
| `em.evaluate(model, df)` | `emr_model.py` | **where the metrics are defined**: ROC-AUC, PR-AUC, calibration (observed/predicted), alert-rate sensitivity; `flat_group_metrics` adds per-sex numbers. `mlflow.log_metrics` just stores what these return |
| `em.build_model(kind, ...)` | `emr_model.py` | the sklearn pipeline (logistic or gradient boosting) |
| `em.load_months`, `em.simulate` | `emr_model.py` | read or generate the monthly extracts |
| `em.roc_figure` | `emr_model.py` | the ROC plot logged as an artifact |
| `train_and_log(...)` | `train.py` | one function that builds, fits, evaluates and logs a run: the "log one run" cell, reusable in a grid |

## Verified results (what you should see)

**Notebook 1** (train Jan–Oct 2024, test Nov–Dec 2024: same as the DVC lesson's v2):

| Run | ROC-AUC | Sensitivity | Girls | Boys |
|---|---|---|---|---|
| baseline-logistic (C=1) | **0.8122** | 0.5702 | 0.534 | 0.600 |
| logistic C=0.1 | 0.8119 | 0.5730 | 0.528 | 0.610 |
| gbm depth 2, lr 0.05 | 0.8071 | 0.5758 | **0.558** | 0.590 |
| gbm depth 3, lr 0.1 | 0.8038 | 0.5675 | 0.546 | 0.585 |
| gbm depth 4, lr 0.1 | 0.8028 | 0.5758 | 0.540 | 0.605 |

Best ROC-AUC isn't the smallest girls/boys gap: a decision for people, not `ORDER BY`.

**Notebook 2** (benchmark: May–Jun 2025, after food distribution stopped; prevalence 24%):

| Model | ROC-AUC | Sensitivity | PPV | Girls | Boys |
|---|---|---|---|---|---|
| champion v1 (trained 2024) | 0.8256 | 0.4897 | 0.5950 | 0.472 | 0.507 |
| challenger (retrained to Apr 2025) | **0.8363** | **0.5206** | 0.6325 | 0.497 | 0.544 |

Promotion rule (ROC-AUC +0.005, no sex group losing >0.02 sensitivity): **PROMOTE**,
so `@champion` moves to version 2.

## Serving: the model as an API *(optional)*

With the environment **activated** (the server starts a helper program, `uvicorn`,
that must come from the same environment), from this folder:
```bash
mlflow models serve -m "models:/gam-risk@champion" --port 5001 --env-manager local
```
In another terminal, score one visit. A 14-month-old girl in Afar, recent diarrhoea,
90 minutes to water, small at birth:
```bash
curl -s -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d '{
  "dataframe_records": [{"age_months": 14, "weight_kg": 8.1, "height_cm": 72.5, "whz": -1.6,
    "muac_cm": 12.7, "recent_diarrhea": 1, "recent_fever": 0, "vaccinated_for_age": 0,
    "currently_breastfed": 1, "prior_sam_admission": 0, "mother_age": 22,
    "water_source_minutes": 90, "household_size": 7, "lean_season": 1, "region": "Afar",
    "residence": "rural", "sex": "F", "birth_size": "small", "mother_education": "none"}]}'
#   {"predictions": [[0.107, 0.893]]}     <- [P(no GAM), P(GAM within 90 days)] = 89% risk
```
`--env-manager local` uses your current environment; without it MLflow tries to build
a fresh one from the model's saved requirements (slow, needs internet).

## Should we actually serve with MLflow? *(optional)*

`mlflow models serve` is a quick, honest way to show "a registered model becomes an API",
and fine for internal tools and pilots. For production, teams usually choose one of:

| Option | When it fits |
|---|---|
| `mlflow models serve` / `mlflow models build-docker` | internal service, the team already runs MLflow; the image includes MLflow (large) |
| Your own small API (e.g. FastAPI) that loads the model file | you want a small, controlled service; common in practice |
| **Batch scoring** (a scheduled job scores each new extract, writes risks back to the EMR/DHIS2) | the clinic doesn't need an instant answer; often the right fit for monthly malnutrition risk |
| Embed in the client app (on-device) | offline use; see below |

In the workshop: show serving, then ask which of these fits the ministry's setup.

## Central MLOps, light deployments *(optional)*

MLflow, DVC and Great Expectations are **training-side** tools: they belong where models
are built, compared and approved (a ministry server, a cloud project). They don't need to
be where predictions happen, and usually shouldn't be: a tablet in a health post offline
for days can't run an MLflow server.

The usual split:
- **Centre:** data versioning (DVC), validation (GX), tracking + registry (MLflow),
  monitoring (Evidently), orchestration (Prefect). Heavy tools, good connectivity.
- **Edge:** only the *approved model*, exported from the registry, plus a tiny check of its
  inputs. Options, lightest first:
  - **A points score.** A logistic regression is a weighted sum. Its coefficients can be
    turned into a simple score (e.g. +3 for recent diarrhoea, +2 for >60 min to water) that
    runs inside any EMR form, on any phone, even on paper. Many clinical risk tools work
    this way, and it's very explainable.
  - **A model file + a small script** (scikit-learn/skops, or ONNX for a tiny runtime with
    no scikit-learn). No MLflow needed to *use* a model, only to manage it.
  - **A container** with the model and a small API, where a local server exists.
- **Back to the centre:** predictions and later outcomes sync back when connectivity
  returns, so drift and performance are still monitored centrally.

So yes, caveat the training: *these tools manage the lifecycle centrally; what runs at the
point of care depends on the deployment context.* Distribute the model and a minimal
input check, not the MLOps stack. Worth asking the ministry: where would predictions run
(central batch, district server, eCHIS tablets), and how often do devices sync?

## Gotchas we hit building this (MLflow 3.17) *(optional)*

- **"Untrusted types" when logging a model.** MLflow 3 saves scikit-learn models with
  *skops* (safer than pickle: loading can't run arbitrary code) and refuses classes it
  hasn't been told to trust. `skops.io.get_untrusted_types()` lists them; we trust exactly
  two numpy/sklearn internals (`train.py`). Don't trust everything blindly.
- **Loaded model returns 0/1 instead of a risk.** By default the generic (pyfunc) model
  calls `.predict()`. We log with `pyfunc_predict_fn="predict_proba"`, so predictions are
  `[P(0), P(1)]` pairs; column 1 is the risk.
- **The signature checks shape, not meaning.** It blocks missing columns and wrong types
  (and, as a side effect of types, a missing value in an integer column). It accepts an
  unknown code (`sex="female"`, silently ignored) and **MUAC in millimetres, which drives
  every child's risk to ~0**. Notebook 2 demonstrates this. Great Expectations is the real
  data check; the signature is the model's last line of defence. Also: integer columns must
  stay integers in the signature; casting them to float made MLflow reject normal input.
- **Runs "disappear".** You started `mlflow ui` from a different folder, so it opened a
  different (empty) `mlflow.db`. Start it from `03-mlflow/`.
- **Start over:** stop the UI, delete `mlflow.db` and `mlruns/` in this folder.
  Re-running notebook 2 without that registers additional versions (harmless, but the
  version numbers won't match the text).

## For a team: a shared tracking server *(optional)*

Locally, MLflow writes to `mlflow.db` + `mlruns/`. A team runs one server that everyone logs to:
```bash
mlflow server --backend-store-uri postgresql://user:pass@dbhost/mlflow \
              --artifacts-destination s3://bucket/mlflow --host 0.0.0.0 --port 5000
```
and points clients at it with `export MLFLOW_TRACKING_URI=http://<server>:5000`. The
*backend store* (a database: runs, params, metrics, registry) and the *artifact store*
(files and models: S3, MinIO, Azure, GCS, or a shared disk) can both live on-premises,
which matters where health data must stay on government infrastructure.

**See it working:** [`central-mlflow/`](../../central-mlflow/) runs one server for three projects
(malnutrition, TB, LLM), each reporting its models and monthly checks to it, with a portfolio notebook for the
organisation's view and the full steps for a real server (PostgreSQL, object storage, logins).

## Exercises

1. In `1_tracking.ipynb`, add a run with `model="logistic", C=0.01`. Where does it rank?
   Find it in the UI with the search box: `params.C = "0.01"`.
2. Log one more metric per run: the share of flagged children who are under 2 years old.
   Where would you add it, and what would a clinic learn from it?
3. Change the promotion rule to also require that sensitivity in **Addis Ababa** doesn't
   drop. Does the challenger still pass? (`metrics_by_group.json` has the numbers.)
4. Roll back: point `@champion` at version 1, reload `models:/gam-risk@champion`, and
   confirm the January 2025 risk scores change back.
5. Use `train.py` from the terminal to log a run, then find it in the UI. What does the
   UI show under *Source* compared with the notebook runs?

Try the exercises first (they are take-home). Stuck, or short on time? [`exercise_solutions.ipynb`](exercise_solutions.ipynb) holds a worked answer to each; it ships without outputs, so run it to see them.
