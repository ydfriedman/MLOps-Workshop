# One MLflow server for every project: the organisation's view

**The question.** In the lessons, every tool ran inside one project folder: its own `mlflow.db`, its own
Great Expectations results, its own Evidently reports. How can an organisation see **all** its projects in one
place? For example: which model is in use for each project, how good it is, and whether last month's data
checks passed?

**The answer shown here.** Run **one MLflow server** that every project reports to, and agree a few
**conventions** so the reports can be compared. Here three projects from this workshop (malnutrition, TB
chest X-ray, LLM guideline assistant) each run their own pipeline and log to the same server. Then
[`portfolio.ipynb`](portfolio.ipynb) shows what a senior person sees: every project's champion model and latest
checks, with the Great Expectations and Evidently evidence one click away.

It is the same idea as DVC's remote: in the DVC lesson the "remote" was a folder next to the repo
(`../dvc-remote`) and the README explained how to swap it for real storage. Here the "remote" is an MLflow
server on your laptop, storing its data in `../mlflow-remote`, and the last section explains how to swap it for
a real server.

```
  malnutrition_pipeline.py ─┐                      ┌──────────────────────────────────────┐
  (GX + Evidently)          │                      │  MLflow server  http://127.0.0.1:5050 │
                            │   MLFLOW_TRACKING_URI │                                      │
  tb_pipeline.py ───────────┼─────────────────────►│  database: runs, metrics, tags,      │──► portfolio.ipynb
  (GX on DICOM headers)     │   (the only setting  │            model registry           │──► the MLflow web UI
                            │    that changes for  │  artifact store: models, GX results, │
  llm_pipeline.py ──────────┘    a real server)    │            Evidently reports        │
  (document fingerprint)                           └──────────────────────────────────────┘
                                                           stored in ../mlflow-remote/
```

## Setup

Everything is already installed with the workshop environment. The TB project needs the study table: if you did
the TB lesson you have it; otherwise run `python scripts/download.py zambia` from the repo root once.

**Terminal 1: the central server.** From the **repo root** (`MLOps-Workshop/`), environment active:
```bash
mkdir -p ../mlflow-remote                       # Windows: mkdir ..\mlflow-remote
mlflow server --backend-store-uri sqlite:///../mlflow-remote/mlflow.db \
              --artifacts-destination ../mlflow-remote/artifacts --port 5050
```
(On Windows, put the command on one line without the `\`.) Leave it running. The web UI is at
http://127.0.0.1:5050; switch the toggle at the top left to **Model training**. Port 5050 keeps it apart from the
lesson's `mlflow ui` on port 5000.

**Terminal 2: three teams report to it.** From the repo root, environment active:
```bash
cd central-mlflow
python malnutrition_pipeline.py     # about 1 minute
python tb_pipeline.py               # about 40 seconds
python llm_pipeline.py              # a few seconds; no API key needed (it reads the lesson's saved results)
```
Each script prints what it logged. Then open **`portfolio.ipynb`** in VSCode (kernel `.venv`) and run it.

The scripts don't need `MLFLOW_TRACKING_URI` to be set: `hub.py` uses `http://127.0.0.1:5050` unless the variable
says otherwise.

## What each project reports

| | Malnutrition (`malnutrition_pipeline.py`) | TB (`tb_pipeline.py`) | LLM assistant (`llm_pipeline.py`) |
|---|---|---|---|
| **Experiment** | `malnutrition/gam-risk` | `tb/cxr-screening` | `llm-cds/guideline-assistant` |
| **Training / evaluation** | logistic model, Jan–Oct 2024 | logistic on image features, Kanyama | the 4 setups of the LLM lesson's notebook 3 (reference-free Ragas checks) |
| **Champion** | `malnutrition.gam-risk@champion` | `tb.cxr-screening@champion` | `llm-cds.guideline-assistant@champion` (a configuration) |
| **Headline metric** | ROC-AUC 0.812 | ROC-AUC 0.804 | answer rate 0.90 (answerable questions not wrongly refused) |
| **Monitoring periods** | 7 months (Jan–Jul 2025) | 3 new sites | 2 weeks |
| **Data gate** | Great Expectations: lesson 02's suite, unchanged | Great Expectations on the DICOM headers | are the guideline documents the ones it was evaluated on? |
| **Drift** | Evidently: lessons 05–06's tests | can a classifier tell the site's images from Kanyama's? | share of questions in everyday words (keyword search fails on them) |
| **Performance** | observed / predicted, 3 months late | sensitivity and ROC-AUC from lab results | pending: needs clinicians' review (LLM lesson, notebook 5) |
| **What happens** | Feb blocked (messy export), Mar drift (new regions), **Jul performance alert** (food distribution stopped in April) | **Chainda South blocked** (different X-ray machine); Chawama and HIV-positive patients: performance alerts | **week 2 blocked**: a revised cholera guideline (simulated) must be re-evaluated first |

Each monitoring run also keeps the tools' own reports as files (`reports/` in the run's artifacts): the full
Great Expectations result (JSON, shown in the MLflow UI) and the Evidently report (HTML, a few MB: download it from
the run's Artifacts tab, or let the portfolio notebook fetch it). They are stored on the server, so anyone with
access can read them without access to the project's machine.

## The conventions are what make it work

A central server alone is just a big pile of runs. What makes the portfolio view possible is that every project
follows the same rules, written once in [`hub.py`](hub.py) and imported by every pipeline:

| Convention | Why |
|---|---|
| One experiment per project, named `<project>/<use case>` | each team has its own space |
| Tags on every experiment, run and model: `project`, `use_case`, `owner`, `risk_tier` | "show me all Tier C projects", "everything the TB team logged" |
| Registered models named `<project>.<model>`, the one in use has the alias **`@champion`** | one way to ask "which model is live?", for every project |
| Each registered model names its `headline_metric` | one number per project, and everyone knows which |
| Every run has a `kind` (training, evaluation, monitoring); monitoring runs carry `period`, `data_gate`, `drift`, `performance` | the same three questions for every project, whatever tools answered them |
| The tools' reports are logged as artifacts of the monitoring run | the evidence sits next to the verdict |

Try exercise 1 below to see what happens to a project that doesn't follow them.

## From the local remote to a real one

On your laptop, the "remote" is a server process and a folder. In an organisation it is a machine that IT runs.
**The projects' code does not change**: only the address they report to.

| | This tutorial | An organisation |
|---|---|---|
| Where the server runs | your laptop, `127.0.0.1:5050` | one server (or virtual machine), e.g. `https://mlflow.eaii.example` |
| Database (runs, metrics, registry) | SQLite file `../mlflow-remote/mlflow.db` | PostgreSQL (or MySQL), backed up nightly |
| Artifact store (models, reports) | folder `../mlflow-remote/artifacts` | object storage: MinIO on your own hardware, or S3 / Azure Blob / Google Cloud Storage |
| Who can use it | anyone on your laptop | logins and permissions (below), HTTPS |
| What a project sets | nothing (the default in `hub.py`) | `MLFLOW_TRACKING_URI`, a username and a password |

**On the server** (once, by IT):
```bash
pip install mlflow psycopg2-binary boto3                     # MLflow + PostgreSQL driver + S3/MinIO client
export MLFLOW_S3_ENDPOINT_URL=http://minio.internal:9000     # only for MinIO; leave out for Amazon S3
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...      # storage keys live on the server only
mlflow server \
  --backend-store-uri postgresql://mlflow:PASSWORD@db.internal:5432/mlflow \
  --artifacts-destination s3://mlflow-artifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts mlflow.eaii.example \
  --app-name basic-auth
```
Put it behind an HTTPS reverse proxy (for example nginx), run it as a service so it restarts on reboot, and back up
both the database and the bucket.

**On every project's machine** (or in its pipeline's settings):
```bash
export MLFLOW_TRACKING_URI=https://mlflow.eaii.example
export MLFLOW_TRACKING_USERNAME=tb-pipeline
export MLFLOW_TRACKING_PASSWORD=...                          # from a secrets store, never in git
```
The server passes models and reports to and from the storage itself (`--artifacts-destination`), so the projects
never need the storage keys.

**Permissions.** `--app-name basic-auth` turns on MLflow's built-in logins. An administrator creates users and gives
each one READ, EDIT or MANAGE on each experiment and each registered model: for example, each team MANAGE on its
own project, the leadership READ on everything, and pipelines EDIT on their own experiment only. Change the default
administrator password before anyone else can reach the server. For single sign-on with an organisation's existing
accounts, put an identity proxy in front, or use a managed MLflow service.

**Data stays where you decide.** Runs hold metrics, settings and reports, not patient records, but a logged model or
a report can contain data from the training set. Keep the server on the organisation's own network and storage, and
decide which artifacts may be logged.

## The other tools in the same picture

The portfolio view stores the *results* of Great Expectations and Evidently in MLflow, which is the simplest way
to get one place. Each tool can also be run centrally on its own:

| Tool | Central option | In this tutorial |
|---|---|---|
| **Great Expectations** | keep each project's Data Docs on shared storage (an S3/MinIO site), or GX Cloud | the full validation result is logged with every monitoring run |
| **Evidently** | the Evidently UI service (self-hosted) or Evidently Cloud: one workspace, a project per team | the HTML report is logged with every monitoring run |
| **Prefect** | one Prefect server for every team's pipelines (the Prefect lesson already uses a server) | the pipelines here are plain scripts |
| **DVC** | one storage remote for all projects; a "data registry" repository for shared datasets | not used here |

## Files

| File | What |
|---|---|
| `hub.py` | the shared conventions: connect to the server, project tags, promote to `@champion`, log a monitoring run |
| `checks.py` | runs Great Expectations and saves Evidently reports, so their results can be logged |
| `malnutrition_pipeline.py`, `tb_pipeline.py`, `llm_pipeline.py` | one pipeline per team; each reuses its lesson's code |
| `assistant_config_model.py` | the LLM project's registry entry ("models from code") |
| `llm_results/` | saved evaluation results of the LLM lesson that `llm_pipeline.py` reads |
| `portfolio.ipynb` | the organisation's view, read only from the server |
| `data/` | local working files (simulated months, report files before upload); not in git |

## Exercises

1. **A project that skips the conventions.** With the server running, log a run from the MLflow lesson to the central
   server:
   ```bash
   cd ../malnutrition/03-mlflow
   MLFLOW_TRACKING_URI=http://127.0.0.1:5050 python train.py --run-name from-the-lesson
   ```
   (Windows PowerShell: `$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5050"; python train.py --run-name from-the-lesson`.)
   The run reaches the server (look in the UI), but does it appear in `portfolio.ipynb`? Why not, and what would
   `train.py` need to add?
2. **A fourth project.** Write `hr_pipeline.py` for the HR shortlisting use case (or any project of yours) with
   `hub.use_project`, a training run, `hub.promote` and one `hub.log_check`. Re-run the portfolio.
3. **A new champion.** The malnutrition model failed its July check. Train a challenger on the months that passed the
   data gate in 2025 (the files are in `data/malnutrition/`), compare it with the champion on the same months, and
   promote it with `hub.promote`. What does the portfolio show now, and where is version 1?
4. **Your own question.** Using `mlflow.search_runs(search_all_experiments=True, filter_string=...)`, list every
   monitoring run whose performance is still `pending`. Who should be chasing those outcomes?

## Start over

Stop the server (Ctrl+C in Terminal 1), then delete its data and the local working files:
```bash
rm -rf ../mlflow-remote central-mlflow/data        # from the repo root
```
