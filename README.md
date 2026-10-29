# MLOps Workshop

Hands-on lessons, one per tool, built around one running example: predicting
which children are at risk of acute malnutrition from (synthetic) Ethiopian
EMR data.

The seven malnutrition lessons are taught in two blocks (data, then model); see [`malnutrition/README.md`](malnutrition/README.md) for what is core and what is optional.

| # | Topic | Folder |
|---|-------|--------|
| 0 | Mock EMR data + drift scenarios; [notebook](malnutrition/00-data/explore_data.ipynb) to explore it | [`malnutrition/00-data/`](malnutrition/00-data/) |
| 1 | DVC: data, model and pipeline versioning ([cheat sheet](malnutrition/01-dvc/DVC_CHEATSHEET.md)) | [`malnutrition/01-dvc/`](malnutrition/01-dvc/) |
| 2 | Great Expectations: data validation as a pipeline gate | [`malnutrition/02-great-expectations/`](malnutrition/02-great-expectations/) |
| 3 | MLflow: experiment tracking, model registry, serving | [`malnutrition/03-mlflow/`](malnutrition/03-mlflow/) |
| 4 | Fairlearn: who does the model miss? Per-group metrics, causes, mitigation | [`malnutrition/04-fairlearn/`](malnutrition/04-fairlearn/) |
| 5 | Evidently: data drift, prediction drift, performance monitoring | [`malnutrition/05-evidently/`](malnutrition/05-evidently/) |
| 6 | Prefect: the monthly pipeline that runs all of the above | [`malnutrition/06-prefect/`](malnutrition/06-prefect/) |
| – | Use case: TB chest X-ray screening (Zambia study: acquisition drift vs population shift) | [`tb-chest-xray/`](tb-chest-xray/) |
| – | Use case: an LLM assistant over the national guidelines. Part 1: build a RAG chatbot, evaluate it without reference answers (Ragas), improve it one part at a time. Part 2: a clinical decision support tool, graded by people and by a checked judge, a cheaper model through a promotion rule | [`llm-cds/`](llm-cds/) |
| – | The organisation's view: one MLflow server for every project. Malnutrition, TB and LLM pipelines report to it (with their Great Expectations and Evidently results); a portfolio notebook shows every project's champion and latest checks; how to swap the local server for a real one | [`central-mlflow/`](central-mlflow/) |
| – | Wrap-up: in groups, write the MLOps plan for one AI project (objective, data, quality gates, metric and promotion rule, fairness principle, monitoring) and present it | [`wrap-up/`](wrap-up/) |

## Where each tool runs: training vs production, centre vs edge

The MLOps tools in this workshop live **where models are built and managed**. They are
not installed where predictions happen at the point of care.

| Tool | Training / retraining (central) | Production, central (server, batch scoring) | Production, edge (offline tablet / health post) |
|---|---|---|---|
| **DVC** | versions training + evaluation data, pipeline, model files | versions production data **once it syncs back** and becomes next round's training data | – |
| **Great Expectations** | gate on every training extract | gate on every scoring batch (a scoring suite: no label column) | – the tool stays central; its **per-record** rules become a small input check or form validation |
| **MLflow tracking** | every training run: params, metrics, data + code version | optional: monitoring metrics over time | – |
| **MLflow registry** | approve a version (`@champion`), with evidence | source of truth for which model to load | indirectly: the exported bundle records which registry version it is |
| **MLflow serving** | – | one option for an API (or your own API, or batch scoring) | – |
| **Fairlearn** | per-group evaluation before promotion | per-group performance once outcomes arrive | – |
| **Evidently** | defines the reference data | drift reports on synced production data | – |
| **Prefect** | orchestrates retraining | schedules batch scoring, monitoring, syncs | – |
| **Labeling (e.g. Label Studio)** | creates labels (images, audio, text) | reviews uncertain or flagged cases → new labels | – |

**What runs at the edge** is small: the **approved model** (a model file, or for a logistic
regression even a points score), an **input check** for single records, and **logging**
of inputs, predictions and the model version for later sync.

**What crosses the boundary:**
- *centre → edge:* model bundle = model (or points table) + input rules + registry version
- *edge → centre:* inputs, predictions, model version used; later, the outcomes

**Two kinds of data checks:** *per-record* rules (MUAC 8–25 cm, sex F/M, age 0–59 months,
required fields) can run anywhere, including on the device or in the data-entry form.
*Batch* rules (duplicates, row counts, outcome rate, number of facilities) need many
records, so they run centrally on synced data. Keep one rules file that generates both,
so edge and centre never disagree. At the edge, a failed check should explain why there's
no risk score and fall back to the standard clinical rule (MUAC cut-offs), not fail silently.

## Setup

### 1. Get your own copy of the repo

1. On GitHub, click **Fork** on this repo: you get `github.com/<you>/MLOps-Workshop`.
2. Clone **your fork** and connect it to the original, so you can pull lesson updates:
```bash
mkdir mlops && cd mlops          # the DVC lesson creates ../dvc-remote next to the clone
git clone https://github.com/<you>/MLOps-Workshop.git
cd MLOps-Workshop
git remote add upstream https://github.com/ydfriedman/MLOps-Workshop.git
git checkout -b work             # do the lessons on this branch; keep `main` = the lessons as published
```
- `origin` = your fork (you push here). `upstream` = the workshop repo (you only pull from it).
- **Getting lesson updates** during the workshop:
  ```bash
  git checkout main && git pull upstream main     # main = latest lessons
  git checkout work && git merge main             # bring them into your work branch
  ```
- **Starting over** (throws away everything on `work`, including DVC data and models):
  ```bash
  git checkout -f main             # -f: discard uncommitted changes
  git branch -D work
  git clean -fdx -e .venv          # delete untracked + ignored files (DVC cache, data, models); keep .venv
  rm -rf ../dvc-remote             # the DVC lesson's local remote
  git checkout -b work
  ```

### 2. Python environment

**Python 3.12.** (3.10+ works with `requirements.txt`, but the lock file and
all numbers in the READMEs were verified on 3.12.) macOS ships Python 3.9,
which is too old.

Easiest: [uv](https://docs.astral.sh/uv/) installs Python and packages, and is fast:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Windows: see the uv docs
uv python install 3.12
uv venv --python 3.12 --seed .venv                  # --seed: include pip (MLflow warns without it)
source .venv/bin/activate                           # Windows: .venv\Scripts\activate
uv pip sync requirements.lock
```

Or with a Python 3.12 you already have:
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
```

Check: `python --version && dvc --version && python -c "import great_expectations as gx; print(gx.__version__)"`
(expect 3.12.x, 3.67.1, 1.24.0).

### Two requirements files
- `requirements.txt`: what the project depends on directly, loosely pinned. Edit this one.
- `requirements.lock`: every package, including dependencies of dependencies,
  pinned to the exact versions the lessons were verified with, for macOS,
  Windows and Linux. Install from this one.

After changing `requirements.txt`, regenerate the lock and re-verify the lessons:
```bash
uv pip compile requirements.txt --universal --python-version 3.12 -o requirements.lock
```
Same idea as DVC: `dvc.lock` pins the data and outputs, `requirements.lock`
pins the environment. Reproducibility needs both.
