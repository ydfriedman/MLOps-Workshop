# Wrap-up: an MLOps plan for an AI project (1 hour)

All week you used one tool at a time. Now put them together: in three groups, write the **MLOps plan** for one AI
project, then present it.

## What to do

1. **Form three groups** (5 min).
2. **Choose a project.** One of your own (best), or one from the list below.
3. **Copy the template.** One person per group types, on their laptop: in VSCode, copy
   [`mlops_plan_template.md`](mlops_plan_template.md) to `plan_<group name>.md` in this folder (or use a copy of the
   shared doc) and fill it in together (30 min).
   [`example_plan_malnutrition.md`](example_plan_malnutrition.md) is a filled-in example from Day 1.
4. **Present** the box at the top, "The plan on one page", in **5 minutes** from that laptop on the projector, then
   2 minutes of questions from the other groups.

Short answers are fine. If you don't know something, write **"?"** and who would know: a plan that shows what is
still unknown, and who must answer it, is a good plan.

## Timing

| Minutes | What |
|---|---|
| 0–5 | form three groups, choose a project |
| 5–35 | fill in the template; start with sections 1–2, end with "The plan on one page" |
| 35–55 | presentations: 5 minutes each, then 2 minutes of questions |
| 55–60 | what was hardest to decide, and who in your organisation would decide it |

## Projects to choose from (if you don't bring your own)

| Project | The decision | Things to think about |
|---|---|---|
| **Malnutrition risk** (Day 1) | which children get a follow-up visit | already worked through in the example: choose another project, or change one thing (a new region? a model that also uses photos?) |
| **TB screening from chest X-rays** (Wednesday) | who gets a lab test for TB | a new X-ray machine at a new site; HIV-positive patients |
| **Amharic speech recognition** (Wednesday) | (depends on its use: transcribing consultations? call-centre messages?) | accents and dialects; what "accuracy" means; recordings are personal data |
| **HR shortlisting** (Thursday) | which applicants get an interview | fairness between groups; what the label is; who can appeal |
| **The guideline assistant** (Thursday–Friday) | what a health worker does for a patient | use section 8 of the template |
| **Crop yield or rainfall forecast** | where to send support before a bad season | the label arrives once a year; the climate changes |
| **Fraud or error detection in claims or payments** | which claims a person checks | few true cases; what a missed case costs |

## Which week's tool answers which question

| Template section | Tool | Lesson |
|---|---|---|
| 2. Data | DVC | malnutrition 01 |
| 3. Data quality gates | Great Expectations | malnutrition 02 |
| 4. Model, tracking and promotion | MLflow | malnutrition 03, central-mlflow |
| 5. Fairness | Fairlearn | malnutrition 04 |
| 6. Monitoring | Evidently | malnutrition 05, TB |
| 7. Pipeline | Prefect | malnutrition 06 |
| 8. LLM projects | Ragas, MLflow traces, rubrics | llm-cds |
