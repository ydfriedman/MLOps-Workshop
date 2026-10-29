# MLOps plan: <project name>

Group members:

Copy this file (for example `plan_<group name>.md`), fill it in, and present it. Short answers are fine: a word, a
number, a sentence. If you don't know something, write **"?"** and who would know: that is part of the plan too.

---

## The plan on one page (fill this in last: it is what you present)

| | |
|---|---|
| **The decision the model supports** | |
| **Who uses it, and what they do with the output** | |
| **The one number we judge it by, and the target** | |
| **The promotion rule** (when a new version may replace the old) | |
| **The most important data quality gate** | |
| **Fairness: which groups, which principle** | |
| **The first alert we would set up** | |
| **The biggest risk, and what we do about it** | |

---

## 1. The objective

| | |
|---|---|
| Project | |
| Owner (a named person or team) | |
| The decision it supports (who decides what, when) | |
| What happens today, without the model (the baseline) | |
| What a wrong output costs: a missed case? a false alarm? | |
| Risk tier (EAII MLOps Practice Guide; Tier C = routine use for decisions about people's health) | |

## 2. The data (DVC)

| | |
|---|---|
| Sources (which systems, who owns them) | |
| What one row is (a visit? a patient? an image? a question?) | |
| The label: what it is, where it comes from, **how long until we know it** | |
| Training period, and test period (later than training?) | |
| What we version with DVC, and where the remote is | |
| Personal or sensitive data: what, who may see it, where it is stored | |

## 3. Data quality gates (Great Expectations)

What a new batch must pass before the model scores it. Three to five expectations.

| Column or check | Expectation | Critical (stop) or warning? |
|---|---|---|
| | | |
| | | |
| | | |

What happens to a batch that fails a critical expectation (quarantine? who is told?):

## 4. The model, tracking and promotion (MLflow)

| | |
|---|---|
| Model type (and the simple baseline to beat) | |
| Metric(s) to log for every run | |
| **The one headline metric** | |
| Threshold or operating point (for example top 20% flagged, or 90% sensitivity), and why | |
| What else every run records (data version, parameters, code version) | |

**The promotion rule, written before seeing any result.** A challenger replaces the champion only if:

1.
2.
3.

Who approves the promotion?

## 5. Fairness (Fairlearn)

| | |
|---|---|
| The groups to compare (sex? region? age? language?) | |
| The principle: demographic parity, equal opportunity, equalized odds, or predictive parity | |
| Why this principle, for this decision | |
| The largest gap we accept (for example a difference in sensitivity of at most 0.05) | |
| What we do if the gap is larger | |

## 6. Monitoring (Evidently)

| What we watch | How (test or threshold) | How often | Who is alerted |
|---|---|---|---|
| Key input feature: | | | |
| Key input feature: | | | |
| The model's predictions (the share flagged, the mean score) | | | |
| Performance, once labels arrive (when?) | | | |

What happens after an alert (investigate? retrain? stop using it?), and who decides:

## 7. The pipeline (Prefect)

| | |
|---|---|
| When it runs (every day? every month?) | |
| Its steps, in order | |
| What stops it (which gate) | |
| Retraining: on a schedule, or when an alert fires? | |

## 8. For LLM projects only (skip otherwise)

| | |
|---|---|
| The test set: how many questions or cases, written by whom, in which languages | |
| How answers are judged: reference-free checks (Ragas), a rubric, people | |
| Who checks the judge, how often | |
| The documents in the index: who approves them, how a revision is handled | |
| Planted-instruction (prompt injection) and out-of-scope tests | |

## 9. Risks

| Risk | What we do about it |
|---|---|
| | |
| | |
| | |
