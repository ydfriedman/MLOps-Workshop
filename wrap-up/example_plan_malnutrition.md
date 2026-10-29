# MLOps plan: GAM risk at routine child visits (example)

A filled-in example, from the malnutrition lessons of this workshop (Day 1). Your plan can be shorter. The data in the
lessons is SIMULATED; a real plan would name real systems and people.

---

## The plan on one page

| | |
|---|---|
| **The decision the model supports** | which children seen at a routine visit get a home follow-up visit, because they are at high risk of acute malnutrition (GAM) in the next 90 days |
| **Who uses it, and what they do with the output** | health extension workers: each month, a list of the 20% of visits with the highest risk, to follow up first |
| **The one number we judge it by, and the target** | ROC-AUC on the latest months not used for training (the first model: 0.81) |
| **The promotion rule** | a retrained model replaces the current one only if it ranks no worse, is better calibrated, and sensitivity for girls and for boys does not drop |
| **The most important data quality gate** | MUAC in centimetres, between 8 and 25: a batch in millimetres is stopped |
| **Fairness: which groups, which principle** | girls and boys; equal opportunity (the same share of future GAM cases caught in each) |
| **The first alert we would set up** | observed / predicted GAM rate outside 0.67–1.5, once outcomes arrive |
| **The biggest risk, and what we do about it** | outcomes arrive 3 months late, so a change (food distribution stops) shows up late: watch the inputs and the share flagged every month meanwhile |

---

## 1. The objective

| | |
|---|---|
| Project | GAM risk score for children 0–59 months at routine visits |
| Owner | the regional nutrition programme's data team |
| The decision it supports | every month, which children the health extension workers visit first |
| What happens today | follow-up of children already wasted (MUAC below 12.5 cm); no ranking of the others |
| What a wrong output costs | a missed child may become malnourished before anyone sees them; a false alarm costs one home visit |
| Risk tier | Tier C: routine use, decisions about children's health |

## 2. The data (DVC)

| | |
|---|---|
| Sources | the EMR's monthly export of child visits (health centres), owned by the regional health bureau |
| What one row is | one visit: age, sex, region, weight, height, MUAC, recent illness, household information |
| The label | GAM within 90 days of the visit, from later visits: **known 3 months late** |
| Training and test periods | train January–October 2024, test November–December 2024 (always later than training) |
| What we version with DVC | each month's raw export, as received; the remote is on the bureau's server |
| Personal data | child records: only the data team; the model and its reports contain no names |

## 3. Data quality gates (Great Expectations)

| Column or check | Expectation | Critical or warning? |
|---|---|---|
| all columns | exactly the expected columns | critical |
| `visit_id` | unique, never empty | critical |
| `muac_cm` | between 8 and 25 (centimetres, not millimetres) | critical |
| `age_months` | between 0 and 59 | critical |
| `weight_kg` | not empty in at least 95% of visits | critical |
| `gam_within_90d` | mean between 0.02 and 0.30 | warning |

A batch that fails a critical expectation is **not scored**: it goes to quarantine, and the data team and the EMR
team are told which expectation failed.

## 4. The model, tracking and promotion (MLflow)

| | |
|---|---|
| Model type | logistic regression; the baseline is "flag MUAC below 12.5 cm" |
| Metrics logged | ROC-AUC, sensitivity and precision at the top 20%, calibration (observed / predicted), all by sex and by region |
| Headline metric | ROC-AUC |
| Operating point | the top 20% of visits by risk are flagged: what the health extension workers can visit in a month |
| Also recorded | the DVC data version, the git commit, all parameters |

**The promotion rule** (Prefect lesson). A retrained challenger replaces the champion only if:

1. its ROC-AUC is at most 0.005 below the champion's;
2. its calibration (observed / predicted) is closer to 1 than the champion's;
3. sensitivity for girls and for boys drops by at most 0.02.

Approved by the programme lead; the model in use always has the alias `@champion` in the registry.

## 5. Fairness (Fairlearn)

| | |
|---|---|
| Groups | sex; region |
| Principle | equal opportunity: the same sensitivity for girls and boys |
| Why | the aim is to catch the children who will become malnourished; missing more girls than boys is the harm to avoid |
| Largest gap accepted | 0.05 in sensitivity |
| If larger | a separate threshold per group, reviewed by the programme lead |

(In the workshop we left the principle open for discussion: another group could argue for another one.)

## 6. Monitoring (Evidently)

| What we watch | How | How often | Who is alerted |
|---|---|---|---|
| `muac_cm` | drift against the same season last year (test: drift score at most 0.1) | monthly | data team |
| `region` | drift (a rollout to new regions changes the mix) | monthly | data team |
| `weight_kg` | share missing at most 5% | monthly | data team, EMR team |
| the predicted risk | drift of the risk scores | monthly | data team |
| performance | observed / predicted GAM rate between 0.67 and 1.5, three months later | monthly | programme lead |

After an alert: the data team investigates within a week; a performance alert starts a retraining, and the
promotion rule decides.

## 7. The pipeline (Prefect)

| | |
|---|---|
| When | on the 2nd of every month |
| Steps | ingest the export → quality gate → version with DVC → score with `@champion` → drift report → check the outcomes of 3 months ago → retrain if needed |
| What stops it | a critical quality gate failure |
| Retraining | when the performance check fails; the promotion rule decides |

## 9. Risks

| Risk | What we do about it |
|---|---|
| outcomes arrive 3 months late | watch inputs and the share flagged monthly; calibration check on every month as soon as it can be done |
| the EMR export changes format (units, columns) | the quality gate stops the batch |
| a change in the real world (food distribution stops) that the inputs don't show | the performance check; a named person reads it every month |
