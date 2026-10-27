# 04 – Fairlearn: who does the model miss?

**Where we are.** Every lesson so far showed the same number: the model catches fewer
malnourished girls than boys. This lesson measures that properly, finds where it comes
from, and asks which principle of fairness the programme should adopt. One notebook: [`fairness.ipynb`](fairness.ipynb) (about 1.5 minutes to run).

**The setting.** A clinic can follow up 20% of visits; the model ranks children by risk
and the top 20% are flagged. The costly error is a **miss** (a child who becomes acutely
malnourished but wasn't flagged).

## Setup

From the repo root, environment active:
```bash
cd malnutrition/04-fairlearn
```
Open `fairness.ipynb` in VSCode (kernel: `.venv`) and run it top to bottom. It reuses the
MLflow lesson's helper module (`../03-mlflow/emr_model.py`) and generates its own data in
`04-fairlearn/data/`: training on 2024, evaluation on a full year (2025, 24,000 visits).

## What the notebook covers, and the verified results

| Section | Finding |
|---|---|
| 1. `MetricFrame` by sex | In 2025 girls and boys develop GAM at about the same rate (~11%), but sensitivity is **0.54 vs 0.61**: about 7 more of every 100 malnourished girls are missed. Flags are equally reliable (PPV ~0.32). |
| 2. Which "fair"? | The four principles, one table at a time: each is a ratio of the same four numbers (A caught, B missed, C false alarm, D correctly left out), with the formula written out. **Two pen-and-paper practices** (100 girls, 100 boys) show that when GAM rates differ, no rule makes all four equal. Then the real clinic per 1,000 girls and 1,000 boys: the model as built follows **predictive parity** (PPV 0.316 vs 0.321) but not equal opportunity (60 of ~110 sick girls caught vs 67 of ~110 boys). By region, alert rates *should* differ, but sensitivity ranges from **0.15 in Addis Ababa to 0.80 in Afar**. |
| 3. Intersections + uncertainty | Bootstrap intervals (`n_boot`). On 2 months, the youngest children show girls 0.22 vs boys 0.74, from 23 cases each: mostly noise. On 12 months the real gap is at **6–23 months** (0.59 vs 0.68). |
| 4. Why fewer girls are flagged | In the 2024 training data boys developed GAM a little more often (11.3% vs 10.7%), as in surveys. The model learns "girl = lower risk" through `sex`; in 2025 the rates are almost equal (11.1% vs 10.9%). Dropping `sex` **halves the gap** (0.071 → 0.036) at no cost in ROC-AUC. **PAUSE: which principle would you adopt** (leave it, equal opportunity, demographic parity)? |
| 5. If we choose equal opportunity | **Per-sex thresholds** within the 20% capacity: equal sensitivity (0.58 / 0.58), overall no worse, but a flag for a girl is less reliable (PPV 0.31 vs 0.34). **The three choices side by side** (one threshold, per-sex thresholds, top 20% of each sex): all catch 127–129 sick children per 2,000; what changes is who, and who gets the false alarms. PAUSE: which row would you sign? Fairlearn's `ThresholdOptimizer` with defaults flags **33%** of visits, because its objective ignores capacity. |
| 6. Decide, document, monitor | Trade-off table; write down the definition and tolerance first; gate promotion on it (MLflow lesson); monitor it (Evidently). |

## Key messages for the room

- **Unfairness usually enters through the data, not the code.** Here a small historical
  difference between girls and boys becomes a lower score for every girl, and the model
  repeats it.
- **Choose the fairness definition with the programme owners**, before looking at models.
  For screening, equal opportunity (equal sensitivity) is the usual priority.
- **Small groups produce noisy gaps.** Always report intervals and case counts.
- **Fairness tools optimise what you tell them.** Encode the real constraint (capacity).
- **The same tools give different answers in different settings.** Using sex at decision
  time is normal in clinical care (growth standards are sex-specific); in hiring (the
  resume-matching use case) protected attributes usually can't be used at all.

## Helper functions in this lesson

| Helper | What it does |
|---|---|
| `sensitivity_with_ci(y, flag, group)` | sensitivity per group with a bootstrap confidence interval, so small gaps are not over-read |
| `summarise(name, score, flag)` | one row of the comparison table: ROC-AUC, alert rate, sensitivity by sex, the gap |
| `top20(score)` | flags the top 20% of scores: the programme's follow-up capacity |
| `equal_sensitivity_flags(...)` | per-group thresholds that give boys and girls the same sensitivity at the same total alerts |

## Why doesn't a better model remove the gap?

The model already adjusts for sex (the `sex_F` coefficient). Interactions and boosted trees (the notebook
runs both) leave a gap of 0.07 → 0.07 / 0.05, against 0.04 without the sex feature. The model is not too simple:
the gap comes from the history in the data, and a flexible model learns it just as well. No model can tell
whether a historical difference is real need or an artefact of how records were made. The fixes are decisions
(principle, threshold, features, review), not model capacity.

## Is the girls-boys difference real?

The direction follows the literature (boys are more often wasted than girls: Thurstans et al., *BMJ Global Health*
2020, pooled OR 1.26; Ethiopian DHS 2000–2016, 11.9% of boys vs 9.9% of girls). The size here is synthetic and
smaller. The sensitivity gap itself is sample-dependent: on this workshop's 24,000 evaluation visits the 90% interval
for the gap is roughly 0.04 to 0.10, and a different training sample would give a different size. Report it
with an interval, as section 3 shows.

## Fairlearn cheat sheet

| Call | What it does |
|---|---|
| `MetricFrame(metrics={...}, y_true, y_pred, sensitive_features=df[["sex"]])` | Any metrics, overall and per group (several columns = intersections). |
| `.overall`, `.by_group`, `.difference()`, `.ratio()` | Overall value, per-group table, largest gap, smallest/largest ratio. |
| `MetricFrame(..., n_boot=200, ci_quantiles=[0.05, 0.95])` → `.by_group_ci` | Bootstrap confidence intervals per group. |
| `selection_rate`, `true_positive_rate`, `false_negative_rate`, `false_positive_rate` | Ready-made metrics (alert rate, sensitivity, miss rate, false-alarm rate). |
| `demographic_parity_difference`, `equal_opportunity_difference`, `equalized_odds_difference` | One-number fairness gaps. |
| `ThresholdOptimizer(estimator, constraints=..., objective=..., prefit=True)` | Learns group-specific thresholds under a fairness constraint. |
| `ExponentiatedGradient(estimator, constraints=TruePositiveRateParity())` | Trains a classifier under a fairness constraint (yes/no decisions, not risk scores). |

## Exercises

1. **Per-region thresholds.** Apply `equal_sensitivity_flags` to `evalset["region"]`
   instead of sex. What happens to Addis Ababa, to Afar and Somali, and to overall
   sensitivity? Who pays for equality, and would the nutrition programme accept that?
2. **Thresholds set in advance.** In practice thresholds are chosen on past data. Choose
   per-sex thresholds on Jan–Jun 2025 and apply them unchanged to Jul–Dec 2025. Does
   equal sensitivity hold? What about the alert rate (the lean season starts in June)?
3. **Intersectional thresholds.** Use `sex` and `age_band` together (6 groups). Using
   section 3's intervals, which of these thresholds would you trust?
4. **Log it.** In the MLflow lesson, add `MetricFrame(...).by_group` for sex and region to
   each run (`mlflow.log_dict`) and the girls/boys sensitivity gap as a metric. Then
   tighten the promotion rule to require a gap below 0.03.
5. **Resume matching.** Sketch the same analysis for shortlisting candidates: what is
   the "miss", which groups, which definition, and which mitigations are allowed?

Try the exercises first (they are take-home). Stuck, or short on time? [`exercise_solutions.ipynb`](exercise_solutions.ipynb) holds a worked answer to each; it ships without outputs, so run it to see them.
