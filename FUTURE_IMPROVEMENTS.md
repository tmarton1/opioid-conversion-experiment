# Future Improvements

Known methodological limitations and cohort inconsistencies in the current experimental design. Each item describes the issue, its root cause, and potential remediation.

---

## 1. Haiku and versatile have incomplete random-dose pair coverage

**Models affected**: `anthropic/claude-3-5-haiku-20241022`, `meta-llama/llama-3.3-70b-versatile`

**Issue**: Both models have 44 / 49 random-dose pairs in `pair_model_analysis.csv`, missing the same 5 OME pairs:
- `im_buprenorphine → oral_morphine`
- `iv_buprenorphine → oral_morphine`
- `iv_methadone → oral_morphine`
- `parenteral_nalbuphine → oral_morphine`
- `transdermal_buprenorphine → oral_morphine`

Neither model ran a vignette experiment. Both are excluded from `CONSISTENT_MODELS`.

**Root cause**: Experiments for these 5 pairs were never run (or failed) for these two models. The reason is undocumented in the codebase.

**Impact**: Both models appear in `ranking_naive_random_doses.csv` (16 models) but are absent from all other analyses. The 13-model `CONSISTENT_MODELS` cohort was defined to exclude them, but the exclusion rationale is implicit rather than stated.

**Remediation**: Re-run the missing 5 pair experiments for both models, then decide whether to include them in `CONSISTENT_MODELS`. Document the exclusion criterion explicitly in `common/models.py`.

---

## 2. jamba-mini has complete coverage but is excluded from CONSISTENT_MODELS

**Model**: `ai21/jamba-mini-1.7`

**Issue**: `jamba-mini-1.7` has all 49 random-dose pairs and all 20 vignette scenarios (complete coverage). It appears in both `ranking_naive_random_doses.csv` and `ranking_naive_vignettes.csv`, but it is absent from `CONSISTENT_MODELS` and therefore excluded from cross-experiment matching, ICC analysis, Wilcoxon condition comparisons, and explain-units analysis.

**Root cause**: Reason for exclusion is undocumented.

**Impact**: A potentially interesting data point (small model vs. large model comparison) is silently dropped from all analyses that use `CONSISTENT_MODELS`.

**Remediation**: Document why jamba-mini was excluded (e.g. anomalous behaviour, post-hoc decision). If the exclusion is no longer justified, add it to `CONSISTENT_MODELS` and re-run analyses.

---

## 3. Explain-units covers only 15 of 19 OME pairs and adds one non-OME pair

**Issue**: `explain_units_pair_model_analysis.csv` contains 15 pairs. It is missing the same 5 OME pairs as haiku/versatile (§1 above), and it includes `oral_morphine → oral_morphine`, which is a within-drug identity conversion (not an OME conversion and not in the main `pair_model_analysis.csv`).

**Missing from explain-units** (4 OME pairs with Nielsen reference data + 1 without):
- `im_buprenorphine → oral_morphine`
- `iv_buprenorphine → oral_morphine`
- `iv_methadone → oral_morphine`
- `parenteral_nalbuphine → oral_morphine`
- `transdermal_buprenorphine → oral_morphine`

**Extra in explain-units**: `oral_morphine → oral_morphine` — a trivial within-drug conversion that serves as a control but dilutes the OME signal.

**Root cause**: The explain-units experiment was apparently designed around Nielsen reference table coverage, which does not include the 5 buprenorphine/methadone/nalbuphine rows. The within-drug control was added intentionally but not flagged as structurally different.

**Impact**: `dosing_mean_tight_drop_pp` (mean accuracy drop from dosing-unit ambiguity) and the recovery statistics include the within-drug pair, which trivially shows perfect accuracy and inflates the apparent baseline. The 5 missing OME pairs are the same ones with incomplete model coverage, so the gap compounds §1.

**Remediation**: Run the 5 missing explain-units pairs once Nielsen reference values are confirmed. Flag `oral_morphine → oral_morphine` as a control row and exclude it from summary statistics.

---

## 4. Cross-experiment matching covers only 11 of 20 vignette scenarios

**Issue**: The cross-experiment analysis in `analysis/cross_experiment_matched.py` matches on all four of (`from_opioid`, `to_opioid`, `from_opioid_unit`, `to_opioid_unit`). Only 11 vignette scenarios survive this filter.

The 9 excluded scenarios and their reasons:

| Scenario | Reason for exclusion |
|---|---|
| `middle-aged-woman-cancer-surgery` | `multiple_from_opioids=True` (compound source) |
| `elderly-female-renal-colic` | `multiple_from_opioids=True` (compound source) |
| `elderly-hip-replacement` | Direction `oral_morphine → oral_oxycodone` not in RD |
| `middle-aged-ankle-surgery` | Direction `oral_morphine → oral_hydrocodone` not in RD |
| `elderly-female-pancreatic-carcinoma` | Direction `oral_morphine → iv_morphine` not in RD |
| `elderly-female-chronic-bowel-obstruction` | Direction `oral_methadone → iv_methadone` not in RD |
| `middle-aged-female-cancer` | Pair `oral_hydromorphone → transdermal_fentanyl` not in RD |
| `middle-aged-carpenter-amputation` | Pair `iv_fentanyl → oral_hydromorphone` not in RD |
| `pediatric-burn-patient` | Drug pair matches (`iv_hydromorphone → oral_morphine`) but dose unit is `mg/4hr` in vignette vs `mg/day` in RD |

**Root cause**: Vignette scenarios were designed to reflect realistic clinical conversion contexts, which include reverse directions (e.g., deescalation) and unusual targets. The RD experiment covers a curated subset of 49 pairs, prioritising common forward conversions.

**Impact**: The 11-pair matched set is a non-representative subset of both experiments. The 4 scenarios with reverse directions (deescalation) are clinically distinct from the forward conversions; excluding them may systematically underestimate cross-experiment agreement for deescalation tasks.

**Remediation**:
- Add the 8 missing RD pairs (4 reverse directions + `iv_fentanyl→oral_hydromorphone` + `oral_hydromorphone→transdermal_fentanyl` + the two compound scenarios, decomposed) to the RD experiment.
- For `pediatric-burn-patient`, either re-run the vignette with `mg/day` dosing or add a `mg/4hr` variant to RD.

---

## 5. Two vignette scenarios have compound source opioids

**Scenarios**: `middle-aged-woman-cancer-surgery`, `elderly-female-renal-colic`

**Issue**: Both scenarios involve a patient on two concurrent opioids. The `from_opioid_unit` field records both units as a comma-separated string (e.g. `mg/day,mg/day`). This means:
- No single RD pair maps to these scenarios (§4 above).
- The ICC and CV analyses still include these scenarios but the `from_opioid` is ambiguous.
- It is unclear whether model responses integrate both source drugs or treat only the primary one.

**Root cause**: Clinical vignettes were taken from real cases and not constrained to single-drug scenarios.

**Impact**: Accuracy scores for these two scenarios are computed against a single expected output, but the input stimulus involves two drugs. Model errors may reflect legitimate ambiguity rather than conversion failure.

**Remediation**: Either decompose each scenario into two single-drug sub-scenarios, or flag them explicitly in vignette analysis as multi-source scenarios and report their accuracy separately.

---

## 6. Random-dose CV reflects dose-range sampling variation, not test-retest reliability

**Issue**: In `pair_model_analysis.csv`, `n_observations` per model × pair × condition cell is not 1 (it ranges from 1 to 68 with median ~25). The experiment samples many different source doses from a dose range for each pair, then asks the model to convert each one. `observed_coefficient_cv` is the coefficient of variation across these dose-sampled responses.

This means CV measures *dose-range response variation* (how consistently a model scales as source dose changes), not *test-retest reliability* (whether the model gives the same answer for the same dose on repeated calls).

**Root cause**: The RD experiment design uses a dose-sampling strategy to cover the clinical range, rather than repeating a fixed dose. A "repetition" in the CV strip figure is a different source dose, not a re-query.

**Impact**:
- CV cannot be interpreted as stochastic noise analogous to the vignette inter-wording variance.
- The CV strip figures (Figs 2 and 9) and CV × accuracy chi-square (§12) are correctly labelled but the asymmetry with vignette ICC is not explicitly stated.
- High CV for a model on a pair could reflect dose-dependent nonlinearity, not inconsistency.

**Remediation**: Add repeated trials with a fixed representative dose per pair (e.g. 3 repetitions of the median dose) to obtain a true stochastic-noise estimate. Report dose-range CV and stochastic CV separately.

---

## 7. Vignette scenarios use clinical conversion directions; RD covers both directions

**Issue**: Vignette scenarios reflect clinician practice: they predominantly describe dose escalation or rotation to a more potent or convenient opioid. The RD experiment includes both directions for most drug pairs (e.g. both `iv_hydromorphone → oral_morphine` and `oral_morphine → iv_hydromorphone`).

Neither the cross-experiment comparison nor the Nielsen effect analysis corrects for this directional asymmetry.

**Root cause**: Vignette scenarios were authored as realistic clinical cases; RD pairs were enumerated exhaustively.

**Impact**: Cross-experiment accuracy differences may partly reflect direction-of-conversion effects rather than task-format effects. In the current 11-pair matched set, all matched pairs happen to be in the same direction across both experiments — but this cannot be taken as validation of direction balance.

**Remediation**: Tag each RD pair as "clinical direction" vs "reverse direction" and report cross-experiment comparisons separately for each subset. Compare accuracy on reverse-direction RD pairs to vignette scenarios that share the same direction.

---

## 8. Nielsen reference table covers OME-routed conversions; non-OME Nielsen effect is indirect

**Issue**: The Nielsen reference table provides conversion factors primarily via the oral morphine equivalent (OME) pathway. For non-OME pairs (30 / 49 RD pairs), the "Nielsen" conversion factor is likely derived by chaining two OME ratios (e.g. A→morphine→B). This indirect derivation is not explicitly documented.

**Root cause**: The Nielsen (2022) guide is organised around oral morphine equivalents. Direct conversion ratios for non-OME pairs (e.g. `iv_hydromorphone → oral_hydrocodone`) are typically not given and must be computed.

**Impact**:
- The "Nielsen effect" for non-OME pairs reflects whether models implicitly use OME-chaining logic, not whether they have memorised a specific direct ratio.
- `rd_nonome_tight_improvement_pp` (60.2 pp) vs `rd_ome_tight_improvement_pp` (72.6 pp): the smaller non-OME improvement is consistent with models already partially knowing these ratios (or with indirect ratios being less reliably derived).
- The expected coefficient for non-OME pairs may have higher uncertainty than for OME pairs.

**Remediation**: Document the derivation of each expected coefficient in the resources folder. For non-OME pairs, flag whether the expected value is a direct citation or a chained OME calculation, and report the two subsets separately in the Nielsen effect analysis.

---

## 9. pediatric-burn-patient uses q4h dosing; incompatible with mg/day RD experiment

**Issue**: The `pediatric-burn-patient` vignette specifies an `oral_morphine` target dose in `mg/4hr` (every-4-hour dosing), while all random-dose pairs in `pair_model_analysis.csv` use `mg/day`. The drug pair (`iv_hydromorphone → oral_morphine`) exists in RD, but the dose-unit mismatch prevents cross-experiment matching (see §4).

**Root cause**: The vignette was written with a q4h dosing order, which is common paediatric practice, rather than a total daily dose.

**Impact**: This scenario is excluded from cross-experiment comparison. The scenario can still be evaluated within the vignette experiment (where the expected coefficient is dose-unit normalised), but the unit mismatch means models that answer in mg/4hr vs mg/day are graded on different bases depending on what normalisation is applied.

**Remediation**: Either rewrite the vignette with a total daily dose target, add a `mg/4hr` variant of `iv_hydromorphone → oral_morphine` to the RD experiment, or document that the expected coefficient for this scenario is already unit-normalised and verify that the evaluator handles `mg/4hr` responses correctly.
