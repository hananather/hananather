# Early empirical findings (pilot runs, 2026-10-23). Read before designing the protocol.

All figures are from the **New England pilot population**: IRS Form 990 full filers in MA, CT, ME, NH, VT and RI that filed for both TY2022 and TY2023.
- N = 20,626 continuing units. Between the two years, 1,115 units stopped filing ("deaths") and 1,414 started ("births").
- Frame = BMF snapshot of 2023-08-16 plus the TY2022 return. Targets = TY2023 return.
- Text = TY2022 mission (Part I line 1 and Part III line 1), plus program descriptions (Part III 4a–4d).
- Median text length: 19–23 words for mission, 71 words for mission + programs.

The pilot Monte Carlo runs used only R = 40–60 replicates, so treat every efficiency number as ±0.15–0.2. They show *direction* and *bugs*, not results.

## Data facts that shape the design
- **NTEE is missing for 31% of units** in the 2023 BMF (6,417 of 20,626).
  - Real register defect; the conventional stratifier carries "MISSING" as its own level.
  - Text-coded NTEE (multinomial logit on embeddings) reaches **69–70%** 5-fold accuracy over the 10 NCCS groups.
- **Skew is extreme.** The top 1% of units hold:
  - 68% of expenses;
  - 66% of government grants;
  - 82% of volunteers. Knights of Columbus (CT) alone reports 2.1M volunteers, 42% of the regional total.
- **Gross errors exist.** A church reported 89,785 employees with $0 salaries; there are negative expenses and grants, and 152 filing dates before fiscal year end. Narrow edit rules E1–E3 were applied and logged (`src/build_population.py::apply_edits`). Influential but plausible values are kept.
- **Near-duplicate texts.** 989 units (4.8%) share a mission text with another unit (union locals, Elks lodges, elderly-housing LLCs, "see schedule o"). Cross-fitting uses grouped folds to keep these together.
- **Leakage profile.** 10–11% of mission texts contain a digit and 0.1% a "$". For program descriptions the figures are 39% and 3.7%. A numbers-masked text variant was embedded.
- **Late filing gives a real nonresponse mechanism.** With "response" = TY2023 return submitted ≤ 300 days after FYE:
  - response rate is 60%, but only 18% on an expense-weighted basis (big organizations file late, on extension);
  - respondents' mean expenses are 0.12× nonrespondents'; mean government grants 0.35×; lobbying share 0.41×.

## Stage-1 population diagnostics (anticipated-variance ceilings)
Setup: n = 1,000, a production design (NTEE10 × 3 size classes, take-all top 1% = 206 units), and Neyman allocation on lagged expenses.

Each model is fitted on the whole population by weighted ridge, with 5-fold cross-fitting over the population and λ picked per y on the population. **This is an optimistic ceiling, not an estimator.**

Values are the ratio AV(strong frame) / AV(frame + text), where "strong frame" = strata + lagged revenue / expenses / assets / employees + stratum × lagged expenses. All text here is LSA (TF-IDF + SVD, 256-d).

- **Size variables (expenses, revenue, employees):** about 1.00. Text adds nothing.
- **Salaries:** 1.39 (full text); 1.19 (mission only).
- **Government grants:** 1.17 (full); 1.07 (mission).
- **Contributions:** 1.17. **Fundraising expenses:** 1.15. **Program revenue:** 1.12.
- **Binary indicators:** 1.06–1.18 (grants to individuals 1.18; has government grants 1.14).
- **Volunteers (raw):** about 1.0. The outliers dominate.
- **Weak frame (no lagged amounts), relative to the strong frame:**

  | | Weak alone | Weak + text |
  |---|---|---|
  | Expenses | 0.23 | 0.45 |
  | Salaries | 0.37 | 0.85 |

  So in a weak register, text roughly doubles efficiency but recovers only about half of what lagged admin data give.
- **Oracle (positive control):** the lagged value of the survey-only variable itself (not available on a real frame) gives **4–6×** for government grants, fundraising and officer compensation. Text is far weaker than last year's value of the same variable.

## Monte Carlo pilot (R = 60, n = 1,000; MC noise about ±0.15)
Design: compromise allocation (Neyman on sqrt(x), min n_h = 10), take-all handled as in production, modelling on the take-some part.

**Bugs found and fixed**
- The GREG numerical epsilon was trace-scaled. It swamped dummy and PC columns next to dollar-valued columns, so embeddings had no effect. Now per-column.
- The expenses-optimal Neyman allocation left 2–4 units in small-unit strata. Yes/no indicators then had about 50% coverage under *every* estimator, HT included. Switched to the compromise allocation.
- The separate-ratio-per-stratum GREG ("sep", 97 columns) is worse than HT for yes/no indicators. It is a bad baseline and was replaced by a family of conventional estimators.

**Results**
- The baseline used is GREG_size_cat (strata + lagged amounts + NTEE + state). Its efficiency relative to HT is about 20× for expenses and 3–9× for most financial variables.
- Text gains relative to GREG_size_cat:
  - ridge-penalized ratio calibration and cross-fitted model calibration (ratio scale) with **LSA full text**: government grants 1.4×, salaries 1.4–1.7×, program revenue 1.2×, contributions 1.2×;
  - **MiniLM mission-only**: smaller gains (1.0–1.3×).
- **Negative controls:** random-384 and row-permuted embeddings give about 1.0. Text-free ML model calibration (MCAL_frame_ratio) gives 1.18 for salaries. So the fair comparator for text-ML is text-free-ML, not GREG.
- **Overfitting demonstration:** model calibration *without* cross-fitting gives 72–78% coverage, against 88–95% with cross-fitting.
- **The synthetic estimator Σ m̂ (no design correction)** is biased up to +15% on yes/no indicators and +7% on volunteers.
- **OLS-GREG with 32 raw PCs added to the weak frame** gives 38–55% coverage (p/n overfitting).

## Nonresponse pilot (late filers as nonrespondents, R = 40)

| Estimator | Bias |
|---|---|
| Naive (no adjustment) | −18% to −48% |
| Weighting class by stratum + calibration to the strong frame (production) | size variables ≈ 0; contributions +7%, fundraising +12%, government grants +11%, lobbying −9%, volunteers +5% |
| Weighting class only, no calibration | government grants +3%, contributions +1% |
| Weighting class + model calibration with LSA text | government grants +7%, contributions +5%, fundraising +9%, volunteers +2% |

- With weighting class only (no calibration), the bias is smaller for contributions and government grants, but variance is much higher for size variables. **Calibrating to lagged size increases nonresponse bias for survey-only y.**
- Adding text removes about 30–60% of the leftover bias, not all of it.
- Propensity models with text PCs did not help beyond the weighting-class adjustment.

## Compute
- One MC cell (30 estimators, 14 y, n = 1,000, N = 20.6k): about 1 s per replicate on a shared 4-core CPU, so R = 1,000 takes about 15–20 min.
- The national population (about 330k) is being downloaded (TY2022 done, TY2023 in progress).
- MiniLM CPU embedding: about 340 s per 20k short texts.
