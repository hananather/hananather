# Benchmark protocol, draft A: production realism

**Status:** draft for review, 2026-10-23. **Lens:** senior business-survey methodologist at a national statistical office (NSO).
**Question the benchmark must answer:** *should we change our production system?* Not "can embeddings predict something?".
**Companion notes:** `notes/early-findings.md` (pilot), `notes/research/01–04`. Code names below match `src/` where the estimator already exists. New code is listed in Appendix A.

---

## Summary

The pilot established that text can reduce variance for some variables. It did so under conditions an NSO would not face. This protocol keeps the pilot's machinery but changes the comparison to the one a production area would actually make. Eight changes matter most.

| # | Pilot choice | Production reality | Protocol change |
|---|---|---|---|
| 1 | "Strong" register = lagged revenue, expenses, assets, employees | A tax-linked register also carries lagged payroll and revenue composition (CRA T3010 lines, PD7A payroll; US: the t−1 return itself) | Add register scenario **REG-S+**. Headline y are those *not* on REG-S+. |
| 2 | One set of weights per y (model calibration per variable) | Production publishes hundreds of cells from **one** weight per unit | Add single-weight estimators (multi-y model calibration, single-λ ridge calibration). Measure damage to the other y. |
| 3 | Embedding, K and penalty grid chosen while looking at the current population | Specification is fixed from the previous cycle | Select on the **previous wave** (TY2021→TY2022). Freeze. Evaluate on TY2022→TY2023. |
| 4 | Take-all units assumed to respond | Only 12% of take-all units file within 300 days; production chases and imputes them | Take-all nonresponse with imputation (ratio or historical vs text-donor). |
| 5 | No outlier treatment | Winsorization of take-some outliers is standard (one take-some unit holds 16% of take-some volunteers) | Winsorized versions of the baseline *and* text arms. |
| 6 | Result = relative efficiency (RE) vs HT or GREG | A decision needs sample size, cost and CV-band movement for the *binding* variable | Cost curve (baseline at six n), binding-constraint sample size, CV-band tables. |
| 7 | Continuing units only, every unit has a t−1 return | Births, deaths and stale lags exist at design time | Coverage arm **U-full**, plus a "no current lag" group with name-only text. |
| 8 | Ceiling diagnostic and Monte Carlo use different model classes | The pilot Monte Carlo (gov. grants 1.4×) exceeded its own ceiling (1.13×) | Same-class ceilings. A Monte Carlo RE above its ceiling is a defect to explain, not a result. |

**A check run for this draft** (Appendix B; anticipated-variance ceiling, n = 1,000, LSA full text) shows why change 1 matters:
- Once lagged salaries, contributions and program revenue are on the frame (REG-S+), text adds **nothing** for those variables: 0.97, 1.03 and 0.95. Under REG-S the pilot had reported salaries 1.41.
- Text still adds 1.20 for government grants, 1.16 for fundraising expenses and 1.14–1.17 for two binary indicators.
- The pre-registered headline set has a ceiling geometric mean of **1.12** with LSA full text and **1.05** with MiniLM mission. That is right at the decision threshold of 1.10 used in §9, *before* any sample-fitting loss.

The honest prior is therefore: material gains in weak-register settings, marginal gains in a good tax-linked register, and nothing once the lagged value of the same variable is available. The benchmark is built to confirm or overturn that prior cleanly.

---

## 1. Research questions and estimands

### 1.1 Research questions

| RQ | Question | Primary output |
|---|---|---|
| RQ1 | Under each realistic register scenario, how much do t−1 text auxiliaries reduce the design MSE of totals for survey-only variables, beyond the best text-free estimator an NSO could field? | RE vs the text-free twin and vs the production estimator, with MC 95% CI |
| RQ2 | Do the gains survive **production constraints**? These are a single weight set, take-all handling, winsorization, nonresponse, in-sample-only tuning and previous-wave freezing. | RE of single-weight / frozen versions; weight diagnostics |
| RQ3 | Is inference still valid? | Coverage and variance-estimator bias under linearization and bootstrap |
| RQ4 | Where else does text help: stratification, nonresponse adjustment or imputation, small areas? | Design RE, residual nonresponse bias, domain ARRMSE and publishable-cell counts |
| RQ5 | How much of any gain needs a modern embedding? Compare cheap lexical features (TF-IDF/LSA), a small local model (MiniLM), an API model (Cohere text) and multimodal input (Cohere image or fused). | Gap closed per rung of the ladder |
| RQ6 | What does a gain buy in production? | Equivalent sample size, cost and burden saved, CV-band shifts |
| RQ7 | Does it replicate? | Same endpoints on a second population (national 990; CRA T3010) |

### 1.2 Estimands

All estimands are finite-population quantities of the TY2023 population U (§2.6). Truth is the full TY2023 file (pseudo-census).

- **Totals:** t_y = Σ_U y_k. These are primary, because NSO business tables are totals.
- **Means:** ȳ = t_y / N with N known from the frame. Under nonresponse, also report the Hájek form t̂_y / N̂.
- **Ratios of totals** (published as shares): R₁ = t_govgrants / t_revenue and R₂ = t_fundraising / t_expenses. Linearized as for any ratio.
- **Proportions:** p = t_b / N for binary b.
- **Domain totals and proportions** for SAE: state × NTEE10 (66 domains in New England, about 560 nationally); state; NTEE10.

### 1.3 Target variables: taxonomy and roles

Classes:
- **R** = register-type: on the frame with a lag under REG-S or REG-S+. In production it would be tax-replaced, not surveyed.
- **S** = survey-only continuous.
- **B** = binary.
- **H** = heavy-tailed stress.

| y (code) | Class | Lag on frame in scenario | Role | Why it is in |
|---|---|---|---|---|
| y_expenses, y_revenue | R | S, S+, P | **Negative control** (text gain should be about 1.00) | Corr. with lag 0.97–1.00 |
| y_employees | R | S, S+, P | Negative control | PD7A-type counts |
| y_salaries, y_contrib, y_prog_rev | R under S+, S under S | S+, P | **Scenario-sensitivity** variables | Pilot "gains" that vanish under S+ (App. B) |
| y_gov_grants | S | P only | **Headline** | Largest surviving text ceiling (1.20) |
| y_fundraising_exp | S | P only | **Headline** | 1.16 |
| y_officer_comp | S | P only | **Headline** | 1.04 (text weak; oracle 9.7) |
| y_volunteers_cap10k | S/H | P only | **Headline** | Typical "satellite account" variable; 41% zero or blank |
| y_bin_has_gov_grants | B | P (derived) | **Headline** | 1.14 |
| y_bin_grants_to_indiv_us | B | none | **Headline** | 1.17 |
| y_bin_lobbying, y_bin_has_volunteers, y_bin_foreign_activities | B | none | Secondary | Rare (4–6%) or common (59%) |
| y_grants_paid | H | P (T3010 gifts to donees) | Stress | Top 1% holds 89% |
| y_volunteers (raw) | H | P | Stress | One take-some unit holds 16% of the take-some total |
| y_placebo (within-stratum permutation of y_gov_grants) | — | — | **Placebo** | Text cannot know it |

**Headline set H** (pre-registered, REG-S+): {gov_grants, fundraising_exp, officer_comp, volunteers_cap10k, bin_has_gov_grants, bin_grants_to_indiv_us}.
**All-y set A**: H ∪ R ∪ {salaries, contrib, prog_rev, lobbying, has_volunteers, foreign_activities}, used for multipurpose damage.

**Canadian mapping caveat.** For *registered charities*, CRA's T3010 carries government revenue by level, compensation and fundraising lines **[verify line list against the CRA Open Data Dictionary v2.0]**. So for charities, government grants would be REG-P, not survey-only. The scenario where the headline set really is survey-only is **non-charity non-profits**. These file at most a T1044, or nothing, so the NSO sees only BR information **[verify T1044 thresholds]**. §2.4 makes this explicit, because it decides where a production use could actually exist.

---

## 2. Real-world scenario specification

### 2.1 The survey being simulated

An NSO runs an **Annual Survey of Non-Profit Organizations** for reference year t = TY2023. Inputs and outputs:
- **Purpose:** the non-profit satellite account and sector tables. It produces totals and shares nationally, by state (province) and by activity group (NTEE10). It is a multipurpose survey with about 15 published variables.
- **Frame:** the register as known at the design date, at reference t−1.
- **Collection:** by questionnaire, from fiscal-year end until a collection cutoff.
- **Estimation:** design-weighted, calibrated to frame totals, with imputation for take-all nonrespondents and weighting-class adjustment for take-some nonrespondents.

The 990 data stand in for both the register (t−1 return, BMF) and the questionnaire (t return). Every unit's t return is eventually observed, so the truth is known.

### 2.2 Timeline (dates are protocol parameters)

| Event | Date | Data used | Note |
|---|---|---|---|
| Register snapshot (codes) | BMF ≤ design date. Default: as built (2023-08-16). Sensitivity: 2023-12-12. | NTEE, subsection, ruling year, foundation, BMF income/asset codes | Must pass leakage test T12: share with REVENUE_AMT = y_t about 0 |
| Lagged admin and text | TY2022 return submitted **≤ design date D = 2024-06-30** (stress case: 2024-03-31) | Lagged amounts; mission and program text | 91% of TY2023 filers have one by 30 Jun 2024 (77% by 31 Mar) |
| Sample selection | D | Design D0–D2 | — |
| Collection window | FYE → FYE + 300 days | y_t for respondents | NR-LATE mechanism (§7.2) |
| Estimation | After collection | Respondent y; frame totals; previous-wave model | — |
| Truth | Full TY2023 file | All y_t | Pseudo-census |
| Previous wave (for freezing and selection) | Frame TY2021 return → y TY2022 | Same pipeline, one year earlier | NE TY2021 download in progress |

### 2.3 What is known when

| Information | Design time | Collection time | Estimation time | Never (truth only) |
|---|---|---|---|---|
| BMF codes, state, ruling year | ✔ | ✔ | ✔ | |
| Lagged size (REG-S) and lagged composition (REG-S+) | ✔ | ✔ | ✔ (and population totals for calibration) | |
| t−1 mission and program text, and their embeddings | ✔ | ✔ | ✔ | |
| Website URL (t−1 return) | ✔ | ✔ | ✔ | |
| Website content or screenshot | (crawl at design) | ✔ | ✔ | |
| Previous-wave **sample** y_{t−1} (survey-only variables) | ✔ for last year's sampled units only | | ✔ | |
| Previous-wave model m̂_{t−1} (trained on last year's sample) | ✔ | | ✔ | |
| y_t for respondents | | ✔ | ✔ | |
| Response indicator, filing date (paradata) | | ✔ | ✔ | |
| y_t for nonrespondents and non-sampled units | | | | ✔ |
| t-return text (same year) | | | | ✔ (allowed only in the leakage control) |
| BMF snapshots after D | | | | ✔ (allowed only in the leakage control) |

Rule: an auxiliary may enter an arm only at the stage where the table marks it. Every arm in the config carries a `stage_available` tag, and the harness asserts it.

### 2.4 Register scenarios

| Scenario | Frame content | Real-world analogue (Canada / US) | What text competes with |
|---|---|---|---|
| **REG-W** (weak) | NTEE10 (31% MISSING), subsection, state, ruling year, **BMF income- and asset-code classes** (no exact lagged amounts) | BR-only units: non-charity NPOs (T1044 or nothing), new registrants, sectors without tax linkage / BMF-only | Codes and size class |
| **REG-S** (pilot "strong") | REG-W + lagged revenue, expenses, assets, employees | BR with T2/T1044 summary financials and PD7A headcount | Lagged size |
| **REG-S+** (**primary**) | REG-S + lagged salaries, contributions and program revenue | BR with T3010 summary lines and PD7A payroll (charities); for US, the t−1 return summary | Lagged size and revenue composition |
| **REG-P** (positive control, and realistic for charities) | REG-S+ + lagged survey-only y (gov. grants, fundraising, officer comp., volunteers) | Full T3010 for charities. Also any unit in last year's sample (panel overlap). | The lagged value of the same y |

**Design follows the register.**
- Under REG-W, size classes and the take-all cut are built from BMF INCOME_AMT/ASSET_AMT. That design is **D1-W**.
- Under S, S+ and P they are built from lagged expenses.
- REG-W cells therefore do not share samples with the others. This is noted in every comparison.

### 2.5 Text sources and their real-world analogues

| Arm | Content | Coverage (NE pilot) | NSO analogue | Available |
|---|---|---|---|---|
| T0 | Organization name (BMF) | 100%, births included | BR legal or operating name | Design |
| T1 | Mission: Part I line 1 + Part III line 1, deduplicated, "see Schedule O" removed | 99.9% of continuing units | BN registration "major business activity" text (CRA RC1); SS-4 line 16–17 | Design |
| T2 | T1 + Part III 4a–4d program descriptions (whitelisted XPaths only) | 99.9% | T3010 Section C program descriptions (public); BR activity detail | Design |
| T2-nonum | T2 with digits and "$" amounts masked | same | Leakage ablation | Design |
| T1_t | Same-year (TY2023) mission | — | none (leak) | **Never**: leakage control only |
| W | Website text or screenshot | 93% have a URL in this population (85% bottom size quintile, 99% top). For all 990/EZ filers it is 42–88% by size. | Enterprise or NPO websites (ISTAT, ESSnet) | Collection; or design if crawled. **Needs network: user action.** |
| PDF | Annual report or accounts pages | Not available for 990 | UK CCEW accounts PDFs; financial statements attached to T3010 **[verify CRA publication]** | Replication R2 only |

Text is near-static: the TY2022 and TY2023 missions are identical for 84% of units. Text can therefore act only as a fine-grained, persistent classification, not as a signal of change.

### 2.6 Population, births, deaths and stale lags

- **U-cont** (primary, as the pilot): units with a full 990 for both TY2022 and TY2023. NE: N = 20,626. National: about 330k.
- **U-full** (coverage arm, Stage 2H):
  - The frame is units with a TY2022 return by D. Units that do not file at t are **out-of-scope deaths**: in the sample they carry y = 0 and they stay in the calibration totals.
  - Units filing at t with no TY2022 return by D are **births or stale lags**. They enter a separate take-some stratum sampled at the overall rate. They carry T0 (name) text only and REG-W variables, using TY2021 lags where those exist (flagged "stale").
  - Units absent from the frame at D (true births) are **undercoverage**. Their share of the TY2023 totals is reported as a fixed bias component. In NE, births are 6.4% of units but only 0.3–0.9% of totals (revenue 0.4%, gov. grants 0.9%).
- **Missing NTEE**: 31% of units (38% in the smallest size quintile, 26% in the largest). The status quo keeps MISSING as its own level. D1-imp replaces it with a text-coded NTEE (§3).
- **Reference periods**: fiscal years end in December (47%), June (33%) and September (8%). The estimand is "tax year 2023" as defined by the IRS. A December-FYE-only sensitivity run checks that FYE mix does not drive results (T08).

### 2.7 Nonresponse

The real nonresponse proxy is late filing:

  r_k = 1{TY2023 return submitted ≤ 300 days after FYE}.

- Overall it gives 60% response. By weight it is far lower: 18% expense-weighted, and **12% among take-all units** against 61% among take-some units. Large organizations file on extension.
- As a *survey* nonresponse analogue it is pessimistic for large units, because NSOs prioritise follow-up of take-all units. The protocol therefore handles take-all and take-some nonresponse differently (§7.2):
  - **take-all nonrespondents** are imputed, as production does;
  - **take-some nonrespondents** are reweighted, or imputed in the imputation arm.
- **Item nonresponse:** blank volunteers are currently coded 0. That conflates item nonresponse with true zeros (41% are zero or blank). Stage 0 separates blank from zero (T27). Volunteers estimands are then defined either on "reported volunteers, blanks imputed" or with blanks treated as item nonresponse.

### 2.8 Edits and outliers (applied identically to every arm)

- **Edits:** the narrow edit rules E1–E3 (`build_population.py::apply_edits`) are applied before anything else, and the edit log is part of the population card.
- **Influential values** are kept in the truth.
- **Winsorization (production estimators):** one-sided type-II winsorization of take-some residuals is offered as an option for *all* estimators. It follows Kokic & Bell / Chambers. The cutoffs K_{h,y} are set from the **previous wave** (frozen), never from the current sample.
- Results are reported with and without winsorization. Text gains that appear only without winsorization are outlier artefacts.

---

## 3. Designs

All designs:
- stratified SRSWOR;
- take-all stratum = top 1% of units on the design size variable (206 units in NE, about 3,300 nationally);
- take-all units excluded from modelling and calibration, as in production: T̂ = Σ_TA y_k + T̂_TS;
- compromise allocation: Neyman on √x with minimum n_h = 10 (`frame.production_design`, alloc_power = 0.5);
- equal number of take-some strata H when designs are compared (the **equal-H rule**), so that a "text design" cannot win simply by having more strata.

| Code | Strata | Size variable | Purpose | Notes |
|---|---|---|---|---|
| **D0** | none (SRSWOR) | — | Textbook reference | Shows what the design alone buys |
| **D1** | NTEE10 (incl. MISSING) × 3 size classes (cum-√f on log lagged expenses) + take-all | lagged expenses | **Status quo** | H = 33 + TA in NE. At n = 1,000, 26 strata sit at 10–19 units. |
| **D1-W** | NTEE10 × 3 BMF income-code classes + take-all on BMF INCOME_AMT | BMF INCOME_AMT | Status quo under REG-W | Required for REG-W cells |
| **D1-H44** | NTEE10 × 4 size classes + TA | lagged expenses | Equal-H comparator for D2a | — |
| **D1-imp** | D1 with MISSING NTEE replaced by text-coded NTEE (multinomial logit on E*, trained on the frame units with a known code; no y used) | lagged expenses | **Text as frame coder** (the most likely first deployment) | H = 30 + TA. Pilot coding accuracy 69–70% on known codes. Accuracy on the MISSING units cannot be measured (they are MNAR by age). |
| **D2a** | NTEE10 × 2 size classes × 2 classes of a **frozen text score** s_k + TA | lagged expenses | Prediction stratification | s_k = predicted government-grant intensity, P̂(gov grants > 0) from a model trained on the **previous-wave sample**. Split at the frame median within each NTEE × size cell. H = 44: compare with D1-H44. |
| **D2b** | k-means (K = 11) on E* replacing NTEE10 × 3 size classes + TA | lagged expenses | "Text industry" (unsupervised) | H = 33 = D1. An expected-null control for y not tied to activity. |
| D1-B (optional) | D1 strata, Bethel multivariate allocation for target CVs on H, using previous-wave S_h | — | Production multipurpose allocation | Only if Stage 2 shows design-stage gains |

**Sample sizes:**
- **NE (N = 20,626):** primary n = 1,000 (take-some f = 3.9%). The cost curve uses the baseline at n ∈ {750, 875, 1,000, 1,250, 1,500, 2,000}, and the text arms at {750, 1,000, 2,000}. The pilot's n = 500 is **dropped**: 33 × 10 = 330 > 294 take-some units, so the minimum cannot be met and the allocation degenerates.
- **National (N ≈ 330k):** n = 10,000 (f ≈ 2–3%). Strata: NTEE10 × 4 size classes × 4 census regions; state enters as a calibration margin.

**Stratum jumpers (reported for every design):**
- 8.4% of take-some units change size class between t−1 and t;
- 20 take-some units reach take-all size at t, holding 1.3% of expenses.

The protocol records each design's share of MSE due to jumpers (T24). This matters for D2a, because classifying on a noisy score *creates* jumpers.

---

## 4. Estimators and baselines

### 4.1 Auxiliary matrices (by register)

Existing names in `experiments.build_mats` are kept. A register suffix is added where the scenario changes them.

| Matrix | Columns | Register |
|---|---|---|
| `strata` | stratum dummies | all |
| `weak` | strata + NTEE10 + state + BMF income-code dummies | W |
| `size` / `size_cat` | strata + 4 lagged size vars (+ NTEE10 + state) | S |
| `size_catP` (new) | `size_cat` + lag_salaries, lag_contrib, lag_prog_rev | S+ |
| `size_catO` (new) | `size_catP` + oracle_lag_* (and lag indicators for binaries) | P |
| `ml_frame_ratio[@reg]` | NTEE10 + state + income code + log1p(lags available in reg) | per reg |
| `mlr+E[@reg]` | `ml_frame_ratio[@reg]` + 32 PCs of E | per reg |
| `size_cat[@reg]+E:ratio32` | `size_cat[@reg]` + PC×lag_expenses + PC×lag_employees | per reg |

### 4.2 Estimator list

Role key:
- **SQ** = status quo (what production runs now);
- **SQ-ML** = text-free machine learning, the *fair twin* of the text arm;
- **TXT** = text;
- **CTL** = control.

| Name | Kind | Model / matrix | Role | Single weight? | Variance |
|---|---|---|---|---|---|
| HT | ht | — | SQ (floor) | yes | linearization |
| GREG_ratio | greg | strata + lag_expenses | SQ | yes | linearization (residual) |
| GREG_size_cat[@reg] | greg | `size_cat@reg` | **SQ: production baseline** | yes | linearization (residual and g-weighted) |
| GREG_size_cat[@reg]-WIN | greg + winsorization | same | SQ (production with outlier treatment) | yes | linearization |
| MCAL_frame_ratio[@reg] | mcal, ratio scale, cross-fitted | `ml_frame_ratio@reg` | **SQ-ML twin** | no (per y) | linearization on cross-fitted residuals |
| MCALm_frame_ratio[@reg] (new) | multi-y model calibration on [X_base, m̂_1..m̂_K], K = 4 key y | same features | SQ-ML twin, single weight | **yes** | linearization |
| MCAL_ratio[@reg]+E | mcal, ratio scale, cross-fitted, grouped folds | `mlr+E@reg` | **TXT primary** | no | linearization; bootstrap (subset) |
| MCALm_ratio[@reg]+E (new) | multi-y model calibration, K = 4 | `mlr+E@reg` | **TXT production primary** | **yes** | linearization; bootstrap (subset) |
| RIDGE_size_cat[@reg]+E:ratio32 | ridge calibration: hard on `size_cat`, soft on PC × size, λ by in-sample CV per y | — | TXT | no | linearization |
| RIDGE1W_size_cat[@reg]+E:ratio32 (new) | ridge calibration with **one** λ chosen by mean CV loss over H | — | TXT single weight | **yes** | linearization |
| GREG_size_cat[@reg]+E:add10 | OLS-GREG + 10 PCs | — | TXT (simple, transparent) | yes | linearization |
| GREG_weak+E:add32 | OLS-GREG, weak + 32 PCs | — | TXT under REG-W; overfitting stress | yes | linearization |
| FROZEN_diff[@reg]+E (exists as kind `frozen`) | difference estimator, m̂ trained on the previous-wave sample | — | TXT deployment | no (per y) | exact for fixed m |
| FROZEN_cal[@reg]+E | calibration on [X_base, m̂_frozen] | — | TXT deployment | no (or multi-y: yes) | linearization |
| MCAL_ratio+E:nocf | mcal without cross-fitting | — | CTL (overfitting demo) | no | linearization |
| SYN_ratio+E | Σ_TA y + Σ_TS m̂ | — | CTL (model-based replacement) | — | none |
| (each TXT arm) with E ∈ {random384, PERM_E*, placebo} | — | — | CTL (negative) | — | — |

The primary and production estimators are pre-registered:

- **Primary text estimator:** MCAL_ratio[S+]+E*.
- **Primary comparator:** MCAL_frame_ratio[S+]. It is the same learner, features, folds and tuning, minus text. Their ratio isolates the text increment.
- **Production estimator:** MCALm_ratio[S+]+E* (single weight).
- **Production comparator:** GREG_size_cat[S+], the system in place today. Their ratio answers "does changing production pay?". Both comparisons are reported. A recommendation requires both to pass (§9).
- **E\*** is the text arm chosen on the **previous wave** by the rule in §6.3. If the TY2021 data are not ready, the fallback is E* = MiniLM on T2. That arm has not yet been run in any Monte Carlo, so choosing it is not contaminated by pilot results. LSA T2 was the best pilot arm and is therefore barred from being primary by fallback.

### 4.3 Tuning rules (non-negotiable)

1. **Never tune on population y.** λ is chosen by K = 5 in-sample CV, weighted by d_k (ratio models: d_k x_k), and **per replicate** (as in `mc._ridge_cv_lambda`). A production-fidelity variant fixes λ from the previous wave (`lam_policy: frozen`). Both are run on the primary cell.
2. **Grouped folds:** units with identical normalized mission text share a fold (`experiments.text_groups`; 4.8% of units are near-duplicates). Stage 0 also adds a cosine > 0.95 near-duplicate clustering on E* (T19). The fold assignment is identical for the twin and the text arm.
3. **PCs:** principal components of E are computed on the frame population (no y involved). For frozen models the PCA rotation is fitted on the *training* wave's frame and reused.
4. **K (number of PCs) and the feature map** (ratio vs raw) are fixed from the previous wave: K ∈ {8, 16, 32, 64}. They are never chosen per replicate or on current-wave results.
5. **Ratio-scale models** use offset x_k + 1 and weights d_k x_k (pilot). This is fixed.
6. **Everything that is fitted** (folds, λ, model) is refitted inside each replicate and inside each bootstrap replicate.

### 4.4 Single-weight (multipurpose) estimators

Production needs one weight w_k per responding unit, satisfying:
- the calibration constraints Σ_s w_k x_k = Σ_U x_k (on X_base = `size_cat@reg`);
- Σ_s w_k m̂_j(k) = Σ_U m̂_j(k), for j = 1..K key variables (K = 4: gov_grants, fundraising_exp, volunteers_cap10k, has_gov_grants).

The m̂_j are cross-fitted ratio-scale models (`mcal`). Weights use chi-square distance with bounds L = 0.2 and U = 5 on w_k/d_k (truncated linear; flag non-convergence). The same w_k is applied to **every** y in set A. Reported:
- RE for H (gain);
- RE for R and the other y (damage);
- the weight diagnostics in §8.7.

### 4.5 Variance estimators

| Code | Formula | Used for |
|---|---|---|
| v_L (exists) | Σ_h N_h²(1 − f_h) s²_{e,h}/n_h, with e = y − x′B̂ (GREG) or e = y − m̂^{cf} (MCAL), on take-some strata | All estimators |
| v_g (new) | Same with g-weighted residuals g_k e_k, g_k = w_k/d_k | Production calibration estimators. This is the StatCan G-Est style. |
| v_R (exists in mc) | v_L with respondent counts r_h in place of n_h | Nonresponse cells |
| v_B (new) | Rao–Wu–Yue rescaling bootstrap, B = 500, with **full re-estimation** (λ, folds, model, calibration) in every bootstrap replicate | Primary and production estimators, on the first 200 MC replicates of the primary cell |
| Frozen | v_L with m fixed (exact for the difference estimator) | FROZEN_diff |

Coverage uses Wald intervals θ̂ ± 1.96 √v̂. For proportions with p < 0.1, also report logit-transformed intervals.

---

## 5. Controls

| Control | Implementation | Expected | If violated |
|---|---|---|---|
| **N1 random embedding** | Gaussian 384-d (`embed_random`), through every TXT arm | RE ∈ [0.97, 1.03]; coverage as the twin | The pipeline extracts "signal" from noise: overfitting or a fold leak |
| **N2 permuted embedding** | E* rows permuted across units | Same as N1 | As above; also catches distribution-only effects |
| **N3 placebo y** | y_placebo = y_gov_grants permuted within strata; and the last digit of revenue | Text RE ≈ 1.00 for every arm | Leakage through design or strata, or a bug |
| **N4 register-type y** | Text arms on expenses and revenue under REG-S+ | RE ∈ [0.95, 1.05] | Leakage, or the frame is mis-specified |
| **P1 oracle lag** | REG-P: lagged same variable on the frame | RE vs S+ of 4–10× for gov. grants, fundraising, officer comp. (ceilings 5.1, 7.7, 9.7) | The harness cannot detect a known large gain |
| **P2 synthetic positive** | y_syn = m*(text) + noise with known R² (e.g. 0.3) planted into the population | Realized RE close to the theoretical (1 − R²_F)/(1 − R²_{F+E}) | Estimator or harness defect |
| **L1 same-year text** | T1_t (TY2023 mission) in place of T1 | RE(T1_t) − RE(T1) small (mission 84% identical) | A large difference means text changes track y changes: timing matters, so document it |
| **L2 numbers-masked** | T2-nonum vs T2 | Difference ≤ 0.03 in RE | The embedding reads amounts |
| **L3 late frame** | BMF 2026 snapshot as frame (INCOME_AMT reflects TY2023+) | Inflated gains for size y | Shows the magnitude of a realistic timing mistake (teaching control) |
| **O1 no cross-fit** | MCAL_ratio+E:nocf | Coverage 72–78% (pilot) | Must stay bad: confirms cross-fitting is necessary |
| **O2 OLS with many PCs** | GREG_weak+E:add32, add128 | Coverage collapse (pilot 38–55%) | Confirms the p/n warning |
| **S1 synthetic estimator** | SYN_ratio+E | Bias up to +15% on binaries (pilot) | Shows why the design correction is needed (role D vs role B) |
| **C1 ceiling consistency** | Monte Carlo RE vs the same-class anticipated-variance ceiling (Stage 1) | MC RE ≤ ceiling × 1.05 | Optimistic tuning, leakage, or a ceiling bug. **Blocking.** |

---

## 6. Embedding ladder and text-field arms

### 6.1 Ladder (one change per rung; same template, learners, folds and designs)

| Rung | Representation | Status | Answers |
|---|---|---|---|
| L0 | No text (twin) | ready | Baseline |
| L0-neg | random384, PERM_E*, empty string | ready | Does the machinery invent gains? |
| L1 | ~30 hand-built keyword flags + text length | to build | Can a methodologist without ML get the gain? |
| L2 | TF-IDF (1–2-grams) → SVD-256 (LSA) | ready (T1, T2) | Lexical signal; deterministic; no model risk |
| L3 | Averaged GloVe / spaCy static vectors | to build (wheels available) | Distributional semantics without context |
| L4 | MiniLM-L6 (384-d) | ready (T1), T2 to embed (~15 min NE) | Smallest contextual model, on-prem |
| L5 | Qwen3-Embedding-0.6B or BGE-M3 | **user action** (Hugging Face blocked) | Strong open weights: the on-prem alternative to Cohere |
| L6 | Cohere embed-v4.0 text, `classification`, 1536 float; sensitivity at 256 (Matryoshka) and with `clustering` | **user action** (api.cohere.com + production key) | Does a frontier API model add over L2–L5? |
| L6b | Cohere embed-v5.0 text | user action | Vendor-version bridging (reproducibility) |
| L7 | Cohere image / fused text+image on website screenshots (990) or accounts PDFs (CCEW) | user action + data | Does layout or visual content add beyond text? |
| L7-ext | **Structured extraction** of lagged amounts from the same PDFs (OCR / table parse) → REG-P-like auxiliaries | with L7 | The real competitor of a page-image embedding: the numbers on the page |

### 6.2 Comparisons and the question each answers

| Comparison | Question |
|---|---|
| L0-neg vs L0 | Is the harness honest? (must be ≈ 1) |
| L2 vs L0 | Is there any text signal beyond the register? |
| L4 vs L2 | Does a contextual model beat bag-of-words? (The skeptic's bet is no: sector and size words are lexical.) |
| L6 vs max(L2, L4, L5) | Is an external API justified? Threshold: +0.05 in GM-RE. |
| L5 vs L6 | Can production stay on-prem? (open within 0.05 of Cohere) |
| T2 vs T1 | Is the gain in the program descriptions (est. 70 words) or in the short activity text (est. 20 words, the BN analogue)? |
| T0 vs T1 | Is a name enough? (relevant for births) |
| T2-nonum vs T2 | Numeric leakage |
| L7 vs L6 on the same units | Multimodal increment |
| L7 vs L7-ext | Embedding of a page vs simply reading its numbers |
| Matched dimension: all rungs projected to K ∈ {16, 64} PCs | Separates "better semantics" from "more dimensions" |

### 6.3 Selection rule for E* (pre-registered)

On the **previous-wave** population (TY2021 frame → TY2022 y), compute the same-class Stage-1 ceiling for every local rung × field × K. Then:

- E* = argmax over (rung, field, K) of GM over H of the ceiling RE vs the twin.
- Ties within 0.02 go to the simpler rung (L2 < L4).
- The choice is frozen in `config/protocol_v1.yaml` before any current-wave Monte Carlo is run.
- When L5 or L6 become available later, they are evaluated as **secondary** arms, not re-selected as primary. Otherwise the primary endpoint would be chosen after seeing current-wave results.

---

## 7. Uses beyond calibration

### 7.1 Stratification (designs D1-imp, D2a, D2b)

- **Estimators:** each design is run with HT, GREG_size_cat[S+] and MCAL_frame_ratio[S+]. The text designs are *not* combined with text estimators in the primary comparison: the design gain is isolated first, then the combination D2a × MCAL_ratio+E* is run.
- **Metrics:**
  - RE(D_x vs D1, or vs D1-H44 for D2a) per y, and GM over A;
  - share of MSE from stratum jumpers;
  - realized vs predicted RE under 1 − ρ²(1 − 1/L²) using population ρ(s_k, y).
- **Production reading:** if D1-imp alone captures most of the weak-register text gain, the realistic deployment is "text-assisted coding of the register". That is a frame-maintenance change, not an estimation change, and much easier to govern.

### 7.2 Nonresponse adjustment and imputation

**Mechanisms** (take-some; take-all handled separately below):

| Code | Mechanism | Target response rate | Text "should" help? |
|---|---|---|---|
| NR0 | MCAR | 65% | no (sanity) |
| NR1 | logit p = a + b·log(1 + lag_exp) + NTEE effects | 65% | no |
| NR2 | logit p = a + γ·u_k, where u_k is a standardized text score (frozen previous-wave model of gov-grant intensity) | 65% | **yes, by construction** |
| NR3 | logit p = a + δ·log(1 + y_gov_grants,t) (MNAR) | 65% | partly, only through correlation with y |
| **NR-LATE** | real: filed ≤ 300 days after FYE | 61% take-some | unknown: **the real test** |
| NR4 | p depends on website presence (missing-text units respond less) | 65% | tests missing-not-at-random auxiliaries |

δ and γ are set so that the respondent/nonrespondent ratio of mean gov. grants is about 0.35, matching NR-LATE.

**Take-all nonresponse (production realism):** under NR-LATE, 88% of take-all units are late. Two scenarios:
- **TA-follow-up:** take-all respond MCAR at 90% (intensive follow-up). Nonrespondents are imputed.
- **TA-late:** the real late-filing indicator.

Imputation methods for take-all nonrespondents:
- **TA-hist (SQ):** y_k = lag_x_k × (Σ_r y / Σ_r lag_x) within NTEE10 among responding take-all and top take-some units, using the matching lag (lag_expenses, or the REG-S+ composition lag).
- **TA-nn-text (TXT):** nearest-neighbour donor among respondents, matching on [log lag size, E* PCs] (Banff-style).
- **TA-nn-frame (SQ-ML):** the same donor method matching on [log lag size, NTEE, state].

**Take-some adjustments:**

| Code | Adjustment | Role |
|---|---|---|
| NR_naive_HT | none | CTL |
| NR_rhg_* | response homogeneity groups = design strata (exists) | **SQ** |
| NR_psc_frame (new) | 5 propensity-score classes, logistic on REG frame (Haziza–Beaumont score method) | SQ-ML |
| NR_psc_text (new) | 5 propensity classes, logistic on frame + E* PCs | TXT |
| NR_pmc_text (new) | 5 **predictive-mean classes** on cross-fitted m̂(frame, E*) for a key y (Little–Vartivarian: predict y, not only response) | TXT |
| NR_rhg_MCAL_ratio+E (exists) | RHG then model calibration with text | TXT |
| NR_ps_text_* (exists) | continuous propensity then calibration (two-step, Haziza–Lesage) | TXT |
| NR_imp_nn_text (new) | donor imputation for take-some nonrespondents, matching on [size, E*] | TXT (production imputation) |

**Metrics:**
- relative bias;
- variance;
- RRMSE;
- **share of residual bias removed**, (|RB_SQ| − |RB_x|)/|RB_SQ|, with MC CI;
- the Kish adjustment factor 1 + CV²(adjustment factors);
- coverage.

Results are shown as a Little–Vartivarian 2 × 2: covariate related to response (yes/no) × related to y (yes/no), using NR0–NR3.

**Pilot facts to beat:** production nonresponse adjustment plus calibration leaves +7% to +12% bias on contributions, fundraising and government grants. Text removed 30–60% of it, and propensity-with-text did not help.

### 7.3 Small area estimation

**Domains:**
- state × NTEE10 (NE: 66 domains, median N_d = 145, 17 with N_d < 30; expected n_d at n = 1,000 has median 8, and 41% of domains have < 5);
- national: state (51) and state × NTEE10 (about 560).

**Estimands:** domain totals of gov_grants and salaries; domain proportions of has_gov_grants and grants_to_indiv_us.

**Estimators:**

| Code | Estimator | Role |
|---|---|---|
| DIR_HT | direct HT | floor |
| DIR_GREG[S+] | domain estimate from the national GREG_size_cat[S+] weights | **SQ** (production publishes these) |
| DIR_MCALm+E* | domain estimate from the single-weight text calibration | TXT |
| SYN_frame / SYN_text | Σ_{U_d} m̂(k) | CTL (bias) |
| FH_frame | Fay–Herriot on the DIR_GREG direct estimates. Area covariates: log N_d, Σ_{U_d} lag_exp, share of take-all, share c3 (and Σ_{U_d} lag composition under S+). Sampling variances smoothed by a generalized variance function, log ψ_d = a + b log n_d + c log N_d, fitted per replicate. | **SQ-SAE** |
| FH_text | FH_frame + one text covariate: Σ_{U_d} m̂_frozen(k) (previous-wave text model; a scalar per domain, so p ≪ m) | TXT |
| FH_textPC | FH_frame + area means of the first 3 PCs of E* | TXT (stress: more covariates) |
| BHF (optional) | unit-level nested-error on the ratio scale y/x | later |

**Metrics:**
- per-domain RB and RRMSE over replicates;
- AARB = (1/D) Σ_d |RB_d| and ARRMSE = (1/D) Σ_d RRMSE_d (domains with θ_d = 0 are excluded and counted);
- distribution of per-domain RE;
- FH interval coverage with the Prasad–Rao MSE;
- the **number of domains with CV ≤ 25%** (bands A–D, publishable).

All of these are reported by n_d class (0, 1–4, 5–19, ≥ 20). Non-sampled domains are reported separately.

**Text aggregates have no sampling error,** because they are computed on all frame units. That is an advantage over web-scraped area covariates, and the Ybarra–Lohr concern does not apply. Model error in m̂_frozen across domains can still create **domain-level bias**, so report per-domain RB for FH_text vs FH_frame.

### 7.4 Frozen / previous-wave deployment (the production pattern)

Each year the NSO has last cycle's *sample* respondents with y_{t−1}, and frame_{t−2} plus text_{t−2}. It trains a head on them and applies it to frame_{t−1} plus text_{t−1} for all of U_t.

Simulation:
1. Draw J = 20 independent D1 samples (n = 1,000) from the previous-wave population (TY2021 frame → TY2022 y).
2. On each, fit the ratio-scale ridge head (frame + E PCs; PCA fitted on the TY2021 frame), with λ chosen by in-sample CV.
3. Apply the head to the TY2022 frame and text, giving m̂_j(k) for U_t.
4. Current-wave Monte Carlo replicate r uses head j = r mod J. The difference estimator is then exactly design-unbiased given the head, and the spread across j measures training-sample variability.
5. Also fit a **census-trained** head on the whole previous-wave population. This is realistic only where y_{t−1} is administrative (REG-P), and it gives an upper bound.
6. Report **model ageing**: the ceiling R² of the frozen head on the wave it was trained for vs the next wave.

---

## 8. Metrics and Monte Carlo protocol

### 8.1 Point and interval metrics

For estimator a, variable y, replicates r = 1..R, estimates θ̂_r, variance estimates v̂_r, truth θ:

- Relative bias: RB = (θ̄̂ − θ)/θ, with MC SE = sd(θ̂_r)/(√R |θ|).
- MSE = R⁻¹ Σ_r (θ̂_r − θ)²; RRMSE = √MSE / |θ|.
- Relative efficiency on paired replicates: RE(a | b) = MSE_b / MSE_a. The MC SE uses the delta method on log RE (implemented in `mc.metrics`):
  Var(log RE) ≈ R⁻¹ [Var(e_a²)/MSE_a² + Var(e_b²)/MSE_b² − 2 Cov(e_a², e_b²)/(MSE_a MSE_b)].
- Coverage: C = R⁻¹ Σ_r 1{|θ̂_r − θ| ≤ 1.96 √v̂_r}, with MC SE √(C(1 − C)/R).
- Variance-estimator relative bias: (v̄̂ − MSE)/MSE. Also report (v̄̂ − Var)/Var, to separate the bias² term.
- Mean relative CI width: 2 × 1.96 × mean √v̂_r / |θ|.
- Estimated CV per replicate: CV̂_r = √v̂_r / θ̂_r. Report its distribution and the band shares (§8.6).
- Failure rate: share of replicates with NaN, non-convergent calibration, or out-of-bounds weights. **> 0.5% is a defect.**

### 8.2 Aggregating across variables

- **GM-RE over a set Y** = exp(|Y|⁻¹ Σ_{y∈Y} log RE_y). Its MC CI comes from a bootstrap over replicate indices (B = 2,000; resampling r keeps the cross-y correlation within a replicate).
- **Worst-case RE over A** (multipurpose risk): min_y RE_y, with its own CI.
- **Win/loss count:** number of y with lower 95% MC bound > 1.05 (win), or upper bound < 0.95 (loss).
- **Binding-variable RE:** the RE for the variable that determines the sample size (§8.5).

### 8.3 Replicates, seeds and common random numbers

- **R:** screening 200; standard 1,000 (coverage MC SE 0.69 pp); **confirmatory primary cell 2,000** (0.49 pp). Bootstrap variance check on 200 replicates × B = 500.
- **Common random numbers:** within a cell, every estimator and every embedding arm sees the same R samples. Register scenarios S, S+ and P share D1 samples. W uses D1-W, so it is not shared.
- **Seeds:** the sample for replicate r comes from `default_rng([cell_seed, r])`. Estimator-internal randomness (folds, CV) comes from `default_rng([cell_seed, r, hash(estimator_name)])`.
- **Code change:** the current `mc.run_cell` passes one rng to all estimators in sequence. Adding an estimator therefore changes the folds of later estimators. Fix this before Stage 2 so that results are invariant to the estimator list.
- **Population randomness:** the population is fixed, so MC CIs quantify sampling variability *for this population only*. Generalization comes from replication (§10.2), not from larger R.

### 8.4 Ceiling consistency

For each Monte Carlo estimator, Stage 1 computes the **same-class** anticipated-variance ceiling: the same features, ratio scale and learner, fitted on the population with population cross-fitting. Report:
- the ratio MC-RE / ceiling-RE (expected 0.7–1.0 at n = 1,000);
- the "fitting loss" 1 − MC-RE/ceiling-RE as a function of n. This is itself a result: how much of the potential gain a sample of size n can realize.

### 8.5 Equivalent sample size, binding constraint and cost

- **Cost curve:** run the production baseline at n ∈ {750, 875, 1,000, 1,250, 1,500, 2,000}. Fit log MSE_y(n) by monotone interpolation in log n.
- **Equivalent n:** the text arm at n has n_eq,y = MSE_base,y⁻¹(MSE_text,y(n)).
- **Take-some saving:** σ_y = 1 − (n − n_TA)/(n_eq,y − n_TA). Take-all is fixed. Without the fpc, σ ≈ 1 − 1/RE.
- **Binding constraint:** the hypothetical survey has pre-registered CV targets, for example:
  - national totals CV ≤ 5% for gov. grants and fundraising, ≤ 10% for volunteers;
  - national proportions CV ≤ 5%;
  - state totals CV ≤ 15%.
  Using the Stage-1 anticipated-variance functions (rescaled by MC/ceiling ratios), compute the minimum n meeting *all* targets under the status quo and under the text system. The **saving is in that minimum n**, not in the y with the best RE.
- **Cost:** ΔC = Δn × c_u + Δn × b × c_b − C_text, where:
  - c_u = variable cost per sampled unit (collection, follow-up, editing). Default **CAD 200 [ASSUMPTION: replace with the program's figure]**.
  - b = respondent burden hours per questionnaire (default 2 h [ASSUMPTION]), with c_b a burden cost.
  - C_text = annual fixed cost of the text pipeline (default 0.3 FTE methodologist/IT plus compute; on-prem MiniLM about 0; Cohere API about $2–5 per national pass).
  - Report the break-even Δn.

### 8.6 CV quality bands

The common StatCan scale is A < 5%, B 5–9.99%, C 10–14.99%, D 15–24.99%, E 25–34.99% (use with caution) and F ≥ 35% (suppressed). **[Confirm with the target program's own quality guideline.]**

For national, state and state × NTEE estimates, report the distribution of bands under the status quo and under the text system. Use the band of the *true* CV (√MSE/θ) and, separately, of the *estimated* CV.

Rule of thumb printed on every table: **RE = 1.2 shrinks CVs by only 1 − 1/√1.2 = 8.7%.** So band changes are rare at national level, and the publishing case lives in the domains.

### 8.7 Weight diagnostics (single-weight estimators)

- share of negative weights; share with w < 1;
- range and quantiles of g = w/d;
- CV of the weights;
- Kish deff_w = n Σ w² / (Σ w)²;
- share of replicates hitting the bounds (L, U);
- max change in any published *register-type* total vs the production weights (they should be identical under exact calibration).

### 8.8 Standard result tables (every cell)

1. RE vs the twin and vs production per y, with MC CI.
2. RB, RRMSE and coverage per y.
3. GM-RE over H and worst-case over A.
4. Weight diagnostics.
5. Ceiling-consistency ratio.
6. Timing.

Each table carries the cell manifest: population hash, design, register, n, R, seeds, config hash, git commit and embedding hashes.

---

## 9. Pre-registered hypotheses, thresholds and decision rules

### 9.1 Decision vocabulary

Decisions use effect-size thresholds with MC confidence bounds, not p-values. With common random numbers and R = 2,000, tiny differences would be "significant".

- **Material:** lower 95% MC bound of RE ≥ 1.10, which is about a 9% take-some sample saving.
- **Production-worthy:** point GM-RE ≥ 1.20 (about 17% saving) **and** lower bound ≥ 1.10.
- **Negligible:** CI inside [0.95, 1.05].
- **Harm:** upper bound < 0.95.
- **Inconclusive:** anything else. Increase R once (to 5,000); if still inconclusive, report it as such.

### 9.2 Hypotheses

Skeptic's prediction = what the user's skeptical prior implies. Ceiling = the Stage-1 value already observed (Appendix B; NE, n = 1,000, LSA T2 unless noted). The ceilings are *optimistic* and must be recomputed same-class on the previous wave (§6.3).

| ID | Hypothesis (cell) | Metric | Pass threshold | Skeptic predicts | Current ceiling |
|---|---|---|---|---|---|
| **P1** (primary: efficacy) | Text improves survey-only y under REG-S+ (D1, n = 1,000) | GM-RE over H, MCAL_ratio[S+]+E* vs MCAL_frame_ratio[S+] | Material | 1.00–1.08 | 1.12 (LSA T2), 1.05 (MiniLM T1) |
| **P2** (primary: production) | Single-weight text system beats production | GM-RE over H, MCALm_ratio[S+]+E* vs GREG_size_cat[S+]; **and** worst-case over A ≥ 0.95; **and** coverage ≥ 93% on all of A | Material + no harm + valid | 1.0–1.15 overall. It may still pass through the ratio-scale ML, not the text, so P1 and P2 are read together. | n/a |
| S1 | No gain on register-type y | RE on expenses and revenue (REG-S+) | Negligible | 1.00 | 0.99 |
| S2 | Text helps the weak register | RE of GREG_weak+E / MCAL_ratio[W]+E* vs the REG-W twin, on expenses and salaries | Production-worthy | Yes (~2×) but irrelevant where admin data exist | expenses 1.9×, salaries 2.3× (REG-S pilot ceilings) |
| S2b | Weak + text vs the strong register | share of the log-efficiency gap REG-W → REG-S+ closed by text | report; ≥ 0.5 = "text substitutes for admin" | ≤ 0.5 | vs REG-S (pilot): expenses 0.46 (weak 0.23 → weak + text 0.45), salaries 0.84 (0.37 → 0.85). Vs REG-S+ the salaries share collapses, because S+ is 15.7× REG-S. |
| S3 | Validity | coverage of every cross-fitted TXT arm on every y in A | ≥ 93.0% (R = 2,000); variance relative bias in [−10%, +10%]; bootstrap within ±5 pp of nominal | holds with cross-fitting | pilot 88–95% (R = 60) |
| S4 | Text-coded NTEE improves the design | GM-RE over A, D1-imp vs D1 (HT and GREG_size_cat) | ≥ 1.05 | ≤ 1.03 | — |
| S5 | Prediction strata help | GM-RE over H, D2a vs D1-H44 | Material | ≤ 1.05 (strata are capped by ρ²) | — |
| S6 | Text reduces nonresponse bias (NR-LATE) | share of residual bias removed vs NR_rhg_GREG_size_cat[S+], on gov_grants, fundraising, contrib, volunteers_cap10k, lobbying | ≥ 30% on ≥ 3 of 5 y with RRMSE not worse | < 30% | pilot 30–60% (R = 40) |
| S7 | SAE | ARRMSE reduction of FH_text vs FH_frame on state × NTEE10, gov_grants and has_gov_grants | ≥ 10%; publishable-domain share +5 pp | < 10% | — |
| S8 | Frozen deployment works | RE of FROZEN_cal+E* vs MCAL_ratio+E* | ≥ 0.90, with coverage ≥ 94% | ≥ 0.9 (it is the honest version) | — |
| S9 | Modern embedding needed | GM-RE(L6 or L4) − GM-RE(L2), on the log scale | ≥ log 1.05 to justify the API | < log 1.05 | MiniLM T1 < LSA T2 so far |
| S10 | On-prem suffices | GM-RE(L6) − GM-RE(L5) | ≤ log 1.05 | yes | — |
| S11 | Multimodal adds (R2 only) | GM-RE(L7 fused) vs L6 text on the same units; and L7 vs L7-ext | ≥ 1.05 over text; and L7 ≥ L7-ext | L7-ext ≫ L7 | — |
| S12 | Replication | P1 and P2 on the national 990 and on CRA T3010 | same verdict category | weaker or null | — |
| C | All controls (§5) | as specified | all pass | — | — |

### 9.3 Multiplicity

- There are two primary endpoints (P1, P2), each a single pre-specified estimator, comparator, y set, cell and E*. **Holm at α = 0.05** is applied to their one-sided "material" tests (lower bound ≥ 1.10). Equivalently, use 97.5% bounds for the first test.
- Secondary hypotheses S1–S12 are reported with 95% bounds and labelled secondary. No secondary result can rescue a failed primary.
- **Everything else is exploratory** and labelled as such: other estimators, y, n, K and arms. Selecting the best of ~30 estimators per y is not a result.
- Controls (C) are gates, not hypotheses. If any blocking control fails, the run is invalid until it is explained.

### 9.4 What falsifies the idea, and what justifies production

**The idea is falsified for annual non-profit surveys with a tax-linked register** if all of the following hold:
1. P1's upper 95% bound is < 1.10 under REG-S+;
2. the controls are clean;
3. the result replicates in direction on a second population.

**It is reduced to frame coding** if the REG-W gains (S2) are ≥ 80% reproduced by D1-imp plus GREG_size_cat using text-coded NTEE dummies alone. In that case text matters only as a classifier.

**It is a non-production curiosity** if P1 passes but P2 fails: the gain does not survive a single weight set or multipurpose constraints.

**Production pilot is justified** if all of the following hold:
- P1 and P2 pass;
- S3 passes;
- the binding-constraint saving is ≥ 10% of the take-some sample;
- the gain replicates in at least one other population;
- an on-prem rung (L2, L4 or L5) is within 0.05 of the best;
- the cost balance (§8.5) is positive.

**Adoption** additionally requires GM-RE ≥ 1.20 on the program's own binding variables. It also requires a parallel run over one production cycle with frozen previous-wave models.

**Niche use is justified** without P1 if S2 (weak register), S6 (nonresponse) or S7 (SAE) pass in a scenario that maps to a real program, for example a non-charity NPO frame. The recommendation is then scoped to that use only.

### 9.5 Why these thresholds

- 1.10 is about the smallest variance reduction that would make a production area re-specify its estimation system, given system change, documentation, QA and governance costs.
- At n = 1,000 (794 take-some units) it saves about 70 questionnaires a year (≈ CAD 14k at the default unit cost), which is near the cost of maintaining the pipeline.
- 1.20 saves about 130. Thresholds are on GM over H because a survey is multipurpose. A single-variable win is reported but does not trigger a recommendation.

---

## 10. Dataset red-team checklist and replication datasets

### 10.1 Tests for notebook 01 (each prints a number and PASS / FLAG / FAIL)

"Blocking" means Stage 2 does not start until the test passes or a documented decision is recorded in `working-log.md`.

**Units and population**

| # | Test | Rule | Observed (NE) |
|---|---|---|---|
| T01 | One row per EIN; drops logged by reason (form type, group return, duplicate, multiple periods, territory) | blocking | parser keeps the latest return per EIN |
| T02 | Amended returns: share of truth from amendments; \|Δy\|/y distribution | FLAG if > 1% of any total changes | `amended_t` stored |
| T03 | Group returns excluded; their revenue share | report | 5.5% nationally |
| T04 | Continuing-unit conditioning: totals for U-cont vs all TY2023 filers | report | births 0.3–0.9% of totals |
| T05 | 990-EZ threshold truncation (gross receipts ≥ $200k or assets ≥ $500k): units entering 990 from EZ counted as "births" | report | — |
| T06 | Geography: BMF vs XML state agreement | ≥ 98% | — |
| T07 | Schema versions: null rate of every y and text XPath by return version | FAIL if any version has > 2× the null rate | — |
| T08 | FYE mix and short or long periods; December-only sensitivity | report | Dec 47%, Jun 33% |

**Timing and leakage**

| # | Test | Rule | Observed (NE) |
|---|---|---|---|
| T10 | Lagged return submitted ≤ D | blocking; coverage by size class | 91% (national, 30 Jun 2024) |
| T11 | Every embedded text comes from a TY ≤ t−1 ObjectId | blocking | — |
| T12 | BMF snapshot equality test: share with REVENUE_AMT = y_t | blocking if > 1% | 0.6% (Dec 2023 BMF) vs 18.3% (2026 BMF) |
| T13 | Previous-wave population: BMF income and asset codes recomputed from the TY2021 return. The 2023 BMF holds TY2022 amounts, which is the y of that wave. | **blocking** | new |
| T14 | Numbers in text: share containing the exact y_t, y_{t−1}, or "$" | report; L2 control | mission 10–11% digits / 0.1% "$"; programs 39% / 3.7% |
| T15 | XPath whitelist: no expense labels or amount siblings in the text | blocking | — |
| T16 | Text staleness: share identical between t−1 and t | report | mission 84% |
| T17 | Memorization probe: predict log(y_t/y_{t−1}) (change) from T0 and T2 per rung; compare with shuffled names | FLAG if any rung's change-R² > 0.02 above the random control | — |
| T18 | Placebo y gain ≈ 1 | blocking (control N3) | — |

**Cross-fitting and modelling**

| # | Test | Rule | Observed (NE) |
|---|---|---|---|
| T19 | Near-duplicate clusters (exact normalized text; cosine > 0.95 on E*). Folds keep clusters together. | blocking | 4.8% exact duplicates |
| T20 | No model sees population y (code audit + unit test that shuffles y outside the sample and checks the estimates are unchanged) | blocking | — |
| T21 | PCA computed on the frame only (no y); for frozen models, the rotation comes from the training wave | blocking | — |
| T22 | Same-class ceiling computed for every Monte Carlo estimator | blocking (C1) | pilot violated: 1.4 MC vs 1.13 ceiling |

**Coverage, design and y quality**

| # | Test | Rule | Observed (NE) |
|---|---|---|---|
| T23 | Text and website missingness by size class, NTEE, state; correlation with y | report; NR4 uses it | URL 85% (smallest quintile) → 99% (largest) |
| T24 | Stratum jumpers: size-class changes t−1 → t; take-some units reaching take-all size; their MSE share | report | 8.4%; 20 units = 1.3% of expenses |
| T25 | Allocation feasibility: Σ min n_h ≤ n_TS for every n in the grid | blocking | n = 500 fails, so it is dropped |
| T26 | Outliers: largest take-some contribution per y; d_k × y_k / T̂ | FLAG if one unit > 5% of a take-some total | volunteers 16.4%, gov. grants 1.5% |
| T27 | Volunteers: blank vs 0; heaping at multiples of 10, 50, 100 | define the estimand | 41% zero or blank |
| T28 | NTEE MISSING by size and ruling year (informative?) | report | 38% → 26% across size quintiles |
| T29 | Late-filer response by stratum, weighted and unweighted | report | 61% take-some, 12% take-all, 18% expense-weighted |
| T30 | Edit log: counts per rule; sensitivity with edits off | report | E1–E3 |
| T31 | Cross-check of totals against the IRS SOI extract (processing-year basis) | FLAG if > 5% difference after alignment | — |
| T32 | Register-scenario sanity: the REG-S+ ceilings for salaries, contrib and prog_rev ≈ 1 (text redundant) | report | 0.97 / 1.03 / 0.95 |

### 10.2 Replication datasets

| ID | Dataset | Why | User action needed |
|---|---|---|---|
| R0 | **National 990** (TY2022 done, TY2023 downloading; TY2021 for the previous wave) | Second population; larger SAE (about 560 domains); 9 Census-division sub-populations give a *between-population* spread of the verdict | none (compute only) |
| R1 | **CRA T3010** (Canada): two consecutive fiscal periods; IDENT, NEW_ONGOING_PROGRAMS, FINANCIAL_SECTION_*, SCHEDULE_3 | Canadian register; **bilingual text** (MiniLM is English-only, so L5/L6 multilingual rungs are needed); government revenue by level shows how large REG-P is for charities | allowlist `open.canada.ca`, or download and upload the CSVs; allowlist `huggingface.co` or provide multilingual weights |
| R2 | **UK Charity Commission (England and Wales)**: register extract + accounts PDFs for a stratified sample of about 5–10k charities at t−1 | The only realistic **PDF-native** population, and so the multimodal test (L7 vs L6 vs L7-ext) | allowlist the CCEW register and extract hosts (confirm hostnames from the download page), or upload the extracts and PDFs; production Cohere key (images: 400/min) |
| R2-alt | ACNC (Australia) AIS + financial reports | fallback for R2 | data.gov.au access |
| W | Website screenshots or HTML for 990 units with a URL | Multimodal on a realistic est-time source | network access for crawling; legal review of robots.txt and ToS |

---

## 11. Notebooks, results layout and staged execution

### 11.1 Layout

```
embedding-auxiliary-variables/
  config/
    protocol_v1.yaml        # frozen pre-registration (hash recorded in working-log before Stage 2)
    cells.yaml              # every MC cell: population, register, design, n, R, seed, estimator list
  src/                      # library (existing + Appendix A)
  scripts/
    run_cell.py             # CLI: run one cell id from cells.yaml -> results/...
    build_embeddings.py     # all API/model calls; never called from notebooks
  results/<stage>/<cell_id>/
    reps.parquet            # long: rep, estimator, y, est, var (+ weights summary)
    metrics.csv             # mc.metrics output with MC SEs
    manifest.json           # pop hash, emb hashes, config hash, git commit, seeds, R, runtime
  data/  (not in git; symlink to /tmp/claude-0/data or a persistent volume)
    pop/<pop_id>.parquet + popcard.json
    emb/<pop_id>/<name>.npy + .json
  notebooks/
    00_protocol_and_config.ipynb      # renders config, checks hash, lists cells and status
    01_data_qa_redteam.ipynb          # Stage 0: T01–T32, population card
    02_stage1_ceilings.ipynb          # Stage 1: previous-wave selection; current-wave predictions
    03_mc_calibration.ipynb           # Stage 2: P1, P2, S1–S3, S8, cost curve
    04_designs_stratification.ipynb   # Stage 2D: S4, S5
    05_nonresponse.ipynb              # Stage 3: S6
    06_sae.ipynb                      # Stage 4: S7
    07_embedding_ladder.ipynb         # Stage 5: S9–S11 (Cohere when available)
    08_replication.ipynb              # Stage 6: S12
    09_decision_report.ipynb          # pass/fail table, cost translation, recommendation
```

**Rules:**
- Notebooks only read `results/`, so each one re-runs in under 2 minutes.
- Heavy runs go through `scripts/run_cell.py` in the background, logged.
- Every figure or table cell prints the manifest hashes it used.
- Iterative review happens at gates. Each gate ends with an entry in `working-log.md` recording the decision and its reason.

### 11.2 Staged plan, with compute estimates

The compute basis is from the pilot: about 1 s per replicate for 30 estimators at n = 1,000 and N = 20.6k, on a shared 4-core CPU. Cost scales roughly linearly in the number of estimators and n. Three cells run in parallel.

| Stage | Content | Cells × R | Est. CPU | Gate |
|---|---|---|---|---|
| **0 Data QA** | build U-cont, U-full, previous wave (needs NE TY2021); T01–T32; MiniLM T2 and T2-nonum embeddings (~15 min each); keyword (L1) and GloVe (L3) arms | — | ~1.5 h | **G0:** blocking tests pass; estimands fixed (volunteers decision) |
| **1 Ceilings** | previous wave: same-class ceilings for rungs L1–L4 × T1/T2 × K × registers; select E*, K and feature map; **freeze config**. Current wave: ceilings as predictions for every MC estimator. | ~60 feature sets × ~40 s | ~1.5 h | **G1:** `protocol_v1.yaml` hashed and logged |
| **2 MC calibration** | 2A primary (S+, D1, n = 1,000, ~35 estimators, R = 2,000); 2B registers W/S/P (R = 1,000); 2C cost curve (baseline only, 6 n, R = 1,000; text arms at 750 and 2,000); 2D designs D0/D1-imp/D2a/D2b/D1-H44 (R = 1,000); 2E λ policy frozen vs CV; 2F bootstrap (200 × 500 on 2 estimators); 2G winsorized variants; 2H U-full coverage arm | ~16 cells | ~7 h → ~3 h wall | **G2:** controls pass; P1/P2 verdicts drafted; review whether S3 holds before any further stage |
| **3 Nonresponse** | NR0–NR4 + NR-LATE × TA-follow-up / TA-late; ~20 estimators; R = 1,000 | 8 cells | ~2.5 h | **G3:** S6 verdict |
| **4 SAE** | reuses the Stage-2 samples (same seeds); FH by REML per replicate (66 domains); R = 1,000 | 2 cells | ~0.5 h | **G4:** S7 verdict |
| **5 Ladder / Cohere** | L5/L6/L6b when available: re-embed NE (~20k texts, < $0.50, about 10 min API); rerun 2A estimators for the new arms (CRN) and Stage-1 ceilings | 2 cells | ~1 h + API | **G5:** S9, S10 (**user action U1**) |
| **6 Replication** | national: build (~4 h CPU MiniLM on 330k), 2A and 2B at n = 10,000, R = 500 (est. 10–20 s/rep → 1.5–3 h per cell); 9 Census-division sub-populations at R = 500 with primary estimators only; CRA T3010 (U2) and CCEW multimodal (U3) as data arrive | ~12 cells | ~15–20 h | **G6:** S11, S12, final decision report |

### 11.3 User actions (blocking only for the stages named)

| ID | Action | Unblocks |
|---|---|---|
| U1 | Allowlist `api.cohere.com`; add a **production** `COHERE_API_KEY` as an environment secret (a trial key's 1,000 calls/month cannot embed the national file) | Stage 5 (L6, L6b, L7) |
| U2 | Allowlist `open.canada.ca` or upload two years of T3010 CSVs | R1 |
| U3 | Allowlist the CCEW hosts or upload the extracts plus a PDF sample | R2, L7 |
| U4 | Allowlist `huggingface.co` or provide Qwen3-Embedding-0.6B / BGE-M3 weights | L5; French text in R1 |
| U5 | Decisions: unit cost c_u and burden b; CV targets of the hypothetical survey; volunteers estimand; NE-only vs national as the primary population | §8.5, G0 |
| U6 | Confirm the GivingTuesday data-lake terms, for redistributing derived tables | publication |

---

## 12. Real-world translation and governance

### 12.1 How a result would be used

| Result pattern | Production action | What it buys |
|---|---|---|
| P1 and P2 pass under REG-S+ | Add a frozen text-score calibration variable to the existing calibration (MCALm or FROZEN_cal). Sample unchanged in year 1 (parallel run), then reduced. | Take-some sample cut σ ≈ 1 − 1/RE_binding (e.g. 1.2 → 17%), or the same n with tighter domain CVs |
| Only S2 passes (weak register) | Use text only for **frame segments without admin data**: non-charity NPOs, births, stale-lag units | Better precision where the register is thin. Often a small share of totals, so assess the share of output first. |
| Only S4 passes (text-coded NTEE) | **Register maintenance:** autocode missing or old activity codes (StatCan already codes BN text) | Better strata and domain classification. No change to estimation. Easiest to govern. |
| S6 passes | Add text-based predictive-mean classes or donor matching variables to the nonresponse and imputation step (Banff donor imputation) | Lower nonresponse bias for survey-only y. Bias, unlike variance, is not fixed by a larger n. |
| S7 passes | Add one text-derived area covariate (domain sum of the frozen text score) to Fay–Herriot | More publishable domain cells (band D or better) |
| Nothing passes | Stop. Document the null result: it settles a recurring proposal. | Avoided system cost |

**Translation rules printed in the decision report:**
- RE r saves 1 − 1/r of the take-some sample *only for the binding variable*;
- CVs shrink by 1 − 1/√r;
- cost saved = Δn × c_u − C_text.

### 12.2 Which NSO data sources would play the role of text

| Source | At design time? | Coverage | Confidentiality | Modality | Best role |
|---|---|---|---|---|---|
| BN registration "major business activity" + main products (CRA RC1); SS-4 line 16–17 (US) | yes | near-universal for BN holders; stale for old units | confidential (Income Tax Act sharing; Statistics Act) | short text | frame coding, prediction strata, calibration score. **The realistic analogue of T1.** |
| T3010 program descriptions (charities) | yes (t−1) | all registered charities (~84k) | public | text, EN/FR | analogue of T2; R1 test |
| BR operating and legal names | yes | universal, births included | confidential | very short text | births and stale units (T0) |
| Organization websites | crawl at design, or est-time | size-graded (≈ 45% for small NPOs → 88% for large) | public, under the NSO web-scraping policy | text + layout | est-time auxiliary; nonresponse covariate; multimodal L7 |
| Annual reports and financial-statement PDFs (UK CCEW accounts; T3010 attachments **[verify publication]**; ACNC) | t−1 | mostly larger units | public (UK/AU) | PDF pages | **lagged numbers**: first extract them (L7-ext), then embed narrative only if extraction leaves residual signal |
| Admin tax and payroll (T2/GIFI, T3010 lines, PD7A) | yes | near-universal | confidential | numeric | **the baseline text must beat** (REG-S+/P) |

### 12.3 Governance

- **Confidentiality:** BN text and register data cannot be sent to an external API. Production options:
  - (a) open-weight models on-prem (L2, L4, L5), which is why S10 is a decision criterion;
  - (b) a vendor private deployment inside the NSO enclave, or a single-tenant deployment with a written Canadian-residency guarantee.
  Public-data experiments (990, T3010, CCEW) may use the Cohere API.
- **Reproducibility and revision policy:**
  - Embeddings are computed **once per cycle**, content-addressed (model id, version, input_type, dimension, preprocessing, input hash; `embed.py` manifests) and frozen with the cycle's estimates.
  - Published estimates are never revised because a model changed.
  - When a model is deprecated (Cohere Embed v2 was shut down in 2026), run a **bridging study**: re-embed the last cycle with the new model, re-estimate, and publish the difference in the quality report.
  - A canary set (1,000 texts) is re-embedded each cycle to detect silent drift (alarm if cosine < 0.999).
- **Methodological documentation:** the text score is one calibration variable among many. Document it as a model-assisted auxiliary. Design-consistency holds whatever its quality, provided it is cross-fitted or frozen. Report its population R² and the weight diagnostics in the methodology note.
- **Fairness and consistency across domains:** check per-domain bias of the text-assisted estimators (S7 metrics), language (French-text units in R1) and size classes (text length grows with size). Domain-level model error is the main way text could *introduce* bias into published small-area tables.
- **Operational cost:** pipeline ownership (frame team), annual retraining on last cycle's sample, QA of text extraction, model monitoring. Budgeted as C_text in §8.5.
- **Responsible-ML framework:** apply the NSO's responsible-ML review (StatCan's *Framework for Responsible Machine Learning Processes* **[confirm current title and version]**). Record the purpose, data, model card, validation (this benchmark) and monitoring plan.

---

## Appendix A: code changes needed (by file)

| File | Change | Needed for |
|---|---|---|
| `experiments.build_mats` | register-specific matrices `size_catP`, `size_catO`, `ml_frame_ratio@S+`, `mlr+E@S+`; T0/T2-nonum arms | REG-S+, REG-P |
| `mc._estimate` | kind `mcal_multi` (multi-y model calibration, bounded chi-square weights, returns one weight vector applied to all y); kind `ridge1w` (single λ by mean CV loss over H) | P2 |
| `mc._estimate` | `win` option: type-II winsorization with frozen cutoffs | §2.8 |
| `mc.run_cell` | per-estimator RNG spawned from (cell_seed, r, name); store a weight summary per replicate | CRN robustness, §8.7 |
| `mc._nr_weights` | take-all nonresponse with imputation (hist, nn-frame, nn-text); propensity-class and predictive-mean-class RHGs; donor imputation for take-some | Stage 3 |
| `survey.py` | g-weighted variance v_g; Rao–Wu–Yue bootstrap wrapper that re-runs `_estimate` | §4.5 |
| `frame.py` | `production_design_weak` (BMF income-code size, D1-W); `imputed_ntee_design` (D1-imp); `score_strata_design` (D2a); `kmeans_design` (D2b); equal-H assertion | §3 |
| `diagnostics.py` | same-class ceilings (ratio-scale features, ratio weights) callable for every mc spec | C1 |
| new `sae.py` | domain estimators; FH with REML, GVF smoothing, Prasad–Rao MSE | Stage 4 |
| new `frozen.py` | previous-wave training (J heads from J previous-wave samples), PCA rotation carried over | §7.4 |
| `build_population.py` | U-full construction (deaths as out-of-scope, births or stale stratum); previous-wave BMF codes from the TY2021 amounts (T13); volunteers blank flag | §2.6, T13, T27 |
| `metrics` | GM-RE with replicate bootstrap; band tables; cost curve and binding-n | §8 |

---

## Appendix B: check run for this draft (REG-S+, ceilings)

Run 2026-10-23 on the NE pilot population. Setup:
- design D1 (n = 1,000);
- anticipated variance of the difference estimator with a population-fitted, 5-fold cross-fitted weighted ridge, λ = best per y;
- features = REG frame + 32 PCs of the embedding, each also multiplied by lagged expenses and lagged employees.

These are **optimistic ceilings, not estimates.** Values are AV(baseline)/AV(baseline + text).

To reproduce: run `diagnostics.evaluate_feature_sets` with:
- the baseline `frame_matrix(pop, des, "strong")`, with the columns lag_salaries, lag_contrib and lag_prog_rev appended for REG-S+;
- text features = `pcs(E, 32)` plus each PC × lag_expenses/sd and × lag_employees/sd;
- the oracle = the four `oracle_lag_*` columns, left unpenalized.

Runtime is about 2.5 min per batch of four feature sets.

| y | REG-S + LSA T2 (pilot frame) | REG-S+ + LSA T2 | REG-S+ + MiniLM T1 | REG-P (oracle lag) vs REG-S+ |
|---|---|---|---|---|
| salaries | 1.41 | **0.97** | — | — |
| contributions | 1.19 | **1.03** | — | — |
| program revenue | 1.04 | **0.95** | — | — |
| expenses | 0.97 | 0.99 | — | — |
| employees | 0.93 | 0.92 | — | — |
| gov. grants | 1.13 | **1.20** | 1.11 | 5.09 |
| fundraising exp. | 1.20 | **1.16** | 1.02 | 7.66 |
| officer comp. | — | 1.04 | 1.00 | 9.66 |
| volunteers (cap 10k) | — | 1.03 | 1.02 | 1.23* |
| grants paid | — | 1.16 | 1.09 | 0.98 |
| has gov. grants | — | 1.14 | 1.10 | 1.01* |
| grants to individuals | — | 1.17 | 1.06 | 1.00* |
| lobbying | — | 1.06 | 1.03 | 1.00* |
| has volunteers | — | 1.06 | 1.03 | 1.00* |
| **GM over H** | — | **1.12** | **1.05** | — |

\* The oracle columns hold raw lagged amounts, not capped values or indicators, so they understate the oracle for volunteers (cap) and for the binaries.

**Interpretation for the protocol:**
1. The pilot's salaries gain (1.4–1.7× in Monte Carlo) is a baseline artefact. It disappears once payroll is on the frame.
2. The surviving text signal sits in government funding, fundraising and grant-making: variables tied to *what kind of organization this is*, which the register does not encode.
3. Even the optimistic ceiling sits near the 1.10 threshold, so the realistic expectation for P1 is "inconclusive to fail" under REG-S+.
4. The lagged value of the same variable (REG-P) dominates text by a factor of 4–8. Wherever an NSO has it (T3010 for charities), text is not worth adding for those variables.
