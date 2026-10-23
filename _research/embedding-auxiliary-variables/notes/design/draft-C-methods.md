# Benchmark protocol, draft C: estimator and experiment design

**Status:** draft for review, 2026-10-23.
**Lens:** survey statistics and ML methods: model-assisted estimation, calibration, cross-fitting, prediction-powered inference (PPI), small area estimation (SAE).
**Scope:** what to estimate, with which estimators, how to tune them, how to split a gain between its sources, and how to tell a real gain from an artefact.

`draft-A-production-realism.md` covers production realism. This draft keeps A's register scenarios, timeline, headline variable set and decision thresholds, so the two can be merged. It departs from A where estimator design calls for it; those places are marked **[differs from A]**.

**Companions:** `notes/early-findings.md`, `notes/research/01–04`. Estimator names match `src/` where the estimator already exists. New code and fixes are listed in Appendix A. Numbers computed for this draft are in Appendix B.

---

## Summary

**1. Three defects in the estimation code change how the pilot should be read. Two were verified for this draft.** Fix them before Stage 2.

- **(a) The ridge penalty does not scale with the weights (verified).**
  - `mc._ridge_fit` adds λ·n to a cross-product matrix whose diagonal is Σw.
  - For ratio-scale learners the weights are d·x̃, so Σw ≈ 7.7 × 10¹⁰ on a real n = 794 take-some sample. On that sample the coefficient norm was **identical for every λ in the grid {0.001, …, 10}** (Appendix B.1).
  - The pilot's best text estimators (`MCAL_ratio+…`), the frozen models and the SAE models were therefore *unpenalized* cross-fitted weighted least squares on about 60 columns.
  - With weights d the penalty is about 25 times weaker than intended. With weights normalized to mean 1 it behaves as intended.
- **(b) The ridge-calibration estimator's λ is chosen on a different scale from the one it is applied on.** `RIDGE_…` tunes with `_ridge_fit` (penalty relative to Σd) but estimates with a penalty relative to the column variance. The applied penalty is about Σd/n ≈ 26 times stronger than the one the cross-validation (CV) scored. Found by inspection; a unit test is in Appendix A.
- **(c) Binary targets go through the ratio-scale learner.** The `MCAL_ratio` specs in `experiments.py` model y/x̃ for binary y as well. `frozen.py` handles binaries correctly.

The pilot's *direction* survives these defects: negative controls are ≈ 1 and cross-fitting fixes coverage. Its magnitudes do not. The K-scan and the full-embedding experiments cannot run until (a) is fixed, because an unpenalized learner with K = 128 PCs or all 384 dimensions will overfit.

**2. The benchmark measures a decomposition, not one number.** "Text gain" is confounded with three other changes that usually travel with it:
- a more flexible learner;
- a ratio-scale working model;
- finer use of register codes that text partly re-expresses. The BMF carries a full NTEE code with 1,571 distinct values. The status quo uses only 10 groups.

The core experiment is therefore a 2 × 2 × 2 factorial on common random numbers: {linear ridge calibration, ratio-scale model calibration} × {NTEE10 only, fine register codes} × {no text, text}. It is reported as main effects on log MSE, with Shapley shares (§4.7, §6).

**3. The fair comparator for "text" is the text-free twin that uses the register's fine codes [differs from A].** A Stage-1 pre-check ran for this draft (Appendix B.2): REG-S+ population, n = 1,000, same-class ridge ceilings with LSA full text. It measured how much of the text ceiling a fine-code twin already captures. **[RESULT: see B.2.]**

**4. The skeptic's arithmetic.** Expected Monte Carlo (MC) gain ≈ ceiling × fitting efficiency.
- The REG-S+ ceiling for the headline set is a geometric mean (GM) of 1.12 with LSA full text (draft A, Appendix B).
- Fitting about 60 free columns on n = 794 inflates residual variance by about p/n. The text-free twin fits about 28.
- The expected MC gain is therefore 1.12 × (1 + 28/794)/(1 + 60/794) ≈ **1.08**. That is under the 1.10 "material" threshold before any other loss.
- Primary endpoint P1 is therefore expected to be *inconclusive or fail* under a good register. The benchmark is powered to show that cleanly (R = 2,000 gives an MC SE of about 0.026 on RE).

**5. A planted-signal positive control calibrates the whole pipeline.**
- Semi-synthetic targets carry a text component with known population partial R² ∈ {0, 0.02, 0.05, 0.10, 0.20, 0.40}. The component is linear in the PCs, nonlinear (clusters), or planted in a *different* embedding from the one used to estimate.
- They show what fraction of the theoretical gain each estimator family recovers at each n.
- They also give a **minimum detectable partial R²**, which turns a null result on real y into a statement about the size of signal that could have been missed.

**6. The production candidate is the frozen previous-wave model.**
- It is trained on last cycle's sample, then used as one extra calibration column (`FROZEN_cal`). It is exactly design-unbiased in its difference form and needs no cross-fitting. Its linearization variance is exact.
- It also changes production software least: one more calibration variable.
- The previous-wave population (TY2021 frame → TY2022 y; N = 20,317; 19,478 units in both waves) is already built.

**7. Validity is a primary outcome, not a footnote.**
- Coverage and variance-estimator bias are tracked as functions of the effective number of fitted parameters p_eff/n.
- They are diagnosed with the studentized-error distribution: bias vs variance underestimation vs skewness.
- They are checked with a bootstrap that re-runs tuning and cross-fitting inside every bootstrap replicate.

**8. Generalization comes from replicate populations, not from larger R.** The MC CI is conditional on one finite population. The verdict is re-estimated on:
- the second wave;
- the nine US Census divisions (national file);
- CRA T3010.

log RE is pooled with a random-effects meta-analysis. It is reported with a **prediction interval for a new population**, which is the number a production decision needs.

---

## 1. Research questions and estimands

### 1.1 Research questions

| RQ | Question | Primary quantity |
|---|---|---|
| RQ1 Signal | What is the population partial R² of t−1 text for each y, given each register scenario and given fine register codes? | Same-class anticipated-variance (AV) ceiling ratio; partial R² on the AV scale |
| RQ2 Realization | What share of that ceiling does each estimator family realize at sample size n? | Fitting efficiency φ = RE_MC / RE_ceiling |
| RQ3 Attribution | How much of a measured gain comes from text, and how much from learner and scale, code granularity, dimension, or the embedding model? | 2³ factorial on log MSE; ladder "gap closed" |
| RQ4 Validity | Do variance estimators and confidence intervals (CIs) stay valid as p_eff/n grows, with and without cross-fitting, and for frozen models? | Coverage; variance relative bias; studentized-error moments; bootstrap vs linearization |
| RQ5 Placement | Where is text information best used: design, estimation, nonresponse adjustment or domains? Are the gains additive or substitutive across stages? | Design × estimator factorial; share of nonresponse bias removed; ARRMSE |
| RQ6 Transport | Does a model frozen from last cycle work this cycle? Does the verdict replicate across populations? | RE of frozen vs cross-fitted; meta-analytic prediction interval |
| RQ7 Value | What does a gain buy? | Equivalent n, take-some saving, CV-band movement |

### 1.2 Estimands

All estimands are fixed finite-population quantities of the TY2023 population U. U-cont has N = 20,626 in New England and about 330k nationally. Truth is the full TY2023 file.

- **Totals** t_y = Σ_U y_k (primary).
- **Means** ȳ_U = t_y/N. N is known from the frame, so a mean's relative efficiency (RE) equals its total's and is not reported separately. The exception is nonresponse, where the Hájek mean t̂_y/N̂ is reported as well.
- **Proportions** p_b = t_b/N, for binary b.
- **Ratios of totals** R = t_a/t_b, linearized with z_k = (a_k − R̂ b_k)/t̂_b:
  - R₁ = gov_grants / revenue (government dependence);
  - R₂ = fundraising_exp / contrib (cost of raising a dollar).

  Ratios are in the protocol because per-y model calibration can improve t̂_a while breaking the coherence of t̂_a/t̂_b. That makes them a multipurpose check.
- **Domain totals and proportions:**
  - state × NTEE10 (66 domains in New England: median N_d = 145, 17 with N_d < 30; about 560 nationally);
  - state;
  - NTEE10.
- **Not estimated:** year-on-year change. It would need a panel, and 84% of mission texts are identical between t−1 and t, so text carries level information, not change information.

### 1.3 Target variables: classes, roles and pre-specified model family

Population facts are for New England U-cont (Appendix B.3).

| y | Class | Zero share | Top-1% share of total | Lag on frame | Role |
|---|---|---|---|---|---|
| expenses, revenue | R | 0.7% | 68% | S, S+, P | **Negative control** (text RE ≈ 1) |
| employees | R | 37% | 51% | S, S+, P | Negative control |
| salaries, contrib, prog_rev | R under S+ / S under S | 29%, 19%, 31% | 62–73% | S+, P | Scenario sensitivity |
| **gov_grants** | S (zero-inflated) | **69%** | 65% | P only | **Headline** |
| **fundraising_exp** | S | 62% | 61% | P only | **Headline** |
| **officer_comp** | S | 51% | 35% | P only | **Headline** |
| **volunteers_cap10k** | S | 41% | 48% | P only | **Headline** (estimand fixed at G0: blank vs zero) |
| grants_paid | H | 70% | 89% | P | Heavy-tail stress |
| volunteers (raw) | H | 41% | 82% (CV 63) | P | Heavy-tail stress |
| **bin_has_gov_grants** | B | p = 0.31 | — | P (derived) | **Headline** |
| **bin_grants_to_indiv_us** | B | p = 0.12 | — | none | **Headline** |
| bin_lobbying, bin_foreign_activities, bin_gaming | B (rare) | p = 0.058, 0.043, 0.027 | — | none | Secondary; rare-proportion coverage |
| bin_has_volunteers, bin_fundraising_activities | B | p = 0.59, 0.21 | — | none | Secondary |
| y_placebo | — | — | — | — | Within-stratum permutation of gov_grants (§5) |
| y_plant(ρ², form) | semi-synthetic | — | — | — | Planted-signal positive control (§5.2) |

- **Headline set H** (same as draft A): {gov_grants, fundraising_exp, officer_comp, volunteers_cap10k, bin_has_gov_grants, bin_grants_to_indiv_us}.
- **All-y set A:** every non-placebo y above.

**The model family is fixed per y type, before any results [differs from the pilot].**

- **Non-negative amounts and counts.**
  - Working model: ratio scale, m(z) = x̃ · g(z), where x̃ = lag_expenses + 1 (pilot rule).
  - g is a weighted ridge of y/x̃ on the features, with weights d·x̃ normalized to mean 1. The ratio scale encodes Var(y | z) ∝ x̃.
  - Secondary: **two-part** m(z) = p̂(z) · x̃ · ĝ₊(z), where p̂ is a ridge-logistic model for 1{y > 0} and ĝ₊ is the ratio-scale ridge fitted on units with y > 0. This is pre-specified for the zero-inflated headline amounts. Text plausibly predicts *whether* an organization receives government grants more than *how much*.
- **Binaries:** ridge-logistic p̂(z) (primary), linear probability (secondary). **Never the ratio scale.**
- **Features shared by every learner of a given scenario:** log1p of each lag the scenario carries; NTEE10, state and BMF income-code dummies. The fine-code factor adds the codes listed in §4.4.

---

## 2. Real-world scenario specification

### 2.1 The survey being simulated

A national statistical office (NSO) runs an annual survey of non-profit organizations for reference year t = TY2023. It is multipurpose, with about 15 published variables, and publishes totals nationally, by state (province) and by NTEE10 group.

- The **frame** is the register as known at design date D = 2024-06-30, with reference t−1.
- **Collection** runs from fiscal-year end to a cutoff.
- **Estimation** is design-weighted. It calibrates to frame totals, imputes take-all nonrespondents and reweights take-some nonrespondents.

The 990 data play both the register (the t−1 return plus the BMF) and the questionnaire (the t return). Every unit's t return is eventually observed, so the truth is known. This mirrors a StatCan design in which:
- the Business Register (BR) and CRA data are the frame;
- lagged administrative data (T2/T3010/PD7A) are auxiliaries;
- Business Number (BN) "major business activity" text or T3010 program text is the candidate text auxiliary.

### 2.2 Information sets: what is known when

Formal rule used by the harness. Each matrix, model and choice carries a `stage` tag, and the harness asserts it.

| Set | Contents | Who may use it |
|---|---|---|
| **I_D** (design, date D) | For every frame unit: BMF codes (NTEE, subsection, ruling year, income and asset codes, state); lags allowed by the register scenario; t−1 text and its cached embeddings; website URL. For the previous-wave sample only: (frame_{t−2}, text_{t−2}, y_{t−1}), plus every model trained on it. | Design (strata, allocation); auxiliaries for calibration (their population totals are computable because every frame unit has them); frozen models |
| **I_E** (estimation) | I_D plus y_k and the response indicator for sampled units, plus paradata (filing date) | Sample-fitted models (cross-fitted), calibration, variance |
| **Truth only** | y_k for non-sampled units and nonrespondents; same-year text; BMF snapshots later than D | Metrics; leakage controls L1 and L3 only |

Three rules follow.

- **R1 (complete auxiliary information).** A variable can be a calibration auxiliary only if it is I_D-measurable for *every* unit of U_TS. Embeddings of frame text qualify. Website screenshots qualify only if crawled at D.
- **R2 (no population y).** No fitted object may touch y outside the sample. Unit test: permute y over non-sampled units and assert that every estimate is unchanged (T-M3).
- **R3 (no current-wave researcher degrees of freedom).** These choices are fixed across replicates:
  - embedding rung;
  - text field;
  - K (number of PCs);
  - feature map;
  - λ policy;
  - learner family.

  They are made on the **previous wave** (TY2021 → TY2022), or pre-registered here. Never on current-wave MC results. Choosing the best of about 30 estimators per y on the current wave is a selection leak, and it is treated as exploratory.

### 2.3 Timeline

The timeline is the same as draft A §2.2 and summarized here.
- **Register codes:** BMF snapshot of 2023-08-16. It must pass leakage test T12: the share of units whose BMF REVENUE_AMT equals y_t is about 0. Observed 0.48%, against 22.8% for the 2026 snapshot.
- **Lags and text:** the TY2022 return submitted by D.
- **Collection:** fiscal-year end (FYE) to FYE + 300 days.
- **Truth:** the full TY2023 file.
- **Previous wave:** the same pipeline one year earlier. Its BMF income and asset codes must be **recomputed from TY2021 amounts**, because the 2023 BMF already carries TY2022 amounts, which are that wave's y (A, T13).

### 2.4 Register scenarios

These are the scenarios of draft A §2.4, restated as feature sets.

| Scenario | Frame content (features of every twin) | Real-world analogue | Text competes with |
|---|---|---|---|
| **REG-W** (weak) | NTEE10 (31% MISSING), subsection, state, ruling year, BMF income- and asset-code classes | BR-only units: non-charity non-profit organizations (NPOs), new registrants | Codes and size class |
| **REG-S** | W + lagged revenue, expenses, assets, employees | BR with T2/T1044 summary financials and PD7A headcount | Lagged size |
| **REG-S+** (**primary**) | S + lagged salaries, contributions, program revenue | Registered charities with T3010 summary lines and PD7A payroll; US: the t−1 return summary | Lagged size and revenue composition |
| **REG-P** (positive control) | S+ + lagged survey-only y (government grants, fundraising, officer compensation, volunteers) | Full T3010 for charities; panel units | The lagged value of the same y |

For registered charities, T3010 carries government revenue, so gov_grants is REG-P there and text has little to add. The scenario where the headline set really is survey-only is **non-charity NPOs and new registrants** (draft A §1.3). This is stated in every result table, because it decides where a production use could exist.

### 2.5 Text sources and their real-world analogues

| Arm | Content | Coverage (NE) | NSO analogue | Stage |
|---|---|---|---|---|
| T0 | Organization name (BMF; **not yet in `ne_pop.parquet`**, add it) | 100%, births included | BR legal or operating name | I_D |
| T1 | Mission: Part I line 1 plus Part III line 1, deduplicated | 99.9%; median 19–23 words | BN registration activity text (CRA RC1), SS-4 lines 16–17 | I_D |
| T2 | T1 plus Part III 4a–4d program descriptions | 99.9%; median 71 words | T3010 program descriptions (public) | I_D |
| T2-nonum | T2 with digits and "$" amounts masked | same | Leakage ablation | I_D |
| T1_t | Same-year (TY2023) mission | — | none | Truth only (leakage control L1) |
| W | Website text or screenshot | URL on 93% of U-cont units (42–88% by size for all 990/990-EZ filers) | NPO or enterprise websites (ISTAT, ESSnet) | I_D if crawled at D. **User action (network).** |
| PDF | Accounts or annual-report pages | None for the 990 | UK CCEW accounts PDFs | Replication R2 only |

Text length grows with organization size, so the auxiliary's information content is itself size-graded. Every gain is therefore reported by size class (§8.2).

### 2.6 Population, births, missing NTEE

- **Primary population: U-cont** (continuing units).
- **Coverage arm: U-full** (draft A §2.6). Units that do not file at t are out-of-scope deaths. Births and units with stale lags form a separate take-some stratum with name-only text.
- Births are 6.4% of units but only 0.3% of expenses.
- NTEE is missing for 31.1% of units, and missingness is informative by age and size. The status quo keeps MISSING as a level. Design D1-imp replaces it with a text-coded NTEE: 69–71% 5-fold accuracy over 10 groups, against 24% for always guessing the largest class.

### 2.7 Nonresponse

The real mechanism is late filing: r_k = 1{TY2023 return submitted ≤ 300 days after FYE}.
- Response is 60.1% of units but only 18.4% of expenses.
- By stratum it is 61% among take-some units and **12% among take-all units**.
- Respondents' mean government grants are 0.35 times nonrespondents'.

Because r_k is fixed per unit, the asymptotic nonresponse bias of any adjustment is a **population quantity**. It can be computed exactly in Stage 1 (§7.2), and the MC adds variance and coverage.

### 2.8 Previous wave

- **File:** `ne_wave2022_pop.parquet` (frame TY2021 → y TY2022, N = 20,317; 19,478 units in both waves).
- **Uses:**
  - (i) selecting E\*, K, the feature map and the λ policy (R3);
  - (ii) training frozen models (§7.4);
  - (iii) a replicate population for the meta-analysis (§8.6).
- The PCA rotation used for frozen models is fitted on the *previous-wave* frame and carried forward.

---

## 3. Designs

### 3.1 Common rules

- **Stratified SRSWOR.**
- **Take-all stratum:** the top 1% of units on the design size variable (206 units in New England, about 3,300 nationally).
- **Take-all units are excluded from modelling and calibration**, as in production: T̂ = Σ_TA y_k + T̂_TS.
- **Compromise allocation:** Neyman on √x with min n_h = 10 (`frame.production_design`).
- **Equal-H rule:** a design that adds a stratification variable is compared with a status-quo design that has the same number of take-some strata.

### 3.2 Design arms

| Code | Strata | Allocation | Question | Status |
|---|---|---|---|---|
| **D0** | none (SRSWOR) | — | The scaling experiment: n can go low because there is no minimum per stratum | exists |
| **D1** | NTEE10 (incl. MISSING) × 3 size classes (cum-√f on log lagged expenses) + take-all; H = 33 + TA | Neyman on √x, min 10 | **Status quo** | exists |
| **D1-W** | NTEE10 × 3 BMF income-code classes + take-all on BMF INCOME_AMT | same | Status quo under REG-W | new |
| **D1-H44** | NTEE10 × 4 size classes + take-all | same | Equal-H comparator for D2a | new |
| **D1-imp** | D1 with MISSING replaced by text-coded NTEE (multinomial logit on E\* trained on known codes; no y) | same | Text as frame coder | new |
| **D2a** | NTEE10 × 2 size classes × 2 classes of a frozen text score + take-all; H = 44 | same | Prediction stratification (Breidt–Opsomer 2008; Rivest 2002) | new |
| **D2b** | **D1 strata, text-informed allocation**: n_h ∝ N_h Ŝ_{e,h}, with Ŝ from a frozen model of the residual scale | compromise over y ∈ H | Text at the allocation step only (Nedyalkova–Tillé 2008: π ∝ σ(x)) | new |
| **D2b-twin** | as D2b, with the residual-scale model fitted without text | same | Isolates the text increment in D2b | new |
| **D2c** | k-means (K = 11) on E\* in place of NTEE10, × 3 size + take-all; H = 33 | same | Unsupervised "text industry". Expected null for y not tied to activity. | new |
| D3 (optional) | πps (conditional Poisson) with π ∝ σ̂(z, e) | — | Only if D2b shows ≥ 1.05 | optional |

**The D2a score s_k** is the frozen previous-wave text model's predicted government-grant intensity ĝ(z_k, e_k) on the ratio scale. It is split at its frame median within each NTEE10 × size-2 cell.

**The D2b residual-scale model** is a frozen previous-wave regression of log|y − m̂| on (z, e), fitted on the previous-wave sample's cross-fitted residuals. Its fitted values give Ŝ²_{e,h,y} = mean_{k∈U_h} σ̂²_y(z_k, e_k).
- The y-specific Neyman allocations n_{h,y} ∝ N_h Ŝ_{e,h,y} for y ∈ H are averaged after normalization (a Kozak-style compromise).
- The minimum n_h = 10 is then imposed and the take-all stratum is kept.
- If text predicts *which kinds of organization have volatile survey-only values*, D2b uses that without touching the estimator.

### 3.3 Sample sizes

| Population | Design | n grid | Note |
|---|---|---|---|
| New England (N = 20,626) | D0 | 250, 500, 1,000, 2,000 | Scaling of φ and coverage in p/n |
| New England | D1 family | **1,000 (primary)**, 750, 1,500, 2,000 (cost curve) | n = 500 is infeasible: 33 × 10 = 330 > 294 take-some units |
| National (N ≈ 330k) | D1 national (NTEE10 × 4 size × 4 Census regions + TA) | 5,000, **10,000**, 20,000 | Replication and SAE |

### 3.4 Analytic anticipated variance before any Monte Carlo

For every design D and working model m, Stage 1 computes

  AV_D(m) = Σ_h N_h² (1 − f_h)/n_h · S²_{y−m,h}

using the same-class population fit (§8.4). This gives a predicted RE for each design arm, and for each design × estimator pair in the factorial below, before the MC is run.

**Design × estimator factorial (Stage 2D).** Designs {D1, D2a, D2b} are crossed with estimators {GREG_size_cat, MCAL twin, MCAL text, FROZEN_cal text}.
- **Hypothesis:** design-stage and estimation-stage uses of the same text information are *substitutive*. RE(D2 + text estimator) is close to max(RE_design, RE_estimator), not their product.
- **Test:** the interaction contrast on log MSE.
- If the uses are substitutive, the recommendation is to use text where it is cheapest to govern.

**Stratum jumpers.**
- 8.4% of take-some units change size class between t−1 and t.
- 20 take-some units reach take-all size at t, holding 1.3% of expenses.
- The share of MSE from jumpers is reported for every design. D2a classifies on a noisy score, so it creates jumpers by construction.

## 4. Estimators and baselines

### 4.1 Notation

- **Sample:** s = s_TA ∪ s_TS, with design weights d_k = N_h/n_h. All fitting uses s_TS only.
- **X_base[@reg]:** the status-quo calibration matrix, `size_cat@reg` = stratum dummies + the lags of the scenario + NTEE10 + state. Its population totals T_X are known.
- **F[@reg]:** the learner's frame features, `ml_frame_ratio@reg` = NTEE10, state and income-code dummies plus log1p of the scenario's lags.
- **C:** fine register codes (§4.4).
- **E:** the embedding (N × D). Z_K holds its first K population principal components, standardized.
- **x̃ = lag_expenses + 1.**
- **Cross-fitting:** fold f(k) ∈ {1, …, 5}. ĝ^{(−f)} is a learner fitted on s_TS without fold f.
  - Sample predictions: m̂_k = x̃_k ĝ^{(−f(k))}(z_k).
  - Population predictions: m̄_k = (1/5) Σ_f x̃_k ĝ^{(−f)}(z_k).

### 4.2 Estimator families (formulas)

**E0 Status quo (single weight).**
- HT: t̂ = Σ_s d_k y_k.
- GREG(X): t̂ = Σ_s d_k y_k + (T_X − Σ_s d_k x_k)′B̂, with B̂ = (Σ_s d x x′)⁻¹ Σ_s d x y. Its calibration weights are w_k = d_k(1 + x_k′λ̂).
- Instances, in `experiments.conventional_specs`: `GREG_ratio`, `GREG_size`, **`GREG_size_cat[@reg]`** (the production baseline).
- `GREG_size_cat-WIN`: one-sided type-II winsorization of take-some values, with cutoffs frozen from the previous wave (draft A §2.8).

**E1 Text inside linear calibration (single weight).** Soft columns: S_K = [Z_K ⊙ x̃, Z_K ⊙ ẽmp] (`size_cat+E:ratioK`), optionally with C ⊙ x̃ added.
- `PCGREG_K`: OLS-GREG on [X_base, S_K] with K ∈ {2, 4, 8}. It is simple and transparent, and expected to lose efficiency as K grows (§1 of note 01: break-even R² rises with p/n).
- `RIDGECAL_K` (`ridge` kind): penalized calibration. B̂_λ = (Σ_s d c c′ + Λ)⁻¹ Σ_s d c y, with c = [X_base, S_K] and Λ = diag(0_{X_base}, λ σ̂²_j Σ_s d).
  - The weights are w = d(1 + c′(Σ d c c′ + Λ)⁻¹(T_c − Σ d c)). They are exact on X_base and approximate on S_K (Guggemos & Tillé 2010).
  - **λ is chosen by CV of this same estimator.** Each fold computes B̂_λ on its training part and scores the out-of-fold loss of §4.5 T3. This fixes defect (b).
- `RIDGE1W_K`: as RIDGECAL, with **one** λ for all y, chosen by the mean normalized CV loss over H. One weight set.
- `PCCAL_pos`: calibration on [X_base, first K′ PCs of S_32]. K′ is the largest value in {0, …, 32} for which all g-weights are ≥ 0 in that replicate (Cardot, Goga & Shehzad 2017). A production-style, data-driven, single-weight rule.

**E2 Model calibration (Wu & Sitter 2001), cross-fitted; per y unless noted.**
- **Calibration variables:** c_k = (x_base,k, m̂_k) for k ∈ s, with totals (T_X, Σ_U m̄). Weights w = d(1 + c′λ̂).
- **Estimate:** t̂ = Σ_s w_k y_k.
- **Residuals for variance:** ê_k = y_k − c_k′B̂ (code: `mcal`).

| Name | Learner features | Learner | Role |
|---|---|---|---|
| `MCAL_frame_ratio[@reg]` | F | ridge-ratio | **Twin** (text-free) |
| `MCAL_frame_codes[@reg]` (new) | F + C | ridge-ratio | **Codes twin** |
| `MCAL_ratio[@reg]+E` | F + Z_K | ridge-ratio | Text |
| `MCAL_codes_ratio[@reg]+E` (new) | F + C + Z_K | ridge-ratio | **Text, primary** (§4.7) |
| `MCAL_2part[@reg](+E)` (new) | as above | two-part | Zero-inflated amounts |
| `MCAL_logit[@reg](+E)` (new) | as above | ridge-logistic | Binaries (replaces the ratio-scale fit for binaries) |
| `MCAL_knn(+E)` (new) | E (cosine) or [F_std, Z_K] | kNN on the ratio scale | Is the signal local (neighbours, templates) rather than linear in PCs? |
| `MCAL_ratio+E:full` (new) | F + all D dims (dual ridge) | ridge-ratio | Is the y-relevant signal beyond the top K PCs? |
| `MCAL_gbm(+E)` (new, exploratory) | F (+ Z_K) | sklearn HistGradientBoosting, fixed hyperparameters | ML flexibility ceiling; R = 200 only |
| `MCALm[@reg](+E)` (new) | as the text primary | per-y learners | **One weight set**: calibrate on [X_base, m̂_1, …, m̂_J] with J = 4 (gov_grants, fundraising_exp, volunteers_cap10k, has_gov_grants), bounded chi-square distance with g ∈ [0.2, 5] |

**E3 Cross-fitted difference (cross-PPI).**
- `DIFF_cf(+E)` (new): t̂ = Σ_{U_TS} m̄_k + Σ_{s_TS} d_k (y_k − m̂_k), with no calibration step.
- It is the design analogue of cross-prediction-powered inference (Zrnic & Candès 2024).
- Comparing it with MCAL shows what the calibration step adds. Calibration should protect the size variables and the binaries' implied counts.

**E4 Frozen previous-wave models (m fixed before sampling).** `frozen.py` and the `frozen` kind exist.
- **`FROZEN_diff(+E)`:** t̂ = Σ_U m_k + Σ_s d_k (y_k − m_k).
  - It is exactly design-unbiased.
  - Its variance is V_HT(y − m), and v_L is unbiased for it under stratified SRSWOR.
- **`FROZEN_ppi(+E)`:** GREG on [stratum dummies, m]. This is PPI++ with a stratum-pooled power parameter λ̂ = within-stratum regression slope of y on m (Angelopoulos, Duchi & Zrnic 2023; Mozer 2026). It protects against a badly scaled frozen model.
- **`FROZEN_cal(+E)`:** GREG on [X_base, m]. **The production candidate.** It needs one extra calibration column in existing software.
- **`FROZEN_calm(+E)`:** calibration on [X_base, m_1, …, m_J], giving one weight set.
- **`FROZEN_census(+E)`:** a head trained on the whole previous-wave population. It is an upper bound, realistic only where y_{t−1} is administrative.
- **Training:** J_train = 20 heads, each trained on an independent D1 sample of the previous-wave population. Replicate r uses head r mod 20 (§7.4).
- **Learner rules** for each head are those of the current-wave learner, after the Appendix A fixes.

**E5 Controls** (§5): `SYN_…` (synthetic Σ_TA y + Σ_TS m̄), `…:nocf` (in-sample fit), `GREG_weak+E:addK` with K ∈ {32, 128} (OLS with many PCs), and the random, permuted, within-cell-permuted and placebo arms.

**E6 Model-based domain estimators** (§7.3): Fay–Herriot and nested-error models.

### 4.3 Learners

| Learner | Target scale | Fit weights (normalized to mean 1) | Hyperparameters (in-sample CV) |
|---|---|---|---|
| ridge-ratio | y/x̃ | d·x̃ | λ ∈ {10⁻⁴, 10^−3.5, …, 10²} (13 values) |
| two-part | 1{y > 0} (logit); y/x̃ on y > 0 | d; d·x̃ | λ_logit, λ_ratio (same grid) |
| ridge-logit | binary | d | same grid |
| linear probability | binary | d | same grid (secondary) |
| kNN | y/x̃ | d in the neighbour average | k ∈ {5, 10, 20, 50}; cosine on E, or Euclidean on [F_std, Z_K] |
| HistGBM (exploratory) | y/x̃ | d·x̃ | max_depth 3, learning rate 0.1, 200 iterations, min_samples_leaf 20; fixed, not tuned |

### 4.4 Feature maps

- **Embedding PCs.** Z_K comes from a population PCA of E, which uses no y and so is a legitimate frame transform. For frozen models the PCA rotation comes from the previous-wave frame.
  - K ∈ {8, 16, 32, 64}. The primary K is chosen on the previous wave (R3).
  - Ratio-scale learners enter Z_K additively on the ratio scale, which is multiplicative in x̃ on the y scale.
  - Linear calibration uses Z_K ⊙ x̃ and Z_K ⊙ ẽmp. The pilot showed that additive raw-scale PCs do worse than plain GREG for skewed y.
- **Full dimension** (`:full`): ridge on all D dims through the SVD of the n_TS × D sample matrix (dual form when D > n_TS).
- **Fine register codes C.** All are BMF fields, so I_D-measurable and available to a text-free modeller:
  - NTEE letter (26 levels + MISSING);
  - NTEE code with ≥ 20 units (206 levels in New England), others pooled to their letter;
  - subsection;
  - ruling decade;
  - BMF asset code.

  Always penalized, never OLS. Linear calibration also uses C ⊙ x̃.
- **Missing text** (births, empty mission): a zero vector plus an indicator column. Never imputed from y.

### 4.5 Tuning rules (non-negotiable)

- **T1. In-sample only.** Every hyperparameter is chosen within each MC replicate by 5-fold CV on s_TS. Nothing is tuned on population y. The one exception is control L5, which tunes on the population deliberately.
- **T2. One penalty parametrization, scale-free [fixes defect (a)].**
  - Columns are standardized with the weighted SD, weights are normalized to mean 1, and the penalty is λ·n_train on the standardized coefficients.
  - The same function is used for CV and for the final fit.
  - Unit tests: predictions are invariant to w → c·w; ‖B̂_λ‖ strictly decreases in λ for every weight vector seen in the harness.
- **T3. The CV loss is the cross-validated anticipated variance on the y scale [differs from the pilot]:**

  L(λ) = Σ_h N_h² (1 − f_h)/n_h · s²_h(e^{oof}), where e^{oof}_k = y_k − m̂^{(−f(k))}_{λ,k} after back-transforming from the ratio scale.

  This is the quantity the estimator's variance depends on. The pilot's loss (a w-weighted squared error on the ratio scale) down-weights large units relative to their contribution to the design variance. For binaries the same formula is used on the 0/1 scale.
- **T4. Folds.**
  - K = 5.
  - **Grouped:** exact normalized-mission duplicates (4.8% of units), plus clusters with cosine > 0.95 on E\*, share a fold.
  - **Design-balanced:** within each stratum, groups are dealt to folds round-robin in random order.
  - **Shared:** within replicate r, the twin and every text arm use the *same* folds, seeded by (cell_seed, r) and not by estimator. This makes paired contrasts sharper.
  - λ is chosen on the whole sample before cross-fitting (one hyperparameter; accepted approximation). Nested CV is an exploratory sensitivity.
- **T5. PCA** on the population frame embedding only.
- **T6. Fixed choices** (rung, field, K, feature map, learner family, λ policy) are set on the previous wave or in this protocol (R3).
- **T7. Bootstrap.** Every fitted object (folds, λ, learner, calibration) is refitted inside each bootstrap replicate.

### 4.6 Variance estimators and intervals

| Code | Formula | Used for | Status |
|---|---|---|---|
| v_L | Σ_h N_h²(1 − f_h) s²_{ê,h}/n_h over take-some strata with n_h ≥ 2, where ê is the final calibration residual (y − c′B̂) or HT's y | all | exists |
| v_g | v_L with g_k ê_k, g_k = w_k/d_k | calibration estimators (StatCan G-Est style) | new |
| v_df | v_L × n_TS/(n_TS − df), with df = tr(C(C′DC + Λ)⁻¹C′D) of the final fit | PCGREG, RIDGECAL, PCCAL. A heuristic for the p/n bias of v_L (Eustache, Dagdoug & Haziza 2025; Bouhadra & Dagdoug 2026; exact forms not retrieved). | new |
| v_B | Rao–Wu–Yue rescaling bootstrap, m_h = n_h − 1: d*_k = d_k[1 − a_h + a_h (n_h/(n_h − 1)) m*_k], a_h = √(1 − f_h), m*_k = number of times k is drawn. B = 300, with **full re-estimation** (T7). | primary and production estimators, first 200 replicates of the primary cell | new |
| v_frozen | v_L with m fixed. Unbiased for V_HT(y − m). | FROZEN_diff | exists |
| v_d | v_L on domain-indicator residuals ê_k 1{k ∈ d} | domain estimators | exists (`sae._dom_var`) |
| v_R | v_L with respondent counts r_h in place of n_h | nonresponse cells (conditional on response homogeneity groups) | exists |
| v_R,B | Bootstrap that re-estimates the propensity model and the outcome model in every replicate | propensity and doubly robust (DR) nonresponse estimators | new |

**Intervals.**
- Wald: θ̂ ± 1.96√v̂ (primary).
- Exploratory: a log-scale interval θ̂·exp(±1.96√v̂/θ̂) for skewed totals, and a logit interval for proportions below 0.10.

**Why coverage fails: the studentized-error diagnostic.**
- For each estimator and y, compute t_r = (θ̂_r − θ)/√v̂_r.
- Report its mean (bias), its SD (> 1 means the variance is underestimated) and its skewness and tail shares (asymmetric failure from heavy tails).
- Undercoverage from skewness affects HT too. Every coverage figure is therefore also reported as a **gap to HT's coverage** for the same y.

### 4.7 Roles, primary pairs and the attribution factorial

**Roles.**
- **SQ (status quo):** HT, GREG_ratio, GREG_size, GREG_size_cat[@reg] and its winsorized version.
- **SQ-ML (text-free twins):** MCAL_frame_ratio, MCAL_frame_codes, MCAL_2part / MCAL_logit / kNN / GBM without text, FROZEN_*_twin.
- **TXT:** every estimator with +E.
- **CTL:** §5.

**Pre-registered pairs.**

| Pair | Text arm | Comparator | Question |
|---|---|---|---|
| **P1 efficacy** [differs from A] | `MCAL_codes_ratio[S+]+E*` (binaries: `MCAL_logit` with the same features) | `MCAL_frame_codes[S+]` | Does text carry information the register's codes do not? |
| P1′ (A's P1, reported alongside) | `MCAL_ratio[S+]+E*` | `MCAL_frame_ratio[S+]` | Text vs a twin with NTEE10 only |
| **P2 production** | `FROZEN_calm[S+]+E*` (frozen, single weight) | `GREG_size_cat[S+]` | Does changing production pay? |
| P2′ (A's P2) | `MCALm[S+]+E*` (cross-fitted, single weight) | `GREG_size_cat[S+]` | The same with current-sample models |

P2 uses the frozen system because it is the deployable one. It is design-unbiased, its variance is exact, and it needs no cross-fitting in production. P2′ is kept for comparison.

**The 2³ attribution factorial (REG-S+, D1, n = 1,000, common random numbers).** Every cell is one estimator.

| A: learner and scale | B: codes | C: text | Estimator |
|---|---|---|---|
| linear ridge calibration | NTEE10 | no | `GREG_size_cat[S+]` (no soft columns) |
| linear | NTEE10 | yes | `RIDGECAL_32[S+]+E*` |
| linear | fine | no | `RIDGECAL[S+]+C` |
| linear | fine | yes | `RIDGECAL_32[S+]+C+E*` |
| ratio-scale model calibration | NTEE10 | no | `MCAL_frame_ratio[S+]` |
| ratio-scale | NTEE10 | yes | `MCAL_ratio[S+]+E*` |
| ratio-scale | fine | no | `MCAL_frame_codes[S+]` |
| ratio-scale | fine | yes | `MCAL_codes_ratio[S+]+E*` |

- **Fit, per y:** log MSE_{abc} = μ + α_a + β_b + γ_c + (αβ) + (αγ) + (βγ) + (αβγ), estimated from the eight MC MSEs.
- **MC uncertainty:** a replicate bootstrap (§8.3).
- **Shapley shares:** each factor's share of log RE(best | production) is its Shapley value, the average of its marginal contribution over the 3! = 6 orderings. This removes the order dependence of a telescoping split.
- **Reading:**
  - a large α (learner and scale) with small γ (text) means "ML, not text";
  - a large γ that disappears once β is present means "text re-expresses the codes";
  - γ surviving β means genuine text information.

A second factorial on the ladder (L2 vs L4, T1 vs T2) is in §6.

---

## 5. Controls

### 5.1 Control table

**Gate** means Stage 2 results are invalid until the control passes or the failure is explained in `working-log.md`.

| ID | Control | Implementation | Expected | If violated |
|---|---|---|---|---|
| **N1** | Random embedding | Gaussian, same dimension as E\* (`embed_random`), through every TXT arm | RE vs twin ∈ [0.97, 1.03]; coverage as the twin | Overfitting or a fold leak. **Gate.** |
| **N2** | Permuted embedding | E\* rows permuted across all units | as N1 | as N1; also catches distribution-only effects. **Gate.** |
| **N2w** (new) | **Within-cell permutation** | E\* rows permuted within NTEE10 × size class × state cells | RE vs twin = the "between-cell" part of the text gain, which the codes could carry. RE(E\*) / RE(N2w) = the within-cell gain. | Interpretation, not failure: the size of the genuinely new information |
| **N2c** (new) | Within-code permutation | E\* rows permuted within fine-NTEE-code × size cells | RE(E\*)/RE(N2c) = text information finer than the 206-level code | interpretation |
| **N3** | Placebo y | gov_grants permuted within design strata; last digit of revenue | RE ≈ 1.00 on every arm | Leakage through strata or a bug. **Gate.** |
| **N4** | Register-type y | text arms on expenses and revenue (REG-S+) | RE ∈ [0.95, 1.05] | Leakage, or a misspecified frame |
| **P1o** | Oracle lag (REG-P) | lagged same-variable on the frame | RE vs S+ of 4–10× for gov_grants, fundraising and officer_comp (ceilings 5.1, 7.7, 9.7) | The harness cannot detect a known large gain. **Gate.** |
| **P2s** | **Planted signal** (§5.2) | semi-synthetic y with known text partial R² | Realized RE tracks the theoretical curve; RE ≈ 1 at ρ² = 0 | Estimator or harness defect. **Gate** for ρ² = 0 and 0.2. |
| **L1** | Same-year text | T1_t in place of T1 | small difference (missions 84% identical) | Text change tracks y change; timing matters |
| **L2** | Numbers masked | T2-nonum vs T2 | ΔRE ≤ 0.03 | The embedding reads amounts |
| **L3** | Late frame | BMF 2026 snapshot as frame | inflated gains on size y | Teaching control: the size of a realistic timing mistake |
| **L4** | Memorization probe | predict log(y_t/y_{t−1}) from T0 and T2 for each rung vs shuffled names | change-R² ≤ 0.02 above random | An API model has seen post-2023 financials |
| **L5** (new) | **Population-tuned λ and K** | λ, K chosen to minimize the population AV (oracle tuning) | MC RE above the in-sample-tuned RE | Quantifies the optimism of oracle tuning, which many published simulations commit |
| **L6** (new) | Ungrouped folds | random folds that ignore near-duplicate clusters | small optimistic shift in RE and coverage | Near-duplicates leak across folds; grouped folds needed |
| **O1** | No cross-fitting | `MCAL_…:nocf` | coverage 72–78% (pilot) | Must stay bad: cross-fitting is necessary |
| **O2** | OLS with many PCs | `GREG_weak+E:add32` and `add128` | coverage collapse (pilot 38–55% at K = 32) | Confirms the p/n warning |
| **O3** (new) | Unpenalized vs penalized at large K | ridge-ratio learner with λ → 0 vs CV λ, at K ∈ {64, 128, full} | The unpenalized version loses efficiency as K grows | Shows what defect (a) would have cost |
| **S1** | Synthetic estimator | `SYN_ratio+E` | bias up to +15% on binaries (pilot) | Shows why the design correction is needed |
| **C1** | Ceiling consistency | MC RE vs same-class ceiling (§8.4) | MC RE ≤ ceiling × 1.05 | Optimistic tuning, a leak, or a ceiling bug. **Gate.** |

### 5.2 Planted-signal family (positive control and power calibration)

**Construction.** On the real population, take a real headline amount y⁰ (gov_grants) and a real binary b⁰ (has_gov_grants). Build:

- **Amount:** y_plant = y⁰ · exp(a·u_k − a²σ²_u/2).
  - u_k is a standardized text index and a is a scalar.
  - The multiplicative form keeps the zeros, the skew and the real frame relationships. Only within-cell variation tied to text is added.
- **Binary:** b_plant = 1{logit⁻¹(logit(p⁰_k) + a·u_k) > U_k}.
  - p⁰_k is a frame-only population logit fit of b⁰.
  - U_k ~ Uniform(0, 1), drawn once with a fixed seed.

**Forms of u (each answers a different question).**

| Form | u_k | Question |
|---|---|---|
| lin-same | w′Z_K(E\*) with fixed random w; orthogonalized against F | Can the estimator recover a linear signal in its own representation? |
| lin-other | the same, built from MiniLM T2 but estimated with LSA (and vice versa) | Representation mismatch: does the signal survive a different embedding? |
| cluster | effect of the 50 k-means clusters of E\*, drawn N(0, 1) | Nonlinear or local signal; kNN vs ridge |
| tail | w′ on PCs 65–128 of E\* | Signal beyond the top K: does truncation lose it? |

**Calibration of a.** For each form, a is set so that the **same-class population partial R² on the AV scale** reaches ρ² ∈ {0, 0.02, 0.05, 0.10, 0.20, 0.40}:

  ρ² = 1 − AV(m_{F+E}) / AV(m_F).

**Use.**
- Plant all forms and levels as extra columns of Y in the same cells. The MC is vectorized over y, so this costs little.
- Report **realized RE vs the theoretical 1/(1 − ρ²)** for each estimator family and n.
- Define the **minimum detectable partial R²**, ρ²_min(n), as the smallest ρ² at which P1's decision rule ("material") passes in at least 4 of 5 independent plantings (different w, U).
  - This runs as a separate "planted cell" with the 10 primary estimators: 4 forms × 6 levels × 5 plantings = 120 extra y columns.
- A null on real y is then reported as "text partial R² < ρ²_min(n) for these y", not as "text has no information".

---

## 6. Embedding ladder and text-field arms

### 6.1 What is held fixed

Every rung uses:
- the same text template (no financial numbers);
- the same frame features, learners, folds, designs and replicates (common random numbers);
- the **same K after projection**. Every rung is reduced to K ∈ {8, 16, 32, 64} population PCs, and also run at full dimension through the dual ridge.

Matched K separates "better semantics" from "more dimensions". Without it, a 1,536-d model is rewarded or punished for its dimension at n ≈ 800.

### 6.2 Ladder

Rung names follow draft A.

| Rung | Representation | Status | Answers |
|---|---|---|---|
| L0 | none (twin) | ready | baseline |
| L0-codes | fine register codes C (no text) | new (§4.4) | Can the register's own codes do what text does? |
| L0-neg | random, permuted, within-cell permuted (N1, N2, N2w) | ready / new | Does the machinery invent gains? |
| L1 | about 30 keyword flags + text length | to build | Can a methodologist without ML get the gain? |
| L2 | TF-IDF (1–2-grams) → SVD-256 (LSA) | ready (T1, T2) | Lexical signal; deterministic |
| L3 | averaged static vectors (GloVe or spaCy) | to build (wheels available) | Distributional semantics without context |
| L4 | MiniLM-L6 (384-d) | ready (T1, T2) | Small contextual model, on-prem |
| L5 | Qwen3-Embedding-0.6B or BGE-M3 | **user action** (Hugging Face blocked) | Strong open weights: the on-prem alternative to Cohere |
| L6 | Cohere embed-v4.0 text: `input_type = classification`, 1,536, float (primary); 256 via Matryoshka and `clustering` (sensitivity) | **user action** (api.cohere.com + production key) | Does a frontier API model add over L2–L5? |
| L6b | Cohere embed-v5.0 text | user action | Vendor-version bridging |
| L7 | Cohere image, or fused text + image, on website screenshots (990) or accounts PDFs (CCEW) | user action + data | Does layout or visual content add beyond text? |
| L7-ext | Structured extraction of lagged amounts from the same PDFs | with L7 | The real competitor of a page embedding: the numbers on the page |
| L8 | Supervised ceiling: a small model fine-tuned on *previous-wave* labels, used frozen | needs a GPU (optional) | How far are off-the-shelf models from what labels allow? |

### 6.3 Comparisons and what each answers

| Comparison | Question | Statistic |
|---|---|---|
| L0-neg vs L0 | Is the harness honest? | RE ∈ [0.97, 1.03] |
| L2 vs L0 | Is there any text signal beyond the register? | RE, P1′ |
| L2 vs L0-codes, and the 2³ factorial (§4.7) | Is the text signal more than finer codes? | γ given β; N2w and N2c ratios |
| L4 vs L2 | Contextual model vs bag-of-words? (Skeptic's bet: no, because sector and size words are lexical.) | Δ log RE |
| L5 vs L6 | Can production stay on-prem? | Δ log RE ≤ log 1.05 |
| L6 vs max(L2, L4, L5) | Is an external API justified? | Δ log RE ≥ log 1.05 |
| T2 vs T1 | Is the gain in the program descriptions (about 70 words) or in the short activity text (about 20 words, the BN analogue)? | Δ log RE |
| T0 vs T1 | Is a name enough (for births)? | RE on the births stratum |
| T2-nonum vs T2 | Numeric leakage | L2 control |
| PCs vs `:full` | Is the y-relevant signal beyond the top K PCs? | Δ log RE at matched learner |
| ridge vs kNN | Linear-in-PCs vs local signal (templates, near-duplicates) | Δ log RE |
| L7 vs L6 on the same units; L7 vs L7-ext | Multimodal increment; page embedding vs extracted numbers | Δ log RE |

**Second factorial: field × model** {L2, L4} × {T1, T2}, with main effects and interaction on log MSE. A model × field interaction (a contextual model helping only with longer text) is plausible and decides which register text an NSO should target.

### 6.4 Scaling experiments

| Factor | Grid | Design | What it shows |
|---|---|---|---|
| n | 250, 500, 1,000, 2,000 (D0); 750–2,000 (D1) | D0, D1 | φ(n) = RE_MC/RE_ceiling; coverage vs p_eff/n |
| K | 2, 4, 8, 16, 32, 64, 128, full | D1, n = 1,000 | The bias–variance trade-off in dimension; the overfitting onset for OLS-GREG vs ridge vs model calibration |
| Dimension within a model | Cohere 256/512/1,024/1,536 (Matryoshka truncation if the equivalence test passes) | D1 | Dimension vs semantics at fixed model |
| Text length | size class (text length grows with size) | D1 | Is the gain concentrated where text is long? |
| National N | n = 5k, 10k, 20k | national D1 | Does φ → 1 as n grows, as theory says? |

**Expected shape, pre-registered.**
- φ rises with n, approaching 1 for ridge and model calibration.
- Coverage of OLS-GREG with K PCs falls as K/n rises, following the §1 table of note 01: about 92% at n = 200, p = 20; about 86% at p = 50.
- Ridge and model calibration hold coverage at K ≤ 64 with cross-fitting.

**Proxy validity.** Across rungs, compute the Spearman correlation between three things:
- (i) the Stage-1 same-class ceiling;
- (ii) the text-coded NTEE accuracy (a cheap proxy task);
- (iii) the MC RE.

If (i) ranks the rungs as (iii) does (ρ_S ≥ 0.8), future model choices (Cohere v5, Qwen) can be screened at Stage 1 alone, without a full MC.

### 6.5 Selection of E\* (pre-registered)

The rule is the same as draft A §6.3.
- On the **previous wave**, compute the same-class Stage-1 ceiling for every local rung × field × K.
- E\* = argmax of the GM over H of the ceiling RE vs the codes twin.
- Ties within 0.02 go to the simpler rung (L2 before L4).
- The choice is frozen in `config/protocol_v1.yaml` before any current-wave MC.
- Later rungs (L5, L6) are **secondary** arms. They are never re-selected as primary.
- **Fallback if the previous wave is not ready:** E\* = MiniLM T2. The best pilot arm (LSA T2) is barred from primary by fallback.

### 6.6 Cohere specifics

These follow note 04.
- **Settings:** `embed-v4.0`, `input_type = classification`, 1,536, float. The texts are embedded once and cached by content hash, with a manifest. The API is never called inside the MC.
- **Cost and time:** New England is about 20k texts per field, under $0.50 and about 10 minutes. The national file is about 330k texts, about $2–3 and about 2.5 hours at 2,000 inputs per minute.
- **Pre-flight tests (cents):** repeatability, batch invariance, Matryoshka equivalence, int8 vs float, and a canary set.
- **Multimodal only where images exist.** Never render 990 XML to images: the page would contain y, and no NSO would do it.

---

## 7. Uses beyond calibration

### 7.1 Stratification and allocation

- **Arms:** D1-imp, D2a, D2b (with its twin), D2c (§3.2).
- **Estimators:** each design is run with GREG_size_cat[S+], MCAL_frame_codes[S+] and FROZEN_cal[S+]+E\*. The design gain is isolated with text-free estimators first; the factorial then tests substitution (§3.4).
- **Metrics:**
  - RE vs D1 (or D1-H44 for D2a) per y, and GM over A and over H;
  - share of MSE from stratum jumpers;
  - realized RE vs the predicted 1 − ρ²(1 − 1/L²), using the population ρ between the score and y within size classes;
  - for D2b, the anticipated vs realized within-stratum S_h.
- **Skeptic's prediction:** stratification is capped by ρ². A score with population partial R² of 0.1 within cells gives at most about 1.07–1.10 from two score classes. D2a ≤ 1.05.
- **The likeliest design-stage win is D1-imp.** It only repairs the 31% MISSING NTEE, which matters for domain estimation by NTEE10 more than for national totals.

### 7.2 Nonresponse adjustment

**Mechanisms.** The real NR-LATE plus simulated mechanisms. The simulated ones fill in the Little–Vartivarian 2 × 2: does the covariate relate to response, to y, to both or to neither?

| Code | Mechanism (take-some) | Response rate | Should text help? |
|---|---|---|---|
| NR0 | MCAR | 65% | no (sanity) |
| NR1 | logit p = a + b·log(1 + lag_exp) + NTEE effects | 65% | no |
| NR2 | logit p = a + γ·u_k, with u_k a frozen text score of gov-grant intensity | 65% | yes, by construction (relates to y and response) |
| NR2′ | logit p = a + γ·v_k, with v_k a text index orthogonal to every y in H | 65% | **no: text predicts response only.** Expected: variance up, bias unchanged. |
| NR3 | logit p = a + δ·log(1 + y_gov_grants,t) (MNAR) | 65% | partly, through correlation with y |
| NR4 | p depends on website presence | 65% | missing-not-at-random auxiliary |
| **NR-LATE** | real: filed ≤ 300 days after FYE | 61% of take-some | **the real test** |

δ and γ are set so that the respondent / nonrespondent ratio of mean gov_grants is about 0.35, matching NR-LATE.

**Take-all nonresponse.** As in draft A §7.2. Take-all units respond at 90% (follow-up scenario) or by the real late-filing indicator. Nonrespondents are imputed by historical ratio (SQ), a kNN text donor (TXT) or a kNN frame donor (SQ-ML).

**Take-some adjustment families.** All outcome models are cross-fitted on the respondents. All propensity models are cross-fitted on the sample.

| Code | Method | Formula | Role |
|---|---|---|---|
| NR_naive | none | Σ_{s_r} d_k y_k | CTL |
| NR_rhg | weighting class = design strata | Σ_c (N̂_c/N̂_{r,c}) Σ_{s_r ∩ c} d_k y_k | **SQ** (exists) |
| NR_rhg + GREG_size_cat | the same, then calibration to T_X | — | **SQ production** (exists) |
| NR_psc(z) / NR_psc(z, e) | score method: 5 quantile classes of a cross-fitted ridge-logistic p̂ (Haziza & Beaumont 2007) | weighting class on the p̂ classes | SQ-ML / TXT |
| NR_pmc(z, e) | **predictive-mean classes**: 5 quantile classes of m̂(z, e) for the key y | weighting class | TXT (Little–Vartivarian: predict y, not only response) |
| NR_joint(z, e) | 3 × 3 cross of p̂ classes and m̂ classes, collapsed to keep ≥ 10 respondents per cell | weighting class | TXT |
| NR_2step(z, e) | inverse propensity, then calibration (Haziza & Lesage 2016) | w⁰ = d/p̂; calibrate w⁰ to (T_X, Σ_U m̄) | TXT (the existing `ps` + `mcal` spec, after the fixes) |
| NR_cal(z, e) | calibration-only (one-step nonresponse calibration on [X_base, m̂]) | — | TXT (exists: `NR_rhg_MCAL_ratio+E`) |
| **NR_DR(z, e)** | augmented IPW with cross-fitted nuisances (Kim & Haziza 2014; Dagdoug & Haziza 2026) | Σ_U m̄_k + Σ_{s_r} (d_k/p̂_k)(y_k − m̂_k) | TXT (new) |
| NR_imp_knn(z, e) | donor imputation: nearest respondent on [log x̃, Z_K] within NTEE10 | Σ_{s_r} d y + Σ_{s_nr} d y_donor | TXT (production imputation) |

Each TXT method has a text-free twin: the same method with (z) only. Propensities are clipped at 0.02.

**Stage-1 exact bias diagnostic (no MC).** Response is deterministic per unit, so each method's limiting bias is a population quantity. Fit the method on the whole of U_TS, as if the population were the sample (d = 1):

- **Weighting class:** B = Σ_c N_c (ȳ_{r,c} − ȳ_c).
- **DR:** B = Σ_{U_TS} (r_k/π*_k − 1)(y_k − m*_k), where π\* is the population-fitted propensity and m\* the outcome model fitted on population respondents.
- **Calibration:** B = Σ_{U_TS} (r_k g*_k − 1) y_k, with g\* the population-limit calibration factors on respondents.

This ranks the methods by achievable bias before any MC, and pre-registers the bias each one should show. The MC then adds variance, coverage and finite-sample behaviour.

The pilot found that calibrating to lagged size *increases* bias for survey-only y. That finding becomes a testable statement: B(NR_rhg + GREG_size_cat) > B(NR_rhg) for gov_grants, contrib and fundraising.

**Metrics.**
- Relative bias.
- **Share of bias removed** vs NR_rhg + GREG_size_cat[S+]: (|B_SQ| − |B_x|)/|B_SQ|, with an MC CI.
- Variance; RRMSE.
- Kish factor 1 + CV²(adjustment factors).
- Coverage, with v_R or the bootstrap v_R,B for propensity and DR estimators.
- The Little–Vartivarian table from NR0–NR3 and NR2′.

### 7.3 Small area estimation

**Domains.**
- New England: state × NTEE10 (66 domains). At n = 1,000 the median expected domain sample is about 8, and 41% of domains expect fewer than 5.
- National: state (51) and state × NTEE10 (about 560).

**Estimands.** Domain totals of gov_grants and fundraising_exp; domain proportions of has_gov_grants and grants_to_indiv_us.

**Estimators.**

| Code | Estimator | Role |
|---|---|---|
| DIR_HT | Direct HT | floor |
| DIR_GREG | Domain estimate from the national GREG_size_cat[S+] weights | **SQ** (production publishes these) |
| DIR_FROZEN_calm+E\* | Domain estimate from the single-weight frozen-text weights | TXT (single weight) |
| MA_dom(+E) | Σ_{U_d} m̂ + Σ_{s_d} d (y − m̂), with a national cross-fitted model (exists: `MA_`) | TXT (design-consistent) |
| SYN(+E) | Σ_{U_d} m̂ | CTL (bias) |
| FH_frame | Fay–Herriot on DIR_GREG domain means. Covariates: intercept + the domain mean of a frozen *frame-only* prediction m̂_F | **SQ-SAE** |
| FH_text | FH_frame + the domain mean of a frozen *text* prediction m̂_{F+E} (p = 3 ≪ m) | TXT |
| FH_textPC | FH_frame + domain means of the first 3 PCs of E\* | TXT (stress: unsupervised area covariates) |
| NER_text (optional) | unit-level nested-error EBLUP on the ratio scale y/x̃ with F + Z_8 | TXT (model-based unit level) |

**Fay–Herriot specification** (replaces the current `sae._fh`, which has one covariate, no intercept and raw ψ):
- θ̂_d = direct mean; ψ_d smoothed by a generalized variance function, log ψ̂_d = a + b log n_d + c log N_d, fitted per replicate over domains with n_d ≥ 2;
- σ²_u by REML, truncated at 0;
- the EBLUP, and the synthetic x_d′β̂ for domains with n_d < 2;
- MSE by Prasad–Rao.
- Covariates come from **frozen** models applied to the full frame. They have no sampling error and do not depend on the current sample.

**Metrics.**
- Per-domain RB and RRMSE over replicates.
- AARB = D⁻¹ Σ_d |RB_d| and ARRMSE = D⁻¹ Σ_d RRMSE_d. Domains with θ_d = 0 are excluded and counted.
- The distribution of per-domain RE.
- Design-based coverage of FH intervals, per domain and averaged.
- MSE-estimator relative bias, averaged over domains.
- The **number of domains with true CV ≤ 25%** (publishable).

Everything is reported by n_d class (0, 1–4, 5–19, ≥ 20), and **non-sampled domains separately**.

**Methods note.** FH's MSE estimator targets the model MSE averaged over domains, while design-based per-domain coverage varies with the domain's own random effect. Average coverage near 95% can hide domains at 60%. The coverage *distribution* is therefore reported, not only its mean.

### 7.4 Frozen previous-wave deployment

**Procedure** (`frozen.py`, after the fixes):

1. Draw J = 20 independent D1 samples (n = 1,000) from the previous-wave population (frame TY2021 → y TY2022).
2. On each sample, fit the learner of §4.3 (y-type-specific) on F + C + Z_K. The PCA rotation is fitted on the TY2021 frame. λ is chosen by in-sample CV.
3. Apply each head j to the current frame (TY2022 frame + TY2022 text, rotated with the *old* PCA) to get m_j(k) for all of U_TS.
4. Current-wave replicate r uses head j = r mod J. This averages over the luck of the training sample. The spread of RE across heads is reported as **training-sample risk**.

**Variants.**
- `FROZEN_diff`, `FROZEN_ppi`, `FROZEN_cal` and `FROZEN_calm` (§4.2), each with a text-free twin.
- `FROZEN_census`: a population-trained head (upper bound).
- `FROZEN_update`: the frozen m as an offset, plus a current-sample cross-fitted NTEE10 × ratio intercept correction. A light transfer-learning step that absorbs drift.
- λ policy frozen (the previous-wave λ) vs re-tuned. In production the head is fixed; the comparison shows whether that costs anything.

**Model ageing.** Report the ceiling R² of each head on the wave it was trained for vs the next wave. A drop of more than 20% of R² flags drift. Text is near-static, so the skeptic expects little ageing for text and more for lag-based features.

**Expected.**
- FROZEN_cal is close to cross-fitted MCAL: RE ratio ≥ 0.9.
- Its coverage sits at nominal minus HT's skewness gap, because the variance is exact for fixed m.

The pilot frozen run is in progress (`mc_ne_frozen_n1000.log`). It uses the unpenalized learner (defect a), so rerun it after the fix.

---

## 8. Metrics and Monte Carlo protocol

### 8.1 Point and interval metrics

For estimator a, variable y, replicates r = 1..R, estimates θ̂_r, variance estimates v̂_r and truth θ:

| Metric | Formula | MC SE |
|---|---|---|
| Relative bias | RB = (θ̄̂ − θ)/θ | sd(θ̂_r)/(√R \|θ\|) |
| MSE, RRMSE | MSE = R⁻¹ Σ(θ̂_r − θ)²; RRMSE = √MSE/\|θ\| | delta method |
| Relative efficiency | RE(a \| b) = MSE_b/MSE_a, on paired replicates | Var(log RE) ≈ R⁻¹[Var(e_a²)/MSE_a² + Var(e_b²)/MSE_b² − 2Cov(e_a², e_b²)/(MSE_a MSE_b)] (`mc.metrics`) |
| Coverage | C = R⁻¹ Σ 1{\|θ̂_r − θ\| ≤ 1.96√v̂_r} | √(C(1 − C)/R) |
| Coverage gap to HT | C_a − C_HT | paired binomial |
| Variance relative bias | (v̄̂ − MSE)/MSE, and (v̄̂ − Var)/Var | delta method |
| Studentized error | mean, SD, skewness of t_r = (θ̂_r − θ)/√v̂_r | bootstrap over r |
| CI width | 2 × 1.96 × mean√v̂_r/\|θ\| | — |
| Estimated CV | CV̂_r = √v̂_r/θ̂_r; band shares (§8.7) | — |
| Fitting efficiency | φ = RE_MC(a \| twin)/RE_ceiling(a \| twin) | delta method (ceiling treated as fixed) |
| Gap closed | G_rung = log RE(rung \| twin)/log RE(best \| twin) | bootstrap over r |
| Failure rate | share of replicates with NaN, non-convergent calibration or out-of-bounds weights | > 0.5% is a defect |

**Robust companions for heavy-tailed y.** MSE is dominated by a few replicates. For the H-class y, also report the median absolute relative error and the MSE with the top 0.5% of |errors| trimmed. They are reported, but they never drive a decision.

### 8.2 Aggregation across y and subgroups

- **GM-RE over a set Y:** exp(|Y|⁻¹ Σ_{y∈Y} log RE_y). Its MC CI comes from a replicate bootstrap (§8.3).
- **Worst-case RE over A:** min_y RE_y, the multipurpose-damage check, with a bootstrap CI.
- **Win/loss count:** y with lower MC 95% bound > 1.05 (win) or upper bound < 0.95 (loss).
- **Binding-variable RE:** the RE of the y that sets the sample size (§8.7).
- **By size class:** the take-some MSE contribution of each size class, for each y, under twin and text. This shows where the gain lives.

### 8.3 Replicates, seeds, common random numbers

**Replicates.**

| Use | R | Coverage MC SE |
|---|---|---|
| Screening | 200 | — |
| Standard | 1,000 | 0.69 pp |
| **Confirmatory primary cell** | **2,000** | 0.49 pp |
| Planted cell | 1,000 | — |
| Bootstrap check | 200 replicates × B = 300 | — |

**Precision.** At R = 60 the pilot's RE SE was about ±0.15. It scales as R^−½, so it is about ±0.037 at R = 1,000 and about ±0.026 at R = 2,000. That resolves 1.10 from 1.05.

**Common random numbers.**
- Within a cell, every estimator and every embedding arm sees the same R samples.
- Within replicate r, all arms share folds (T4).
- Register scenarios S, S+ and P share D1 samples. W uses D1-W.

**Seeds.**
- The sample for replicate r comes from `default_rng([cell_seed, r])`.
- Folds come from `default_rng([cell_seed, r, 1])`.
- An estimator's other randomness (kNN ties, bootstrap) comes from `default_rng([cell_seed, r, hash(name)])`.
- Adding or removing an estimator must not change any other estimator's numbers. This is regression-tested.

**Replicate bootstrap** for GM-RE, Shapley shares and factorial contrasts: resample replicate indices (B = 2,000). This keeps the within-replicate correlation across y and estimators.

### 8.4 Ceilings (Stage 1) and the consistency gate

**Same-class AV ceiling.** For each MC estimator, take the same features, the same target scale, the same loss weights (ratio: x̃) and the same learner family. Fit it on the whole of U_TS with population 5-fold cross-fitting, with λ chosen on the population: an oracle, labelled as such. Then

  AV(m*) = Σ_h N_h²(1 − f_h)/n_h · S²_{y−m*,h}, and RE_ceiling(a | b) = AV(m*_b)/AV(m*_a).

The current `diagnostics.py` uses a raw-scale ridge with weights d − 1. It is *not* same-class for the ratio-scale MCAL arms. That explains at least part of the pilot inconsistency: MC RE of 1.43 for gov_grants against a ceiling of 1.13. The other candidates are R = 60 noise and the different baselines.

**Learning-curve ceiling (Stage 1b; new and cheap).**
- Draw B = 50 design samples of size n. Fit the learner on each, as the MC would, without calibration or variance.
- Evaluate AV(m̂_s) on the population with m̂_s held fixed.
- The mean over the samples is the anticipated variance of a "frozen-from-a-sample-of-size-n" model. It predicts φ(n) at about 1/20 of the MC cost.

**Gate C1.** MC RE ≤ same-class ceiling × 1.05, with MC uncertainty. A violation is a defect to explain, not a result.

### 8.5 The p/n arithmetic used for pre-registered predictions

For a cross-fitted learner with p_eff effective parameters fitted on n_TS units, the out-of-sample residual variance is about S²_e(1 + p_eff/n_TS). This is the OLS/ridge excess-risk approximation; the GREG analogue is in note 01 §1. Hence

  RE_MC(text | twin) ≈ RE_ceiling × (1 + p_twin/n_TS)/(1 + p_text/n_TS).

With p_twin ≈ 28 and p_text ≈ 60 at n_TS = 794, the factor is 0.96. Under ridge, p_eff = df < p, so the factor is closer to 1. That is the case for penalization once defect (a) is fixed. The planted cell measures the factor directly.

### 8.6 Replicate populations and meta-analysis

The MC CI quantifies sampling variability for one fixed population. A production decision needs to know how the verdict varies across populations and years. Primary endpoints and key secondaries are therefore re-estimated on:

| Population | Status | Notes |
|---|---|---|
| p1: New England TY2022 → TY2023 (primary) | built | |
| p2: New England TY2021 → TY2022 (previous wave) | built | Primary arms only, with frozen heads trained on a TY2020 → TY2021 wave if that is buildable; otherwise cross-fitted only |
| p3–p11: the nine US Census divisions | after the national build | New England is one of them, so p1 overlaps p3 |
| p12: CRA T3010 | user action | |

**Pooling.** A random-effects meta-analysis of log RE_p (REML), with within-population variance = MC SE². It gives:
- the pooled log RE;
- τ, the between-population SD;
- a **95% prediction interval for a new population**.

The populations overlap (New England is inside the national file; waves share units), so τ is read as *heterogeneity across settings*, not as a sampling model. A decision of "production-worthy" (§9) requires the prediction interval's lower bound > 1.0.

### 8.7 Equivalent sample size, binding constraint, cost and CV bands

- **Cost curve.** Run the production baseline at n ∈ {750, 875, 1,000, 1,250, 1,500, 2,000} and interpolate log MSE_y(n) monotonically in log n.
- **Equivalent n:** n_eq,y solves MSE_base,y(n_eq) = MSE_text,y(n).
- **Take-some saving:** σ_y = 1 − (n − n_TA)/(n_eq,y − n_TA). Without the fpc, σ ≈ 1 − 1/RE.
- **Binding constraint.** Pre-registered CV targets for the hypothetical survey:
  - national totals CV ≤ 5% (gov_grants, fundraising) and ≤ 10% (volunteers);
  - national proportions CV ≤ 5%;
  - state totals CV ≤ 15%.

  Compute the minimum n meeting *all* targets under each system, using the Stage-1 AV functions rescaled by the MC/ceiling ratio. The saving is in that minimum n.
- **Cost:** ΔC = Δn × c_u − C_text.
  - c_u = cost per sampled unit; default CAD 200 **[assumption; replace with the programme's figure]**.
  - C_text = the pipeline's annual cost.
- **CV bands** (StatCan: A < 5%, B 5–10%, C 10–15%, D 15–25%, E 25–35%, F ≥ 35%; **confirm with the programme's guideline**). Report the band distribution of the true CV and of the estimated CV for national, state and state × NTEE10 estimates.
- **Rule printed on every table:** RE = 1.2 shrinks CVs by only 1 − 1/√1.2 = 8.7%.

### 8.8 Weight diagnostics (single-weight estimators)

For RIDGE1W, PCCAL_pos, MCALm and FROZEN_calm:
- share of negative weights and of w < 1;
- quantiles of g = w/d;
- CV of the weights;
- Kish deff = n Σw²/(Σw)²;
- share of replicates hitting the bounds;
- calibration non-convergence rate.

### 8.9 Standard result tables (every cell)

1. RE vs the codes twin, vs the NTEE10 twin, and vs production, per y, with MC CI.
2. RB, RRMSE, coverage, coverage gap to HT, variance relative bias, studentized-error moments.
3. GM-RE over H and worst case over A.
4. φ and the ceiling-consistency ratio.
5. Weight diagnostics.
6. Timing.

Each table carries the cell manifest: population hash, design, register, n, R, seeds, config hash, git commit, embedding hashes.

---

<!-- PART4 -->
