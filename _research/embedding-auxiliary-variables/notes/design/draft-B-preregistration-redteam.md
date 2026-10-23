# Draft B: Pre-registered benchmark protocol, with red team, for "embeddings as auxiliary variables"

**Status:** draft for review, 2026-10-23. Lens: adversarial reviewer and meta-scientist.
**Aim of this draft:** make the benchmark *falsifiable*. Every hypothesis has a numeric pass/fail threshold fixed before the run, the skeptic's predicted magnitude next to it, the strongest text-free comparator, and a control that would expose a leak or an overfit.

**Inputs read:**
- `notes/early-findings.md` and `notes/working-log.md`;
- research notes 01–04 (note 06 did not exist when this was written);
- all of `src/`;
- the skeptic's script `scratchpad/skeptic/falsify_pop.py` (redundancy controls adopted in Stage 1).

**New facts computed for this draft** (scratchpad script `draftB/facts.py`):
- per-division population sizes;
- how many TY2022 returns had been filed by each candidate design date;
- take-all and zero shares by y.

**Conventions:**
- **[USER]** marks a step that needs the user's action (network allowlist, API key, download, sign-off).
- **[VERIFY]** marks a factual claim that must be checked against a primary source before it is relied on.
- **[CODE]** marks an implementation change needed in `src/`. The full list is in Appendix A.

---

## Executive summary (one screen)

1. **The question.** Once the frame's conventional information is already used *as well as a competent methodologist would use it*, how much **design-based** variance reduction does an embedding of frame-time text add? The text is register descriptions, mission statements, websites or report PDFs. Variance reduction is measured as RE on MSE, with valid coverage, for totals and domain totals in an annual non-profit survey.
2. **Primary contrast: text vs its text-free twin.** The headline comparison is the *same* estimator and learner with and without text. Text versus HT, or text versus a deliberately thin GREG, is not a headline result. The "pilot gains" against `GREG_size_cat` mix three things: the text, ratio-scale ML, and a baseline that never saw half the BMF fields.
3. **Confirmation on held-out populations.** New England has been looked at and is now a development population. Confirmation runs on **8 held-out US Census divisions**, 14.7k–58.2k continuing units each and 285.6k in total. Each division is one replicate population. Generalization is judged across divisions, not across Monte Carlo replicates.
4. **Four frame tiers, tied to real registers.**

   | Tier | Contents | Real-world analogue |
   |---|---|---|
   | F0 | codes only | BR unit without tax accounts; new charity |
   | F1 | summary register: codes plus lagged revenue, expenses, assets, employees | BR with T2/T1044 and PD7A |
   | F1p | F1 plus payroll | PD7A payroll |
   | F2 | full prior-year return | prior-year T3010 for registered charities |

   The pilot's "oracle lag" is not an oracle in Canada. StatCan holds the prior-year T3010, so F2 is the *realistic* frame for every T3010 item.
5. **Gates come before claims.** No efficiency claim is made unless these all pass: negative controls (random, permuted and within-stratum-permuted embeddings), placebo y, a known-answer synthetic y, an overfitting demo, a check that the Monte Carlo gain does not exceed the anticipated-variance ceiling, and coverage ≥ 93%.
6. **Pre-registered, falsifiable core claim (H1).** Under F1, the geometric-mean RE of the primary text estimator against its twin, over the four "detail" y, is ≥ 1.10 averaged across divisions, with a lower bound above 1.
   - Pilot-implied (proponent) range: 1.15–1.30.
   - **The skeptic predicts 1.00–1.08.** This draft argues that most of the pilot's gain will be absorbed by foundation code, subsection, fine NTEE and ratio-scale modelling.
7. **Production needs more than H1.** It also needs:
   - a **single weight per record** (multipurpose calibration);
   - gains that survive a **frozen model trained on the previous wave's sample**;
   - an **open-weight model within 5% of Cohere** (confidential register text cannot go to a SaaS API);
   - replication on **CRA T3010**.

---

## Red-team findings that change the setup (read before the protocol)

These come from auditing the pilot code and data against the real-world scenario. Each is fixed or tested in the protocol.

| # | Finding | Evidence | Consequence if ignored | Fix in protocol |
|---|---|---|---|---|
| RT-1 | **The baseline is starved of BMF fields.** The BMF has `FOUNDATION`, `AFFILIATION`, `CLASSIFICATION`, `ACTIVITY`, `DEDUCTIBILITY`, `ORGANIZATION`, `ACCT_PD`, `FILING_REQ_CD`, `GROUP`, `ASSET_AMT`, `INCOME_AMT`, `REVENUE_AMT`, full `NTEE_CD` and `SORT_NAME`. `ne_pop.parquet` keeps none of foundation, affiliation, classification, activity, ACCT_PD or the BMF amounts. `frame_matrix`/`build_mats` use only NTEE10, state and income code. | BMF header checked; parquet has 53 columns and no `foundation`. The foundation code encodes the public-support basis: 170(b)(1)(A)(vi) donations versus 509(a)(2) gross receipts, hospital, school, church **[VERIFY code meanings in the IRS EO BMF dictionary]**. That is close to the *revenue mix* the text is credited with predicting (contributions, program revenue, government grants). | Text gains on D-class y are inflated. | Build the **BMF-max** frame (§2.4). The twin and B* use it. Hypothesis H8 measures how much of the pilot gain it absorbs. |
| RT-2 | **The pilot frame assumes every unit's prior-year return is available at design time.** | Among continuing units, the share with the TY2022 return submitted by the design date (by units / by revenue) is 70.0% / 48.1% at 2023-12-31, 80.5% / 56.1% at 2024-03-31, and 94.4% / 87.7% at 2024-06-30. Large units file late: 96.0% (≤$200k) against 90.2% (>$25M) at 2024-06-30. These figures come from the latest submission per EIN, so they are lower bounds. | Lagged amounts *and* text are overstated exactly where the totals are. | An availability flag `avail_D`. Fallbacks: BMF amounts, then TY2021 text, then name only (§2.3). |
| RT-3 | **The headline pilot gains are on variables a Canadian frame already carries.** Salaries (1.4–1.7×) are on PD7A payroll. Government revenue, donations and program revenue are on the prior-year T3010 for charities. | Working-log reasoning; T3010 line list in note 03 §6. | Gains on y that a real NSO would not survey, or would calibrate to lagged values. | Tiers F1p and F2. Classify each y by register status (§1.3). Hypotheses H5 and H7. |
| RT-4 | **The population conditions on the future.** U = units filing a full 990 in *both* years. Units that died or switched to 990-EZ are dropped (1,115 deaths and 1,414 births in NE). | `build_population.build` uses an inner merge. | Inflates the R² of lagged size, for all estimators. It hides "frame units that become zero". | Sensitivity population P2, defined on the frame at t−1 (§2.6). |
| RT-5 | **New England is burned for confirmation.** The pilot, the K choices (10/32 PCs), "LSA full text works best" and the estimator forms were all chosen on NE. | Pilot logs. | Garden of forking paths. | NE = development only. Confirmation = 8 held-out divisions with frozen specifications (§9.5). |
| RT-6 | **Model calibration gives y-specific weights.** An NSO publishes one weight per record. | `mc._estimate` calibrates on [X_base, m̂_j] separately for each y. | The gains are not deployable as designed. | Multipurpose estimators `MCALM_*` and `RIDGEM_*` (§4). Hypothesis H14. |
| RT-7 | **The 32 unsupervised PCs are a bottleneck.** A better embedding may put y-relevant signal outside its top-32 variance directions. | `build_mats` uses `pcs(E, 32)`. | The ladder is biased *towards the skeptic*: models would look equal because the PCs hide their differences. | Add a supervised full-dimension ridge arm, `:fullD` (§6). |
| RT-8 | **The ladder comparison in the pilot is confounded.** It compares "LSA full text" with "MiniLM mission". MiniLM also truncates at 256 tokens, which bites hardest on large units' long program text; Cohere has a 128k context. | `early-findings.md`; `embed.embed_minilm(max_len=256)`. | Model quality is confounded with text field and with context length. | Hold the field fixed for every rung. Log the truncation rate by size class (§6). |
| RT-9 | **The weak-frame scenario borrowed the strong-frame design.** `GREG_weak` ran on a design stratified and take-all'd on *lagged expenses*. | `frame.production_design(size_col="lag_expenses")` is used for every scenario. | Weak-register efficiency is overstated and text's weak-frame gain is mismeasured. | Each tier's design uses only that tier's information (§3.1). |
| RT-10 | **Common random numbers break across runs.** `run_cell` uses one `rng` both to draw samples and for CV folds inside estimators. | `mc.run_cell` / `_estimate(…, rng)`. | Adding or removing an estimator changes every later sample. Paired contrasts across runs (ladder rungs run in separate cells) are no longer paired. | Separate seed streams (§8.3) **[CODE]**. |
| RT-11 | **The Monte Carlo gain and its ceiling use different baselines.** The anticipated-variance (AV) ceiling is AV(strong)/AV(strong + text) on an additive scale. The MC RE is against `GREG_size_cat` with a ratio-scale model. For government grants the MC gain (1.4) is above the "ceiling" (1.17). | `diagnostics.py` vs `mc.py`. | Gate G5 (gain must not exceed ceiling) cannot be applied. | Compute the ceiling for the *exact twin pair* on the same scale **[CODE]**. |
| RT-12 | **Late filing inverts the usual survey response gradient.** Big units file late on extension; in surveys big units usually respond *more*, because they are chased. The harness then assumes take-all units respond. | Response rate 60% by count, 18% expense-weighted. | NR results may not transfer. | Simulated mechanisms NR-sim A/B/C. A sensitivity with take-all nonresponse handled by ratio imputation from lagged values (§7.2). |
| RT-13 | **The lag and y come from the same instrument.** Correlation between lagged and current values (0.99 for revenue) is higher than survey-vs-register correlation in practice, where concepts differ. | Note 03 §0. | Favours the conventional baseline: **conservative for text**. | State it in the translation section (§12). Do not "correct" it. |
| RT-14 | **Edit rule E1 copies the lagged value into y_employees.** | `apply_edits` E1. | Slightly inflates the R² of lagged size for employees. | Report the edit count. Sensitivity run with no edits. |
| RT-15 | **The `ridge` kind tunes λ on ungrouped folds.** | `_ridge_cv_lambda(…, groups=None)` is called from the `ridge` branch. | Near-duplicate texts give optimistic λ. | Use grouped and stratified folds everywhere **[CODE]**. |

---

## 1. Research questions and estimands

### 1.1 Research questions

| RQ | Question | Answered by |
|---|---|---|
| RQ0 (validity) | Can embedding-assisted estimators be design-valid at realistic n? That means bias negligible against RMSE, coverage ≥ 93%, and the variance estimator unbiased to within ±15%. | Gates G1–G6 |
| RQ1 (calibration) | What efficiency does text add over the strongest text-free alternative, by frame tier and y class? | H1–H8 |
| RQ2 (heterogeneity) | Does the gain replicate across populations (8 divisions; CRA T3010)? Is it predictable from population diagnostics? | Between-division analysis; prediction-accuracy check |
| RQ3 (representation) | How much of the gain is keyword-level, lexical (LSA), contextual (MiniLM), frontier (Cohere text) or visual (page images)? | Ladder, H9 and H15 |
| RQ4 (other uses) | Stratification, nonresponse adjustment, small area estimation, frozen previous-wave deployment. | H10–H13 |
| RQ5 (translation) | What does a gain buy: sample size, cost, burden, CV bands? Is it deployable under confidentiality and reproducibility constraints? | §12; H14; governance criteria |

### 1.2 Estimands

All estimands are finite-population quantities of the pseudo-census U at reference year t (TY2023).
- **θ₁ (primary): the total.** t_y = Σ_{k∈U} y_k.
- **θ₂: the mean.** ȳ_U = t_y/N. N is known, so every *relative* metric for θ₂ is identical to θ₁'s. **No separate mean tables are produced**: doing so only multiplies tables. Means are reported only as proportions for binary y.
- **θ₃: domain totals** t_{y,d} for planned domains (state × NTEE10) and unplanned small domains (ZIP3). Used in Stage 4 (SAE).
- **θ₄: ratios of totals**, R_y = t_y / t_revenue. Examples are the government-funding share of revenue and the program-revenue share. These are typical non-profit satellite-account indicators. Estimated as the ratio of the two estimated totals, with variance linearized on z_k = (y_k − R̂ x_k)/t̂_x. Secondary.
- **θ₅: change**, t_{y,t} − t_{y,t−1}. Used *only* as a memorization probe (test RT-G2). Text should carry no information about change beyond what the frame has.

### 1.3 The y set, by register status (fixed now; nothing is added or dropped later)

The class depends on whether a *real* NSO frame would carry the variable. That determines the realistic comparator.

| Class | Meaning | y (code names) | Realistic comparator tier | Role |
|---|---|---|---|---|
| **R**: register-type | On summary registers through tax or payroll accounts | `y_revenue`, `y_expenses`, `y_employees`; `y_salaries` (payroll: PD7A in Canada) | F1 (salaries: F1p) | **Negative control.** Expected RE_inc ≈ 1. |
| **D**: detail admin items | On detailed returns (990 Part VIII/IX, T3010 Section D/Schedule 6), not on summary registers | **Headline D4:** `y_gov_grants`, `y_contrib`, `y_prog_rev`, `y_fundraising_exp`. **Secondary:** `y_officer_comp`, `y_grants_paid` | F1 (headline); F2 for registered charities | **Headline for H1.** |
| **S**: survey-type | On no summary register; on the 990 checklist in the US but not on any Canadian summary register **[VERIFY which are on the T3010]** | **Headline S5:** `y_volunteers_cap10k`, `y_bin_has_volunteers`, `y_bin_grants_to_indiv_us`, `y_bin_foreign_activities`, `y_bin_lobbying`. **Secondary:** `y_bin_fundraising_activities`, `y_bin_gaming`, `y_bin_has_gov_grants`, `y_bin_has_employees` | F1 | **Headline for H2.** |
| **H**: heavy-tail stress | Dominated by a handful of units | `y_volunteers` (raw). Knights of Columbus CT holds 42% of the NE total. | F1 | Robustness only; never in aggregate scores |
| **P**: placebo | Text cannot know these | `PLC_perm` (y_gov_grants permuted within design stratum × lagged-size decile); `PLC_noise` (lognormal noise, σ = 2); `PLC_bern` (Bernoulli(p_h) within strata) | F1 | Gate G3 |
| **K**: known answer | Synthetic, with a known text-explained share | `KA_text` (§5.4) | F1 | Gate G6 |

**Why these y.** R-class y are where a real NSO already has admin data, so if text "helps" there, something is leaking. D-class y are the revenue-by-source and expenditure-by-function items that non-profit satellite accounts and surveys actually collect **[VERIFY current StatCan program content]**. S-class y are the purest survey items. They are binary or count-valued, and they test whether text describes *what an organization does* well enough to predict activities.

**Facts that shape interpretation** (NE, take-all = top 206 by lagged expenses):

| y | Take-all share of total | Share of zeros |
|---|---|---|
| revenue | 0.68 | 0.01 |
| prog_rev | 0.71 | 0.31 |
| salaries | 0.59 | 0.29 |
| contrib | 0.56 | 0.19 |
| volunteers raw | 0.47 | 0.41 |
| fundraising | 0.43 | 0.61 |
| gov_grants | 0.38 | 0.69 |
| officer_comp | 0.23 | 0.51 |
| volunteers_cap10k | 0.09 | 0.41 |

Text can act only on the take-some share. For government grants that is 62% of the total; for revenue it is 32%. The zero shares mean that a two-part (hurdle) learner is a plausible sensitivity (§4.3).

---

## 2. Real-world scenario specification

### 2.1 The scenario being imitated

An NSO runs an **annual survey of non-profit organizations for reference year t = TY2023**.
- **Design date:** D = **2024-06-30**.
- **Collection:** July 2024 to about May 2025.
- **Estimation date:** E ≈ mid-2025.
- **Pseudo-census truth:** the TY2023 Form 990 file as finally indexed (2026-08-25).
- **"Response":** a TY2023 return filed by its own deadline, defined in §2.5.

The Canadian analogue is StatCan's Business Register (BR) with CRA Business Number (BN) data:
- the frame holds NAICS, the BN "major business activity" text, T2/T1044 financials and PD7A payroll;
- for registered charities, the prior-year T3010 including program descriptions;
- the survey collects what the register lacks.

### 2.2 Timeline and information sets

| Object | Testbed source | Available at D (design)? | Available at E (estimation)? | Real-world analogue |
|---|---|---|---|---|
| Register codes | BMF snapshot: latest that predates TY2023 filings. Primary is 2023-12-12 **[download from data lake]**; fallback is the 2023-08-16 snapshot used in the pilot. Sensitivity: 2024-04-11, with the equality test reported. | Yes | Yes | BR codes (NAICS, legal form, status) |
| Register amounts | BMF `ASSET_AMT`, `INCOME_AMT`, `REVENUE_AMT` (from the latest processed return, 1–3 years old) | Yes | Yes | BR size measures (lagged revenue, employment) |
| Lagged summary financials | TY2022 return: revenue, expenses, assets, employees. **Only if the first submission is ≤ D** (`avail_D`). | 94.4% of units, 87.7% of revenue | Same values (the frame is not updated) | T2/GIFI, T1044, PD7A employment |
| Lagged payroll | TY2022 salaries (F1p only) | As above | As above | PD7A payroll |
| Full lagged return | All TY2022 items, including lagged values of every y (F2 only) | As above | As above | Prior-year T3010 (charities) |
| Frame-time text | Mission and program text from the **latest return submitted ≤ D**: TY2022 if `avail_D`, else TY2021, else none. Name always. | Yes, by that rule | Yes | BN activity text (confidential); T3010 program descriptions (public) |
| Same-year text | TY2023 mission | **No**: it arrives with y | Arrives with y | Leakage control only |
| Website URL | XML `WebsiteAddressTxt` of the latest return ≤ D | Yes (flag only) | Yes | BR contact data; web-scraping inventory |
| Website content or screenshots | Not available: needs crawling, which is blocked | — | Would be estimation-time | ISTAT/OBEC web intelligence |
| Survey y | TY2023 return values | No | For respondents only | Survey responses |
| Response status | TY2023 `SubmittedOn` vs FYE | No | Yes | Collection paradata |

### 2.3 Text and lagged-amount availability rules (realism fix for RT-2)

For each unit k, at design date D:
1. If the TY2022 return was submitted ≤ D, set `avail_D = 1`. Lagged amounts and text come from that return. Use the **first** original submission; later amendments are ignored.
2. Else, if a TY2021 return exists, set `avail_D = 0`. Lagged amounts come from BMF `REVENUE_AMT`/`ASSET_AMT`/`INCOME_AMT`, flagged as `lag_source = BMF`. Text comes from the TY2021 return; mission text is near-static (identical year-on-year for 70–93%).
3. Else: BMF amounts and **name-only text** (T0), with `text_missing = 1`.

Every auxiliary matrix carries the `avail_D` and `text_missing` indicators. It also carries their interactions with size.

The TY2021 download exists for NE. A national TY2021 download is needed for the confirmatory divisions **[network: reachable S3; about 14 GB; Stage 0]**.

### 2.4 Frame tiers (strong vs weak registers)

Each tier defines **both the design and the estimator inputs**. A tier never borrows the design of a richer tier (RT-9).

| Tier | Contents | Size measure s_k used for design | Real NSO case it represents | How common in practice |
|---|---|---|---|---|
| **F0** codes-only | BMF codes: NTEE10 (MISSING as a level), subsection, foundation, affiliation, classification, deductibility, organization, ACCT_PD, ruling-year class, filing-requirement code, GROUP flag, state; size codes `ASSET_CD`, `INCOME_CD` | Class midpoint of `INCOME_CD` | New registrations; units with no tax or payroll accounts; 990-N postcard filers; small charities on a register with no financials | Many small units, a small share of totals |
| **F1** summary register (**primary**) | F0, plus BMF amounts, plus lagged revenue, expenses, assets and employees (§2.3 rules), plus `avail_D` and `has_website` | Lagged expenses if `avail_D`, else BMF `INCOME_AMT` | StatCan BR with T2/T1044 and PD7A employment; US Census BR | Default for business-type frames |
| **F1p** | F1 plus lagged salaries | As F1 | F1 plus PD7A payroll amounts (Canada) | Canada: most employers |
| **F2** full prior-year return | F1p plus lagged values of *every* y, including the binaries | As F1 | Registered charities with the prior-year T3010 at StatCan | Charities: realistic |

**BMF-max encoding** (used by B* and the twins):
- full `NTEE_CD`, collapsed to codes with ≥ 20 units, with the rest pooled into "OTHER" and MISSING kept;
- NTEE letter (26 levels);
- foundation code;
- subsection;
- affiliation;
- classification;
- the first activity code (3-digit, collapsed at ≥ 20);
- ACCT_PD (FYE month);
- ruling-year bins (≤1970, 71–90, 91–2000, 01–10, 11–15, 16+);
- `GROUP` ≠ 0, a group-exemption member, which indicates a chapter;
- `has_website`.

Low-frequency levels are pooled by a fixed rule (≥ 20 units) that never looks at y.

### 2.5 Nonresponse = late filers (a real mechanism, with caveats)

- **NR-real (primary).** A unit responds if its TY2023 return is submitted ≤ 300 days after FYE. That gives 60% response by count and 18% expense-weighted in NE. Sensitivity: 180 and 365 days.
- **Treatment of take-all units:**
  - *Primary:* take-all units are assumed to respond. This is the pilot convention: NSOs chase large units.
  - *Sensitivity NR-TA:* take-all units follow the same rule. Missing take-all units are imputed by ratio imputation from their lagged value, which is standard practice.
- **Caveat (RT-12).** Late filing is an administrative process and its size gradient is the reverse of a typical voluntary survey. So results under NR-real are reported **alongside** the simulated mechanisms of §7.2, and conclusions require agreement between them.

### 2.6 Population definitions

- **P1 (primary): continuing units.** Full 990 in both TY2022 and TY2023; one unit per EIN; group returns excluded; state from the BMF; edits E1–E3 applied. This is the pilot's U, restricted to `avail_D`-consistent frame variables.
- **P2 (sensitivity): frame-defined population.** All units whose latest return at D is a full 990 for TY2022. The target is their TY2023 value:
  - unit filed a full 990: its values;
  - unit filed a 990-EZ: the EZ values where the item exists (revenue, expenses, assets, contributions, program revenue, salaries, grants paid); other y are set to missing and estimated on P1 only;
  - unit filed nothing for TY2023: y = 0 and flagged dead.

  This is what an NSO actually estimates for frame units **[CODE; needs the TY2023 990-EZ XML for about 1–2k NE EINs; reachable]**.
- **Births** (units filing a full 990 at t with no TY2022 full return; 1,414 in NE) are frame undercoverage. They are reported as a coverage share of each total, which is the same for all estimators. They are out of scope for efficiency claims.

### 2.7 What is real, what is simulated, what is absent

| Real | Simulated | Absent (state as a limitation) |
|---|---|---|
| Population, skew, outliers, missing NTEE, text quality, frame-time availability, late filing, near-duplicate chapter texts | Sample selection (designs), simulated NR mechanisms, placebo and known-answer y | Survey measurement error and mode effects; differences between register and survey concepts (RT-13); website content; real PDFs (except CCEW in Stage 6); French text (except CRA in Stage 6) |

---

## 3. Designs

### 3.1 Common rules

- **Sampling fraction.** n = round(1000 × N / 20,626), which is f ≈ 4.85%. That gives NE n = 1,000 (pilot-compatible). Across the 8 divisions n runs from 711 (East South Central) to 2,822 (South Atlantic).
- **Take-all stratum.** Top 1% of units by the tier's s_k (§2.4). These units have weight 1 and are excluded from modelling and calibration, as in `mc.Cell`.
  - Sensitivity S-TA: take-all units *are* used to train models, but not in the calibration totals.
- **Strata.** Group × 3 size classes. Size classes come from cum-√f boundaries on log1p(s_k) over the whole population. The group is NTEE10 under D1.
- **Allocation (identical across D1, D1-imp, D2a and D2b).** Neyman on s_k^0.5, minimum n_h = 10, iterative take-all for over-allocated strata (`frame.production_design`, `neyman_allocation`). Only the *stratum definition* changes between D1 and D2. This isolates stratification from allocation.
- **Every design uses only design-time information** for its tier.

### 3.2 Design list

| ID | Strata | Allocation | Purpose | Status |
|---|---|---|---|---|
| **D0** | none (SRSWOR) | n | Reference only; not realistic for business surveys. Estimates the efficiency gap that stratification closes. | Dev (NE) only |
| **D1** | NTEE10 (incl. MISSING) × 3 size classes, plus take-all | §3.1 | **Production status quo.** Primary design for all calibration hypotheses. | Dev + confirmatory |
| **D1-imp** | As D1, but NTEE10 for MISSING units (31% in NE) is **imputed from text**. The imputer is a multinomial logit on LSA-256 T2, trained on frame units with a known NTEE; the pilot's 5-fold accuracy is 69–70%. | §3.1 | "Text as autocoder": the *productivity* use, measured in variance terms | Dev + confirmatory |
| **D2a** | K = 11 k-means clusters of the T2 embedding (same group count as D1), × 3 size classes, plus take-all | §3.1 | Unsupervised text grouping vs the official classification | Dev + confirmatory |
| **D2b** | Within each size class: k-means (K = 11) on the vector of **predicted ratios** m̃_j(k)/s̃_k for j ∈ D4 ∪ S5, from a **frozen previous-wave model with text** (§7.4) | §3.1 | Supervised, prediction-based stratification (multipurpose) | NE dev (needs TY2021); confirmatory only if national TY2021 is downloaded **[network]** |
| **D2b-ctrl** | As D2b, but the frozen model has **no text** | §3.1 | Isolates text's contribution to D2b | As D2b |
| D3 (optional) | Pareto-πps on √s_k | — | PPS sensitivity | Deferred |

**Checks on D1-imp.**
- Imputation accuracy is checked on *truly missing* units where possible: units with NTEE missing in the 2023 BMF but coded in the 2026 BMF. That is a realistic "later truth". The 2026 codes are used **only** for this check, never as a frame variable.
- The imputer uses no survey y.

**Stratification and calibration overlap.** If you stratify on text and then calibrate on text, the gain is counted twice. So every stratification result is reported as a **2 × 2**: {D1, D2x} × {B0 conventional estimator, primary text estimator}.

---

## 4. Estimators and baselines

### 4.1 Common estimator anatomy

For each sample s, write s_TS for its take-some part. Every estimator has the form

```
T̂ = Σ_{TA} y_k + T̂_TS
```

The take-some components are as follows.

```
HT:    T̂_TS = Σ_{s_TS} d_k y_k
GREG:  T̂_TS = Σ_s d y + (T_X − Σ_s d x)' B̂,   B̂ = (Σ_s d x x' + Λ)^{-1} Σ_s d x y
       Λ = 0 for OLS-GREG.
       Ridge: Λ = diag(0_hard, λ·Σd·Var_d(x_j)) on the soft columns.
MCAL (model calibration, Wu–Sitter):
       z_k = (x_base,k, m̂_k); GREG on z with population totals (T_xbase, Σ_{U_TS} m̄_k)
       m̂_k for k∈s is cross-fitted: fit on the other K−1 folds.
       m̄ for k∈U is the average of the K fold models.
Ratio-scale learner (":ratio" / "_ratio"):
       m̂_k = s̃_k · ĝ(f_k), s̃_k = s_k + 1
       ĝ is a weighted ridge of y/s̃ on features f, with weights d·s̃.
DIFF (frozen):
       T̂_TS = Σ_{U_TS} m̃_k + Σ_s d (y − m̃), with m̃ trained on independent previous-wave data.
       Exactly design-unbiased.
FCAL (frozen calibrated): GREG on (x_base, m̃).
SYN (synthetic): T̂_TS = Σ_{U_TS} m̂_k (no design correction).
MCALM / RIDGEM (multipurpose, one weight per record):
       calibrate once on z_k = (x_base,k, m̂_{1,k}, …, m̂_{q,k}) for all q target y;
       the m̂ columns are ridge-penalized with one common λ (below).
```

### 4.2 Full estimator list

Names match `src/experiments.py` where the estimator exists. "New" means **[CODE]**. `{e}` is an embedding name and `{F}` a frame tier.

| Name | Family | Auxiliary / features | Single weight? | Notes |
|---|---|---|---|---|
| `HT` | status quo | — | yes | |
| `GREG_ratio` | status quo | strata + s_k | yes | |
| `GREG_size` | status quo | strata + 4 lagged amounts | yes | |
| **`GREG_size_cat` (B0)** | **status quo, pre-specified baseline** | strata + lagged amounts + NTEE10 + state | yes | The pilot baseline; fixed as B0 for F1 |
| `GREG_weak` (B0 for F0) | status quo | strata + NTEE10 + state + income-code dummies | yes | |
| `GREG_sep` | status quo (reported, flagged poor) | separate ratio by stratum | yes | Worse than HT for binaries in the pilot |
| `GREG_bmfmax` (new) | status quo, strengthened | size_cat + subsection + foundation + NTEE letter + avail_D + has_website (p ≈ 70) | yes | OLS-GREG; p/n ≈ 0.09 at n_ts ≈ 800 |
| `RIDGE_bmfmax` (new) | status quo, strengthened | hard: size_cat; soft: all BMF-max dummies (p ≈ 300–500) | yes | λ by in-sample CV |
| `GREG_F2_multi` (new) | status quo under F2 | size_cat + lagged value of every y | yes | |
| `GREG_F2_y` (new) | status quo under F2 | size_cat + lagged value of target y + stratum × lagged y | per y | |
| `MCAL_frame_raw`, `MCAL_frame_ratio` | text-free ML | `ml_frame`, `ml_frame_ratio` | per y | Pilot text-free ML |
| **`MCAL_bmfmax_ratio` (new) = TWIN** | **text-free ML twin of the primary** | log lagged amounts, BMF-max dummies, size × NTEE letter, avail_D | per y | Same learner, folds and λ grid as the primary |
| `MCALM_bmfmax_ratio` (new) | text-free ML, multipurpose twin | as twin | **yes** | |
| `MCAL_F2_ratio` (new) | text-free ML twin under F2 | twin + log lagged y | per y | |
| **`MCAL_bmfmax_ratio+{e}` (new) = PRIMARY TEXT ESTIMATOR** | **text** | twin features + 32 population PCs of {e} | per y | Primary {e} = `lsa256_full` (T2), fixed now (§6) |
| `MCAL_bmfmax_ratio+{e}:fullD` (new) | text | twin + all D dims (supervised ridge; dual form) | per y | Answers RT-7 |
| **`MCALM_bmfmax_ratio+{e}` (new)** | **text, production form** | as primary | **yes** | H14 |
| `MCAL_F2_ratio+{e}` (new) | text under F2 | F2 twin + PCs | per y | H5 |
| `GREG_size_cat+{e}:add10` | text, naive | OLS-GREG with 10 PCs added | yes | Pilot form; p/n demonstration |
| `RIDGE_size_cat+{e}:ratio32` | text | ridge calibration, soft PC × size | per y (λ) | |
| `RIDGEM_size_cat+{e}:ratio32` (new) | text, multipurpose | as above with one common λ | **yes** | |
| `MCAL_raw+{e}`, `MCAL_ratio+{e}` | text (pilot forms) | `ml+{e}`, `mlr+{e}` | per y | Kept for continuity with the pilot |
| `GREG_weak+{e}:add32` | text under F0 (naive) | OLS-GREG with 32 PCs | yes | Overfitting demonstration (OF2) |
| `MCAL_F0_ratio`, `MCAL_F0_ratio+{e}` (new) | twin / text under F0 | F0 features, s = INCOME_CD midpoint | per y | H6 |
| `DIFF_frozen_frame`, `DIFF_frozen+{e}`, `FCAL_frozen_frame`, `FCAL_frozen+{e}` (new) | frozen deployment | §7.4 | DIFF: yes; FCAL: per y | H13 |
| `MCAL_bmfmax_ratio+{e}:nocf` | control (overfitting) | no cross-fitting | per y | OF1 |
| `SYN_bmfmax_ratio+{e}` | control (synthetic) | — | — | Bias demonstration |

**Comparators used in the hypotheses.**
- **B0:** pre-specified status quo for the tier: `GREG_size_cat` (F1), `GREG_weak` (F0), `GREG_F2_multi` (F2).
- **Twin:** the identical learner without text. RE_inc = MSE(twin)/MSE(text). **This is the primary contrast.**
- **B\*:** the ex-post best, per y and per population, by MSE among *all* status-quo and text-free ML estimators of the tier. Picking it on the same replicates favours B*, so it is **conservative for text**. The skeptic should accept B* as fair.

### 4.3 Tuning rules (no exceptions)

1. **Everything is tuned inside the sample.** λ is chosen by K = 5-fold CV within s_TS. The CV loss is d-weighted squared error on the learner's scale. The grid is `LAMS = (1e-3, 1e-2, 1e-1, 1, 10)` × n, as in `mc._ridge_fit`.
   - Per-y estimators get a per-y λ.
   - Multipurpose estimators get **one** λ that minimizes Σ_j CV_j / CV_j(null), where the null model is the intercept-only model.
2. **Folds are grouped *and* stratified.** All units in a near-duplicate text cluster (`experiments.text_groups`) share a fold. Folds are balanced across design strata. The same folds serve λ-CV and cross-fitting **[CODE: the `ridge` kind too]**.
3. **Fixed now, never tuned:** K_add = 10 and K_ratio = 32 PCs; ratio offset 1; K = 5 folds; ridge as the learner; the BMF-max pooling rule (≥ 20 units).
   - Learner sensitivity: `HistGradientBoostingRegressor` and a two-part hurdle model (logit for > 0 × ridge on log y | y > 0). Dev only, R = 200, D4 y only. CPU cost rules them out of confirmation.
4. **PCA is computed on the frame embeddings of U** (no y involved). Each population gets its own PCA, as an independent deployment would.
5. **TF-IDF/LSA is fitted on the t−1 frame texts of the population only**, never on t texts.
6. **Nothing is chosen on Monte Carlo results after the freeze** (§9.5).

### 4.4 Variance estimators

| ID | Estimator | Where |
|---|---|---|
| **V1** (primary) | Stratified linearization with finite-population correction on the final calibration residuals: v̂ = Σ_h N_h²(1 − n_h/N_h) s²_{e,h}/n_h, with m̂ treated as fixed (`survey.strat_var_total`) | All cells |
| V2 | Respondent-count version for NR (`mc._var_counts`) | Stage 3 |
| V3 | Exact HT variance of the residuals y − m̃ | Frozen DIFF |
| **V4** (validation) | Rao–Wu rescaling bootstrap, B = 200, re-running the *whole* pipeline (λ-CV, cross-fitting, calibration) in each bootstrap replicate | Sub-study: R = 200, primary and twin only, NE, F1, D1 **[CODE]** |

- Intervals are normal (z = 1.96). Sensitivity uses t with n_TS − H degrees of freedom.
- A calibration failure (`LinAlgError` or non-finite output) is **counted**, never silently dropped. If any estimator fails in more than 1% of replicates it is flagged. RE is computed on the replicates where both members of the contrast succeeded **[CODE: `metrics` must report failure counts]**.

---

## 5. Controls

Each control has a predicted result and a failure meaning. A failed gate stops the efficiency claims for that cell until it is explained.

### 5.1 Negative controls (the text must do nothing)

| ID | Construction | Expected | Pass rule | If it fails |
|---|---|---|---|---|
| NC1 | `random384`: Gaussian vectors (seeded), run through the same PCA and estimator | RE_inc = 1 | RE ∈ [0.97, 1.03] and its 95% MC CI inside [0.95, 1.05]; coverage ≥ 93% | The pipeline rewards dimensionality: overfit |
| NC2 | `perm_{e}`: real embedding rows permuted globally | 1 | as NC1 | as NC1 |
| NC3 | `permh_{e}`: rows permuted **within design strata**. This keeps the NTEE × size composition, so it is a stronger test. | 1 | as NC1 | Gain comes from features correlated with the strata, used non-linearly |
| NC4 | Placebo y `PLC_perm`, `PLC_noise`, `PLC_bern` with the real text | 1 | as NC1 | Leakage or overfitting |
| NC5 | Constant embedding (every unit identical) | Identical to the twin, to machine precision | Exact equality | Code-path bug |
| NC6 | R-class y under F1 (revenue, expenses, employees) | ≈ 1 | RE ∈ [0.95, 1.05] | Leakage (timing or numbers) suspected first |

### 5.2 Positive controls (the harness must detect real signal)

| ID | Construction | Expected |
|---|---|---|
| PC1 | Oracle lag: F2 lagged y in GREG and in MCAL | Large: 4–6× in the pilot for government grants, fundraising and officer compensation. Sets the scale. |
| PC2 | Known-answer y `KA_text` (§5.4) | The harness recovers the predicted RE within ±10% |
| PC3 | NR-sim-B, where response depends on a text-predicted quantity | Text NR adjustment removes ≥ 70% of the bias |

### 5.3 Leakage and overfitting controls

| ID | What it probes | Construction | Interpretation |
|---|---|---|---|
| LK1 | Same-year information | Same-year text, TY2023 mission (`text_mission_t` → `lsa256_mission_t`) | The gain over T1 (t−1 mission) is the value of contemporaneous text. It is *not* deployable. Report it only as an upper bound. |
| LK2 | Numbers in text | `T2m`: digits and "$" amounts masked; re-embedded | A gap of more than 3% in RE between T2 and T2m means numeric leakage. The masked version then becomes primary. |
| LK3 | Frame snapshot timing | Share of units with BMF `REVENUE_AMT` = y_revenue,t | Must be < 1% for the primary snapshot (Dec 2023 measured 0.62%) |
| LK4 | Gain above the theoretical ceiling | MC RE_inc vs AV-ceiling(twin, twin+text), same scale and same features | MC RE_inc > ceiling × 1.05 + 2·MCSE raises a leakage alarm (gate G5) |
| LK5 | Memorization by pretrained models | Predict log(y_t/y_{t−1}), the change, from the embeddings, beyond the frame. Compare against shuffled names, and MiniLM (pre-2022 training data) against Cohere (2025). | Change should be unpredictable. Cohere ≫ MiniLM on change means contamination is suspected. |
| LK6 | Near-duplicate leakage | Grouped vs ungrouped folds | The difference in RE and coverage quantifies the optimism |
| OF1 | No cross-fitting | `…:nocf` | Coverage drop expected (pilot 72–78%). Demonstration only. |
| OF2 | p/n | `GREG_weak+{e}:add32`, `GREG_size_cat+{e}:add10` | Coverage loss and variance inflation (pilot 38–55% coverage). Demonstration only. |
| OF3 | Small-n penalty | n-grid {0.5, 1, 2} × n (NE) | The gain should shrink as n falls. Gains that grow at small n are suspicious. |
| SYN | Model-based aggregation | `SYN_bmfmax_ratio+{e}` | Shows the bias of the "replace the survey" route (pilot: +15% on binaries) |

### 5.4 Known-answer synthetic y (gate G6)

Construct

```
y^KA_k = s̃_k · exp(β' Z_k + γ' F_k + ε_k),  ε ~ N(0, σ²)
```

where Z is the first 8 PCs of `lsa256_full`, F is the twin features, and the parameters are set so that the population partial R² of Z given F, on the ratio scale, is 0.20.

Theory predicts AV-RE_inc ≈ 1/(1 − 0.20) = 1.25. **The harness must recover MC RE_inc within [1.12, 1.38]**, which is ±10% of the predicted RE. Run it under D1, F1 and n = 1000 in every population. This tests that the machinery *can* find a gain of the size H1 asks about. Without it, a null H1 could mean an insensitive harness.

---

## 6. Embedding ladder and text-field arms

### 6.1 Text fields (all frame-time, by the §2.3 rule)

| Arm | Content | Real-world analogue | Coverage |
|---|---|---|---|
| **T0** name | BMF `NAME` + `SORT_NAME` | Every register has names. This is the only text for births and new units. | 100% |
| **T1s** registration-length | First 12 words of the T1 mission. A crude stand-in for the BN "major business activity" / SS-4 line 17 text **[approximation; say so]**. | BR activity text (confidential) | t−1 or t−2 filers |
| **T1** mission | Part I line 1 + Part III line 1, deduplicated (`text_mission`) | T3010 program description (short) | as above |
| **T2** mission + programs (**primary**) | `text_full`: T1 + Part III 4a–4d, whitelisted XPaths | T3010 program descriptions, C2 | as above |
| T2m | T2 with numbers masked | — | as above |
| T3 (optional) | T2 + Schedule O | — | Needs a parser **[CODE]** |
| Tt (control) | TY2023 mission | — | Leakage control LK1 only |
| Tw (absent) | Website text | Enterprise websites | Needs crawling **[USER: network]** |
| Tpdf (Stage 6) | t−1 accounts PDF: text layer, page images, or fused | CRA-filed financial statements; CCEW accounts; SEDAR+ | CCEW only |

### 6.2 Representation rungs

Every rung keeps the field (T2), estimator spec, folds, seeds and PCA size (K = 32) fixed. Each rung also runs the `:fullD` arm.

| Rung | Representation | Cache name | Where it runs | Status |
|---|---|---|---|---|
| R0 | none (the twin) | — | local | ready |
| R0-neg | random384, perm, permh, constant | `random384`, … | local | ready / [CODE] for permh |
| R1 | Hand-built features: log length, has-text, has-website, 30 keyword flags (hospital, university, school, church, housing, foundation, PTA, VFW, Lions, Rotary, union, local, chapter, …), chapter-pattern flag | `kw30_T2` | local | [CODE] |
| **R2** | TF-IDF (1–2 grams) → LSA at 64 and 256 dims | `lsa256_full`, `lsa64_full` | local | ready (256) |
| R3 | GloVe/spaCy averaged | `glove300_T2` | local | [CODE] |
| R4 | all-MiniLM-L6-v2, max_len 256 (truncation rate logged by size class) | `minilm_full` | local CPU (about 340 s per 20k short texts; T2 is longer, about 2–3×) | ready (NE) |
| R5 | Qwen3-Embedding-0.6B (on-prem analogue; with and without instruction) | `qwen06_T2` | needs Hugging Face; a GPU is advisable | **[USER: allowlist huggingface.co; GPU optional]** |
| R6 | Cohere embed-v4.0 text, `input_type=classification`, 1536 float. Factors: clustering / search_document; Matryoshka 256/512/1024; int8. | `cohere4c1536_T2` | API | **[USER: allowlist api.cohere.com + production COHERE_API_KEY]** |
| R6b | Cohere embed-v5.0-pro text | `cohere5p_T2` | API | [USER] |
| R7 | Cohere multimodal on CCEW accounts PDFs: page images, fused text+image, with/without masked numbers | §10.3 | API | [USER] Stage 6 |
| R7-open | Qwen3-VL-Embedding-2B on the same pages | — | HF + GPU | [USER] |

**Primary representation, fixed now:** `lsa256_full` (R2 on T2).
- Rationale: it was best in the pilot, it is deterministic and on-prem, and it is the cheapest credible text representation. A gain from it is the most deployable.
- That choice was made on NE; confirmation is on the divisions.

### 6.3 What each comparison answers

| Comparison | Question | Metric | Decision use |
|---|---|---|---|
| R2 vs R0 (twin) | Does any frame-time text add efficiency? | RE_inc | H1, H2 |
| R1 vs R2 | Is it just keywords a methodologist would write? | RE(R2)/RE(R1) | If ≤ 1.03, use keywords and skip embeddings |
| R4 vs R2 | Contextual vs lexical | ratio of RE | |
| R5 vs R4 | Model scale (on-prem) | ratio | |
| R6 vs max(R2, R4, R5) | **Does a frontier proprietary model add anything?** | ratio | H9; governance |
| R6 input_type and dims factors | Sensitivity to analyst choices | spread of RE | A spread > 5% calls for a pre-registered setting |
| T1 vs T2 | Does richer program text matter? | ratio | Which register text to acquire |
| T1s vs T1 | Can register-length text carry the signal? | ratio | **Realism**: BN-like text is short |
| T0 vs none | Value of names (universal; covers births) | RE_inc | Births/new-unit strategy |
| T2 vs T2m | Numeric leakage | ratio | LK2 |
| Tt vs T1 | Timing leakage and staleness | ratio | LK1 |
| `:fullD` vs PC32 | Does the PC bottleneck hide model differences? | ratio | Interpreting the ladder (RT-7) |
| R7 vs R6 on the same PDFs | Does the image add beyond the text layer? | ratio | H15 |

**Cheap representation diagnostics (no Monte Carlo), run per rung in Stage 1:**
- linear-probe accuracy for NTEE10;
- linear-probe accuracy for imputing truly missing NTEE (checked against the 2026 BMF);
- population cross-fitted partial R² on the ratio scale for each y, beyond BMF-max;
- **redundancy**: the R² of the 32 PCs on BMF-max dummies. Above 0.8 means the text is mostly a code proxy.

---

## 7. Uses beyond calibration

### 7.1 Stratification

- **Designs:** D1 (status quo), D1-imp (autocoded NTEE), D2a (unsupervised text clusters), D2b and D2b-ctrl (frozen-model prediction strata, with and without text).
- **Estimators held fixed:** HT, B0 and the primary text estimator. Results are reported as a 2 × 2 (§3.2).
- **Metric: multipurpose efficiency.**
  - Report RE(D_x vs D1) per y.
  - Report the geometric mean (GM) of RE over D4 ∪ S5 ∪ R.
  - Report the **minimum over all y**. Stratifying for one y can hurt others; the minimum is the "no harm" criterion.
- Population-level anticipated variance for each design is computed in Stage 1 (`diagnostics.anticipated_var`). It predicts the MC result.

### 7.2 Nonresponse adjustment

**Mechanisms.**
- **NR-real:** late filers (§2.5). The response set is deterministic given the population, so the only randomness is sampling.
- **NR-sim** (Bernoulli response redrawn in every replicate from its own seed stream; overall rate calibrated to 60%):
  - **A, MAR-frame** (negative control for text): logit p_k = α + 0.5·z(log1p s_k) + γ_NTEE10. Large units respond *more*, as in surveys. Text should add nothing beyond the strata, frame and size adjustment.
  - **B, MAR-text** (positive control): logit p_k = α + 0.5·z(log1p s_k) + 0.8·z(u_k). Here u_k is the frozen previous-wave text model's predicted government-grant ratio. Response depends on something text knows and that relates to y.
  - **C, MNAR:** logit p_k = α + 0.5·z(log1p s_k) + 0.8·z(r_k). Here r_k is the residual of log1p(y_gov_grants,t) given F1. Nothing fully fixes this; the question is whether text reduces the damage.

**Estimators.**
- `NR_naive_HT`, `NR_rhg_HT`, `NR_rhg_GREG_size_cat` (production), `NR_ps_frame_GREG_size_cat`, `NR_rhg_MCAL_frame_ratio`;
- text versions `NR_rhg_MCAL_ratio+{e}`, `NR_ps_text_GREG_size_cat+{e}`, `NR_ps_text_MCAL_ratio+{e}`;
- **new** `NR_rhg_MCAL_bmfmax_ratio` (twin) and `NR_rhg_MCAL_bmfmax_ratio+{e}`;
- **new** `NR_tRHG_HT`: response homogeneity groups = strata × tertiles of the cross-fitted text-predicted government-grant ratio. This is the Little–Vartivarian "predict y" route.
- **Reference text-free estimator for bias removal:** the better (lower |RB|, chosen per y ex post, which is conservative) of `NR_rhg_HT`, `NR_rhg_GREG_size_cat` and `NR_rhg_MCAL_bmfmax_ratio`. The pilot found that calibrating to lagged size *increases* nonresponse bias for survey-only y, so both calibrated and uncalibrated references are needed.

**Metrics.**
- RB, RRMSE, coverage (V2).
- **Bias-removal fraction** BRF = 1 − |RB_text| / |RB_ref|.
- Propensity diagnostics: AUC with and without text; the correlation between text-predicted y and the response indicator within strata.
- A covariate reduces bias only if it predicts *both* response and y, so the report also shows the partial correlation of m̂_text with y among *respondents*.

### 7.3 Small area estimation

**Domains.**
- Planned: **state × NTEE10**. NE has 66 domains; the divisions have 44–99.
- Unplanned small areas: **ZIP3** (first three ZIP digits; about 60 in NE, average n_d ≈ 16). No crosswalk is needed.
- Each domain total = take-all part (known) + take-some part (estimated).

**Estimators** (twin = text-free, text = primary representation):

| Name | Definition |
|---|---|
| `DIR` | Domain HT (domain total) |
| `MAD_twin`, `MAD_text` | Model-assisted domain GREG: Σ_{U_d} m̂ + Σ_{s_d} d (y − m̂), with the cross-fitted model from §4 |
| `SYN_twin`, `SYN_text` | Σ_{U_d} m̂ |
| `FH_twin`, `FH_text` | Area-level Fay–Herriot on domain means ȳ_d (REML, σ_v² truncated at 0). Direct = DIR/N_d. Sampling variances are GVF-smoothed: log v̂_d on log n_d and log N_d. Covariates: area means of the F1 frame variables plus the area mean of m̂_twin; `FH_text` adds the area mean of m̂_text. MSE by Prasad–Rao (or Datta–Lahiri for REML). Domains with n_d < 2 get the regression-synthetic prediction and are reported as non-sampled. |

A unit-level BHF on the log scale is an optional extension.

**Metrics.**
- Per domain: RB_d, RRMSE_d, coverage_d (FH: MSE-based intervals; DIR/MAD: V1-based).
- Aggregated: ARB = mean_d |RB_d|, ARRMSE = mean_d RRMSE_d, medians, and the 90th percentile of RRMSE_d.
- Each reported separately by n_d class {0, 1–4, 5–19, ≥ 20}, for NTEE-MISSING domains, and for sampled vs non-sampled domains.
- **CV-band migration:** the count of domains moving between quality bands (§8.6), and the share of domains publishable (CV < 25% and < 35%).
- **Heterogeneity red-team check:** correlation across domains between RB_d of `SYN_text` and the domain's median text length or share of boilerplate. Text models may be biased exactly where the text is poor.

R = 500.

### 7.4 Frozen / previous-wave deployment (closest to NSO practice)

**Previous wave (NE; national only if TY2021 is downloaded).**
- U′ = continuing TY2021 → TY2022 units.
- Frame′ = BMF codes, TY2021 lagged amounts (`avail` rule at D′ = 2023-06-30) and TY2021 text.
- y′ = TY2022.
- **Limitation:** no BMF snapshot before 2023-08 exists, so Frame′ codes come from the 2023-08 snapshot. That slightly post-dates D′. Codes are near-static, but it is reported.

**Training data, as a real NSO would have it.** A previous-wave sample s′ is drawn under D1(Frame′) with the same n, and the respondents are taken by the 300-day rule. **Not the whole previous-wave population**: if the NSO had y′ for everyone, it would be on F2.

**Model.** m̃ = the primary learner (ratio-scale ridge, twin features ± PCs) fitted on s′ with design weights, with λ chosen by grouped CV inside s′. Features have the same *definition* in both waves: "lagged expenses" means TY2021 in training and TY2022 in application.

**Application.**
- m̃ is applied to Frame_t for all of U.
- Estimators: `DIFF_frozen+{e}` (exactly design-unbiased; V3), `FCAL_frozen+{e}`, the text-free versions, and the in-sample `MCAL_bmfmax_ratio+{e}` for comparison.
- s′ is redrawn in every replicate (unconditional; primary). Sensitivity: results conditional on 20 fixed s′ draws, showing the spread an NSO faces from a single previous sample.

**Drift.** Report the population R² of m̃ in wave t against its in-sample R² in wave t−1.

**Panel overlap.** Overlap between s′ and s is ignored (they are independent draws). Stated as a limitation.

---

## 8. Metrics and Monte Carlo protocol

### 8.1 Per-cell metrics

A cell is one (population, tier, design, n, mechanism). θ is the true total; θ̂_r and v̂_r are the replicate-r estimate and variance estimate, r = 1..R.

| Metric | Formula | MC standard error |
|---|---|---|
| Relative bias | RB = (θ̄̂ − θ)/θ | sd(θ̂)/(√R·\|θ\|) |
| MSE, RRMSE | MSE = R⁻¹Σ(θ̂_r − θ)²; RRMSE = √MSE/\|θ\| | SE(MSE) = sd((θ̂ − θ)²)/√R; SE(RRMSE) = SE(MSE)/(2√MSE·\|θ\|) |
| Relative efficiency | RE_{A vs B} = MSE_B / MSE_A, paired over replicates | Var(log RE) ≈ Var(a)/(R·MSE_A²) + Var(b)/(R·MSE_B²) − 2Cov(a,b)/(R·MSE_A·MSE_B), with a = (θ̂_A − θ)², b = (θ̂_B − θ)² (`mc.metrics`). Cross-check by bootstrapping replicates (B = 2000). |
| Coverage | C = R⁻¹Σ 1{\|θ̂_r − θ\| ≤ 1.96√v̂_r}; also lower and upper non-coverage separately | √(C(1 − C)/R) |
| Variance-estimator relative bias | RB(v̂) = (mean v̂ − MSE)/MSE. Also reported against the empirical variance. | Delta method |
| CI relative width | 2·1.96·mean√v̂ / \|θ\| | — |
| Tail-robust error | Median \|θ̂ − θ\|/θ; 95th percentile of \|θ̂ − θ\|/θ | Bootstrap |
| Weight diagnostics (single-weight estimators) | % negative weights; min and max g-weight; Kish deff = 1 + CV²(w) in the take-some part; calibration failure rate | — |
| Gap-closure (F0) | G = log RE(F0+text vs F0 twin) / log RE(F1 twin vs F0 twin) | Delta |
| Fine-code absorption | Φ = 1 − log RE(text vs BMF-max twin) / log RE(text vs NTEE10-only twin) | Delta |
| Bias-removal fraction | BRF = 1 − \|RB_text\|/\|RB_ref\| | Delta |

### 8.2 Aggregation across y and across populations

- **Across y:** within a class, the geometric mean GM = exp(mean_j log RE_j) over that class's headline set (D4, S5, R). H-class and P-class y are never pooled into scores.
- **Multipurpose:** the minimum over all non-placebo y of RE (the no-harm criterion).
- **Across populations:** for each division g, L_g = log GM_g. The confirmatory statistic is L̄ = mean_g L_g, with SE = sd(L_g)/√8 and a t-distribution with 7 df. MC error is inside each L_g, so it is counted.
- **Consistency:** the count of divisions with GM_g ≥ 1.05. This is descriptive: the sign test at 6/8 has p = 0.14, so it is not treated as a test.
- All per-y and per-division results are reported in full. Nothing is hidden behind aggregates.

### 8.3 Monte Carlo protocol

**Common random numbers.**
- Sample draw r of cell c uses `rng_s = default_rng(SeedSequence([seed_c, r, 0]))`.
- Estimator e in replicate r uses `rng_e = default_rng(SeedSequence([seed_c, r, hash32(e)]))`.
- Simulated response uses `[seed_c, r, 1]`; the previous-wave sample uses `[seed_c, r, 2]`.
- So every estimator and every embedding rung, even in separate runs, sees **identical samples** (fixes RT-10).
- `seed_c = hash32(population_id, design_id, n)`. Note that the tier is *not* part of the seed: tiers that share a design share samples. Tiers with different designs (F0 vs F1) are compared only through their own twins.

**Number of replicates (fixed before running; no optional stopping).**

| Run | R |
|---|---|
| Confirmatory primary and gate cells | 1000 (coverage MC SE ≤ 0.69 pp) |
| SAE | 500 |
| Development / exploratory (NE) | 200–500 |
| Bootstrap validation | 200 × B = 200 |

R for the confirmatory cells is checked against a precision target **estimated on NE dev**: MC SE of log RE_inc ≤ 0.015 for each D4 y. If the target is not met at R = 1000 in dev, R_conf is raised (to 2000) *before* the freeze, for all cells alike.

**Reproducibility.**
- Re-running 10 replicates of any cell from its manifest must reproduce the estimates bit-for-bit.
- Each run writes a manifest with: git commit, spec hash, data and embedding manifest hashes, seeds, package versions, wall time and failure counts.

**Sanity tests on every cell:**
- HT's RB is within ±3 MC SE.
- `DIFF_frozen` RB is within ±3 MC SE (exact unbiasedness).
- HT coverage is 93–97% for non-heavy-tailed y.
- Any y where HT coverage is below 92% is flagged **coverage-uninterpretable**. For those y the coverage gate becomes *relative*: the text estimator's coverage must be ≥ the twin's − 2 pp.

### 8.4 Theory-based predictions (calibrating expectations)

Stage 1 computes AV-ceilings for every twin pair, on the *same scale and features* as the MC estimator **[CODE: ratio-scale variant of `diagnostics.evaluate_feature_sets`]**:

```
AV-RE_inc = AV(twin) / AV(twin + text)
```

- NE dev gives the empirical shrinkage factor κ = median_y (log RE_MC / log AV-RE) for the primary pair.
- The **pre-registered prediction** for each confirmatory division and y is log RE_pred = κ · log AV-RE(division).
- Secondary outcome: the median |log RE_MC − log RE_pred| ≤ 0.05. In words, population diagnostics predict MC gains to within about 5%. If this holds, an NSO can screen variables with cheap diagnostics instead of a Monte Carlo.

### 8.5 Equivalent sample size and cost

- **Analytical** (fixed design, take-some part). The text estimator needs n_eq to match the baseline at n_TS:

  ```
  1/n_eq − 1/N_TS = RE · (1/n_TS − 1/N_TS)
  ```

  Saving = 1 − n_eq/n_TS. Example: RE = 1.20, n_TS = 794, N_TS = 20,420 gives n_eq ≈ 666, a 16% saving.
- **Empirical (preferred: it includes the model-fitting penalty at smaller n).** Run the primary text estimator at n_TS × {0.6, 0.7, 0.8, 0.9, 1.0}. Interpolate on log RRMSE vs log n to find n where RRMSE_text(n) = RRMSE_B0(n_TS). This runs in the confirmatory divisions for the primary estimator only, at R = 500.
- **Binding-constraint rule (real-world).** A multipurpose survey's n is set by the y with binding precision targets. Achievable saving = min over binding y of saving_y. The binding set is **[USER: pick, e.g., D4 + S5]**. Gains on non-binding y buy nothing in sample size.
- **Cost:** saving × n × c_unit + burden hours saved. c_unit and hours per questionnaire **[USER]**. Fixed costs are listed separately: embedding compute (§11), governance (§12.3) and maintenance.

### 8.6 CV quality bands

Report each domain estimate's CV = √MSE/θ against the publication bands of the target program. As a placeholder **[VERIFY against the StatCan program's standard]**:

| Band | CV |
|---|---|
| A | < 5% |
| B | 5–< 10% |
| C | 10–< 15% |
| D | 15–< 25% |
| E (use with caution) | 25–< 35% |
| F (suppress) | ≥ 35% |

National and division totals will mostly sit in A or B. Bands matter for **domains** (§7.3). The reported outcome is the count of domains that move up a band, and the change in the publishable share.

---

## 9. Pre-registered hypotheses

### 9.1 Validity gates (all must pass in a cell before its efficiency results are interpreted)

| Gate | Criterion |
|---|---|
| G1 design sanity | HT and B0: \|RB\| ≤ 3 MC SE. Coverage 93–97% (or flagged coverage-uninterpretable, §8.3). |
| G2 negative controls | NC1–NC3 and NC5: RE_inc ∈ [0.97, 1.03] with 95% MC CI inside [0.95, 1.05]. Coverage ≥ 93%. |
| G3 placebo y | NC4: same rule |
| G4 text estimator validity | Primary text estimator: coverage ≥ 93% (or relative rule); RB(v̂) ∈ [−15%, +15%]; \|RB\| ≤ 0.10 × RRMSE; failure rate ≤ 1% |
| G5 ceiling | RE_inc,MC ≤ AV-RE_inc × 1.05 + 2·MCSE for every headline y |
| G6 known answer | KA_text recovered within [1.12, 1.38] |
| G7 negative-control y | NC6 (R-class under F1): RE_inc ∈ [0.95, 1.05] |
| G8 leakage | LK2 gap ≤ 3%; LK3 < 1% |

If a gate fails: the cell is quarantined, the cause is diagnosed and logged in `prereg/deviations.md`, and the fix is applied to **all** cells. Results are reported before and after.

### 9.2 Primary confirmatory hypotheses

**Setting:** 8 held-out divisions, P1, tier F1, design D1, f ≈ 4.85%, R = 1000. Primary text = `MCAL_bmfmax_ratio+lsa256_full`; twin = `MCAL_bmfmax_ratio`.

| H | Statement | Metric | Proponent / pilot-implied | **Skeptic's prediction** | **Pass (supports)** | **Falsified if** |
|---|---|---|---|---|---|---|
| **H1** | Text adds efficiency for detail items | GM over D4 of RE_inc | 1.15–1.30 (pilot: 1.2–1.7 vs B0, minus about 1.1–1.2 for ratio-ML) | **1.00–1.08**: foundation code, subsection and fine NTEE absorb most of it | exp(L̄) ≥ 1.10 **and** Holm-adjusted one-sided 95% lower bound on L̄ > 0 | Upper 95% bound on exp(L̄) < 1.10, or exp(L̄) < 1.05 |
| **H2** | Text adds efficiency for survey-type items | GM over S5 of RE_inc | 1.10–1.20 (AV ceilings for binaries 1.06–1.18) | **1.00–1.06** | as H1 | as H1 |
| **H3** | The gain is practical, against the best conventional or text-free estimator chosen ex post | GM over D4 of RE(primary vs B\*) | 1.10–1.25 | **0.95–1.05** | exp(L̄) ≥ 1.10 and lower bound > 0 | exp(L̄) < 1.03 |

**Multiplicity for the primary family:** Holm across H1, H2 and H3, at α = 0.05 one-sided, using t₇ tests on L_g. The gates are equivalence checks, not α-spending tests. Per-y results are descriptive, with BH-FDR q-values for "RE_inc > 1".

### 9.3 Secondary hypotheses (pre-specified; reported with unadjusted intervals; labelled secondary)

| H | Statement | Metric | Skeptic's prediction | Supporting threshold |
|---|---|---|---|---|
| H4 | No gain where the register already knows (also gate G7) | RE_inc for R-class under F1 | 0.98–1.02 | — (gate) |
| H5 | Text still helps with the full prior-year return (T3010-like frames) | GM over D4 of RE(`MCAL_F2_ratio+e` vs `MCAL_F2_ratio`) | **≤ 1.03** | ≥ 1.05 |
| H6 | Weak registers: text gives large gains, but closes only part of the gap to admin data | GM over R ∪ D4 of RE(F0 + text vs F0 twin); gap-closure G | RE 1.5–2.0, **G ≤ 0.5** (pilot: about 0.5) | RE ≥ 1.5 and G ≥ 0.5 |
| H7 | The salaries gain disappears once payroll is on the frame | RE_inc for salaries under F1p | **≤ 1.05** (pilot under F1: 1.4–1.7) | ≥ 1.10 |
| H8 | Much of the pilot-style gain is a classification proxy | Φ (fine-code absorption) over D4 | **Φ ≥ 0.5** | Φ < 0.3 means text carries information beyond codes |
| H9 | Representation matters | (a) RE(R4)/RE(R2); (b) RE(R6)/max(RE(R2), RE(R4), RE(R5)) on D4 ∪ S5 | (a) 0.95–1.05; (b) **≤ 1.05** | (b) ≥ 1.10 means the frontier model is worth its governance cost |
| H10 | Text-informed stratification helps, without harm | GM over D4 ∪ S5 ∪ R of RE(D2b vs D1) under B0; min_y RE | **≤ 1.05**; D1-imp vs D1 ≤ 1.03 | GM ≥ 1.10 and min_y ≥ 0.95 |
| H11 | Text reduces nonresponse bias (NR-real) | BRF and RRMSE ratio on D4 | BRF 0–40%; RRMSE ratio ≥ 0.95 | BRF ≥ 30% on ≥ 2 of 4 D4 y **and** RRMSE ratio ≤ 0.90 **and** coverage not worse by > 2 pp. Must also show BRF ≥ 70% under NR-sim-B and \|BRF\| ≤ 10% under NR-sim-A. |
| H12 | Text improves small area estimates | ARRMSE(FH_text)/ARRMSE(FH_twin), planned domains; ARB not worse | **≥ 0.95 overall**; ≤ 0.90 only in NTEE-MISSING domains | ≤ 0.90 overall, with ARB(text) ≤ ARB(twin) + 1 pp |
| H13 | Frozen deployment keeps the gain | log RE(`FCAL_frozen+e` vs `FCAL_frozen_frame`) / log RE_inc (in-sample); coverage of DIFF | **50–80% retained** | ≥ 80% retained; DIFF coverage ≥ 94% |
| H14 | The gain survives a single weight | log RE(`MCALM…+e` vs `MCALM…` twin) / log RE_inc (per-y); min_y RE vs B0 | **< 60% retained** | ≥ 75% retained and min_y RE(vs B0) ≥ 0.95 |
| H15 | Page images add beyond text (CCEW) | RE(fused image+text vs text-layer embedding), numbers masked | **≤ 1.02** | ≥ 1.05, and it persists with numbers masked |
| H16 | Diagnostics predict Monte Carlo results | median \|log RE_MC − log RE_pred\| | — | ≤ 0.05 |

### 9.4 Verdict rules (what would falsify the idea, and what would justify production)

**Gates.** If any of G2–G8 fails in more than one division after diagnosis, the verdict is "harness not validated". No claims are made either way.

**FALSIFIED (calibration use).** H1 *and* H2 fail at F1, *and* H3 fails. If, in addition, H6's gap-closure G < 0.25, then text is not worth deploying even for weak registers.

**NICHE (weak-register only).** H1, H2 and H3 fail at F1, but H6 passes. The recommendation is to use text only for frame segments without admin data (births, F0-type units, non-filers), and to use names (T0) for births.

**SUPPORTED FOR A PRODUCTION TRIAL.** All of the following hold:
1. H1 or H2 passes, **and** H3 passes;
2. H14 passes (single-weight form);
3. H13 passes (frozen, previous-wave form);
4. the gates pass in ≥ 7 of 8 divisions;
5. **governance:** the best open-weight or on-prem representation (R2, R4 or R5) is within 5% of the best overall (RE ratio ≥ 0.95);
6. **replication:** the CRA T3010 run (§10.2) gives GM RE_inc ≥ 1.05 over its D-class y with the gates passing;
7. **domain no-harm:** fewer than 5% of planned domains show an RRMSE increase greater than 10%.

**SUPPORTED FOR CHARITY SURVEYS WITH T3010-LIKE FRAMES.** The above **plus** H5.

Anything between these verdicts is reported as **inconclusive**, with the binding failure named.

### 9.5 Controls on researcher degrees of freedom

| Decision | Frozen value | Recorded in | Change process |
|---|---|---|---|
| Populations | Dev: NE. Confirmatory: the other 8 divisions, P1. Sensitivity: P2. | `prereg_v1.yaml` | None |
| y set and classes | §1.3, including placebo and known-answer y | yaml | None. No y may be dropped; all are reported. |
| Volunteers treatment | Raw = H-class; cap at 10,000 = S-class | yaml | None |
| Frame tiers and BMF-max encoding | §2.4 | yaml + code hash | — |
| Design date and snapshot | D = 2024-06-30; BMF primary = Dec 2023 (fallback Aug 2023) | yaml | — |
| Response definition | 300 days; sensitivity 180 and 365 | yaml | — |
| Designs and n rule | §3 | yaml | — |
| Primary estimator, twin, B0 and B* definition | §4.2 | yaml | — |
| Learner, λ grid, K, PCs, offset, folds | §4.3 | yaml | — |
| Primary embedding and text field | `lsa256_full`, T2 | yaml | — |
| Edit rules | E1–E3 as in `apply_edits` | code hash | Sensitivity: no edits |
| R, seeds | §8.3 | yaml | Only the pre-freeze R adjustment |
| Thresholds | §9.1–9.4 | yaml | None |

**Freeze procedure.**
1. After Stage 2-dev, write `prereg/prereg_v1.yaml`.
2. Compute its SHA-256.
3. Commit it to git **[USER: sign-off]**.
4. Confirmatory run scripts refuse to start unless the yaml hash and the code commit match the manifest.

Stage 1 diagnostics for the confirmatory divisions are computed **in the same automated job** as their Monte Carlo, after the freeze, so nobody tunes against them. Later stages (NR, SAE, Cohere, replications) each get an addendum `prereg_v1.k.yaml`, frozen before their confirmatory runs. Deviations go in `prereg/deviations.md` with date, reason and impact, and results are reported both ways.

---

## 10. Dataset red-team checklist and replication datasets

### 10.1 Notebook 01 tests

Each test prints a number and PASS/FAIL. FAIL blocks Stage 1 unless a logged deviation accepts it. "Obs." gives values already observed (NE or national).

**A. Units and population**

| ID | Test | Pass rule | Obs. |
|---|---|---|---|
| A1 | One row per EIN; log of drops by reason (form type, group return, duplicates, multiple periods, territory) | Drop log sums exactly | TY2023: 8,369 EINs with more than one return |
| A2 | Amended truth: share of y from amended returns; \|y_amended − y_orig\|/y_orig | Total revenue changed by < 1% | — |
| A3 | Group returns excluded; revenue share | Reported | 5.5% of revenue |
| A4 | FYE mix and short or long periods | Reported; Dec-FYE-only sensitivity | Dec 59%, Jun 25% |
| A5 | BMF state vs XML state agreement | > 98% | Index state missing for about 16k (national) |
| A6 | **Survivorship (RT-4):** deaths, EZ-switchers and births as shares of t−1 and t totals, by y | Reported; P2 built | NE: 1,115 deaths, 1,414 births |
| A7 | Division assignment by BMF state; per-division N | Matches the expected counts | 14.7k–58.2k |

**B. Timing and leakage**

| ID | Test | Pass rule | Obs. |
|---|---|---|---|
| B1 | BMF snapshot date ≤ D; share with `REVENUE_AMT` = y_revenue,t | < 1% | Dec-2023: 0.62%; 2026: 18.3% (fail) |
| B2 | **Frame-time availability (RT-2):** first TY2022 submission ≤ D, by size class; `avail_D` built | Reported; used in F1 | 94.4% of units / 87.7% of revenue (lower bound) |
| B3 | The t−1 return used is the first original filing, not a later amendment | 100% | — |
| B4 | Every embedded text's ObjectId belongs to a return submitted ≤ D | 100% | — |
| B5 | Exact match of y_t or y_{t−1} values, with and without commas, in the text; share of texts with digits or "$" | Exact-match share < 1%; T2m built | Mission: 10–11% have a digit; programs: 39% |
| B6 | XPath whitelist (no `ExpenseAmt`, `GrantAmt` or `RevenueAmt` siblings) | 0 violations | — |
| B7 | Staleness: share of identical text in t−1 and t | Reported | Mission 70–93%; programs 38–86% |
| B8 | Near-duplicate clusters, including clusters that straddle take-all and take-some | Groups built; size distribution reported | 989 units (4.8%) |
| B9 | TF-IDF/LSA vocabulary fitted on frame (t−1) text only; PCA on U only | Asserted in code | — |
| B10 | Edit rules: counts, and effect on totals and on lagged-y R² | Reported; no-edit sensitivity | — |
| B11 | "See Schedule O" variants and boilerplate after `clean_text` | Share < 1%, or handled | — |

**C. Frame completeness and baseline strength**

| ID | Test | Pass rule | Obs. |
|---|---|---|---|
| C1 | NTEE missing, by size, age and subsection | Reported | 31% NE |
| C2 | **Baseline starvation (RT-1):** all BMF-max fields are present and parsed; AV(BMF-max twin) vs AV(NTEE10 twin) per y | Fields present; ratio reported | — |
| C3 | **Text–code redundancy:** R² of the 32 text PCs on BMF-max dummies; linear-probe accuracy for NTEE | Reported (> 0.8 means a code proxy) | — |
| C4 | Foundation code × revenue-mix cross-tabulation (contributions share, program-revenue share) | Reported **[VERIFY code meanings]** | — |
| C5 | Register-status classification of each y, checked against a T3010 line list and PD7A content | Signed off **[USER]** | — |

**D. y quality**

| ID | Test | Pass rule | Obs. |
|---|---|---|---|
| D1 | Negatives, zeros, top-k shares overall and *in the take-some part* | Reported | Top 1% hold 68% of expenses |
| D2 | Prior-year column at t vs current-year column at t−1 | > 98% exact | 98.3% |
| D3 | Volunteers: blank vs 0, heaping, the KoC-type outlier | Rule documented | KoC = 42% of the NE total |
| D4 | Null rates per schema version (y and text XPaths) | No version with more than 2× the median null rate | TY2023 mixes 4 versions |
| D5 | Cross-check against the SOI extract (processing-year caveat) | Revenue totals within 2% | — |
| D6 | Gross errors after editing (e.g., employees with $0 salaries) | Reported | — |

**E. Text quality**

| ID | Test | Pass rule |
|---|---|---|
| E1 | Length distribution by size class, NTEE and `avail_D` source (TY2022 vs TY2021 vs name) | Reported |
| E2 | MiniLM truncation rate (> 256 tokens) by size class | Reported |
| E3 | Non-English share | Reported |
| E4 | Duplicate share in the take-some part | Reported |

**F. Response mechanism**

| ID | Test | Pass rule |
|---|---|---|
| F1 | Response rates by size, NTEE and state; expense-weighted rate | Reported (60% / 18%) |
| F2 | Respondent vs nonrespondent means of y within strata (nonresponse bias potential) | Reported |
| F3 | Late-filing gradient vs typical survey gradient (RT-12) | Stated |

**G. Model and memorization**

| ID | Test | Pass rule |
|---|---|---|
| G1 | Placebo y built and frozen | — |
| G2 | Change-prediction probe (LK5) for every rung | Out-of-fold R²(change \| frame + text) − R²(change \| frame) ≤ 0.01 for local models; Cohere is compared against them |
| G3 | Name-shuffle probe | ≈ 0 |

**H. Simulation hygiene**

| ID | Test | Pass rule |
|---|---|---|
| H1 | CRN: the same samples across separate runs (hash of s_r) | Identical |
| H2 | Bit reproducibility of 10 replicates | Identical |
| H3 | Exact unbiasedness of DIFF_frozen | \|RB\| ≤ 3 MC SE |
| H4 | HT coverage per y | 93–97%, or flagged |

### 10.2 Replication R1: CRA T3010 (Canada)

**Why.** It is the closest real analogue for a StatCan reader:
- program descriptions in English and French;
- government revenue split by level;
- staff counts;
- an annual census of about 84k charities.

It also forces the **F2** question, since the lagged T3010 is held for everyone.

**Data [USER].** `open.canada.ca` is blocked in this sandbox. Download two consecutive fiscal-period years (e.g., FPE 2021 and 2022) of the following files from dataset `2dc3c94d-f98c-4144-98c8-3c7da3d844d5`, and place them in `data/raw/cra/` with checksums:
- `IDENT`;
- `NEW_ONGOING_PROGRAMS`;
- `FINANCIAL_SECTION_A_B_AND_C`;
- `FINANCIAL D_AND _SCHEDULE_6`;
- `SCHEDULE_3_COMPENSATION`;
- the program-area tables.

**Setup.**
- U = designation C charities with filings in both years.
- y:

  | Line | Item |
  |---|---|
  | 4700 | total revenue (R-class) |
  | 4540 / 4550 / 4560 | government revenue by level (D) |
  | 5100 | total expenditures (R) |
  | 300 / 370 | staff (R/S) |
  | 390 | compensation (R) |
- Frame: category, sub-category, designation, province, lagged 4700, plus F2 lagged y.
- Text: t−1 program descriptions (EN/FR), embedded with a multilingual model.
  - MiniLM is English-only, so LSA per language plus Cohere or Qwen are needed.
  - Report the French subset separately.
- **Do not use name embeddings as frame-time:** names are backfilled. Only 1.4% of BNs show any name change over 2020–24, which is the warning sign.

**Red team.**
- Section D vs Schedule 6 structural missingness (revenue < $100k).
- Arithmetic-violation flags (12.8% of BN-years in the GovAlta work).
- Form-version changes.

**Protocol.** Identical code and thresholds; frozen as `prereg_v1.6.yaml`.

**Compute.** N ≈ 84k is roughly 4 NE-equivalents. About 3 h for Stage 2 on 3 cores.

### 10.3 Replication R2: UK Charity Commission for England and Wales (the genuine multimodal test)

**Why.** It is a real register with accounts PDFs (trustees' annual reports and accounts, including scans). Structured annual-return y give a pseudo-census for: total gross income, government grants and contracts income, volunteer counts, and employees for Part B filers.

**Data [USER].**
- Allowlist the CCEW extract host (`ccewuksprdoneregsadata1.blob.core.windows.net`) and the register site, or download outside the sandbox.
- Confirm the scraping and terms-of-use position for per-charity PDF retrieval, with polite rate limits.

**POC scope.**
- About 5,000 main charities (`linked_charity_number = 0`) with income £25k–£10M in 2–3 regions.
- Accounts PDFs for financial period t−1, and the annual return at t.

**Arms.** All on the *same* documents:
1. PDF text layer → LSA, MiniLM and Cohere text;
2. OCR text, for scans;
3. page images (first 5 pages) → Cohere v4/v5 image;
4. fused text + image;
5. page images with numbers redacted (boxes over digit tokens from the text layer or OCR) — the visual analogue of LK2;
6. `charitable_objects` (registration-time text) as the "register text" arm.

**Comparators.** The same F tiers, built from the CCEW register: classification, area of operation, registration date and lagged income.

**Leakage specific to PDFs.** The accounts *contain* lagged financial amounts, so the image arm is always compared against a frame that **includes the same lagged numbers**. If the image only re-reads last year's income, it adds nothing an NSO lacks.

**Cost.** About 25k images × about 1.2k image tokens ≈ 30M tokens ≈ **$14** with v4. About 1 h at 400 images/min. Image-token count is measured in a 100-page pilot first.

**Governance note.** Trustee names in the PDFs are personal information. They are handled as such and never published.

**Fallback.** ACNC (Australia) AIS plus financial reports, if CCEW access fails **[VERIFY route]**.

---

## 11. Notebook and results layout; staged execution plan

### 11.1 Layout

```
embedding-auxiliary-variables/
  prereg/
    prereg_v1.yaml            # frozen spec: populations, y classes, tiers, designs, estimators, tuning, R, seeds, thresholds
    prereg_v1.sha256
    prereg_v1.k.yaml          # stage addenda (NR, SAE, ladder, replications)
    deviations.md
  src/                        # library and run scripts only; notebooks never run long MC loops
    run_stage.py              # CLI: --stage --population --tier --design --n --R --spec prereg_v1.yaml (checks hash)
    ... (existing modules + Appendix A changes)
  data/                       # manifests and small derived tables; large files by pointer + sha256
    MANIFEST.md               # where raw/derived data live (/tmp/claude-0/data is session-scoped: persist derived artefacts!)
  notebooks/
    00_environment_and_manifests.ipynb     # versions, data hashes, embedding manifests, reproducibility check
    01_dataset_redteam.ipynb               # §10.1, all populations, PASS/FAIL table
    02_population_diagnostics.ipynb        # AV ceilings (twin pairs), partial R², redundancy, κ, predictions table
    03_mc_calibration_dev_NE.ipynb         # exploratory; ends with the freeze proposal
    04_mc_stratification.ipynb             # D0/D1/D1-imp/D2a/D2b, 2x2 tables
    05_mc_nonresponse.ipynb                # NR-real, NR-sim A/B/C, NR-TA sensitivity
    06_sae.ipynb                           # planned + ZIP3 domains, FH twin/text, CV-band migration
    07_frozen_previous_wave.ipynb          # TY2021->2022 training, deployment at 2022->2023
    08_embedding_ladder.ipynb              # R0..R6, fields, fullD, Cohere factors
    09_confirmatory_divisions.ipynb        # H1-H3 tests, forest plots by division, verdict
    10_replication_cra_t3010.ipynb
    11_replication_ccew_multimodal.ipynb
    12_synthesis_and_translation.ipynb     # equivalent n, cost, CV bands, governance checklist, final verdict
  results/
    stage{K}/{population}/{tier}_{design}_n{n}[_{mech}]/{run_id}/
      reps.parquet        # long: rep, estimator, y, est, var, failed
      metrics.parquet     # mc.metrics output + failure counts + weight diagnostics
      diag.parquet        # AV ceilings and predictions for the cell
      manifest.json       # git commit, prereg hash, seeds, data/embedding hashes, versions, runtime
```

**Notebook rules.**
- Notebooks read `results/` and cached embeddings only, and assert manifest hashes.
- Every notebook ends with a **decision cell**: what was learned, what was decided, which working-log entry records it.
- The dev notebooks (03–08 on NE) are labelled EXPLORATORY in their first cell. Notebook 09 is the only CONFIRMATORY one for H1–H3.

### 11.2 Staged plan

Baseline cost: about 1 s per replicate for 30 estimators at n = 1,000 (pilot). Cost scales roughly with n·p² and with the number of estimators, and is nearly independent of N (population sums are precomputed). The CPU is shared (4 cores), so use **at most 3 worker processes**.

"NE-equivalents" = Σ_g n_g/1000. The 8 divisions sum to about 13.9 NE-equivalents.

| Stage | Content | Gate to proceed | Compute (CPU core-hours, approx.) | Needs |
|---|---|---|---|---|
| **0 Data QA** | Stage 0 build tasks:<br>• finish the national TY2023 download;<br>• download national TY2021 (about 14 GB) and the TY2023 990-EZ for P2;<br>• download the Dec-2023 BMF;<br>• rebuild the population with BMF-max fields, `avail_D` and lagged y for F2.<br>Run notebook 01 on NE and all divisions (automated; prints PASS/FAIL). | All A–H PASS, or a logged deviation | Parsing about 330k × 2 returns: about 3–4 h; QA: < 1 h | Network to S3 (reachable) |
| **1 Diagnostics** | NE: AV ceilings for every twin pair and tier; redundancy; ladder probes (local rungs); κ from Stage 2-dev | Ceilings computed; predictions table written | < 1 h | — |
| **2-dev** (NE, exploratory) | Tiers F0/F1/F1p/F2 × D1 × n = 1000, R = 500. D0 and n ∈ {500, 2000} at R = 300. Ladder cell (local rungs, fullD), R = 500. Known-answer and controls. Bootstrap sub-study (V4). | Gates pass in dev; R precision target set; **freeze prereg_v1 [USER sign-off]** | About 6 h | — |
| **2-conf** (8 divisions) | F1 and F2 × D1, R = 1000, about 45 estimators. Diagnostics are computed in the same job. Empirical n-grid (primary only, R = 500). F0 and F1p cells at R = 500. | — | Embedding (MiniLM on 330k T2 texts): about 4–6 h; LSA: minutes. MC: about 13.9 × 2 × 1000 × 1.5 s ≈ 12 h, plus about 4 h for F0/F1p and the n-grid. **About 16 h, ≈ 6 h wall time on 3 cores.** | Stage 0 national data |
| **2b Stratification** | NE dev, then divisions (D1-imp and D2a; D2b only if national TY2021) | — | About 4 h | — |
| **2c Frozen** | NE (TY2021 → 2022 → 2023), R = 1000; divisions if national TY2021 | — | About 2 h (NE), about 8 h (divisions) | TY2021 data |
| **3 NR** | NR-real (300/180/365 days), NR-sim A/B/C, NR-TA; NE dev, then divisions for NR-real | Addendum v1.3 frozen after NE | About 6 h | — |
| **4 SAE** | Planned + ZIP3 domains; NE dev, then divisions; R = 500 | Addendum v1.4 | About 3 h | — |
| **5 Cohere ladder** | Embed NE (20.6k; about $0.2; about 10 min) and then national (about 330k; about $2–4 text; 2.5 h at production limits). Repeatability, batch, Matryoshka, int8 and canary tests (< $0.50). MC ladder cells R = 1000 with CRN. | Addendum v1.5 | MC about 6 h | **[USER] allowlist + production key** (a trial key is capped at 1,000 calls/month and cannot finish one pass) |
| **6 Replications** | CRA T3010 (R1); CCEW multimodal (R2) | Addendum v1.6 | R1 about 3 h; R2 embedding about 1 h + MC about 1 h | **[USER] downloads, allowlists, terms of use** |

**Total compute:** roughly 60–80 core-hours over all stages, which is about 1–2 weeks of wall-clock on a shared machine at 3 workers. The binding constraints are user actions and network access, not CPU.

---

## 12. Real-world translation and governance

### 12.1 How a finding would be used

1. **Sample size and burden.** Express each confirmed RE as the empirical equivalent-n (§8.5), under the binding-constraint rule.
   - Example: RE = 1.20 on every binding y at n_TS ≈ 794 in an NE-sized population is about 130 fewer questionnaires a year. At about 5 hours each **[USER: real burden estimate]** that is about 650 respondent hours.
   - The take-all units, which carry most of the burden, are untouched. Burden reduction falls on small units, which is often a policy goal in itself.
2. **Same n, better domains.** Report the gain as more publishable domain cells (CV-band migration, §8.6). For many NSOs this is the more valuable outcome.
3. **Nonresponse follow-up.** If H11 holds, text-informed RHGs or propensities could target follow-up. If BRF is small, the honest message is "text does not fix nonresponse".
4. **Where not to use it.**
   - For R-class y with admin data: never.
   - For T3010-covered items of registered charities: only if H5 passes.
5. **The skeptic's framing, made concrete.** "Variance reduction, not productivity" is tested directly:
   - D1-imp vs D1 measures the *productivity* use (autocoding NTEE) in variance terms;
   - the calibration hypotheses measure the *variance-reduction* use.

   The two are reported separately, so neither result can stand in for the other.

### 12.2 Which NSO data would play the role of text

| Role in the testbed | Canada | US / other | Availability / timing | Notes |
|---|---|---|---|---|
| T1s / T1 (short frame-time description) | CRA BN registration "major business activity" text (BR, confidential); NAICS autocoding input | IRS SS-4 line 16–17; ABN main activity (Australia) | Design time; possibly stale for old units | Plain text: **no multimodal case** |
| T2 (program descriptions) | T3010 Section C2 program descriptions (public, EN/FR) | Form 990 Part III; CCEW `charitable_objects` / activities | Design time (prior year) | CRA replication R1 |
| T0 (names) | BR legal and operating names | BMF names | Universal | Only text for births |
| Websites | Enterprise websites under the StatCan web-scraping policy (notify, API first, no personal information) | ISTAT/OBEC web intelligence | Estimation time; size-graded coverage (NE: about 45% of small vs 88% of large units have a URL) | Not testable here (no crawl) **[USER: network]** |
| PDFs (annual reports, financial statements) | Financial statements filed with the T3010 (held by CRA **[VERIFY StatCan access]**); SEDAR+ for public companies | CCEW accounts; Companies House; ACNC | Estimation time or prior year | **The only real multimodal case**. CCEW replication R2. |
| Charity registers | CRA charities listing | CCEW, ACNC | Design time | Replication frames |

### 12.3 Governance requirements (deployment criteria, not afterthoughts)

1. **Confidentiality.** BN registration text and BR content are confidential. They may not go to a SaaS embedding API. None of Cohere's managed-cloud options processes in Canada (note 04 §2.4). The permissible routes are:
   - on-prem open-weight models (Qwen3-Embedding / Qwen3-VL-Embedding, Apache-2.0);
   - LSA;
   - a Cohere private deployment or Model Vault with a written residency guarantee.

   Hence verdict criterion 5 in §9.4: open-weight within 5% of the best.
2. **Reproducibility and model lifecycle.**
   - Content-addressed embedding cache (note 04 §6); vectors stored with manifests.
   - A canary set re-embedded at each cycle (alarm at cosine < 0.999).
   - Models are deprecated: Cohere's Embed v2 was shut down on 2026-04-04.
   - **A model change is treated like a classification revision:** a dual-run bridging cycle that re-estimates with the old and new embeddings, and reports the break in the series.
3. **Documentation (UNECE uncertainty quantification handbook).** Report:
   - the estimator (model-assisted, with design consistency regardless of model correctness);
   - the variance method, including the V4 bootstrap evidence;
   - the coverage evidence from this benchmark;
   - the model card for the embedding;
   - the cross-fitting and tuning rules.
4. **Explainability.**
   - For LSA, publish the top terms of the directions that carry weight.
   - For any model, show which unit groups have g-weights moved by text, and by how much.
5. **Fairness across domains.** Text quality is size- and language-graded: small units have short texts, and Canada has French texts. Domain-level bias of text models (§7.3 heterogeneity check) and the French subset (R1) are mandatory reporting.
6. **Stability.**
   - Year-to-year volatility of weights and of domain estimates, compared between the frozen and re-trained models.
   - A text model that reshuffles weights every year creates artificial changes in the series.
7. **Do-no-harm guarantee and its limits.**
   - Model-assisted estimators stay design-consistent even when the text model is wrong.
   - They do **not** protect against a poor variance estimate at small n, or against bias in synthetic or SAE components.
   - The benchmark reports exactly those failure modes (OF1, OF2, SYN, H12).

### 12.4 Known limitations that bias the results, and in which direction

| Limitation | Direction for text |
|---|---|
| Lagged values and y come from the same instrument (RT-13) | Conservative: understates text's value |
| No survey measurement error | Neutral to slightly conservative |
| No website content (the most common real estimation-time text) | Unknown |
| US non-profit sector only, until R1/R2 | Unknown transfer |
| Previous-wave codes from a slightly later BMF (§7.4) | Slightly favourable to the frozen models |
| P1 conditions on survival (RT-4) | Favours lagged-size baselines; checked by P2 |
| Late filing as nonresponse (RT-12) | Unknown; triangulated with NR-sim |

---

## Appendix A. Code changes required (all marked [CODE] above)

1. **RNG streams (RT-10).** `mc.run_cell` draws samples from a per-replicate `SeedSequence([seed_c, r, 0])`. Each estimator gets its own stream. Add a hash of s_r to `reps.parquet` for the CRN check (H1).
2. **BMF-max frame (RT-1).** `build_population.build` keeps:
   - foundation, affiliation, classification, activity (first code), deductibility, organization, ACCT_PD, FILING_REQ_CD, GROUP, ASSET_AMT, INCOME_AMT, REVENUE_AMT, SORT_NAME, full NTEE_CD;
   - `frame.build_mats` adds matrices `bmfmax`, `bmfmax_ratio` and `F0_ratio`, using the fixed pooling rule.
3. **Frame-time availability (RT-2).** Add `avail_D` from the first TY2022 submission ≤ D (the manifest keeps only the latest return, so the index is needed). Add the BMF-amount fallback, the TY2021 text fallback and the `text_missing` flag.
4. **Lagged y for F2:**
   - `lag_*` for every y, including the binaries. `parse990_min` already parses them for t−1; `build` drops them.
   - Rename the pilot's `oracle_lag_*` to `lag_*`. It is a tier, not an oracle.
5. **Tier-consistent designs (RT-9).** `production_design(size_col=…)` per tier; the F0 size measure is the `INCOME_CD` midpoint.
6. **Grouped and stratified folds everywhere (RT-15).** Including the `ridge` kind's λ-CV.
7. **Multipurpose estimators (RT-6).** `MCALM_*` (joint calibration on all m̂ with common-λ ridge on the m̂ columns) and `RIDGEM_*` (common λ).
8. **Full-dimension supervised arm (RT-7).** Ridge in dual form (n × n kernel per fold, shared SVD over λ).
9. **Twin-pair AV ceilings on the ratio scale (RT-11),** in `diagnostics.py`.
10. **Failure accounting** in `mc.metrics`: failure counts, and RE computed on the replicates where both members of the contrast succeeded.
11. **Frozen previous-wave pipeline** (§7.4); **NR-sim** mechanisms (§7.2); **SAE module** (DIR, MAD, SYN, FH-REML with GVF and Prasad–Rao/Datta–Lahiri MSE).
12. **Rao–Wu bootstrap** (V4) and the **known-answer y** generator (§5.4).
13. **P2 population builder** (deaths as zeros; EZ-switcher values).
14. **`run_stage.py`**, which refuses to run confirmatory cells unless the prereg hash matches.
15. **Persistence.** Derived populations and embeddings currently live in `/tmp/claude-0/data` (session-scoped). Copy them to a persistent location with sha256 manifests before Stage 2.

## Appendix B. `prereg_v1.yaml` skeleton

```yaml
version: 1
frozen_on: null            # set at freeze; sha256 in prereg_v1.sha256
populations:
  dev: [NE]
  confirmatory: [MidAtl, ENC, WNC, SAtl, ESC, WSC, Mountain, Pacific]
  definition: P1           # P2 = sensitivity
design_date: 2024-06-30
bmf_snapshot: 2023-12-12   # fallback 2023-08-16; sensitivity 2024-04-11
response_rule_days: 300    # sensitivity [180, 365]
y:
  R:  [y_revenue, y_expenses, y_employees, y_salaries]
  D4: [y_gov_grants, y_contrib, y_prog_rev, y_fundraising_exp]
  D_secondary: [y_officer_comp, y_grants_paid]
  S5: [y_volunteers_cap10k, y_bin_has_volunteers, y_bin_grants_to_indiv_us, y_bin_foreign_activities, y_bin_lobbying]
  S_secondary: [y_bin_fundraising_activities, y_bin_gaming, y_bin_has_gov_grants, y_bin_has_employees]
  H:  [y_volunteers]
  P:  [PLC_perm, PLC_noise, PLC_bern]
  K:  [KA_text]
tiers: [F0, F1, F1p, F2]   # primary F1
designs: {primary: D1, others: [D0, D1-imp, D2a, D2b, D2b-ctrl]}
n_rule: "round(1000 * N / 20626)"
estimators:
  primary_text: "MCAL_bmfmax_ratio+lsa256_full"
  twin: "MCAL_bmfmax_ratio"
  B0: {F0: GREG_weak, F1: GREG_size_cat, F1p: GREG_size_cat_pay, F2: GREG_F2_multi}
  Bstar: "ex-post per-y min-MSE over status-quo + text-free ML of the tier"
tuning: {learner: ridge, lambdas: [1e-3, 1e-2, 1e-1, 1, 10], folds: 5, folds_grouped: true,
         folds_stratified: true, K_add: 10, K_ratio: 32, ratio_offset: 1.0, pool_min_count: 20}
R: {confirmatory: 1000, sae: 500, dev: 500, bootstrap: [200, 200]}
seeds: "SeedSequence([hash32(pop,design,n), r, stream])"
gates: {G2_equiv: [0.97, 1.03], G2_ci: [0.95, 1.05], coverage_min: 0.93, var_rb: [-0.15, 0.15],
        G5_slack: 1.05, G6_range: [1.12, 1.38], G7_equiv: [0.95, 1.05], LK2_max_gap: 0.03, LK3_max: 0.01}
hypotheses:
  H1: {metric: "GM_D4 RE(primary vs twin)", pass_point: 1.10, pass_lb: 1.00, falsify_point: 1.05, test: "t7 on log GM, Holm{H1,H2,H3}"}
  H2: {metric: "GM_S5 RE(primary vs twin)", pass_point: 1.10, pass_lb: 1.00, falsify_point: 1.05}
  H3: {metric: "GM_D4 RE(primary vs Bstar)", pass_point: 1.10, pass_lb: 1.00, falsify_point: 1.03}
  # H4-H16 as in section 9.3
verdict_rules: "section 9.4"
```
