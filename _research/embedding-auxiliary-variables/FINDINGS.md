# Findings: text embeddings as auxiliary variables

**Question.** Can text embeddings of unit descriptions serve as auxiliary variables for stratification, nonresponse
adjustment, calibration and small area estimation, and reduce variance enough to matter in a real survey?

**Status.** New England proof of concept under the pre-registered protocol v1.0 (`notes/benchmark-protocol.md`), with
every departure logged in `prereg/deviations.md`. It is a development population, so generalization is **not**
assessed. That is what the 8 held-out Census divisions are for (§9).

<!-- BOTTOM LINE: PENDING -->

---

## 1. How the idea works

### 1.1 Where text can enter, and how

Every use runs through one object: a prediction m̂(z, e) of the survey variable y from register variables z and the
text embedding e. It is fitted on the sample and cross-fitted. The text never replaces the survey; it only shapes
weights or predictions.

| Use | How the embedding enters | Estimator in this benchmark | What protects validity |
|---|---|---|---|
| Calibration / model-assisted estimation | m̂ becomes one more calibration column, with known population total Σ_U m̂ | Model calibration (Wu & Sitter) on [register, m̂]; difference / PPI form; single-weight version with frozen heads | Design-consistent for *any* m̂. A bad model costs efficiency, not bias. Cross-fitting stops overfit (pitfall demo: coverage 72–78% without it) |
| Production deployment | Heads trained on **last cycle's respondents**, applied to this cycle's frame | `FROZEN_*` arms | m̂ is fixed before the sample is drawn, so it is just another register column |
| Stratification / allocation | Strata on a frozen predicted score; text-coded activity code where it is missing; allocation by predicted residual SD | D1-imp, D2a, D2b (protocol §3.2; not run here, D-12) | Design-unbiased by construction |
| Nonresponse | Text in the response propensity or the response homogeneity groups; text-assisted imputation for take-all units | `NR_ps_text_*`, `NR_rhg_*` | Only removes bias if text predicts both response and y |
| Small area estimation | Domain mean of m̂ as the Fay–Herriot covariate | `FH_text` vs `FH_twin` | Model-based. Bias is possible, so it is reported per domain |

### 1.2 What a gain requires

For a model-assisted estimator the variance is approximately the design variance of the residual e = y − m̂. So the
relative efficiency of text over a text-free twin with the same learner, codes and folds is

  RE(text | twin) ≈ AV(e_twin) / AV(e_text) = 1 / (1 − ρ²),

where ρ² is the share of the twin's residual (anticipated) variance that the text explains. This sets the bar:

| Target RE | Partial ρ² needed | Take-some sample saved at equal precision | CV shrinks by |
|---|---|---|---|
| 1.05 | 0.048 | 4.8% | 2.4% |
| 1.10 (production bar) | 0.091 | 9.1% | 4.7% |
| 1.20 (adoption bar) | 0.167 | 16.7% | 8.7% |

Two structural facts make ρ² small for business-type populations. Both were found and quantified here:

1. **The baseline already holds most of the explainable variance.** With last year's revenue, expenses, assets,
   employees and payroll on the register, residuals of financial y are mostly *year-to-year change* and *idiosyncratic
   amounts*. A mission statement is a near-static description: 84.5% of mission texts are identical in TY2022 and TY2023.
   It can tell you *what kind* of organization a unit is, which the codes partly do already, but not how its funding
   moved this year.
2. **Heavy tails cap what any unit-level covariate can explain.** Anticipated variance is dominated by a few large
   take-some units with idiosyncratic amounts. In a zero-inflated, heavy-tailed y such as government grants, even a
   *planted* multiplicative text signal could not be made to explain more than a few percent of AV (D-3). The
   explainable share of a multiplicative signal is capped near 1/(1 + CV²) of the amount distribution.

Text must also enter on the **ratio scale** (y/x̃ with weights d·x̃), not as additive terms beside dollar-valued
columns. The useful signal is "this kind of organization gets a high share of its budget from government", not a
dollar amount. This was learned in the synthetic check and the pilot: additive embedding terms were swamped by size.

---

## 2. The testbed

| Element | Choice | Real-world analogue |
|---|---|---|
| Population | 20,626 New England non-profits that filed a full Form 990 for TY2022 and TY2023 | National non-profit survey population |
| Frame at design date D = 2024-06-30 | IRS BMF snapshot 2023-08-16 (all codes and amounts). The TY2022 return counts **only if filed by D**; otherwise BMF amounts and TY2021 text or the name. Return available at D for 91.9% of units, but only 68.8% of revenue: big organizations file late. | Business register + prior-year tax data, as known at design time |
| Register tiers | F0 codes and BMF amounts only · F1 + lagged revenue / expenses / assets / employees · **F1p + payroll (primary)** · F2 + full prior return | BR-only frames; BR + T2/GIFI; BR + PD7A; charities with T3010 |
| Design D1 | NTEE10 (MISSING as a level) × 3 size classes + take-all top 1%; compromise allocation; n = 1,000 (take-some 794 of 20,420) | Production business-survey design |
| Targets (TY2023) | 10 headline variables **not on the F1p register**: government grants, contributions, program revenue, fundraising expenses, officer compensation, volunteers (capped at 10k), and four yes/no items (has volunteers, grants to individuals, foreign activities, lobbying). Four register-type controls: revenue, expenses, employees, salaries. | Survey-only content |
| Text | TY2022 mission + program descriptions (median 71 words); LSA (TF-IDF + SVD) and MiniLM, run locally | Registration "nature of business" text; T3010 program descriptions |
| Embedding E* | Selected on the **previous wave** (TY2021 → TY2022) by the pre-registered rule: LSA of mission + programs, K = 16 components | Fixed before any current-wave result |
| Nonresponse | Real: a unit responds if its TY2023 return was filed within 300 days of fiscal year end. 61% of take-some units respond; 13% of take-all. | Late or missing questionnaires |
| Monte Carlo | R = 2,000 design-based replicates; common random numbers across all arms; one shared fold vector per replicate | — |

**Primary comparisons**
- **P1 efficacy:** `MCAL_bmfmax@F1p+E*` vs `MCAL_bmfmax@F1p`. Identical learner, all IRS codes, identical folds; the only difference is the text.
- **P2 production:** `FROZEN_calm_bmfmax@F1p+E*` vs `GREG_size_cat@F1p`. One bounded weight set with heads trained on last year's respondents, against today's production GREG.

Both use GM-RE over the 10 headline variables with paired-bootstrap one-sided 95% bounds.
- **Production bar:** MET if point ≥ 1.10 and lower bound ≥ 1.05; EXCLUDED if upper bound < 1.10.
- **Any gain:** YES if lower bound > 1.00; NO if upper bound < 1.05.

---

## 3. Data red team: what makes this dataset trustworthy, and its flaws

About 45 checks are in `notebooks/01_data_and_red_team` (`results/01_red_team_checks_ne.csv`).

**Passed**
- Two independently written XML parsers agree on 100% of target and text fields.
- Parsed revenue matches the GivingTuesday index on 100% of units.
- The 2023-08 BMF leaks target-year revenue for only 0.48% of units, against 22.8% for a careless 2026 BMF build.
- Text states the exact target value for ≤ 0.005% of units.
- Placebo y gains nothing from text (−0.004).
- A memorization probe (text predicting revenue change) gives ≈ 0.
- No single take-some unit holds more than 1.9% of any headline total.

**Flaws that are real-register features (kept and modelled)**
- NTEE is missing for 31.1% of units. Text-coded NTEE reaches about 70% accuracy over 10 groups, against about 24% for chance.
- 11.5% of units sit in near-duplicate text groups (chapters, lodges, template missions). Cross-fitting uses grouped folds.
- Valid URLs exist for 72.5% of units (51% of the smallest quartile, 90% of the largest), not the 92.8% a naive "field non-empty" check suggests.
- Big organizations file late. Prior-year returns cover 68.8% of revenue at D, and only 13% of take-all units respond within 300 days.
- Volunteer counts are heaped and 19.7% blank. The estimand is capped at 10k and blank is treated as 0.
- The 2023-08 BMF has **zero or missing amounts for 722 units holding 6.7% of expenses** (Harvard, Tufts, Northeastern; BCBS-MA is absent from the snapshot). This matters for any design built on BMF amounts alone (Stage 2B, D-28a).

**Leakage controls built into the runs**
- Design-date availability: only information filed by D.
- Previous-wave BMF amounts and codes recomputed from lags (B5): the raw BMF leaks the previous wave's target for 16.9% of units.
- A numbers-masked text arm.
- Placebo targets and random, permuted and constant embeddings.

<!-- RESULTS: PENDING -->

---

## 6. Real-world translation: what data would play the role of the text

| Source | Available at design time? | Coverage | Confidentiality | Modality | Best role, given these results |
|---|---|---|---|---|---|
| BN registration "major business activity" text (CRA RC1); US SS-4 | Yes | Near-universal; stale for old units | Confidential (Statistics Act); cannot go to a SaaS API | Short text | Frame coding: fill missing activity codes (31% here) and support births. Weaker than 990 mission text, so expect smaller gains than measured here. |
| T3010 program descriptions (Canadian charities) | Yes (t−1) | About 84k registered charities | Public | Text | Charity frames already carry the full prior-year return (F2). There the lag of y dominates (G9: 3–10×) and text has little left to explain. |
| BR legal and operating names | Yes | Universal, births included | Confidential | Very short text | Births and stale units |
| Organization websites (Common Crawl ≤ D, or a crawl) | At D, or at estimation time | Size-graded: 51% of small units to 90% of large (NE valid URLs) | Public, under the NSO's web-scraping policy | Text + layout | Survey-only *web/ICT* items (e-commerce, AI use, services offered), the ISTAT precedent. Not financial y. |
| Annual reports and financial-statement PDFs (CCEW, ACNC) | t−1 | Mostly larger units | Public (UK, AU) | PDF pages: **where Cohere's multimodal model applies** | **Extract the numbers first.** A grant line in last year's accounts *is* the lag of y, the strongest auxiliary measured here. Embed the narrative only if residual signal remains. |
| Admin tax and payroll (T2/GIFI, T1044, T3010 lines, PD7A) | Yes | Near-universal | Confidential | Numeric | **The baseline text must beat.** Linking one more admin line (the lag of y) was worth 3–10× here. |
| Last cycle's survey responses (rotation overlap) | Yes, for panel units | Previous sample | Confidential | Numeric | Removes most of what text could add for repeat units |

### 6.1 The multimodal (Cohere) question

The Form 990 data contain no PDFs or images. Rendering the XML return to images would put y on the page, and no NSO
would do it. The realistic multimodal case is the prior-year financial statements and annual reports of larger units
(UK Charity Commission, ACNC, company filings). There the protocol's rule applies: compare the multimodal embedding
against **structured extraction of the numbers on the page** (L7 vs L7-ext). The extraction route is cheaper, more
explainable, and turns the PDF into exactly the admin lag that dominates here. Cohere embed-v4/v5 would add value
only if the narrative explains residual variance *after* the extracted numbers. Nothing in these results suggests a
large residual, but it is untested (U3).

### 6.2 Governance, if anything is deployed

- **Confidentiality.** BN and BR text cannot go to a SaaS API, and none of Cohere's managed options documents Canadian processing (note 04). The permissible routes are on-prem open weights (LSA and MiniLM were used here) or a private deployment with written residency.
- **Reproducibility.** Content-addressed embedding caches, frozen per cycle, with a bridging study at model deprecation (Cohere Embed v2 was shut down 2026-04-04).
- **Documentation** in the spirit of the UNECE handbook on uncertainty quantification in ML-enhanced inference:
  - the model-assisted consistency argument;
  - coverage evidence from this benchmark;
  - a model card for the heads.

---

## 7. Limitations and their direction

| Limitation | Direction for text |
|---|---|
| 990 mission and program text is richer than BN registration text | Favourable to text |
| Lags and targets come from the same instrument (no survey measurement error) | Conservative: understates text slightly |
| Primary F1p ignores panel values of y | Favourable to text |
| E* is LSA, not a modern API model (Cohere could not be reached: U1) | Possibly conservative. But the MiniLM rung (L4) and the K = 64 rung bracket the effect of a better representation, and the population ceilings cap it (§4.1) |
| Late filing as nonresponse: an inverted size gradient | Unknown |
| One development population, reused for design | Guarded by the 8 held-out divisions, which are not yet run |
| Not run in this session: designs D1-imp/D2a/D2b, PANEL, cost curve, bootstrap variance, divisions (D-12) | Those hypotheses stay open |

<!-- USER ACTIONS / NEXT STEPS: PENDING -->
