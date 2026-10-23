# Deviations log (protocol v1.0, `notes/benchmark-protocol.md`)

Each entry records:
- what the protocol says;
- what was done;
- why;
- the expected direction of any bias it introduces.

Entries are added **before** the affected results are read. Entries marked *post hoc* were discovered while running and are applied to every cell.

| ID | Protocol | Done instead | Why | Expected effect |
|---|---|---|---|---|
| D-1 | R = 2,000 for the NE proof-of-concept cell | R = 2,000, split over 3 worker processes. Replicate r's sample depends only on (seed, r), and draws for skipped replicates are made and discarded. | Wall time | None: identical samples |
| D-2 | Binaries: ridge-logistic (primary), linear-probability ridge (secondary) | Linear-probability ridge only | IRLS per λ × fold × y would triple the cost of the cell | Slightly conservative for text on binaries (LPM is less flexible). Design consistency is unaffected (model-assisted). |
| D-3 (*post hoc*) | Planted family: y⁰·exp(a·u − a²/2) on the permuted gov-grants base, with a calibrated to ρ² ∈ {0, .05, .10, .20} | That form is kept as a **diagnostic**, reporting the achieved ρ² at a ∈ {0.5, 1, 2, 3}. The calibrated levels use a lognormal-ratio family: y = x̃·c·exp(σ(ρu + √(1−ρ²)ε) − σ²/2), σ = 0.5. | On a zero-inflated, heavy-tailed base, the explainable share of AV for any multiplicative signal is capped near 1/(1 + CV²). Achieved cross-fitted ρ² was −0.03 to −0.04 at every a, so the targets are unattainable. | None on real-y results. The finding itself is reported, since it explains why text gains on such y must be small. |
| D-4 (*post hoc*) | Stage-1 same-class ceilings fitted on the population with oracle λ | Same, with **5-fold cross-fitting over the take-some population** (ceilings and planted calibration) | In-sample population fits gave ρ² ≈ 0.03–0.05 for a planted target with *no* text signal. The few dominant units are fitted by the extra columns. | Ceilings fall (best GM ceiling over H: 1.070 → 1.040). Fixing this makes gate G5 stricter. |
| D-5 | E* = argmax previous-wave GM ceiling; ties within 0.02 go to the simpler rung, then smaller K | Rule applied as written: E* = LSA T2, **K = 16** (ceiling 1.022; best was LSA T2 K = 64 at 1.040). K = 64 is run as a secondary arm, `Estar64`. | The tie tolerance (0.02) is as large as the effect here | The primary text arm may be slightly weaker than the best available. The secondary arm shows by how much. |
| D-6 | Population sum of predictions = average of the fold models' (clipped) predictions | Prediction of the fold-averaged coefficients, then clipped | 5× fewer full-population products | None for the linear part. Negligible through clipping. |
| D-7 | Linear factorial "codes" arm: C (raw 0/1, λ_C) and C⊙x̃ | C plus (first 40 code columns)⊙x̃ | Cost (C has about 330 columns) | Slightly weaker linear-codes cell. Affects only the factorial A×B cells. |
| D-8 | J = 20 frozen heads, trained on previous-wave respondents with RHG weights | As specified. LSA is fitted on the previous wave's frame text and applied to the current text; PCA rotation is from the previous wave. | — | — |
| D-9 | BMF-max C includes the filing-requirement code | Not included (FILING_REQ_CD was not carried into the population tables) | Oversight in the rebuild; not consequential | Very slightly weaker twin, so slightly favourable to text |
| D-10 | T5 nested-CV check; T6 λ-stability logging | Not run in this session | Time | Unknown, small |
| D-11 | Variance v_g for single-weight arms; replicate bootstrap v_B (Stage 2E) | Linearization v_L only | Time | Coverage of single-weight arms may differ slightly |
| D-12 | Stages 2B–2K (F0/D1-W, scaling, bootstrap variance, D2 designs, cost curve, PANEL, sensitivities, P2 population) | Not run in this session. Stage 2A (with F1/F2 tier arms), the planted lin-same family, and the Stage 3/4 core are run. | Time and compute | Those secondary hypotheses remain open |
| D-13 | Stage 0c: rerun the exact pilot v1 cell after the fixes | Not rerun separately. The Stage 2A cell contains the F1-tier twin and text arms, the pilot-equivalent comparison. The pre-fix pilot v1 results are archived for comparison. | Avoid a duplicate long run | None |
| D-14 (*post hoc*) | Matrix C includes a **valid-URL** flag (test C3: contains a dot; not N/A, NONE or "see schedule o") | Stage 2A used `has_website` (field non-empty): 92.8% against 72.5% valid | Flag rebuilt after the Stage 2A launch | One weakly informative dummy. Negligible. Fixed for later stages. |
| D-15 | Lags and text from the *first original* TY2022 submission ≤ D (test B3) | From the most recent TY2022 submission (originals and amendments). `avail_D` uses that submission's date. | The index keeps the latest return per EIN-year | Conservative: a unit whose amendment came after D is treated as unavailable even if its original was filed before D. Affects about 1.5% of units. |
| D-16 (*post hoc*) | Stage 1 selection on the previous wave with B5 applied | First run omitted B5. Rerun with B5: E* unchanged (LSA T2, K = 16; GM ceiling 1.0227; best 1.0404). | Caught by audit against §10.1 | None (selection unchanged) |

## Adversarial-review round (2026-10-23), applied before any R = 2,000 result was read

Four lens reviewers raised 29 findings and adversarial verifiers confirmed 18. The Stage 2A run in progress was **stopped and restarted** with every fix below. Stage 3/4 runs had not started.

| ID | Finding | Action |
|---|---|---|
| D-6 (withdrawn) | Clipping the fold-averaged model biased the calibration total downward by −0.05 to −0.08 SE (2–4 MC SE), more for the text arm | **Reverted** to the protocol's average of clipped fold predictions |
| D-17 | ρ²_min used an undocumented 2-of-3 rule | Rule = ceil(0.8 × plantings) = 3 of 3 (protocol 4 of 5 with 5 plantings); 2-of-3 reported as a sensitivity. Note: with planted levels {0, .05, .10, .20} and a 1.10 bar, ρ²_min ≤ 0.10 needs φ ≥ 0.905 at ρ² = 0.10, so **FALSIFIED is nearly unreachable by construction**. This is a protocol-level issue flagged for the user at FREEZE-2. |
| D-18 | G9 cannot pass on masked F2: masking late filers' survey lags caps officer_comp at about 1.26× | New **F2o** oracle tier with the survey lags unmasked, used for G9. Realistic masked F2 is kept for S-F2. |
| D-19 | Gates G5 and G6 were never evaluated, and G2(N4)/G3/G4 were coded narrower than the protocol | G5 (against Stage-1-style ceilings), G6 (planted), and G3/G4 over every text arm and comparator are now implemented |
| D-20 | Fold groups covered exact mission duplicates only, and frozen training used ungrouped folds | Groups = exact duplicates ∪ cosine > 0.95 clusters on E* ∪ shared valid URLs (11.5% of units in multi-unit groups, up from 4.8%). Frozen training uses the same groups. |
| D-21 | TY2021 fallback text used without checking that it was filed by D (56 units) | Fallback requires a TY2021 submission ≤ D, otherwise name only |
| D-22 | Response timing came from the latest (possibly amended) TY2023 submission | Response timing now uses the **first** TY2023 submission (424 units reclassified) |
| D-23 | Units unavailable at D still carried website information from their post-D return | Website is set unknown for units unavailable at D |
| D-24 (Stage 3) | Take-all nonrespondents were imputed from availability-masked (zeroed) lags, biasing every arm by up to −33% | Imputation now uses the **unmasked own TY2022 value × NTEE10 trend** among responding take-all units, with a fallback to lag_expenses × ratio. Take-some bias is reported separately. |
| D-25 (Stage 3) | Propensities were fitted in-sample on about 450 standardized columns (in-sample AUC 0.90 against 0.58 cross-fitted) | Propensities are cross-fitted on the shared folds, with C chosen by inner CV log-loss and categorical columns on the raw scale |
| D-26 (Stage 3) | The variance of the propensity arms ignored within-stratum weight variation | Residuals are multiplied by g_k = dr_k / mean_h(dr) in v_R |
| D-27 (Stage 4) | SAE twin and text arms got different fold vectors | One shared fold vector per replicate |

Rejected review findings (11) and their refutations are in `notes/review/adversarial-review-round1.json`.
