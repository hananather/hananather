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

## Stage 2B (weak register F0), logged 2026-10-23 before any Stage 2B result was read

| ID | Protocol | Done instead | Why | Expected effect |
|---|---|---|---|---|
| D-28 | 2B: D1-W, F0 twins and text, R = 1,000 (previously listed under D-12 as not run) | Run in `src/stage2b_ne.py`, R = 1,000, 3 workers. F0 = BMF-max codes + state + BMF income code + log BMF amounts (income, revenue, assets); `size_cat@F0` = strata + raw BMF amounts + NTEE10 + state. x̃ = BMF INCOME_AMT. Arms: HT, GREG_size_cat@F0, MCAL_frame@F0 (±E*), MCAL_bmfmax@F0 (±E*, E*64, L4, N2perm). Text = the same E* (TY2022 990 mission + programs). On the same D1-W samples: GREG_size_cat@F1p and MCAL_bmfmax@F1p (±E*) with x̃ = lag_expenses, for the gap closure S2b. | Pre-registered secondary; cores freed by Stage 2A | 990 text is richer than BN registration text, so S2 is favourable to text (§12.4) |
| D-28a | D1-W size measure = INCOME_AMT | Two variants. **w**: as specified. **wprof** (sensitivity): the 206 largest units by TY2022 expenses get that as their design size, emulating large-enterprise profiling; x̃ is unchanged. | Found while smoke-testing, before any 2B efficiency number was read: the 2023-08 BMF has zero or missing amounts for 722 units holding 6.7% of expenses (Harvard, Tufts, Northeastern; BCBS-MA is absent from the snapshot). D1-W puts them in the bottom size classes with weights of 57–109. No NSO would sample its largest enterprises that way, so the variance under **w** is dominated by a register defect rather than by estimation. | **w** reports the pre-registered S2. **wprof** isolates what text adds to estimation once the design is sane. Both are reported. |
| D-28b | S2b gap closure G (definition not fixed in §9.4) | G = log GM-RE(F0+E* ∣ F0 twin) / log GM-RE(F1p ref ∣ F0 twin) over R ∪ H_D. F1p ref = the better of GREG_size_cat@F1p and MCAL_bmfmax@F1p, i.e. what the lags are worth at best. Both are reported. | Needs a fixed definition | G ≥ 0.5 reads "text substitutes for admin lags" |

## Stage 3 (nonresponse), logged 2026-10-23

| ID | Protocol | Done instead | Why | Expected effect |
|---|---|---|---|---|
| D-29 (*post hoc*, against text) | S6 reference = the text-free method with the smallest Stage-1 limiting abs(bias), fixed in addendum v1.3 | The notebook first coded the status quo (weighting class + GREG) as the S6 reference. **Corrected after the TA-late results were read**, to the protocol's min-bias rule. The limiting bias is estimated by the take-some MC RB (R = 1,000) over the text-free arms run: NR_rhg_HT, NR_SQ_rhg_GREG, NR_rhg_MCAL_bmfmax, NR_ps_frame_MCAL_bmfmax. The status-quo comparison is kept as a secondary. Coverage gaps use the take-some part, since the take-all imputation error is outside every arm's variance. | The original code did not implement §7.2 | The correction makes S6 harder for text: weighting class + HT has the least bias for 4 of the 5 S6 variables |
| D-30 | Stage 3: NR0–NR4 simulated mechanisms (NR1/NR2 are S6's gates), the 180/365-day cutoffs, TA-nn donors, and the NR_psc / pmc / joint / DR / knn families | Only NR-LATE at 300 days, with TA-hist-own, weighting class, propensity and model-calibration arms (7 estimators) | Time | S6 cannot formally pass without the NR1/NR2 gates. Text-assisted methods not anchored on size calibration (DR, predictive-mean classes) remain untested. |

### Errata (2026-10-23, from the pre-run review of Stage 2B and the Stage 3 analysis)

- **D-14:** the valid-URL flag was **not** rebuilt. Every stage, including 2A, 2B, 3 and 4, uses `has_website` (field non-empty: 92.8%, against 72.5% valid). This affects one weakly informative dummy in matrix C and is rated negligible, as before.
- **D-28a:** "the 206 largest units by TY2022 expenses" means by frame `lag_expenses`. That is TY2022 expenses where that return was filed by D, otherwise BMF REVENUE_AMT (86 of the 206). Take-all is re-ranked on the combined size: 171 of the 206 profiled units are take-all, including every zero-BMF giant, and the other 35 slots go to units with large BMF income. The count of zero-amount units is 722, not 703.
- **D-29:** BRF, the reference choice, coverage and the take-some RRMSE ratio use the take-some part. Take-some estimates are identical under TA-follow-up and TA-late (same samples, folds and response), so the "both take-all scenarios" clause binds only through the **total** RRMSE ratio. That ratio is now also required. BRF and RRMSE ratios carry paired-bootstrap 90% intervals. S6 is printed as "point conditions met, not a formal pass" because the NR1/NR2 gates were not run (D-30).

## Stage 2F (design-stage text), logged 2026-10-23 before any Stage 2F result was read

| ID | Protocol | Done instead | Why | Expected effect |
|---|---|---|---|---|
| D-31 | 2F: D1-H44, D1-imp, D2a, D2b, D2b-twin, D2c × 3 estimators, R = 1,000 (previously under D-12 as not run) | `src/stage2f_ne.py`: D1, D1-imp, D1-H44 and D2a only. HT and GREG_size_cat@F1p at R = 2,000; MCAL_bmfmax@F1p at R = 500 (first 500 replicates of the same stream). D1-imp codes MISSING NTEE10 by multinomial logit on Z_16(E*), trained on frame units with a known code. D2a's score = the mean of the 20 frozen previous-wave text heads' predicted gov grants ÷ (lag_expenses + 1). D2b, D2b-twin and D2c are not run. | One core free before Stage 2B | S4: GM over all 15 real y of RE(D1-imp ∣ D1) for HT and for GREG, threshold ≥ 1.05. S5: GM over H of RE(D2a ∣ D1-H44), production bar. M8 is read from the MCAL rows. |
