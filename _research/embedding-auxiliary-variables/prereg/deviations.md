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
