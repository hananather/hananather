# Working log (decisions and why)

Running record of design decisions, so they can be reviewed and challenged. Newest last.

## 2026-10-23: environment and data access
- Sandbox egress is restricted. Reachable hosts: PyPI, GitHub release assets and raw files, some public S3 buckets, and some GCS buckets.
- Blocked: Hugging Face, api.cohere.com, irs.gov, census.gov, sec.gov, open.canada.ca, data.gov, Kaggle, Zenodo.
- No `COHERE_API_KEY` is set. **Cohere embed-v4 cannot run here until both of these happen:**
  1. `api.cohere.com` is added to the environment's allowed domains;
  2. the key is added as an environment secret.
- The **GivingTuesday 990 data lake** (`s3://gt990datalake-rawdata`) is reachable. It contains:
  - yearly e-file indexes;
  - per-return XML;
  - IRS Business Master File (BMF) snapshots back to 2023-08;
  - IRS SOI extracts.
- Full Form 990 filers per tax year: 336,521 (2021), 349,091 (2022), 355,712 (2023).
- The 2022 XML for full 990s totals 14.4 GB; median file is 33 KB.
- New England (MA, CT, ME, NH, VT, RI): 21.7k returns per year, about 1 GB.
- Local embedding models available:
  - all-MiniLM-L6-v2 ONNX (from the Chroma S3 bucket), about 600 short texts/s on CPU;
  - spaCy and GloVe static vectors (GitHub releases);
  - TF-IDF + SVD.
  - WordLlama needs Hugging Face, so it is unavailable.

## Population and frame timing (chosen to mimic a real survey operation)
- **Frame reference = tax year t-1 = 2022.** The frame holds:
  - BMF snapshot of 2023-08-16 (the earliest available, closest to a real design date): NTEE, subsection, ruling year, foundation code;
  - from the TY2022 return: lagged size (revenue, expenses, assets, employees) and the mission/program text.
- **Survey reference = tax year t = 2023.** Targets y come from the TY2023 return.
- Using the 2026 BMF for frame variables would leak future information: its amounts come from TY2024/25 filings. The 2023-08 snapshot avoids this.
- Population U = units filing a full 990 in both years (continuing units). Deaths and births are tracked separately for a frame-coverage analysis.

## Synthetic sanity check of the estimator code (src/survey.py)
- Setup: stratified Neyman design with a take-all stratum, n = 800, N = 20k, R = 1000.
- HT and GREG: unbiased, coverage about 95%.
- Ridge-difference without cross-fitting on 128-d noisy embeddings: +6.7% relative bias and 51% coverage. This is the overfitting pitfall, and the harness detects it.
- An additive linear model in the embedding on the raw scale did worse than plain GREG, for a y whose ratio y/x depends on text. Embeddings should enter as **text-dependent ratios** (features x·PC or a ratio-scale model), not as additive terms. This matters for skewed business-type populations.

## 2026-10-23 (afternoon): protocol runs, gate diagnosis, extra stages
- **Stage 2A (R = 2,000, 35 estimators, 30 targets)** finished after 2.8 h on 3 workers.
  - Four gates have failing items. Each was diagnosed in notebook 03 §1b; the findings are in FINDINGS §4.2.
  - Strict §9.6 reading: HARNESS NOT VALIDATED. Both the strict and the diagnosed readings are reported; neither is hidden.
- **RIDGECAL defect (D-32).** Replaced by LINCAL, rerun on the same samples (R = 2,000). That did not fix it either (D-33): linear calibration on the full code block explodes in about 0.4% of replicates. Attribution now uses the stable codes × text 2 × 2 within the ML learner.
- **Stage 2B weak register (D-28).**
  - Run as pre-registered (D-1W). Smoke testing exposed a BMF defect: 722 units, 6.7% of expenses, with zero amounts. So a profiling sensitivity (wprof) was pre-declared before any result was read.
- **Stage 2F design-stage text (D-31)** was added on a free core. It covers the "stratification" use from the claim, which was otherwise untested.
- **Stage 3, S6 reference corrected to the protocol's minimum-bias rule (D-29).** This happened after the TA-late results were read, and the correction goes against text. A pre-run review then added bootstrap intervals, the total-RRMSE condition, and the gate caveat (D-30).
- **Review rounds.**
  - A pre-run review of the 2B runner and the NR/2B analysis: 6 minor findings confirmed and fixed, 1 refuted. Errata were added to D-14 (the valid-URL flag was never rebuilt), D-28a and D-29.
  - A claim-by-claim verification of FINDINGS.md against the results.
