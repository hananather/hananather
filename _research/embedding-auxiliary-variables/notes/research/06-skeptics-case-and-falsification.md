# 06 — The skeptic's case against "embeddings as auxiliary variables", and how to falsify it

> Claim under test: *"Text embeddings of business descriptions, websites or area profiles can serve as covariates for stratification, nonresponse adjustment, calibration and small area estimation. This is variance reduction, not productivity."*

This note argues the case against the claim as hard as it can be argued, and then turns every argument into a test with a pass/fail threshold. The benchmark can then show whether the claim survives. Wherever I could, I checked the arguments against data. That was a first pass on the New England 990 population that the project has already built (§4). Several skeptical priors survive that pass, and one of them does not.

**How the evidence was gathered (read first).**
- **Literature.** The workflow's shared WebSearch budget ran out before this note started, and WebFetch failed with DNS errors from this sandbox. So literature is cited in one of two ways:
  - from sibling notes 01–04, where it was verified earlier in this session (tagged **[n01]**, **[n02]**, **[n03]**, **[n04]**);
  - from memory, tagged **[UNVERIFIED]**.

  I did not run new literature searches.
- **Data.** Every number in §4 is new and was computed for this note. The population is the New England 990 build: N = 20,626 "continuing" organizations that filed a full 990 for both TY2022 and TY2023, built by `src/build_population.py`. Frame variables come from the 2023-08 BMF and the TY2022 return; targets come from TY2023.
- **Text arms (TY2022 return).**
  - LSA-256 = TF-IDF + SVD on mission plus program descriptions;
  - MiniLM-384 on the mission text only.
- **Design.** The production design from `src/frame.py`: NTEE10 × 3 size classes, a take-all stratum holding the top 1% (206 units), and Neyman allocation on √size.
- **Scripts.** All in `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/skeptic/`: `falsify_pop.py`, `falsify_bin.py`, `describe.py`, `numeracy.py`, `pn_mc.py`.
- **Limits.** These results are exploratory and were *not* pre-registered. They cover one region, use local embeddings only (no Cohere), and are mostly population-level ceilings. They are evidence about the direction of effects, not a final verdict.

---

## 1. Bottom line

1. **There is only one currency: incremental residual variance over the strongest conventional estimator.**
   - The incremental relative efficiency is RE = 1/(1 − partial R²).
   - Reaching RE 1.10, about 9% fewer sample units, requires the text to explain 9% of the variance the frame *leaves over*. Reaching RE 1.25 requires 20%.
   - Gains measured against HT, against SRS, or against a frame that lacks prior-year admin values are not evidence for the claim.
2. **On financial size variables the skeptic is right, and it holds up empirically.**
   - Against a register holding lagged revenue, expenses, assets and employees, plus stratum × lagged expenses, the text arms reach a population-ceiling RE of **0.93–1.01** for revenue, expenses and employees (LSA 0.93–0.97, MiniLM 0.97–1.01). That is no gain, and with LSA a loss. Random vectors do as well: 0.90–0.96.
   - The cause is structural. Once take-all and ratio-on-lag have done their work, the residual is year-over-year change, mostly in a few large units. The text is static: the mission is unchanged for 70–93% of units ([n03]). Static text cannot predict change.
3. **A gain over a "strong" frame is real but conditional, and it disappears the moment the register carries the lag of the same variable.**
   - Against the strong frame, the LSA text arm reaches these population ceilings:

     | y | RE |
     |---|---|
     | Salaries | 1.41 |
     | Fundraising expenses | 1.20 |
     | Contributions | 1.19 |
     | Grants to individuals (yes/no) | 1.17 |
     | Has government grants (yes/no) | 1.14 |
     | Government grants | 1.13 |

   - Add the prior-year value of the same y to the frame, as a CRA T3010-style charity register or a payroll-carrying business register would have it, and the text increment collapses:
     - salaries 1.003, fundraising 1.001, contributions 0.97, government grants **0.93**;
     - lagged indicator in place: grants to individuals 1.007, foreign activities 0.999, lobbying 1.000.
   - **The text is a substitute for a missing administrative lag, not a complement to it.** This is the most important result for deciding where the idea could ever matter.
4. **The "it just re-encodes the industry code" argument is partly *falsified*.**
   - Replace each unit's text vector by the centroid of its 3-character NTEE code, and the gain largely disappears (salaries 1.03, contributions 0.99). Keep only the *within-code* residual, and the gain survives (salaries 1.43, contributions 1.10, grants to individuals 1.12).
   - Fine-code dummies added under the same penalty do not reproduce it. The meta-features (text length, website, digits) give 1.00–1.04.
   - So the signal is genuinely semantic and finer than the code. It is simply redundant with the *lagged y*.
5. **The negative controls behave.** Random vectors, globally permuted embeddings and stratum-permuted embeddings give RE 0.82–1.00, and two placebo y give 0.98–1.00, all against the strong frame. No spurious gain appears at the population level with cross-fitting. **But noise blocks cost up to 18% efficiency** (government grants with random vectors: 0.82). A text arm that adds nothing is not neutral; it is harmful.
6. **Tuning on the population is a measurable optimism.** Choosing the ridge penalty per y on the population, rather than fixing it in advance, inflates the text RE by up to 0.09 (government grants 1.13 vs 1.05; contributions 1.19 vs 1.14). Every ceiling in sibling notes and diagnostics that uses a "best λ per y" is optimistic by about that much.
7. **High-dimensional raw embeddings in OLS-GREG fail as theory predicts** (MC, n = 500, R = 500; §4.5). The figures are the same with random vectors, so the cause is p/n, not the text.

   | Arm | RE vs base GREG | Coverage | Variance estimator | Negative weights |
   |---|---|---|---|---|
   | Raw 309–437 dimensions | 0.00–0.18 | 0–5% | ≈ −100% | 43–44% |
   | 32 additive PCs | 0.60–0.86 | 73–82% | — | 10% |
   | In-sample ridge | — | 2–4% | — | — |
   | Cross-fitted ridge on raw additive dimensions | 0.15–0.84 | Restored (84–96%) | — | — |

   Cross-fitting is necessary for valid inference, but it creates no efficiency gain on its own.
8. **Embedding a document to recover a number is the wrong tool** (§4.4).
   - Even the bare string "Total revenue: $X" is read back by MiniLM only to a median ±41%.
   - As a raw-scale ratio auxiliary it is **6.7× less efficient** than the extracted number.
   - Inside 50–180 words of mission text, most of the number is lost.
   - For PDFs, extract fields and embed only the narrative.
9. **Leakage on this dataset is mainly about timing.** Exact y values never appear in the TY2022 mission and program text (0.0% in every size class, §4.3), so that is not where it comes from. Numbers do appear: 15–50% of texts have ≥ 4-digit numbers. The serious traps are:
   - a current BMF as the frame;
   - same-year text;
   - third-party web pages that quote 990 financials;
   - tuning on the truth;
   - folds that split near-duplicate chapter texts.
10. **Coverage works against the claim.** Text and websites are most often present for large units.
   - Valid website URLs in this population cover 51% of units under $200k against 92% above $25M. [n02] found about 45% against about 88%.
   - Those large units are already take-all or well predicted by lagged size. The take-all 1% hold 68% of revenue.
   - Text is thinnest exactly where the frame is weakest: births, 990-N filers and micro units.
   - **Red-team catch:** the population's `has_website` flag counts "N/A" and "NONE" as websites, giving 92.8% coverage instead of the true 72.5%. It must be rebuilt before use (§4.3).
11. **The skeptic would concede usefulness in production only in narrow cases** (§8):
    - variables with **no** administrative lag (mission and activity variables in a new or redesigned survey);
    - **births** and units without admin history;
    - **text-defined domains** that no frame code captures;
    - NR adjustment, where the text predicts *both* response and y.

    Each needs a pre-registered RE ≥ 1.15 against the strongest conventional estimator, that replicates on a second population, passes clean negative controls and leakage audits, and runs on in-enclave models.

---

## 2. The arithmetic every claim must survive

**Identity.** Take a stratified design with take-all strata TA and take-some strata TS, and a model-assisted (difference, GREG or model-calibration) estimator with working model m(·). Then

  V(t̂) ≈ Σ_{h∈TS} N_h² (1 − f_h) / n_h · S²_{y−m, h}.

Text can only help through S²_{y−m,h} *inside take-some strata*. The take-all units, which hold most of the total in skewed business and nonprofit populations, contribute zero design variance whatever the auxiliary ([n01] §1).

**Incremental efficiency.** Let B be the strongest conventional working model and B+T the model with text added. Then

  RE_T|B = S²_{y−m_B} / S²_{y−m_{B+T}} = 1 / (1 − R²_{T|B}),

where R²_{T|B} is the partial R² of the text given B, computed on take-some residuals and weighted by N_h²(1−f_h)/n_h.

| Partial R² of text given B | RE | Sample-size saving at equal precision |
|---|---|---|
| 0.02 | 1.02 | 2% |
| 0.05 | 1.05 | 5% |
| 0.09 | 1.10 | 9% |
| 0.20 | 1.25 | 20% |
| 0.33 | 1.50 | 33% |
| 0.50 | 2.00 | 50% |

**Ceiling vs achievable.** A population-fitted m with an oracle-chosen penalty is a *ceiling*. A sample-fitted m with p_eff effective parameters loses roughly a factor (1 + p_eff/n) of residual variance. This is a heuristic, the familiar GREG finite-sample penalty ([n01] §2.3). Here it is applied to the text block's extra p_eff only. Consider a text block with p_eff ≈ 30 at n = 500:
- a ceiling RE of 1.10 becomes about **1.04** in practice;
- a ceiling of 1.05 becomes about **0.99**.

*So any ceiling below about 1.07 at n ≤ 1000 should be read as "no practical gain".*

**Where the residual variance lives.** For revenue, the TY2023 vs TY2022 raw-scale correlation is 0.991 ([n03]), so a ratio estimator on lagged revenue leaves about 1.8% of the variance. That residual is year-over-year change: mergers, capital campaigns, one-off grants and restatements, concentrated in a few large take-some units. A static description cannot predict a capital campaign. That is the structural reason for H1 below.

---

## 3. The eight skeptical arguments, each with its test

### 3.1 Redundancy: embeddings mostly re-encode what the frame already has

**Argument.**
- Industry codes are *made from the same text*. StatCan assigns NAICS to the BN "major business activity" text with an automated classifier (G-Code). The US Census auto-coder assigns NAICS from the SS-4 business description for more than 80% of new applications ([n02] §2.4).
- An embedding of that description is therefore largely a continuous version of the code.
- For size variables, which dominate business-survey variance, the frame's lagged admin size (GST/HST revenue, T2/GIFI, PD7A payroll) already explains most of the variance:
  - on the 990, the raw-scale correlation between TY2023 and TY2022 revenue is 0.991, so R² ≈ 0.98 ([n03]);
  - a comparably high R² for StatCan BR tax data is **[UNVERIFIED]**, though StatCan has used tax data in place of survey data since the mid-1960s ([n02]).

**Where incremental signal could plausibly exist** (the skeptic's honest list):
- (a) **Mission-driven or composition variables** that no register carries: volunteers, government-grant dependence, program type, grants to individuals, the share of program-service revenue.
- (b) **Cost-structure ratios.** Salaries/expenses differs sharply between a grant-maker and a service provider within one NTEE major group, and text says which one a unit is.
- (c) **Units without admin history** (births). Only registration text and the name exist.
- (d) **Text-defined domains** that no code captures, such as food banks, arts education, AI-using firms or social enterprises.
- (e) **Within-code heterogeneity** in catch-all codes (NTEE "Z", "Y", "S"; NAICS 541990 "all other professional services").
- (f) **Nonresponse propensity**, if response depends on organizational type in ways the codes miss.

**Evidence (§4.1–4.3).**
- Redundancy *with the code* is weaker than the skeptic feared: the signal is within-code.
- Redundancy *with the lagged y* is total.
- For size variables without their own lag (salaries, when the register has no payroll), text adds a lot (1.41). With payroll on the register it adds nothing (1.003).

**Tests.**
- H1, H3, H4 (§6).
- Mandatory controls: code-centroid and within-code-residual embeddings, fine-code dummies, and a baseline that adds the lag of the same y.

### 3.2 Embeddings encode semantics, not magnitudes

**Argument.**
- Sentence and document embedders are trained contrastively for *semantic similarity* (retrieval, clustering, classification).
- A number reaches the model as sub-word tokens. "$1,234,567" and "$9,234,567" share most tokens, and pooling one vector over a document dilutes the one token sequence that carries the magnitude.
- Probing studies of the 2019 generation of models found that embeddings encode some numeracy for small, in-range numbers and *fail to extrapolate*: Wallace, Wang, Li, Singh & Gardner (2019), "Do NLP models know numbers? Probing numeracy in embeddings," EMNLP **[UNVERIFIED]**; Naik, Ravichander, Rose & Hovy (2019), "Exploring numeracy in word embeddings," ACL **[UNVERIFIED]**.
- A linear head on a pooled vector cannot implement "read the number and take its log".

**Consequence.** If a PDF of financial statements carries a size variable, the right pipeline is **field extraction**:
- XBRL, GIFI or the 990 XML where they exist;
- otherwise OCR or table parsing, or an LLM extraction with validation rules;
- then use the extracted number as a ratio or GREG auxiliary.

Embed only what extraction cannot capture: narrative, funding-source descriptions, program content. For StatCan this is not hypothetical. T2/GIFI financials already arrive as *structured fields* from CRA, and the T3010 has structured financial lines. Rendering structured records as page images in order to embed them would be a step backwards ([n03] §4.8 makes the same point for the 990).

**Evidence (§4.4).** On the NE population, a MiniLM embedding of the literal string "Total revenue: $X" was compared with the extracted number and with a digit-count feature.
- The embedding reads the bare number back to a median ±41%.
- As a raw-scale ratio auxiliary, it is **6.7× less efficient** than the extracted number.
- Embedded inside the mission text, most of the number is lost (R² on the log scale drops from 0.95 to 0.31–0.42).

**Tests.** H16 (extraction ≥ page embedding for size y) and H17 (multimodal increment). Arms B8 and NC7 (digits-masked text) are mandatory.

### 3.3 High-dimensional overfitting

**Argument.** With p = 1024–1536 embedding dimensions and n = 500–2000:
- OLS-GREG and calibration on raw dimensions are singular (p ≥ n) or near-interpolating;
- the residuals shrink, so the linearization variance collapses;
- calibration weights go negative;
- efficiency falls below HT (Cardot, Goga & Shehzad 2017; Chauvet & Goga 2022; Eustache, Dagdoug & Haziza 2025; Bouhadra & Dagdoug 2026; all [n01]).

Under a Gaussian homoscedastic model, note 01 derives E[v̂]/V ≈ (1 − p/n)/(1 + p/(n − p − 2)):
- p/n = 0.17 (p = 85, n = 500): about 0.69, giving nominal 95% CIs that cover about 90%;
- p/n = 0.87 (p = 437, n = 500): about 0.014, so the CIs are useless.

The MC on the heavy-tailed 990 population is *worse* than this Gaussian prediction (§4.5):
- p = 85: variance estimator −49% to −58%, coverage 73–82%;
- p = 437: −100%, coverage 0–1%.

Flexible learners such as kNN, forests or GBMs fitted *in-sample* on near-unique embeddings interpolate. The estimator then collapses towards the synthetic Σ_U m̂ and its variance estimate collapses with it ([n01] §2.4). Cross-fitting (Zrnic & Candès 2024; Dagdoug & Haziza 2026; An, Dagdoug, Haziza & Tillé 2026; [n01]) or a model frozen from an *independent* prior cycle is required.

**Spurious in-sample gains.** Any R² or RE computed on the data used to fit the embedding model is optimistic. Tuning λ, k or the model choice on the population (or on the evaluation replicates) adds further optimism, measured at up to 0.09 in RE here (§4.1).

**Tests.**
- H7 and H8, plus negative controls NC1–NC4 and NC10.
- The harness must reproduce the known failure (non-cross-fitted, raw-dimension GREG) as a *positive control for the harness itself*.

### 3.4 Leakage taxonomy specific to text auxiliaries

| Code | Leak | Mechanism | 990 example | Detection test | Rule |
|---|---|---|---|---|---|
| **L1** | Contemporaneous same-record text | Text written *with* or *from* y in the same reporting act | TY2023 Schedule O or Part III narrative used for TY2023 y; "$" amounts appear in 35–53% of Schedule O for >$5M filers ([n03]) | Compare same-year text (t) with lagged text (t−1). A gain at t well above the gain at t−1 is leakage. | Only text dated before the frame reference date |
| **L2** | Self-reported size in the text | "serving 5,000 clients with 200 staff and a $10M budget" | Program descriptions contain ≥2-digit numbers in 20–68% of cases ([n03]) | Digits-masked arm (NC7). Test whether the text predicts lagged y better than y_t − y_{t−1}. | If the gain is in the digits, attribute it to a numeric auxiliary and extract it |
| **L3** | Third-party derived text | Web profiles (charity directories, nonprofit explorers, Wikipedia infoboxes, LinkedIn headcounts) that *quote the current return* | A website or directory page crawled in 2025 quoting TY2023 revenue | Crawl date ≤ reference date. Strip numbers. Domain blacklist of aggregator sites. | Only first-party text archived before the reference date |
| **L4** | Temporal leakage in the frame | Frame snapshot taken after the reference period | The May-2026 BMF carries the TY2023 revenue for **18.3%** of TY2023 filers, against 0.6% for the Dec-2023 BMF ([n03]) | Assert snapshot date ≤ design date. Share with REVENUE_AMT == y_t ≈ 0. | Archived snapshots only |
| **L5** | Model memorization | The embedding model's pretraining corpus contains pages that state the target | A 2025 API model trained on web pages that publish 990 financials **[UNVERIFIED mechanism]** | NC5: the text must not predict the *change* log(y_t/y_{t−1}). Compare a pre-cutoff model (MiniLM, released before TY2023) with a post-cutoff model (embed-v4). | A gain that appears only for the newest model is suspect |
| **L6** | Tuning leakage | λ, k, model, input_type, text field or dimension chosen on population truth or on evaluation replicates | Per-y best λ inflates RE by up to 0.09 (§4.1) | Fix the tuning on an independent population or year pair (TY2021→22), then freeze | Pre-register all choices |
| **L7** | Fold leakage through near-duplicates | Chapters and affiliates share template missions; random folds put copies in both train and test | About 5% of names match chapter patterns ([n03]); identical normalized missions, §4.3 | Group k-fold by text-duplicate cluster | Mandatory |
| **L8** | Supervised clustering on the population | Strata or post-strata defined by clustering embeddings *with y* (or a model fitted on population y) | — | Any stratifier built from y must be trained on an independent cycle | Unsupervised clustering or a frozen external model only |
| **L9** | Survivorship in the population definition | U = "filed in both years" conditions on a t-outcome. Deaths and late filers are excluded, and births have no text. | The continuing-unit population | Report a frame-coverage analysis that includes deaths, births and late filers | Production gains must be evaluated on the *frame* population |

### 3.5 Coverage and missing-not-at-random text

**Facts** ([n02], [n03]):
- **Websites.**
  - EU enterprises with 10+ persons: 78% overall; 76.7% for small against 95.7% for large. Micro firms are far lower.
  - US nonprofit e-filers: about 45% below $200k revenue against about 88% above $25M.
  - NE continuing full-990 filers (this note, §4.3): valid URLs for 51% below $200k against 92% above $25M.
- **Frame-time text.**
  - Among all BMF entities, only about 26.5% have a TY2022 990/990-EZ, and so any text.
  - The roughly 1.0M 990-N filers have none.
  - At a realistic design cutoff, 9–23% of current filers have no prior return, concentrated in small units.

**Consequences.**
1. **Calibration needs Σ_U x_k.** An auxiliary that is missing for part of U can still be used with a missingness indicator and a zero-filled vector (both totals are then known). Imputing the embedding instead makes the "known total" model-dependent.
2. **Web text crawled at estimation time for sampled units only is *not* complete auxiliary information.** It cannot be used for calibration or GREG at all, only for NR modelling or imputation. Using it for calibration requires crawling the whole frame, every cycle.
3. **The text sits where it is least needed.** Large units are either take-all or tightly predicted by lagged size, and they have the most text. Units with weak frame information (births, micro units, 990-N filers) have the least. The benchmark must report the gain *by coverage stratum* and *at realistic coverage*, never on a population pre-filtered to units that have text.
4. **MNAR.** Having a website correlates with size and with the outcome. A "has text" indicator is therefore an auxiliary in its own right, and the meta-features arm (B6) must be in every comparison. Here the meta-features alone gave RE 1.00–1.04 (§4.1).

### 3.6 Variance estimation, coverage, weights

- **Model-fitting variability.** Linearization treats m̂ as fixed. That is valid asymptotically for cross-fitted or externally frozen models. It is not valid for in-sample flexible fits, for small domains, or when p_eff/n is not small ([n01]: Eustache et al. 2025; Bouhadra & Dagdoug 2026, where a cross-validated variance estimator is proposed).
- **Fold-split noise.** The cross-fitted estimator depends on the random fold split. Report the between-split SD and use the median over at least 5 repeated splits (Chernozhukov et al. 2018, [n01]).
- **Small domains.** A GREG domain estimator uses the global model's synthetic part. Text models with many effective parameters transfer model error into domains of 5–30 sampled units. Report domain-level RB and coverage, not only the averages.
- **Weights.**
  - Calibration on many PCs produces negative and extreme g-weights (Cardot et al. 2017, [n01]: they choose the number of PCs so that all weights stay positive).
  - **Single-weight practice.** An NSO publishes *one* weight per microdata file. Model calibration on a y-specific m̂ gives y-specific weights.
  - If text helps only through y-specific m̂, the gain does not survive conversion to a single multipurpose weight. The **multipurpose damage** to the other published y must be measured (H9).

### 3.7 Operational realism

- **Reproducibility.**
  - API models are retired. Cohere shut down its v2 embed models on 2026-04-04, and its lifecycle runs Active → Legacy → Deprecated → Shutdown ([n04]).
  - A frame embedded with vintage A and births embedded with vintage B are in different spaces. Every model change needs a bridging study, like a questionnaire redesign.
  - The production path is open weights (or a private deployment) with stored vectors and hashes (`src/embed.py` already hashes its inputs).
- **Confidentiality.** Register text (BN activity descriptions, survey write-ins, tax schedules) cannot go to an external API. Cohere embed jobs keep uploaded datasets for 30 days unless they are deleted ([n04]). For StatCan, the confidential route is private deployment, single-tenant Model Vault with Canadian residency confirmed in writing, or open weights in the enclave ([n04]). A proof of concept on public 990 or T3010 text does not show that the production path is feasible.
- **Cost.** Compute is trivial: about $2 per text pass over 300k units; $60–$450 per page-image pass ([n04]). The real costs are the URL-finding, crawl and linkage pipeline (ISTAT found URLs for about 75–81% of enterprises, [n02]), governance, versioning and bridging studies. Each must be set against the sample units saved: n × (1 − 1/RE) × cost per unit.
- **Drift.**
  - The frame text dates from the frame reference time, and websites change. If the text is crawled at survey time it becomes est-time (L1/L4).
  - If it is frozen at design time it ages. Mission text is unchanged year to year for 70–93% of units ([n03]), so ageing is slow, which also means the text carries no change signal.
- **A strong conventional design must be the comparator.** That means stratification by industry × size with take-all large units, Neyman allocation, and ratio or GREG on lagged admin size. Under such a design here, GREG on the strong frame already reaches RE 6.0 over HT for expenses and 2.8 for revenue (project diagnostics `/tmp/claude-0/data/results/diag_ne_n1000.csv`; with the stronger ridge-fitted frame of §4.1, 16.0 and 7.4). Text is judged on what is left.

### 3.8 Benchmark-design traps (publication-bias style)

| Trap | How it inflates the result | Guard |
|---|---|---|
| Cherry-picked y | With about 15 y × 4 uses × 5 estimators × 3 text arms × 4 dimensions × 3 models, roughly 10⁴ comparisons. At MCSE(RE) ≈ 0.03, dozens will "win" by chance. | Pre-register a primary family of ≤ 12 (y, use) pairs with Holm correction; report every y in an appendix |
| Cherry-picked population | One region, one year pair, nonprofits only | Replicate on a second region, a second year pair and a second register (CRA T3010) |
| Weak baseline | RE vs HT, or vs SRS, or vs a frame without lagged y. "Improvements" over weak baselines are a known ML failure mode: Lin (2019), "The neural hype and comparisons against weak baselines," *SIGIR Forum* **[UNVERIFIED]**; Armstrong, Moffat, Webber & Zobel (2009), "Improvements that don't add up," CIKM **[UNVERIFIED]**. | The primary comparator is the best of the B2–B4 ladder (§7), frozen in advance |
| Tuning on evaluation samples | Optimistic λ, k, model or field choices | A tuning population (TY2021→22, or another region) disjoint from the evaluation population |
| Reporting ceilings as achievable | Population-fit ceilings lack the (1 + p/n) penalty | Primary results come from design-based MC with sample-fitted, cross-fitted models |
| Point RE without MCSE | MC noise looks like signal | Paired (common random numbers) MCSE for log RE; R ≥ 2000 |
| Means only, no domains | Hides small-domain bias | Domain RB, RRMSE and coverage distributions |
| Different text fields across models | "Cohere beats LSA" while comparing different inputs | Hold the field fixed when comparing models (here LSA used mission+programs and MiniLM mission only, so their RE are *not* comparable) |
| Garden of forking paths | Undisclosed analytic choices. Gelman & Loken (2014), "The statistical crisis in science," *American Scientist* **[UNVERIFIED]**. | Pre-registration (§6) plus a deviations log in `working-log.md` |
| Leakage disguised as gain | §3.4 | Leakage audit before reporting any RE > 1.05 on a negative-control y |

---

## 4. First evidence from the New England 990 population (exploratory)

### 4.0 What was computed

All numbers in §4.1–4.2 are **population-level anticipated-variance ceilings** for the production design at n = 1000. They come from the design-variance identity in §2, with these model settings:
- weights d_k − 1;
- 5-fold cross-fitting over the whole population;
- weighted ridge;
- the frame block left unpenalized and the text block penalized.

Text enters as 32 PCs in a "ratio" block [Z, Z·x_lag-expenses, Z·x_lag-employees], so the text can shift a unit's ratio of y to size. Each cell is RE = AV(baseline)/AV(arm). The λ is the per-y best of {1e-4, …, 1e-1}, which is optimistic (L6). A fixed-λ column is shown for comparison. Treat these as **upper bounds** on what a sample-fitted estimator can reach (§2).

The frame ladder:

| Frame | Contents |
|---|---|
| **strong** | Stratum dummies, state, NTEE10, BMF income code, lagged revenue/expenses/assets/employees, and stratum × lagged expenses (separate-ratio-type) |
| **strong+lagy** | strong + lagged salaries, contributions and program revenue, each also × stratum. This is what a register with payroll and receipted-gift data would hold. |
| **strong+lagy+oracle** | strong+lagy + lagged volunteers, government grants, officer compensation and fundraising expenses. This is the **T3010-style register-plus** case: the whole prior-year return is administrative data. |

### 4.1 Population ceilings: text vs negative controls vs redundancy controls (RE relative to the *strong* frame)

| y (TY2023 total) | RE strong vs HT | **LSA text** | LSA, fixed λ | MiniLM (mission only) | Random vectors | Global permutation | Permuted within strata | **NTEE-code centroid of LSA** | **Within-code LSA residual** | Meta-features only | LSA over fine code + meta | **LSA over strong+lagy** | **LSA over register-plus** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Revenue | 7.43 | **0.96** | 0.93 | 0.97 | 0.96 | 0.92 | 0.97 | 0.94 | 0.95 | 0.98 | 1.01 | 1.03† | 1.03† |
| Expenses | 15.95 | **0.97** | 0.93 | 0.99 | 0.90 | 0.92 | 0.96 | 0.96 | 0.97 | 0.96 | 0.99 | 0.98 | 0.97 |
| Employees | 38.22 | **0.93** | 0.87 | 1.01 | 0.93 | 0.97 | 0.95 | 1.02 | 0.97 | 1.00 | 0.93 | 0.92 | 0.92 |
| Salaries | 3.92 | **1.41** | 1.41 | 1.04 | 0.82 | 0.98 | 0.87 | 1.03 | 1.43 | 1.04 | 1.40 | **1.003** | 1.006 |
| Program-service revenue | 3.15 | 1.04 | 1.00 | 1.02 | 0.96 | 0.86 | 0.94 | 0.95 | 1.00 | 0.96 | 1.02 | **0.96** | 0.95 |
| Contributions | 1.15 | **1.19** | 1.14 | 1.03 | 0.93 | 0.89 | 0.95 | 0.99 | 1.10 | 1.00 | 1.15 | **0.96** | 0.97 |
| Government grants | 1.14 | **1.13** | 1.05 | 0.95 | 0.82 | 0.89 | 0.97 | 0.93 | 1.07 | 1.01 | 1.11 | 1.19 | **0.93** |
| Fundraising expenses | 1.23 | **1.20** | 1.20 | 1.05 | 0.96 | 0.92 | 0.98 | 1.09 | 1.10 | 1.04 | 1.10 | 1.16 | **1.001** |
| Officer compensation | 1.11 | 1.05 | 1.02 | 0.98 | 0.93 | 0.95 | 0.96 | 0.97 | 1.04 | 1.01 | 1.05 | 1.03 | **0.99** |
| Volunteers (capped 10k) | 1.00 | 1.04 | 1.03 | 1.02 | 0.99 | 0.99 | 0.99 | 1.01 | 1.02 | 1.00 | 1.03 | 1.03 | **1.005** |
| Has government grants (0/1) | 1.03 | **1.14** | 1.14 | 1.11 | 1.00 | 1.00 | 1.00 | 1.06 | 1.08 | 1.01 | 1.08 | 1.12 | 1.06‡ |
| Grants to individuals (0/1) | 1.00 | **1.17** | 1.17 | 1.06 | 1.00 | 1.00 | 1.00 | 1.05 | 1.12 | 1.02 | 1.14 | 1.16 | 1.16‡ |
| Has volunteers (0/1) | 1.01 | 1.06 | 1.06 | 1.03 | 1.00 | 1.00 | 1.00 | 1.01 | 1.05 | 1.04 | 1.03 | 1.05 | 1.05‡ |
| Foreign activities (0/1) | 1.01 | 1.03 | 1.03 | 1.04 | 0.99 | 0.99 | 1.00 | 1.01 | 1.02 | 1.00 | 1.02 | 1.04 | 1.04‡ |
| PLACEBO: revenue permuted within strata | 0.99 | 0.99 | 0.98 | 0.98 | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 1.00 | 0.98 | 0.99 | 0.99 |
| PLACEBO: log-normal noise | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Column definitions:
- Columns 3–11 (LSA text through meta-features only) are RE relative to the *strong* frame.
- Column 12 ("LSA over fine code + meta") is RE relative to strong + fine code + meta.
- Columns 13–14 are RE of (frame + LSA) relative to the same frame without LSA.
- **Fine code** = NTEE 3-character codes with ≥20 units, the rest pooled, entered with the same ratio block and penalty as the text. **Meta-features** = log word counts, has-website, number of programs, "$" present, digit count. **Within-strata permutation** keeps each stratum's (NTEE10 × size class) mix of texts but breaks the unit-to-text link.
- † For revenue, the richer frames are *worse* than strong (strong+lagy: 0.86; register-plus: 0.85; the unpenalized interaction columns overfit the heavy tail). So "text over that frame" is measured against a degraded frame. Against the best frame (strong), the best text arm reaches only 0.97.
- ‡ Register-plus does not include the lagged *indicator* itself; see §4.2.

What this already says:
1. **Size variables (revenue, expenses, employees): text RE < 1 against the best frame.** The best text arm reaches 0.97, 0.99 and 1.01; the LSA text arm 0.93–0.97. That is no better than random vectors (0.90–0.96). The skeptic's prior holds, and the text is *harmful* when added naively.
2. **Redundancy with the industry code is not what limits the text.** The code-centroid version of the text loses almost all the gain; the within-code residual keeps most of it. Text over fine code + meta-features is still 1.10–1.40 for salaries, contributions, government grants and fundraising, and 1.02–1.14 for the binary indicators. **The skeptic's "it just re-encodes NTEE" prior is falsified for these y.**
3. **Redundancy with the lagged y limits it completely.** With the lag of the same y on the frame:
   - salaries 1.003, contributions 0.96, program revenue 0.96, government grants 0.93, fundraising 1.001, officer compensation 0.99, volunteers 1.005.
   - Against the *best* frame per y, the only remaining ceilings above 1.05 are the activity indicators. Those also vanish once the lagged indicator is added (§4.2).
4. **Negative controls behave correctly.** Placebo y gives 0.98–1.00. Random and permuted vectors give 0.82–1.00, never > 1.00. The harness does not invent gains at this scale, but **a useless 96-column block costs up to 18%**.
5. **Fixed vs tuned λ.** Tuning λ on the population adds up to +0.09 RE (government grants 1.05 → 1.13). That optimism must be removed by pre-specifying the tuning or tuning on an independent population.
6. **MiniLM on mission only is far weaker than LSA on mission+programs** (salaries 1.04 vs 1.41). Most of the signal is in the *program descriptions*, not in the mission sentence. Before crediting the model, the field must be held fixed (trap in §3.8).

### 4.2 Activity indicators: is the text gain just a stand-in for last year's checkbox?

The lagged indicator comes from the TY2022 return, parsed into `/tmp/claude-0/data/parsed/ne_2022.parquet`. Same design and model settings as §4.1.

| y (share at t) | Agreement of y_t with y_{t−1} | RE text vs strong | RE lagged indicator vs strong | **RE text vs strong + lagged indicator** |
|---|---|---|---|---|
| Grants to individuals (12.1%) | 97.5% | 1.167 | 3.84 | **1.007** |
| Foreign activities (4.3%) | 99.4% | 1.033 | 4.45 | **0.999** |
| Lobbying (5.8%) | 99.3% | 1.064 | 5.77 | **1.000** |
| Has government grants (31.1%) | 90.7% | 1.144 | 2.38 | **1.017** |
| Has volunteers (58.7%) | 91.8% | 1.062 | 3.04 | **1.003** |
| Fundraising activities (20.8%) | 95.2% | 1.077 | 3.30 | **1.005** |

**Verdict.** In every case the text gain is what remains when a register lacks last year's value. When the register has it, the text adds ≤ 1.7%. For an NSO this is a design question, not a modelling question: *is there a prior-cycle value of this variable anywhere?* If there is, text embeddings are the wrong investment.

### 4.3 Redundancy, magnitude, coverage and leakage diagnostics

Script: `describe.py`; population N = 20,626.

**(a) How much of the embedding *is* the industry code?** The between-group share of embedding variance (the share explained by group means) is:

| Grouping | LSA-256, all dimensions | MiniLM-384, all dimensions | LSA, 32 PCs | MiniLM, 32 PCs |
|---|---|---|---|---|
| NTEE10 (frame stratifier) | 3.0% | 4.3% | 6.3% | 5.5% |
| NTEE 3-character code | 14.3% | 15.7% | 25.2% | 19.9% |
| NTEE10 × BMF income code (≈ strata) | 4.6% | 5.5% | 8.7% | 7.0% |
| State | 0.6% | 0.6% | 0.4% | 1.0% |

- NTEE10 is predictable from 32 text PCs with 67% 5-fold accuracy (LSA and MiniLM alike), against a majority-class rate of 35%. NTEE is missing for 31% of this population.
- **Verdict.** At most a quarter of the leading embedding variance is fine-code variance. The embedding is *not* mainly a re-encoding of the code. This agrees with §4.1, where the within-code residual carries the gain.

**(b) Does text recover size?** Cross-validated R² for log(1 + lagged revenue). The number sits on *the same return* as the text, but is not *in* the text:

| Features | R² |
|---|---|
| NTEE10 + state | 0.04 |
| Fine code | 0.11 |
| Meta-features (length, website flag, digits) | 0.11 |
| MiniLM 32 PCs | 0.15 |
| LSA 32 PCs | 0.23 |
| Fine code + meta + LSA | 0.28 |
| The lagged number itself | 1.00 |

Text carries a weak, coarse size signal (R² ≈ 0.2–0.3, against 1.0 for the number). Part of that signal is simply length: the Spearman correlation of log word count with log revenue is 0.36. This is why size variables gain nothing once the lagged number is on the frame.

**(c) Incremental R² on the log scale for TY2023 y** (5-fold CV ridge, unweighted; complements the design-weighted §4.1). Frame = NTEE10 + state + income code + log lagged size.

| y | Frame | + fine code + meta | + fine code + meta + LSA | + lag of y | + lag + code + meta + LSA |
|---|---|---|---|---|---|
| Revenue | 0.779 | 0.779 | 0.779 | (in frame) | 0.779 |
| Expenses | 0.871 | 0.871 | 0.871 | (in frame) | 0.871 |
| Employees | 0.951 | 0.951 | 0.952 | (in frame) | 0.952 |
| Salaries | 0.636 | 0.647 | 0.663 | 0.941 | 0.941 |
| Program-service revenue | 0.347 | 0.416 | 0.491 | 0.913 | 0.913 |
| Contributions | 0.227 | 0.303 | 0.387 | 0.841 | 0.843 |
| Government grants | 0.197 | 0.265 | 0.335 | 0.687 | 0.697 |
| Volunteers (capped) | 0.128 | 0.206 | 0.249 | 0.784 | 0.786 |
| Fundraising expenses | 0.253 | 0.345 | 0.429 | 0.876 | 0.878 |

- Partial R² of LSA given code + meta: 0.10–0.13 for the composition y (contributions, government grants, program revenue, fundraising); 0.05 for volunteers and salaries.
- Partial R² given the lag: **0.00–0.03**.
- The lag of y is worth 3–6 times what the text is worth, and the text adds almost nothing on top of it.

**(d) Coverage, numbers in text, staleness, duplicates, concentration** (by lagged revenue class):

| Class | n | Share of revenue | `has_website` flag | **Valid URL** | Median words | Has "$" | Has number with ≥ 4 digits | Exact lagged or current revenue/expenses in text | Mission unchanged t−1 → t |
|---|---|---|---|---|---|---|---|---|---|
| <$200k | 4,896 | 0.2% | 85.8% | **50.7%** | 50 | 2.8% | 14.9% | 0.0% | 88.1% |
| $200–500k | 5,196 | 0.7% | 91.0% | **69.0%** | 60 | 2.8% | 17.4% | 0.0% | 84.3% |
| $0.5–1M | 2,954 | 0.8% | 93.9% | **75.5%** | 70 | 2.8% | 20.6% | 0.0% | 83.2% |
| $1–5M | 4,468 | 3.9% | 97.2% | **85.2%** | 95 | 3.8% | 27.0% | 0.0% | 81.2% |
| $5–25M | 2,072 | 8.4% | 99.0% | **91.2%** | 122 | 6.1% | 34.6% | 0.0% | 80.5% |
| >$25M | 1,040 | **86.0%** | 99.8% | **91.8%** | 184 | 11.1% | 50.2% | 0.0% | 78.8% |

**Red-team findings from this table:**
1. **`has_website` in `ne_pop.parquet` is wrong.**
   - 21.7% of the non-empty website values are placeholders: "N/A" (4,011), "NONE", "NA", "SEE SCHEDULE O". Valid-URL coverage is therefore **72.5%**, not 92.8%.
   - The gradient is steep: 51% of units under $200k against 92% above $25M.
   - The flag must be rebuilt (valid URL = contains a dot and is not a placeholder) before it is used as a frame variable, a meta-feature, or a coverage stratum.
   - The meta-features arm in §4.1 used the defective flag. Its 1.00–1.04 should be re-run with the corrected one.
2. **8.4% of the valid URLs are shared by ≥ 2 EINs** (hospital systems, Scottish Rite, Tufts Medicine and the like). Website text would be identical across affiliates, which is the L7 fold-leakage risk.
3. **The take-all 1% (206 units) holds 68% of revenue, and the top class holds 86%.** All text gains are fought over the remaining third of the total.
4. **Exact y values never appear in the mission and program text** (0.0% in every class). Numbers do appear: 15–50% have ≥ 4-digit numbers, 3–11% have "$". The digit-masking ablation (NC7) is still required, because heavy units often state budgets or client counts.
5. **4.8% of units share an identical normalized mission with at least one other unit.** The largest cluster (112 units) is the *empty* mission, where text exists only in the programs. Group k-fold on duplicate clusters is required.

### 4.4 Semantics, not magnitudes: embedding a sentence that literally states the number

This is the cleanest possible version of "embed the financial statement". The prior-year revenue X is written out as text, embedded with MiniLM, and a 5-fold CV ridge on the 384 dimensions is asked to read it back. Random subsample of 6,000 units; scripts `numeracy.py` and `numeracy_raw.py`.

| Input to the model | R² for log(lagged revenue), which *is* the number in the text | R² for log(TY2023 revenue) | Median multiplicative error |
|---|---|---|---|
| Extracted number (field extraction) | **1.000** | 0.704 | ×1.00 |
| Digit count of X only | 0.932 | 0.670 | ×1.68 |
| emb("Total revenue: $X") | 0.945 | 0.676 | ×1.41 |
| emb("Total revenue: $X. " + mission) | 0.420 | 0.302 | ×2.81 |
| emb(mission + " Total revenue: $X.") | 0.308 | 0.235 | ×2.95 |
| emb(mission), no number | 0.165 | 0.165 | ×2.79 |

Within one digit-count class (5, 6 or 7 digits), emb("Total revenue: $X") reaches R² 0.90, 0.80 and 0.94. So MiniLM reads more than the order of magnitude when the number stands *alone*. That is more numeracy than the skeptic expected.

The surrounding text matters far more than that:
- Wrapped in a 50–180-word mission, the number is mostly lost: R² falls from 0.95 to 0.31–0.42, and the error is close to the no-number arm. A real financial-statement page is the worst case, with hundreds of numbers pooled into one vector.

What matters for estimation is the **raw scale**, where take-some ratio estimation lives:

| Ratio auxiliary (take-some units, n = 6,000) | Residual-variance reduction vs no auxiliary |
|---|---|
| Exact extracted number | **18.7×** |
| Number read back from the embedding of the bare sentence | 2.8× |

So **RE(extraction)/RE(embedding) = 6.7**, even in the most favourable case for the embedding. Only 15% of the read-back values fall within 10% of the truth; the median error is ±41%.

**Verdict on §3.2: confirmed.**
- Numbers must be extracted, not embedded.
- H16's threshold (ratio ≥ 2) is already exceeded on the easiest possible input.
- Any multimodal "embed the PDF" arm must be benchmarked against *extracted fields from the same PDF*, not against "no auxiliary".

### 4.5 Raw embedding dimensions in OLS-GREG at n = 500 (design-based MC)

**Setup.**
- R = 500 replicates, common random numbers across arms.
- Production design at n = 500: 34 strata, 206 take-all units.
- **Base** = stratum dummies + 4 lagged size variables + NTEE10 + state (p = 53).
- **Text** = MiniLM mission embeddings (raw 384 dimensions, or the first 10/32 PCs) entered *additively*, or LSA-256 raw.
- **Negative control** = 384 Gaussian columns.

The ridge-difference arms put the raw 384 dimensions into a weighted ridge with the base unpenalized. They are either fitted in-sample (λ = 1e-3) or 5-fold cross-fitted, with λ picked by inner CV on the sample. Variance is the stratified residual linearization. Script: `pn_mc.py`. Results: `pn_mc_R500_n500.csv`.

| Arm (p) | Metric | Revenue | Salaries | Contributions | Government grants | Has government grants | Grants to individuals |
|---|---|---|---|---|---|---|---|
| GREG base (53) | Coverage % / variance-estimator relative bias % | 87 / −19 | 93 / −5 | 80 / −25 | 84 / −5 | 93 / −11 | 92 / −12 |
| GREG base + PC10 (63) | RE vs base | 0.88 | 0.90 | 0.93 | 0.84 | 0.99 | 0.92 |
| | Coverage % | 85 | 91 | 81 | 82 | 89 | 88 |
| GREG base + PC32 (85) | RE vs base | 0.64 | 0.62 | 0.70 | 0.60 | 0.86 | 0.81 |
| | Coverage % / variance-estimator relative bias % | 78 / −50 | 82 / −49 | 74 / −50 | 73 / −52 | 78 / −58 | 79 / −56 |
| GREG base + LSA-256 raw (309) | RE / coverage % | 0.00 / 4 | 0.01 / 5 | 0.00 / 4 | 0.01 / 4 | 0.18 / 5 | 0.15 / 4 |
| GREG base + MiniLM-384 raw (437) | RE / coverage % | 0.00 / 1 | 0.00 / 0 | 0.00 / 1 | 0.00 / 0 | 0.10 / 0 | 0.08 / 1 |
| GREG base + **RANDOM-384** (437) | RE / coverage % | 0.00 / 1 | 0.00 / 1 | 0.00 / 1 | 0.00 / 1 | 0.14 / 1 | 0.10 / 1 |
| Ridge-difference raw 384, **in-sample** | RE / coverage % / RB % | 0.00 / 2 / −33 | 0.00 / 2 / −32 | 0.00 / 2 / −189 | 0.01 / 3 / +113 | 0.35 / 4 / +7 | 0.26 / 2 / +21 |
| Ridge-difference raw 384, **cross-fitted** | RE / coverage % / RB % | 0.16 / 87 / +1.1 | 0.41 / 93 / −0.4 | 0.15 / 84 / +15.6 | 0.54 / 88 / +1.8 | 0.84 / 96 / +0.4 | 0.80 / 94 / −0.4 |
| HT | RE vs base | 0.09 | 0.52 | 1.73 | 0.92 | 1.01 | 1.02 |

**Share of negative calibration weights:**

| Arm | Negative weights |
|---|---|
| Base | 0.3% |
| + PC10 | 1.3% |
| + PC32 | 10.2% |
| + raw 309 / 384 dimensions | **43–44%** |
| + random 384 dimensions | **43%** |

How to read this:
1. **Raw embedding dimensions in OLS-GREG are a disaster, and the disaster is caused by p, not by the text.**
   - The text arms and the random-vector arm are indistinguishable: RE ≈ 0, coverage 0–5%, the variance estimator underestimates by about 100%, and 43–44% of the weights are negative.
   - This is exactly the (1 − p/n)/(1 + p/(n − p − 2)) prediction of §3.3.
2. **Even 32 additive PCs cost 14–40% efficiency and 10–20 points of coverage at n = 500.** Ten additive PCs cost 1–16%. Additive embedding terms on the raw dollar scale are the wrong functional form for skewed populations. The ratio form of §4.1 is needed (the project's working log reached the same conclusion).
3. **In-sample fitting reproduces the overfitting trap:**
   - coverage 2–4%;
   - variance estimator −100%;
   - relative bias up to ±190% for heavy-tailed y.
4. **Cross-fitting restores near-nominal coverage for the binary y (94–96%), but not efficiency.** On raw additive dimensions the cross-fitted ridge is still 16–84% as efficient as the base GREG. For contributions, a heavy-tailed total, the relative bias is +15.6% because of extrapolation on outliers.

   *Cross-fitting is necessary for validity, but it does not create a gain.*
5. **For contributions, plain HT beats the 53-column base GREG by 1.73×.** Lagged size is a poor working model for contributions at n = 500, and the extra columns cost more than they return.

   This is the weak-baseline trap in miniature. "Text-GREG beats GREG" can simply mean "this GREG was a bad comparator". B* must be chosen per y on the tuning population, from a ladder that includes HT/Hájek and post-stratification.
6. **The base itself under-covers heavy-tailed totals** (contributions 80%, government grants 84%) at n = 500. Text does not fix this. The benchmark must report it, so that an embedding arm is not credited with "improving coverage" merely by changing the residual tails.

These arms are the **positive controls for the harness (PC3, NC10)**. Every confirmatory run must reproduce them.

### 4.6 Status of the skeptic's claims after the first pass

| Claim | Status on NE (exploratory) |
|---|---|
| Size variables gain ≈ 0 | **Confirmed** (0.93–1.01 against the best frame) |
| Text = industry code | **Falsified** for composition and activity y (the within-code residual carries the gain) |
| Text = substitute for the missing lag of y | **Confirmed** for every y tested (≤ 1.7% increment once the lag is present) |
| Negative controls show spurious gains | **Not observed** with population cross-fitting (≤ 1.00). Noise blocks cost up to 18%. |
| Tuning optimism | **Confirmed**, up to +0.09 RE |
| p/n failure of raw dimensions | **Confirmed**: RE ≈ 0, coverage 0–5%, 43–44% negative weights; identical with random vectors |
| Additive PCs harmless | **Falsified**: 32 additive PCs cost 14–40% at n = 500 |
| Cross-fitting fixes everything | **Falsified**: it restores coverage, not efficiency |
| Magnitudes not recoverable from embeddings | **Confirmed, with nuance**: a bare number is read back to ±41% (R² 0.95 on the log scale), but that is 6.7× less efficient than extraction on the raw scale, and it degrades to R² 0.3–0.4 inside real text |
| Website coverage as recorded on the frame | **Data defect found**: the flag gives 92.8%, true coverage is 72.5% |

---

## 5. (a) The skeptic's prior, with predicted magnitudes

Each figure is the RE of the **best cross-fitted, design-based text-augmented estimator** against the **best conventional estimator B*** under the same design (§7). Totals; n = 1000 under the production design. The prior was derived from theory and sibling-note evidence. **Honesty note:** the numbers were written down *alongside* the first §4 run, not blind to it. P1–P7 are therefore partly informed by NE and are not a clean test. P8–P15 are genuine out-of-sample predictions. The last column records what §4 already shows. *Sample-level* RE is expected to fall roughly a factor (1 + p_eff/n) below the population ceilings.

| # | y type / use | Comparator B* | Prior median RE | Prior 80% interval | Reasoning | NE ceiling (§4) |
|---|---|---|---|---|---|---|
| P1 | Size y with an admin lag (revenue, expenses, assets; salaries when payroll is on the register; employees when PD7A/headcount is) | B2/B3 | **1.00** | 0.97–1.02 | Residual = change; text is static | 0.93–1.01; salaries 1.003 with lag. **Confirmed.** |
| P2 | Size or cost-structure y *without* its own lag (salaries without payroll; employees without a lag) | B2 | 1.03 | 0.98–1.10 | Text weakly signals labour intensity | **Salaries 1.41: the prior is too low.** Employees 0.93. |
| P3 | Composition or mission y without its own lag (government grants, contributions, program revenue, fundraising) | B2 | 1.08 | 1.00–1.25 | Activity type → funding mix | 1.04–1.20 (fixed λ 1.00–1.20). Consistent. |
| P4 | Same y with its own lag (T3010-style register-plus) | B3 | **1.01** | 0.97–1.04 | Text substitutes for the lag | 0.93–1.001. **Confirmed.** |
| P5 | Heavy-tailed, heaped counts (volunteers) | B3 | 1.00 | 0.90–1.05 | Reporting noise dominates; 25–53% blanks for small units | 1.04 (S1), 1.005 (register-plus) |
| P6 | Activity indicators without a lag (grants to individuals, foreign, lobbying, has government grants) | B2 | 1.08 | 1.00–1.20 | Text-defined, poorly coded by NTEE | 1.03–1.17. Consistent (upper half). |
| P7 | Activity indicators with the lagged indicator | B3 | **1.01** | 0.99–1.04 | Checkboxes are sticky (91–99% agreement) | 0.999–1.017. **Confirmed.** |
| P8 | Births / no admin history (name or registration text only), within the births group | industry × region × size class | 1.10 | 1.00–1.30 | The only stratum where frame information is weak | Not tested (continuing population) |
| P8b | Same, effect on the *overall* total | B2 | 1.01 | 1.00–1.02 | Births are 3.7–9% of units and a smaller share of totals | Not tested |
| P9 | Text-defined domain totals (membership from text vs best code proxy) | Code-proxy domain | RRMSE −20% to −50% | — | This is mostly *bias* correction of the domain definition, not variance reduction | Not tested |
| P10 | Stratification on a text-informed m̂ (multipurpose design) | Industry × size, cum-√f / LH | 1.02 primary y | 0.98–1.06; ≥ 2% loss on ≥ 1 other y | Stratification gain is capped by ρ²; multipurpose trade-offs | Not tested |
| P11 | Nonresponse adjustment with a *real* response mechanism (late filing) | RHG on size × NTEE × prior lateness | NR-bias reduction ≤ 10% | 0–25% | Response is driven by size, burden and history, not mission (Little & Vartivarian 2005, [n01]) | Not tested |
| P12 | Unit-level SAE, state × NTEE10 domains | EBLUP/GREG on frame | Domain MSE −0% to −15% (sampled); coverage worse | — | Domain random effects absorb much; gains mostly in non-sampled domains, where bias risk dominates | Not tested |
| P13 | Neural embedding over TF-IDF/LSA on the *same* field | Same estimator with LSA | 1.00 | 0.97–1.05 | Short, formulaic texts; lexical features suffice | Not comparable yet (different fields) |
| P14 | Image/PDF-page embedding over the text embedding of the same document | Same model, text mode | 1.00 | 0.98–1.02 | Text-native documents carry no visual signal | Not tested (990 has no PDFs) |
| P15 | Page embedding vs extracted numeric field, for size y | Extracted field as ratio auxiliary | RE(embedding)/RE(field) ≤ 0.5 | — | §3.2 | 0.15 (= 1/6.7) even for the bare sentence. **Confirmed.** |

---

## 6. (b) Pre-registration: hypotheses with pass/fail thresholds

### 6.0 Protocol fixed before any confirmatory run

**Populations.** New England TY2022→23 is now *used*, by §4 and by the project's diagnostics. It therefore becomes the **tuning population**. Every model, λ, k, text field, dimension and input_type is chosen there and then frozen.

| Role | Population |
|---|---|
| **Tuning** | New England TY2022→23 |
| **Evaluation E1** | All other states, TY2022→TY2023 continuing units, *plus* a frame-coverage version that includes births and late filers |
| **Replication E2** | Same states, TY2021→TY2022 |
| **Replication E3** | CRA T3010 (user download), 2022→2023 |
| **H17 only** | A PDF-native population (UK CCEW accounts or Companies House) |

**Designs.**
- The production design (§0) at n ∈ {500, 1000, 2000}, scaled so that the take-some sampling fractions stay comparable across populations.
- SRSWOR is reported only as a secondary design.

**Replicates.**
- R = 2000, with common random numbers across arms.
- R = 5000 for revenue, government grants and the other heavy-tailed totals.
- With paired arms, SE(log RE) ≈ √(4(1 − ρ)/R), where ρ is the correlation between the arms' squared errors. At ρ ≈ 0.9 and R = 2000 that is ≈ 0.014, enough to separate RE 1.05 from 1.10.

**Primary metric.** RE = MSE(B*)/MSE(arm), with a paired-bootstrap 95% CI over replicates.

**Secondary metrics.**
- RB, RRMSE;
- coverage of the 95% CI;
- variance-estimator relative bias;
- CI length;
- weight diagnostics;
- domain distributions.

**Primary family** (Holm, one-sided α = 0.05; 12 tests). The six headroom y below, each tested under two comparators:
- the y: government grants, contributions, program-service revenue, fundraising expenses, has government grants, grants to individuals;
- the comparators: **S1** (B* chosen from B0, B1, B2 and B4) and **S3** (B* chosen from B0, B1, B3 and B4).

Every other test is secondary or a gate.

**Equivalence margin.** δ = 0.03 on RE (TOST).

**Deviations.** Logged in `notes/working-log.md` with the date and the reason. Any analysis not listed here is labelled exploratory.

### 6.1 Hypotheses

| ID | Hypothesis (whose claim) | Metric and comparator | **PASS** | **FAIL** | Skeptic's prediction | NE evidence |
|---|---|---|---|---|---|---|
| **H1** | Size null (skeptic) | RE vs B* for revenue, expenses, assets, employees | RE ∈ [0.97, 1.03] by TOST | RE > 1.05 triggers a **leakage alarm**, audited (L1–L9) before any interpretation | Pass | Ceilings 0.93–1.01 |
| **H2** | Headroom gain, S1 (proponent) | RE vs B*(S1), headroom y | ≥ 2 of 6 y with RE ≥ 1.10 and Holm-adjusted lower bound ≥ 1.05 | Otherwise | Borderline: ceilings 1.04–1.20 shrink to about 1.0–1.12 at n = 1000 | Ceilings 1.04–1.20 |
| **H3** | Survives register-plus (proponent) | RE vs B*(S3), lagged same y or indicator in B* | ≥ 1 of 6 y with RE ≥ 1.05 and lower bound ≥ 1.02 | Otherwise | **Fail** | 0.93–1.017 |
| **H4** | Beyond codes (proponent) | RE(B*+text) / RE(B*+fine code+meta); and against the code-centroid arm | Both ratios ≥ 1.05 for every y that passes H2 | Otherwise | Pass (prior was fail; updated) | 1.10–1.40 and 1.05–1.38 |
| **H5** | Neural over lexical (proponent of embeddings) | RE(neural) / RE(LSA), **same field**, same estimator | ≥ 1.03 with lower bound ≥ 1.00 | Otherwise | Fail | Not yet comparable |
| **H6** | Negative controls (gate) | NC1–NC4, every y | RE upper bound ≤ 1.02; placebo y within [0.97, 1.01] | Any NC with lower bound > 1.00 **invalidates the run** | Pass | Pass (0.82–1.00) |
| **H7** | Inference validity (gate) | Cross-fitted text estimators | \|RB\| ≤ 1% (or ≤ 2 MCSE); coverage 93–96.5% at n = 1000 and ≥ 92% at n = 500; variance-estimator relative bias within ±10% | Otherwise the arm is excluded | Pass for PC-based and cross-fitted arms | Smoke (R = 60): cross-fitted coverage 85–97%; R too small |
| **H8** | Harness sensitivity (gate) | PC1 injected signal; PC3 known failure; NC10 in-sample fits | PC1 recovered within ±0.03; PC3 shows RE < 1 and coverage < 90%; NC10 shows coverage < 90% for ≥ 1 y | Otherwise the harness **cannot be trusted** | Pass | PC3 reproduced (RE ≈ 0, coverage 0–5%). NC10: in-sample ridge coverage 2–4% (§4.5); smoke-run model calibration 72–77%. PC1 not yet run. |
| **H9** | Single-weight compatibility (proponent) | One set of text-calibrated weights for all published y | Negative weights ≤ 1%; g ∈ [0.2, 5]; ≥ 50% of the H2 gain kept on targets; ≤ 2% RE loss on every other y | Otherwise | Fail (multipurpose damage) | — |
| **H10** | Leakage audit (gate) | (a) T2 vs T2-nodigits; (b) text(t) vs text(t−1); (c) NC5 change-y; (d) model vintage | (a) \|ΔRE\| ≤ 0.02; (c) partial R² ≤ 0.01; (d) Δ ≤ 0.03 on NC5. (b) is reported only, because same-year text is banned anyway | Any breach: the arm is excluded and the cause documented | Pass for t−1 text; possible breach for post-2024 API models on (d) | — |
| **H11** | Realistic coverage (proponent) | H2 effect re-estimated on the *frame* population (births, no-text units; indicator method) | RE ≥ 1.07 | Otherwise | Fail (dilution) | — |
| **H12** | Nonresponse (proponent) | Response = TY2023 return submitted by cutoff c (real mechanism from `SubmittedOn`); text added to propensity or RHG vs B-NR (size × NTEE × state × prior-year lateness) | \|RB\| reduced by ≥ 25% for ≥ 2 headroom y, RRMSE up by ≤ 2% for every primary y | Otherwise | Fail (≤ 10%) | — |
| **H13** | Stratification (proponent) | Strata on m̂(text) trained on the tuning population, vs cum-√f / LH on size × NTEE | Anticipated CV at fixed n down ≥ 5% for ≥ 1 primary y; no other y up > 3% | Otherwise | Fail | — |
| **H14** | SAE (proponent) | State × NTEE10 domains with ≥ 1 sampled unit; best frame-only GREG/EBLUP vs the same with text | Median domain RRMSE down ≥ 10%; median domain coverage ≥ 90%; ≤ 5% of domains with \|RB\| > 10% | Otherwise | Fail in sampled domains; gain only in non-sampled ones, with bias | — |
| **H15** | Births (proponent) | Group without a t−1 return; name or registration text vs industry × region × size class | RE ≥ 1.20 on the group subtotal for ≥ 1 size y | Otherwise | Borderline | — |
| **H16** | Extraction beats embedding (skeptic) | Size y: extracted field vs an embedding of the document containing it | RE(field)/RE(embedding) ≥ 2 and RE(field+embedding)/RE(field) ≤ 1.02 | Otherwise | Pass | Ratio 6.7 on the bare sentence (§4.4) |
| **H17** | Multimodal increment (proponent of the Cohere image mode) | Image or fused embedding vs a text embedding of the same document, same model, PDF-native population | ≥ 1.03 with lower bound ≥ 1.00 | Otherwise | Fail for text-native documents | — |
| **H18** | Replication (gate) | Every PASS in H2, H3, H12, H14, H15 re-run on E2 and E3 | Same sign; \|ΔRE\| ≤ 0.05 | Otherwise the claim is downgraded to "population-specific" | — | — |
| **H19** | Model and vendor robustness (gate) | Swap model version (embed-v4 ↔ v5; MiniLM ↔ another open model), input_type and dimension | \|ΔRE\| ≤ 0.03 for every passing y | Otherwise production would depend on a vendor choice; record this as a risk | — | — |

**Decision rule.**
- **"Useful in production"** requires all of:
  - the gates H6, H7, H8 and H10 pass;
  - at least one of H3, H12, H14 or H15 passes for at least one *published* y;
  - H18 and H19 pass;
  - the operational conditions in §8 hold.
- **"Only a substitute for missing admin data"** is the verdict if H2 passes but H3 fails. The recommendation is then to *acquire or link the lagged administrative variable*. Here that variable gave RE 2.4–14.6, against the text's 1.0–1.4.
- **"Not useful"** is the verdict if H2 fails.

---

## 7. (c) Mandatory baselines, negative controls, positive controls and reporting rules

### 7.1 Baseline ladder (all mandatory; the text arm must be "best baseline + text")

| ID | Baseline | Real-world analogue | Role |
|---|---|---|---|
| B0 | HT (Hájek for means) under the **production** design | Expansion estimator | Reference. It is never the *headline* comparator, but it is eligible as B* when it beats every model-assisted baseline on the tuning population (heavy-tailed y) |
| B1 | Post-stratified / GREG on industry × region × size class | Weak register (births, S1 frames) | Comparator for births (H15) |
| B2 | Separate-ratio / GREG on lagged admin size (revenue, expenses, assets, employees) × stratum, plus take-all | BR with GST/T2/PD7A; BMF with prior return | **Primary comparator in S1** |
| B3 | B2 + lag of the same y (and lagged indicators), × stratum | T3010-style register-plus; BR with payroll for salaries | **Primary comparator in S3** |
| B4 | Text-free ML: cross-fitted model calibration with GBM or RF on all frame variables (NTEE × size, ruling year, county, subsection, foundation code), ratio scale, **same tuning budget as the text arms** | What a strong methodologist does without text | Guards against "the text won because the model was flexible" |
| B5 | Fine industry code (NTEE 3-character / NAICS 6-digit) in the same penalized ratio block; code-centroid embedding; text-imputed code for missing codes, used as a code | "Better coding" rather than embeddings | Redundancy control (H4) |
| B6 | Cheap text: TF-IDF/LSA on the same field; keyword flags; meta-features (length, has website, numbers present) | No neural model | H5 and the MNAR control |
| B7 | Open small embedding (MiniLM, bge, e5) on the same field | In-enclave model | Must be beaten before any API model is credited |
| B8 | Extracted numeric fields from the document (PDF arm) | Document AI or XBRL/GIFI | H16 |
| B9 | Oracle lagged survey y; injected-signal auxiliary | Upper bound / calibration of the harness | PC1 and PC2 |

**Choosing B\*.** For each y, B* is chosen on the *tuning population* from {B0, B1, B2, B4} (S1) or {B0, B1, B3, B4} (S3) and then frozen. HT or post-stratification can beat a many-column GREG for heavy-tailed y (contributions: HT 1.73× better than the base GREG at n = 500, §4.5). The text arm uses the same estimator family, the same penalty-selection rule and the same CV budget.

### 7.2 Negative controls (mandatory; expected value in brackets)

| ID | Control | Expected value |
|---|---|---|
| NC1 | Gaussian random vectors with the same dimension, the same number of PCs and the same penalty | RE ≤ 1.00 |
| NC2 | Embeddings permuted across all units | ≤ 1.00 |
| NC3 | Embeddings permuted *within* design strata (industry × size), which separates composition from unit-specific signal | ≤ 1.00 |
| NC4 | Placebo y: y permuted within strata, and i.i.d. noise | [0.97, 1.01] |
| NC5 | Change y, log(y_t/y_{t−1}), as the memorization probe | Partial R² ≤ 0.01 |
| NC6 | Same-year text (t) vs lagged text (t−1) | Measures L1. Same-year text is never used in confirmatory arms. |
| NC7 | Digits- and amounts-masked text | \|ΔRE\| ≤ 0.02 |
| NC8 | Shuffled-name and name-only (T0) probes | T0 ≈ B1 |
| NC9 | Pre-cutoff vs post-cutoff embedding model on NC5 | Δ ≤ 0.03 |
| NC10 | In-sample (not cross-fitted) version of every flexible text estimator | **Must** lose coverage (harness check) |
| NC11 | Synthetic Σ_U m̂ with no design correction | **Must** show bias |
| NC12 | k-means-on-embedding strata vs random strata of the same sizes | Size of the gain from unsupervised text strata |

### 7.3 Positive controls

| ID | Control | Requirement |
|---|---|---|
| PC1 | Injected signal: y* = y + c · σ_e · g(PCs), where g is a fixed random linear map of the text PCs, scaled to partial R² ∈ {0.05, 0.10, 0.20} | The harness must recover RE within ±0.03 of the finite-sample prediction (1 + p_eff/n)⁻¹ / (1 − R²) |
| PC2 | Oracle lagged y | Large gains must appear: here RE 2.4–14.6 |
| PC3 | OLS-GREG on raw dimensions with p/n ≥ 0.5 | Must show RE < 1 and coverage < 90% |

### 7.4 Reporting rules

1. **Report everything.** Every y and every arm goes in an appendix table, not only the winners.
2. **Primary vs secondary comparators.** The primary RE is against B*. RE against HT is secondary and labelled as such.
3. **Monte Carlo error.** Give an MCSE for every metric and a paired bootstrap CI for RE.
4. **Weights.** Report % negative weights, min/max g, weight CV and Kish deff, for each weight set.
5. **Domains.** Show distributions of RB, RRMSE and coverage across domains, separately for sampled and non-sampled domains.
6. **Coverage strata.** Show results by text-coverage stratum, and under the realistic-coverage scenario.
7. **Ceilings.** Show population ceilings beside the MC results, both with fixed λ and with tuned λ.
8. **Provenance.** For every embedding matrix record the model id, version, date, input hash, input_type and dimension (`src/embed.py` already writes these).
9. **Cost table.** Weigh embedding, pipeline, governance and bridging-study costs against the n(1 − 1/RE) sample units saved.

---

## 8. (d) Conditions under which the skeptic concedes

The skeptic concedes that embeddings are **useful in production** for a specific (survey, y, use) only if **all** of the following hold.

| # | Condition | Threshold | Why |
|---|---|---|---|
| C1 | **Effect size** at the survey's real n | Any one of:<br>• RE ≥ 1.15 vs B* with lower 95% bound ≥ 1.08 (≥ 13% fewer units, or ≥ 7% lower CV);<br>• NR-bias reduction ≥ 25%;<br>• median domain RRMSE down ≥ 15% | Below this the gain is smaller than the year-to-year noise in collection costs and response rates, and the pipeline is not worth running |
| C2 | **No cheaper admin lag** | The lag of y cannot be obtained or linked at reasonable cost, *or* H3 passes | The lagged admin value achieved RE 2.4–14.6 here, against the text's ≤ 1.4. Buy the lag first. |
| C3 | **Validity** | Gates H6, H7, H8 and H10 pass; coverage ≥ 93%; \|RB\| ≤ 1% overall and in ≥ 90% of published domains | A variance gain with an invalid CI is not a gain |
| C4 | **Replication and robustness** | H18 (a second population and a second year pair) and H19 (model and version swap, \|ΔRE\| ≤ 0.03) | Guards against a population-specific or vendor-specific artefact |
| C5 | **Dissemination compatibility** | H9 (one weight set) passes, *or* the program publishes y-specific model-assisted estimates | NSO microdata carry one weight |
| C6 | **Operational** | All of:<br>• the model runs in-enclave (open weights, or private deployment with Canadian residency in writing);<br>• version-pinned, with stored vectors and input hashes;<br>• a bridging plan for model changes;<br>• text available **at frame time for the whole frame**, not crawled at estimation time for the sample only;<br>• pipeline cost < value of the sample saved | §3.7 |

**Where the skeptic expects a concession to be plausible** (ranked):
1. **New content with no administrative counterpart, where the text covers the whole frame at design time.** Examples:
   - charity or nonprofit surveys of volunteering and program activity, using the T3010 program descriptions ([n02]: ongoing programs described in Section C2) as auxiliaries for a survey of volunteers or program types;
   - ICT, AI-use or innovation surveys, using website text. This is the use with the strongest external precedent: CBS and ZEW predicted innovation status from website text ([n02]). Website text is size-graded and estimation-time, so it is limited to model-assisted estimation with a whole-frame crawl.
2. **Births and units without admin history**, using the BN "major business activity" text, which every registrant gives ([n02]). The gain is confined to a small stratum. It matters for timely estimates of new businesses and for stratum assignment.
3. **Text-defined domains** that no code captures, such as food-security organizations, arts education or AI developers. Here the text changes *what can be estimated* (domain membership), and that is closer to a bias correction than to variance reduction.
4. **Code repair.** NTEE is missing for about 30% of BMF rows ([n03]), and StatCan notes that BR NAICS codes may be outdated or wrong ([n02]). Text-imputed codes feeding the *conventional* stratifier or post-stratifier are a measurable, low-risk use. (This is a productivity use, which the claim disowns. Its variance effect, fewer stratum jumpers, can still be measured.)

**Where the skeptic will not concede, whatever the benchmark shows on public data:**
- Financial size variables when tax data (GST/HST, T2/GIFI, PD7A) are on the BR.
- Any y whose prior-cycle value is held by any administrative source.
- Area-level SAE that uses *text* area profiles (Wikipedia and similar), which quote census figures (L3; [n03]). Use satellite or POI embeddings with census or admin truth instead.
- Any configuration in which the embedding must be produced by an external API from confidential register text.

---

## 9. Real-world mapping: where, if anywhere, this would live at an NSO

| NSO program (Canadian framing) | Text actually held at design time | Admin lag of y on the register? | Text coverage | Skeptic's verdict | Benchmark analogue |
|---|---|---|---|---|---|
| Annual business / industry surveys (financial y) | BN "major business activity" text from registration; codes made from it ([n02]) | **Yes**: GST/HST revenue, T2/GIFI, PD7A payroll | Near-universal for BN units | **No gain expected** (P1, H1) | 990 revenue, expenses, salaries vs B3 |
| Same surveys, *births* in the current cycle | BN text and name only | No | All births | Plausible but small (P8, H15) | 990 units without a t−1 return |
| Charities / nonprofit satellite account (financial y) | T3010 program descriptions (public) | **Yes**: the whole prior T3010 | ~84k registered charities ([n03]) | **No gain** vs register-plus (P4, P7) | 990 S3 scenario (register-plus) |
| Nonprofit volunteering / program-activity content | T3010 program text | **No** (volunteers are not on T3010 **[UNVERIFIED]**) | All registered charities | **Most plausible use** (P3, P6; H2). On NE, though, volunteers reach only 1.04, activity indicators 1.03–1.17 and funding-mix dollars 1.13–1.20 (ceilings, before the finite-sample penalty). | 990 volunteers, program type and grants to individuals under S1 |
| ICT, AI-use, e-commerce and innovation surveys | Websites (crawled), BN text | No | Websites size-graded: EU 77% small vs 96% large; Canada 2013 42% small vs 91% large ([n02]) | Plausible, if the whole frame is crawled at design time; ISTAT/CBS precedent ([n02]) | Needs a web-crawl population (not reachable from this sandbox) |
| Area-level SAE (poverty, income) | Area profiles (Wikipedia), satellite or POI embeddings | Census and admin covariates are strong | Complete | Text profiles **leak** (L3). Satellite embeddings are acceptable. | County SAE with census truth ([n03] §2.10) |
| Financial statements as PDFs (public companies; UK charities' accounts) | PDF pages | Structured filings usually exist (XBRL, the CCEW annual return) | Larger entities only | Extract fields (H16). Image embeddings for narrative only (H17). | UK CCEW accounts PDFs ([n03] R2) |

**The general rule an NSO methodologist can take away.** First list every administrative source that holds a prior-cycle value of y. If one exists, link it: that is the variance reduction, and it is large. Text embeddings are a candidate only for y with **no** administrative counterpart, for **births**, and for **domains that the codes cannot define**. Even there, the gain must beat an equally flexible text-free model (B4) and cheap lexical features (B6).

---

## 10. References

**Tags.**
- **[n01]–[n04]**: verified earlier in this session by the sibling notes `01-methods-literature.md`, `02-applied-precedents-and-real-data-sources.md`, `03-dataset-candidates-and-red-team.md` and `04-embedding-models-cohere-and-alternatives.md`. Full bibliographic detail and URLs are in those notes.
- **[UNVERIFIED]**: cited from memory. It could not be checked here because the shared search budget was exhausted and WebFetch failed with DNS errors.

**Survey estimation with many or ML auxiliaries [n01]**
- Breidt, F.J., Opsomer, J.D. (2017). Model-assisted survey estimation with modern prediction techniques. *Statistical Science* 32(2):190–205. doi:10.1214/16-STS589.
- Breidt, F.J., Opsomer, J.D. (2008). Endogenous post-stratification in surveys: classifying with a sample-fitted model. *Annals of Statistics* 36(1):403–427. doi:10.1214/009053607000000703.
- McConville, K.S., Breidt, F.J., Lee, T.C.M., Moisen, G.G. (2017). Model-assisted survey regression estimation with the lasso. *JSSAM* 5(2):131–158. doi:10.1093/jssam/smw041.
- Ta, T., Shao, J., Li, Q., Wang, L. (2020). Generalized regression estimators with high-dimensional covariates. *Statistica Sinica* 30:1135–1154. doi:10.5705/ss.202017.0384.
- Cardot, H., Goga, C., Shehzad, M.-A. (2017). Calibration and partial calibration on principal components when the number of auxiliary variables is large. *Statistica Sinica* 27:243–260.
- Chauvet, G., Goga, C. (2022). Asymptotic efficiency of the calibration estimator in a high-dimensional data setting. *JSPI* 217:177–187. doi:10.1016/j.jspi.2021.07.011.
- Eustache, E., Dagdoug, M., Haziza, D. (2025). On high-dimensional variance estimation in survey sampling. *Scand. J. Statist.* 52(2):924–959. doi:10.1111/sjos.12776.
- Bouhadra, Dagdoug (2026). High-dimensional variance estimation for the generalized regression estimator. arXiv:2607.05570.
- Dagdoug, M., Goga, C., Haziza, D. (2023). Model-assisted estimation through random forests in finite population sampling. *JASA* 118(542):1234–1251. doi:10.1080/01621459.2021.1987250.
- Dagdoug, M., Haziza, D. (2026). Machine learning methods for finite population parameter estimation in survey sampling. arXiv:2604.01160.
- An, Dagdoug, Haziza, Tillé (2026). Agnostic model-assisted estimation with machine learning for survey data. arXiv:2609.10904.
- Nascimento Silva, P.L.D., Skinner, C.J. (1997). Variable selection for regression estimation in finite populations. *Survey Methodology* 23(1):23–32.
- Wu, C., Sitter, R.R. (2001). A model-calibration approach to using complete auxiliary information from survey data. *JASA* (volume and DOI [UNVERIFIED] in [n01]).
- Montanari, G.E., Ranalli, M.G. (2005). Nonparametric model calibration estimation in survey sampling. *JASA* 100:1429–1442. doi:10.1198/016214505000000141.
- Haziza, D., Beaumont, J.-F. (2017). Construction of weights in surveys: a review. *Statistical Science* 32(2):206–226.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal* 21(1):C1–C68.

**Prediction-powered inference [n01]**
- Angelopoulos, A.N., Bates, S., Fannjiang, C., Jordan, M.I., Zrnic, T. (2023). Prediction-powered inference. *Science* 382(6671):669–674. doi:10.1126/science.adi6000.
- Zrnic, T., Candès, E. (2024). Cross-prediction-powered inference. *PNAS* 121(15):e2322083121. doi:10.1073/pnas.2322083121.
- Mozer (2026). PPI is the difference estimator. arXiv:2603.19160.
- Egami, N., Hinck, M., Stewart, B., Wei, H. (2023). Using imperfect surrogates for downstream inference. NeurIPS 36.

**Nonresponse and stratification [n01]**
- Little, R.J.A., Vartivarian, S. (2005). Does weighting for nonresponse increase the variance of survey means? *Survey Methodology* 31(2):161–168.
- Haziza, D., Lesage, É. (2016). A discussion of weighting procedures for unit nonresponse. *JOS* 32(1):129–145. doi:10.1515/jos-2016-0006.
- Kern, C., Klausch, T., Kreuter, F. (2019). Tree-based machine learning methods for survey research. *Survey Research Methods* 13(1):73–93. doi:10.18148/srm/2019.v13i1.7395.
- Lavallée, P., Hidiroglou, M. (1988). On the stratification of skewed populations. *Survey Methodology* 14(1).
- Rivest, L.-P. (2002). A generalization of the Lavallée and Hidiroglou algorithm for stratification in business surveys. *Survey Methodology* 28(2):207–214.

**Text and web data in official and economic statistics [n01, n02]**
- Hoberg, G., Phillips, G. (2016). Text-based network industries and endogenous product differentiation. *JPE* 124(5):1423–1465. doi:10.1086/688176.
- Daas, P.J.H., van der Doef, S. (2020). Detecting innovative companies via their website. *Statistical Journal of the IAOS* 36(4):1239–1251. doi:10.3233/SJI-200627.
- Kinne, J., Lenz, D. (2021). Predicting innovative firms using web mining and deep learning. *PLoS ONE* 16(4):e0249071. doi:10.1371/journal.pone.0249071.
- Kinne, J., Axenbeck, J. (2018). Web mining of firm websites. ZEW Discussion Paper 18-033.
- Cuffe, J. et al. (2019/2022). Using public data to generate industrial classification codes. In *Big Data for Twenty-First-Century Economic Statistics*, NBER. https://www.nber.org/chapters/c14278
- UNECE (2024). Canada SBR classification (NAICS from the BN major business activity text via G-Code). https://unece.org/sites/default/files/2024-10/3.4_Canada%20SBR%20classification_0.pdf
- US Census Bureau, Business Formation Statistics methodology (the auto-coder assigns NAICS to more than 80% of applications). https://census.gov/econ/bfs/pdf/pr_methodology.pdf
- Website-coverage statistics (Eurostat, ISTAT, ABS, StatCan 2013) as compiled in [n02] §5.1.

**Embedding models and operations [n04]**
- Cohere model lifecycle and deprecations (Embed v2 retired 2026-04-04); embed-jobs dataset retention (30 days); pricing (embed-v4.0 at $0.12 per 1M text tokens, $0.47 per 1M image tokens). See [n04] §2.3–2.5 for URLs.

**Leakage, numeracy, benchmarking practice [UNVERIFIED]**
- Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM TKDD* 6(4).
- Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9).
- Wallace, E., Wang, Y., Li, S., Singh, S., Gardner, M. (2019). Do NLP models know numbers? Probing numeracy in embeddings. EMNLP-IJCNLP.
- Naik, A., Ravichander, A., Rose, C., Hovy, E. (2019). Exploring numeracy in word embeddings. ACL.
- Lin, J. (2019). The neural hype and comparisons against weak baselines. *SIGIR Forum* 52(2).
- Armstrong, T.G., Moffat, A., Webber, W., Zobel, J. (2009). Improvements that don't add up: ad-hoc retrieval results since 1998. CIKM.
- Gelman, A., Loken, E. (2014). The statistical crisis in science. *American Scientist* 102(6).

**Project artefacts used here**
- Population: `/tmp/claude-0/data/pop/ne_pop.parquet` (built by `src/build_population.py`).
- Embeddings: `/tmp/claude-0/data/emb/ne/{lsa256_full,minilm_mission}.npy`.
- Diagnostics: `/tmp/claude-0/data/results/diag_ne_n1000.csv`; smoke MC `/tmp/claude-0/data/results/mc_ne_smoke3_metrics.csv` (R=60).
- This note's scripts and outputs: `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/skeptic/`:
  - `falsify_pop.py` → `falsify_pop.csv`;
  - `falsify_bin.py` → `falsify_bin.csv`;
  - `describe.py` → `describe2.out`;
  - `numeracy.py` → `numeracy2.out`;
  - `numeracy_raw.py`;
  - `pn_mc.py` → `pn_mc_R500_n500.csv`. These live in the session scratchpad. Copy them into `src/` if they should persist.
