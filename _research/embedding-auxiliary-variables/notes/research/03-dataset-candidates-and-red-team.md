# 03 — Dataset candidates and red team for "embeddings as auxiliary variables"

Status: research note, 2026-10-23. Scope: find real finite populations where (a) a survey-type y is known for every unit, (b) text, PDF or image auxiliaries exist for (nearly) every unit and were produced before or independently of y, and (c) conventional frame variables exist, so that a design-based Monte Carlo can measure the **incremental** value of embeddings over current practice. The note ranks the candidates, recommends a primary dataset and replication datasets, and gives a construction recipe and red-team checklist for the primary.

**How this was verified (read first).**
- **Web search budget was exhausted before this note started.** The shared 200-call WebSearch limit had already been used by sibling agents. WebFetch failed with `getaddrinfo ENOTFOUND` on every government host I tried (register-of-charities.charitycommission.gov.uk, open.canada.ca). So I could not read the publishers' own pages.
- **What I did instead:**
  1. Sandbox `curl` reachability tests for every candidate (§1).
  2. Direct listing and download from the GivingTuesday S3 data lake, plus my own computations on 2.1M index rows, two BMF snapshots, and 838 downloaded 990 XMLs (§2.1, §5). All are reproducible from the scripts listed in §9.
  3. GitHub code search (GitHub MCP), then `curl` of the raw files from raw.githubusercontent.com. This surfaced third-party mirrors of publisher data dictionaries: the CCEW register field definitions, CRA's *Open Data Dictionary v2.0* for T3010, ACNC AIS loaders that quote data.gov.au licence records, NSF award XML schema notes, and the EDGAR employee-count tagging notes. **These are secondary sources.** They are cited by GitHub URL and should be confirmed against the publisher before anything is published.
  4. Literature citations already found by search in this session, carried over from note 02 and marked "(via note 02)".
- Anything from memory that I could not confirm is marked **[UNVERIFIED]**.
- **Adversarial verification pass (2026-10-23).** Every measured number was re-computed with independent scripts, every GitHub source was re-fetched, every bucket listing was re-done, and the literature was checked against GitHub-hosted bibliographic records (WebSearch was still exhausted and WebFetch reached only github.com). Corrections are applied in place; see **Verification log** at the end.

---

## 0. Bottom line for a skeptic

1. **IRS Form 990 e-file (GivingTuesday data lake) is the only candidate that is fully reachable from the sandbox and meets every hard requirement.**
   - Size: TY2023 has ≈349k full Form 990 returns (non-group, one per EIN), 331k of them domiciled in the 50 states plus DC.
   - Skew: extreme. The top 1,000 units hold 49.5% of total revenue and the revenue CV is 18.
   - Frame: a pre-cutoff IRS Business Master File snapshot serves as the frame.
   - Panel: 96.3% of TY2023 filers link to a TY2022 return.
   - Auxiliary: prior-year mission and program text is present on 100% of linked returns (measured on the 418-pair sample of 990→990 links; the 13,507 units whose TY2022 return was a 990-EZ have only the EZ purpose text, not Part III 4a–4e).
   - **It is text-only.** There are no PDFs or images in the public XML, so it cannot test the multimodal (Cohere embed-v4 image) claim.
2. **For financial y the skeptic is right on this data.** Among linked units, TY2023 total revenue correlates **0.991** with TY2022 revenue on the raw scale (expenses 0.995, assets 0.994). Lagged administrative values leave almost no room for any covariate on design-weighted totals. Total revenue should be the **negative control**, not the headline.
3. **The headroom is in y variables that the frame does not carry and that change from year to year.** Indicative out-of-sample R² on log1p(y_t), from n=418 stratified pairs (volunteers: the 307 non-blank cases):

   | y_t | Frame only | Frame + lagged y |
   |---|---|---|
   | Government grants (Part VIII 1e) | 0.12 | 0.54 |
   | Contributions | 0.25 | 0.66 |
   | Employees | 0.34 | 0.97 |
   | Volunteers | ≈0 | 0.87 |

   The realistic text use cases are therefore:
   - **(S1)** y absent from the register (volunteers, government funding, program mix), where text competes only with NTEE, state and size;
   - **(S2)** y with lagged admin values, where only government grants and contributions have meaningful residual variance;
   - **(S3)** births and units without a prior return.
4. **Leakage is mostly a timing problem, not a "text states y" problem.** Fewer than 1% of texts contain the exact y value. The big traps are elsewhere:
   - **(i)** Using a *current* BMF snapshot as the frame. The May-2026 BMF carries the TY2023 revenue itself for **18.3%** of TY2023 filers, against 0.6% in the December-2023 BMF.
   - **(ii)** Using the same-year Schedule O. **Corrected:** Schedule O proper contains "$" amounts in 17–25% of t−1 returns over $5M. The earlier 35–53% counted `ExplanationTxt` from every schedule (D, H, J, I, A, …), not only Schedule O.
   - **(iii)** Using a register "current activities" field that is overwritten over time (UK CCEW `charity_activities`), or a backfilled current name (CRA T3010: only 1.4% of BNs show any name change over 2020–2024).
   - **(iv)** Embedding-model memorization. A 2025 embedding model may have been trained on web pages, such as nonprofit profile sites, that quote TY2023 financials **[UNVERIFIED but must be tested]**.
5. **Text is near-static.** Between TY2022 and TY2023 the mission text is identical for 70–93% of units and program text for 38–86%, depending on size. Text therefore acts as a **fine-grained, persistent classification** (between-unit signal), not as a signal of change. That is consistent with it helping stratification and SAE more than it helps estimate year-over-year change.
6. **Coverage of frame-time text is size-graded and realistic.**
   - With a survey design cutoff of 30 Jun 2024, 91% of TY2023 990 filers already have a TY2022 return. That falls to 77% at a 31 Mar 2024 cutoff, and the shortfall is concentrated in small units (92% linkage below $50k against 99.2% above $25M).
   - On the full BMF frame, text exists for only about 26.5% of entities (note 02; this is the May-2026 BMF. On the Dec-2023 BMF frame it is 28.6%). The 990-N postcard filers have a name and sometimes a URL, but no text.
7. **Recommended replications.**
   - **(R1, Canadian) CRA T3010 open data.** About 84k charities a year. Program descriptions up to 2,500 characters, in English or French. Revenue from government split by federal, provincial and municipal level. Full-time and part-time staff counts. OGL-Canada licence. Blocked in the sandbox; the user must download it.
   - **(R2, genuine PDF/image) UK Charity Commission (England and Wales).** Structured annual-return y (income, government grants and contracts income, volunteers; employees for Part B filers) plus per-charity accounts PDFs, some of them scans. Blocked, and the PDFs must be scraped per charity.
   - **ACNC Australia** is the fallback for R2: AIS staff, volunteers and government revenue under CC BY.
8. **Not recommended as primaries.**
   - SEC EDGAR: the population is small and made up of take-all units, and Item 1 states headcount, which leaks the y.
   - Inside Airbnb: y is model-derived from reviews, the photo rights are unclear, and the NSO fit is weak.
   - NIH/NSF awards: y is driven by the funding mechanism, and award populations are selected.
   - USAspending and SBA PPP: no usable text.
   - County SAE with Wikipedia: Wikipedia pages quote census numbers, and SAIPE/ACS values are themselves model-based or survey estimates, not truth. For area-level work, satellite embeddings (AlphaEarth, reachable from the sandbox) are the defensible auxiliary, and the truth must be a census or an admin total.

---

## 1. Sandbox reachability (tested 2026-10-23, `curl -s -o /dev/null -w '%{http_code}' -m 20`)

| Host / URL tested | Code | Meaning |
|---|---|---|
| `https://gt990datalake-rawdata.s3.amazonaws.com/` (GivingTuesday 990 data lake) | **200** | Reachable: index, XML, BMF, SOI extracts, 990-N postcards, revocations |
| `raw.githubusercontent.com/...` | **200** | Reachable (docs and small mirrors only) |
| `storage.googleapis.com/alphaearth_foundations/` | **200** | Reachable; listing shows `satellite_embedding/v1/annual/2017/10N/…` (UTM-zone tiles) |
| `sentinel-cogs.s3.us-west-2.amazonaws.com/` | **200** | Reachable; prefix `sentinel-s2-l2a-cogs/` |
| `overturemaps-us-west-2.s3.amazonaws.com/` | **200** | Reachable; releases up to `release/2026-10-21.0/` (POI names, categories, websites) |
| `storage.googleapis.com/gcp-public-data-landsat/index.csv.gz` | 200 | Reachable |
| `openalex.s3.amazonaws.com/` | 200 | Reachable (works and abstracts; no award amounts) |
| `noaa-ghcn-pds.s3.amazonaws.com/` | 200 | Reachable (irrelevant here) |
| `irs-form-990.s3.amazonaws.com/index_2020.json` (old IRS AWS dataset) | 404 | Discontinued |
| `nccsdata.s3.amazonaws.com/`, `naip-*.s3.amazonaws.com/`, `commoncrawl.s3.amazonaws.com/` | 403 | Blocked |
| irs.gov, apps.irs.gov, open.canada.ca, apps.cra-arc.gc.ca, canada.ca, ccewuksprdoneregsadata1.blob.core.windows.net, register-of-charities.charitycommission.gov.uk, data.gov.au, acnc.gov.au, sec.gov, insideairbnb.com, data.insideairbnb.com, api.reporter.nih.gov, reporter.nih.gov, nsf.gov, files/api.usaspending.gov, data.sba.gov, www2.census.gov, api.census.gov, en.wikipedia.org, dumps.wikimedia.org, propublica, huggingface.co, nccs.urban.org, 990data.givingtuesday.org, statcan.gc.ca, kaggle, zenodo, companieshouse, data/index.commoncrawl.org, api.cohere.com, api.openalex.org | 000 | Blocked at the proxy |

Implication: the proof of concept must be built on the 990 data lake (plus AlphaEarth and Sentinel imagery if an area-level arm is wanted). Every replication dataset requires the user to download data outside the sandbox, or to allowlist the host.

---

## 2. Candidate assessments

Legend for timing. **Frame-time** means the auxiliary exists before the survey is designed. **Same-filing** means the text is produced in the same document as y, which creates a leakage risk.

### 2.1 IRS Form 990 e-file — GivingTuesday 990 data lake (PRIMARY)

**Access and format (verified by listing the bucket).**
- **Yearly index CSVs:** `Indices/990xmls/yearly/{YEAR}_efiledata_xmls_created_on_2026-08-25.csv`, about 300 MB per year. Columns include EIN, FormType, TaxYear, TaxPeriod, DocStatus, GroupReturnForAffiliates, ReturnTs, SubmittedOn, ReturnVersion, OrganizationName, LegalDomicileState, TotalRevenueCY, TotalExpensesCY, TotalAssetsBkEOY, Website, YearFormed and URL.
- **Per-return XML:** `EfileData/XmlFiles/{ObjectId}_public.xml`, averaging about 51 KB (838 files = 43 MB).
- **Bulk IRS release zips:** `EfileData/XmlZips/{YYYY}_TEOS_XML_{MM}{A-D}.zip`, 56 MB–1.4 GB each, organised by **IRS release month, not tax year**. (Some January releases run to letter H; older files are named `{YYYY}_TEOS_XML_CT{n}.zip` and `download990xml_2020_{n}.zip`.)
- **BMF snapshots:** `EfileData/BMF/IRS_BMF_{Y}_{M}_{D}_raw.csv`, about 317–339 MB each (one, 2026-05-14, is 219 MB). 30 snapshots from 2023-08-16 to 2026-05-28 were listed.
- **IRS SOI financial extracts:** `EfileData/Extracts/Data/{YY}eoextract990.xlsx` (files 18–23, 308–380 MB) plus `.dat` files for 2012–2017, with documentation `EfileData/Extracts/Documentation/{YY}eofinextractdoc.xlsx` (`.xls` for 12–17). **Corrected:** the files are labelled by IRS **processing year**, not tax year (the 2023 documentation reads "Total record count (Processing Year 2023): 338,048").
- **Other files:** 990-N postcards (`EfileData/Postcards/990N_Postcards_*_raw.txt`, pipe-delimited, about 245–282 MB) and revocations.

**Licence.** IRS e-file returns of exempt organisations are public-inspection documents **[UNVERIFIED wording of the statute and the GivingTuesday terms of use]**. Confirm the GivingTuesday redistribution terms before publishing derived data.

**Population (measured).**

| Index | Rows | Form 990 | 990-EZ | 990-PF | 990-T |
|---|---|---|---|---|---|
| TY2023 | 710,328 | 355,712 | 207,031 | 124,543 | 23,042 |
| TY2022 | 700,822 | 349,091 | 206,160 | 122,858 | 22,713 |

- After deduplication and removal of group returns, TY2023 has **≈349k Form 990 units** (348,907–348,997 depending on dedup order).
- 331,124 of them are in the 50 states plus DC. 17,139 have a missing `LegalDomicileState` in the index (a defect; take state from the BMF instead). 734 are foreign or in territories.
- 501(c)(3) organisations in the 50 states plus DC: 255,248.
- (Re-computation gives 331,051–331,133 in the 50 states plus DC, 17,122–17,130 with missing state and 255,191–255,256 501(c)(3) units, depending on dedup order. The differences are below 0.05%.)

**Skew (TY2023 Form 990).**
- Revenue shares: top 10 units 5.7%, top 100 19.6%, top 1,000 49.5%, top 1% (about 3,500 units) 67.5%. Revenue CV is 18.3.
- 1,500 units report negative revenue and 3,683 report zero.
- Across years the top 1% hold 66.8–67.4% of revenue (TY2021 67.0%, TY2022 66.8%, TY2023 67.4% on the deduplicated non-group population). **Corrected:** the earlier 68–69% applies only to all Form 990 index rows, including group returns and duplicate or amended rows. Either way, the population is well suited to testing take-all plus ratio/GREG designs against text-assisted designs.

**Candidate y (Form 990 core; XPaths verified in the downloaded XML; concordance variable names from the NODC master concordance).**

| y | XML element | NODC variable |
|---|---|---|
| Employees, Part I line 5 | `TotalEmployeeCnt` | `F9_01_ACT_GVRN_EMPL_TOT` |
| Volunteers, Part I line 6 | `TotalVolunteersCnt` | `F9_01_ACT_GVRN_VOL_TOT`, "required = FALSE" (the concordance also marks employees and government grants FALSE, so the flag does not single out volunteers) |
| Government grants, Part VIII 1e | `GovernmentGrantsAmt` | `F9_08_REV_CONTR_GOVT_GRANT` |
| Contributions | `CYContributionsGrantsAmt` | |
| Program service revenue | `CYProgramServiceRevenueAmt` | |
| Salaries and benefits | `CYSalariesCompEmpBnftPaidAmt` | |
| Total revenue, expenses, assets | | |

The SOI extract adds W-3 employees (`noemplyeesw3cnt`), compensation, grants paid, fundraising and many Part IX lines. Binary items such as Part IV indicators (foreign activities, hospital, school, grants to individuals) give a set of proportion targets.

**Text fields, provenance and timing.**

| Text field | XML element(s) |
|---|---|
| Part I line 1, mission or most significant activities | `ActivityOrMissionDesc` |
| Part III line 1, mission | `MissionDesc` |
| Part III 4a program description | `IRS990/Desc` (direct child, next to `ExpenseAmt`, `GrantAmt`, `RevenueAmt`) |
| Part III 4b–4e program descriptions | `ProgSrvcAccomActy2Grp/Desc`, `…3Grp/Desc`, `…OtherGrp/Desc` |
| Schedule O | `IRS990ScheduleO//ExplanationTxt`. The same element name also occurs in Schedules A, C, D, F, H, I, J, K, M and R (218 of 418 t−1 returns), so restrict by parent. |
| Website | `WebsiteAddressTxt` |
| Name | `OrganizationName` (also on the BMF) |
| 990-EZ mission | `PrimaryExemptPurposeTxt` |

Provenance and timing:
- All of this text is **written by the filer in the same filing as that year's y**. It becomes frame-time only if it comes from the **t−1 return** and is used for a **t target**.
- **Pitfall:** a naive `//Desc` XPath also catches `OtherExpensesGrp/Desc` (expense line labels), so whitelist the XPaths.
- 8.4% of missions are 5 words or fewer, including pointers such as "SEE SCHEDULE O". Resolve these pointers through Schedule O `FormAndLineReferenceDesc`.

**Measured red-team facts.** These come from n=418 pairs: TY2023 Form 990 non-group returns linked to a TY2022 Form 990, with 60 drawn per TY2023 revenue class (seed 11; some XMLs failed to parse).

| Revenue class | Employees >0 | Volunteers blank | Gov. grants >0 | Mission identical t vs t−1 | Program text identical | Program text has "$" | Program text has ≥2-digit numbers | Schedule O has "$" (t−1; corrected: Sched. O only, all-schedule `ExplanationTxt` in parentheses) | Website in XML (t−1; real URL, excluding "N/A" and blank) |
|---|---|---|---|---|---|---|---|---|---|
| <$50k | 0.14 | 0.36 | 0.05 | 0.93 | 0.86 | 0.03 | 0.20 | 0.03 (0.07) | 0.31 |
| $50–200k | 0.31 | 0.53 | 0.14 | 0.92 | 0.71 | 0.03 | 0.27 | 0.05 (0.07) | 0.54 |
| $200–500k | 0.60 | 0.40 | 0.23 | 0.88 | 0.68 | 0.00 | 0.33 | 0.03 (0.05) | 0.55 |
| $0.5–1M | 0.77 | 0.27 | 0.30 | 0.90 | 0.68 | 0.05 | 0.35 | 0.05 (0.12) | 0.73 |
| $1–5M | 0.75 | 0.25 | 0.37 | 0.87 | 0.60 | 0.03 | 0.42 | 0.13 (0.18) | 0.78 |
| $5–25M | 0.90 | 0.03 | 0.52 | 0.70 | 0.38 | 0.10 | 0.50 | 0.17 (0.35) | 0.95 |
| >$25M | 0.85 | 0.03 | 0.42 | 0.80 | 0.40 | 0.17 | 0.68 | 0.25 (0.53) | 0.80 |

(The `WebsiteAddressTxt` element itself is non-blank in 90% of these TY2022 returns, but 93 of 418 values are the placeholder "N/A".)

Further findings from the same sample:
- **Exact-value leakage is rare.** The share of texts containing the exact value (≥100) of the same-year y: revenue 0.2%, expenses 0.0%, employees 0.2%, volunteers 0.7%. The t−1 text contains the t−1 value at 0.2–1.0%.
- **Revision consistency is high.** The prior-year column on the t return equals the t−1 return's current-year revenue in **98.3%** of cases (98.5% within 1%). The index `TotalRevenueCY` equals the XML `CYTotalRevenueAmt` in 100% of the sample.
- **Volunteers are a weak y.**
  - Item non-response: 25–53% are blank below $5M, against 3% above $5M. A blank is not a zero.
  - Heaping: among positive values, 31% are multiples of 10, 15% multiples of 50 and 7% multiples of 100. The median is 30 and the 90th percentile is 455.
  - Employees show far less heaping (5% are multiples of 10).
- **Lag structure.** Year-to-year correlations (TY2022 to TY2023):

  | y | Identical t vs t−1 | log1p correlation |
  |---|---|---|
  | Employees | 54% | 0.984 |
  | Volunteers | 69% | 0.948 |
  | Government grants | 64% | 0.774 |
  | Program service revenue | 39% | 0.965 |
  | Contributions | 20% | 0.852 |

  Clarification: "identical" counts blank–blank pairs as identical, and the correlations treat blank as 0. Volunteers are blank in both years for 25% of pairs (item non-response). The government-grants line is absent, which means zero, in both years for 62%. Among pairs with a value in both years, only 61% (volunteers) and 6% (government grants) are identical, so the 64% grant persistence is mostly persistent zeros.

- **Index-level lag (all linked TY2023 units).** Revenue: raw correlation 0.9913, log 0.805, median absolute % change 18%. Expenses: 0.9952 and 0.900. Assets: 0.9941 and 0.857. Exact revenue carry-forward: 0.68%.
- **Text length.** Median TY2022 words: mission 30, program 29, Schedule O 117 (corrected; 144 counts all-schedule `ExplanationTxt`). Program or mission text mentioning "volunteer": 7.2%. Mentioning government, grant or contract terms: 18.4% (21.8% with a regex that also matches federal, Medicaid and HUD).

**Duplicates, amendments, group returns.**
- TY2023 Form 990/990-EZ: 562,743 rows and 554,007 unique EINs. 8,369 EINs have more than one return (up to 7). 40% of the multi-return rows are amended returns, and 1,258 EINs have more than one distinct TaxPeriod (short years or a change of fiscal year). 359 EINs filed **both** a 990 and a 990-EZ for TY2023.
- Group returns (`GroupReturnForAffiliates=True`): 589 TY2023 Form 990 returns covering **5.5% of total revenue**. Their subordinates do not file separately, so the unit definition is ambiguous; exclude them.
- DocStatus single-flag values in the TY2023 index (all form types, not Form 990 only): AmendedReturn about 8.5k (plus combinations), InitialReturn about 22k, FinalReturn about 6.6k, ApplicationPending about 1.1k. Among Form 990 rows alone, counting combinations: Amended 5,971, Initial 6,201, Final 2,093, ApplicationPending 902.
- `SubmittedOn` contains impossible dates (years 1900–8202 across the TY2021–TY2023 indices; 1924–7202 in TY2023), so use `ReturnTs` for timing.

**Panel and coverage.**
- 96.3% of TY2023 Form 990 non-group filers have a TY2022 return (Form 990 322,482; 990-EZ 13,507; none 12,918).
- Linkage by TY2023 revenue: <$50k 92.0%, $50–200k 95.1%, $200–500k 95.2%, $0.5–1M 96.8%, $1–5M 98.0%, $5–25M 98.8%, >$25M 99.2%.
- Of the unlinked units, 34% carry the InitialReturn flag and 38% have YearFormed ≥ 2022 (births).
- **Frame-time availability depends on the design cutoff.** Share with a TY2022 return submitted by: 31 Dec 2023 66.3%; 31 Mar 2024 76.7%; **30 Jun 2024 91.0%**; 30 Sep 2024 94.9%; 31 Dec 2024 95.6%.
- **When TY2023 truth becomes complete.** TY2023 Form 990 returns arrive by quarter: 2024Q1 12.1k, Q2 53.5k, Q3 45.2k, **Q4 127.9k**, 2025Q1 38.5k, Q2 51.9k, Q3 13.5k, then a trickle into 2026. Truth is effectively complete only by late 2025.
- **Reference-period heterogeneity.** Fiscal-year-end months: Dec 59.3%, Jun 24.9%, Sep 5.1%, Aug 2.5%, Mar 2.0%, May 1.9%.

**Frame (BMF) facts (measured).**
- The December-2023 BMF (`IRS_BMF_2023_12_12_raw.csv`, 1,852,262 rows) contains 97.2% of TY2023 Form 990 filers.
- Its financial fields come mostly from TAX_PERIOD 2022 (290k), with some from 2023 (32k) and 2021 (11.6k). About 12.8k are missing.
- **Leakage check.** BMF `REVENUE_AMT` equals the TY2023 revenue for 0.62% of units in the Dec-2023 snapshot (chance or short-period returns), against **18.3%** in the 2026-05-28 snapshot. The latter's TAX_PERIOD is mostly 2024–2025, so it is post-outcome and must not be used as a frame.
- Correlation of Dec-2023 BMF revenue with TY2023 revenue: raw 0.986, log 0.668.
- **NTEE is missing for 30.1%** of TY2023 Form 990 filers. By ruling year: up to 1990, 60.1% (re-computed 60.3%); 1991–2000, 22.7%; after 2000, about 2–5% (re-computed 2.6% for 2001–2010 and 3.1% after 2010). By subsection: 501(c)(3) 15.1%, (c)(4) 62%, (c)(5) 86%, (c)(6) 70%, (c)(7) 76%.
- **So the conventional NTEE stratifier is itself incomplete, and text-imputed classification is a realistic use.** This is the Ma (2021) NTEE-from-text precedent, via note 02.
- State × NTEE-major domains (excluding missing NTEE): 1,394 domains, median 68 units, 412 with fewer than 30 units. That is a natural SAE structure. (These counts include territories and foreign addresses in BMF `STATE`. Restricted to the 50 states plus DC, as in §4.3, there are 1,324 domains, median 76, 348 with fewer than 30 units.)

**Real-world NSO fit.** High for frame-time text auxiliaries in a non-profit or charity survey. The analogue at StatCan is BR units with CRA T3010 program text and BN registration "major business activity" text (note 02). **Caveat:** 990 program text is richer than a BN registration description. A "name-only" arm and a "short mission only" arm should bracket what an NSO actually holds.

**Verdict.** PRIMARY. Text-only. Cheap to scale. Every leakage channel can be tested.

### 2.2 CRA T3010 Registered Charity Information Return — Canada (REPLICATION R1)

**Access.**
- Open Government Portal dataset `2dc3c94d-f98c-4144-98c8-3c7da3d844d5`, via the CKAN `datastore_search` API, per GovAlta/agency-26-hackathon ATTRIBUTIONS.md and CRA/README.md.
- **Blocked from the sandbox** (open.canada.ca returns 000).
- File layout per CRA's *Open Data Dictionary v2.0* (mirrored at GovAlta/agency-26-hackathon `CRA/docs/guides-forms/OPEN-DATA-DICTIONARY-V2.0 ENG.md`), years 2013–2023 in the dictionary:
  - `IDENT_<year>.csv`: Category, Sub-Category, Designation, Province
  - `WEBURL_<year>.csv`
  - `NEW_ONGOING_PROGRAMS_<year>.csv`: Program Type OP/NP/NA, **Description, Text 2,500, "either in English or French and no translation is provided"**
  - `FINANCIAL_SECTION_A_B_AND_C_<year>.csv`: Program Area #1–3 Code, %, 60-character description
  - `FINANCIAL D_AND _SCHEDULE_6_<year>.csv`: 4540 federal, 4550 provincial and 4560 municipal government revenue; 4570 any government; 4700 total revenue; 4950/5100 expenditures; 5000/5010/5020 program, management and fundraising
  - `SCHEDULE_3_COMPENSATION_<year>.csv`: 300 permanent full-time positions, 370 part-time or part-year employees, 390 total compensation
  - plus directors, qualified donees, gifts in kind, and Schedule 2 (outside Canada)

**Licence.** Open Government Licence – Canada (per GovAlta ATTRIBUTIONS.md).

**Population.**
- About 84,000 identification records per complete year. Dec-31 year-end filers number about 48,500. The 2024 file was partial at 71,954 (GovAlta CRA/README.md).
- The y is known for every filer. Small charities (revenue < $100k) file Section D with fewer lines, and Schedule 3 exists only when field 3400 = yes (GovAlta DATA_DICTIONARY.md). Items therefore have structural missingness by size.

**Candidate y.** Total revenue (4700); government revenue by level (4540, 4550, 4560); total expenditures (5100); full-time and part-time staff (300, 370); compensation (390); program-area percentages. Government revenue by level is a **realistic, policy-relevant y that is not trivially carried by the lag**, but this was **not measured** here.

**Text and timing.** Program descriptions are per filing (keyed by BN and FPE), so the t−1 description is cleanly frame-time for a t target. Bilingual (EN/FR) text is a genuine test of multilingual embeddings, and highly relevant to StatCan.

**Leakage and quality (GovAlta KNOWN-DATA-ISSUES.md, third-party, reproducible SQL).**
- **C-7:** the current legal name is backfilled onto historical years. Only 1,308 of 91,129 BNs (1.4%) show any name change in 2020–2024. (The same repository's `CRA/README.md` data-quality summary gives 0.86%, so the source is internally inconsistent; both figures are low.) **Name embeddings would therefore be post-hoc in T3010.**
- **C-1:** 54,010 arithmetic violations across 30,856 BNs (12.8% of 240,714 BN-years). Examples: sub-lines of 4950 exceed the total (24,960 rows), Schedule 6 compensation ≠ Schedule 3 line 390 (13,504), total expenditures not reconciling (6,697).
- **C-2:** 1,075 unit-error candidates (expenditure/revenue > 100×, compensation > total expenditures, money fields > $10B).
- **C-8:** form revision v24→v27 (fields added or removed in 2023–2024).
- WEBURL has exact-duplicate rows.

**Frame.** Category and sub-category (247 sub-categories), designation (charitable organisation, public foundation, private foundation), province, registration date, and lagged revenue. CRA BN registration text is **not** public. T3010 program text is the public proxy.

**Verdict.** REPLICATION R1. It has the best NSO fit for a StatCan audience, but it is text-only and must be downloaded by the user.

### 2.3 UK Charity Commission for England and Wales register (REPLICATION R2, multimodal)

**Access.**
- Full register download: JSON or TXT zips at `https://ccewuksprdoneregsadata1.blob.core.windows.net/data/json/publicextract.{table}.zip` (URLs seen in matthewgall/charitylens SEEDING.md and tompickup23 `fetch_charities.py`).
- Table sizes: `charity` about 250 MB zip (about 395k rows including removed and linked charities); `charity_annual_return_partb` about 200 MB; `charity_trustee` about 90 MB (charitylens SEEDING.md).
- **Accounts PDFs are not in bulk.** They are fetched per charity from `https://register-of-charities.charitycommission.gov.uk/en/charity-search/-/charity-details/{regno}/accounts-and-annual-returns`, which is keyed by organisation number (different from charity number for about 66%), through a redirecting download link. Some PDFs are scans that need page images (sspencespence/ftbuild-bank-well-church-switch `church_charity_extractor.py` and `.claude/skills/extract-pdf-accounts/SKILL.md`).
- **Blocked from the sandbox.**

**Field definitions** (drkane/charity-lookups `ccew-fields.md`, a mirror of CC data definitions):
- `charity`: `charity_activities` ("the trustees' description of what they do and who they help"), `latest_income`, `latest_expenditure`, `charity_contact_web`, company number. Linked charities share a number; operating charities have `linked_charity_number = 0`.
- `charity_governing_document`: `charitable_objects`, `area_of_benefit`. Charitable objects are the closest frame-time analogue of BN registration text (used by charity-classification/ukcat). **Corrected:** they are not strictly registration-time text. The extract holds the current objects with no field-level date, and `governing_document_description` covers "the original document and any subsequent amendments", so objects can change after registration.
- `charity_annual_return_parta`, up to 5 periods (`fin_period_order_number`): `total_gross_income`, `total_gross_expenditure`, `charity_receives_govt_funding_contracts` and `…_grants`, `count_govt_contracts` and `count_govt_grants`, `income_from_government_contracts` and `income_from_government_grants`, `count_volunteers` (the trustees' optional estimate; "not obliged to"), and salary-band counts above £60k.
- `charity_annual_return_partb`: detailed income and expenditure, reserves, assets, `count_employees`, consolidated-accounts flag. Part B applies only to larger charities: above £500k income according to sspencespence (charitylens does not state a threshold); the threshold is **[UNVERIFIED from CC]**.
- `charity_classification`: What/Who/How codes. `charity_area_of_operation` is also available. A third-party script notes that a plain employee count is not in Part A (tompickup23).

**Licence.** Open Government Licence; tompickup23 `fetch_charities.py` cites v3.0 (third-party; **[UNVERIFIED against CC]**). The accounts PDFs contain trustee names (personal data), so keep them out of any repository (sspencespence SKILL.md).

**Population.** Main registered charities. Size roughly 170k (charitylens SEEDING.md, third-party; **[UNVERIFIED against CC]**). Income is known for annual-return filers; there is an income threshold for annual returns and accounts filing **[UNVERIFIED: £10k and £25k]**.

**Leakage and timing (critical).**
- The extract is a **snapshot**. `charity_activities` is the *current* description and may have been rewritten after the outcome year. A t−1 activities text needs an **archived** extract from before t **[UNVERIFIED whether CC or NCVO archives exist]**.
- The t−1 accounts PDF is dated, so it is safe. **Corrected:** charitable objects are a current snapshot with no field-level date. They are probably stable for most charities **[UNVERIFIED amendment rate]**, but `governing_document_description` should be checked for amendments after t−1.
- The accounts PDF for year t **is** the source of y_t (identical numbers), so only the t−1 PDF may be used.

**Multimodal value.** Genuine. Accounts and trustees' report PDFs (text layer or scan) contain tables, charts and narrative. This is where Cohere embed-v4's page-image input is plausibly better than text extraction. It is the strongest real PDF population found.

**Verdict.** REPLICATION R2. The user must scrape the PDFs politely, about 0.5 s apart, as the third-party code does. Start with Part B charities (larger, with `count_employees`) to keep the PDF count to the low tens of thousands **[UNVERIFIED count]**.

### 2.4 ACNC Charity Register and Annual Information Statement — Australia (R2 fallback)

**Access.**
- data.gov.au. Register package `b050b242-4487-4306-abf5-07ca073e5594`: 66,399 rows, updated weekly, about 15 MB CSV, of which 607 have their ABN withheld.
- AIS: one package per year from 2013 to 2024 (CSV from 2019, XLSX before). AIS row counts: 2024 53,697; 2023 53,198; 2022 52,744. Reporting groups (ABN `91111111xxx`, 296–328 per year) should be skipped. All per jaketracey/opax `docs/DATA-TAX-CHARITY.md`, which says it was verified on 2026-09-29.
- **Blocked.**

**Licence.** CC BY 3.0 AU for the register. AIS years range from CC BY 2.5 AU to CC BY 4.0. **AIS 2024 shows "notspecified"** on data.gov.au; the ACNC site states CC BY 4.0, but its disclaimer says "personal, non-commercial use only" (opax doc). **This licence ambiguity must be resolved before using it.**

**Fields** (opax doc; cam874/charityscanner `database/importer.js`; ben-gy/au-charities `pipeline/aggregate.mjs`):
- **y:** revenue from government (defined as all Commonwealth, state, territory or local funding including contracts), donations and bequests, goods and services, total revenue, employee expenses, total expenses, assets; staff full-time, part-time and casual; FTE; volunteers (`staff - volunteers`); key management personnel paid.
- **Text:** `how_purposes_were_pursued`, `conducted_activities`, and international activities descriptions in the AIS; purposes, beneficiaries, operating states and website in the register.
- **Frame:** charity size (small, medium, large), state, purposes and beneficiaries flags, registration date.

**PDF route.** Financial reports for medium and large charities through the ACNC register **[UNVERIFIED bulk access; probably per-charity only]**.

**Risks.** Withheld records (a coverage hole). Basic religious charities do not provide financial information: per the opax doc (third-party), 8,449 AIS 2024 rows are basic religious charities and 7,829 of them carry 0 revenue, so zeros there are not figures (**[UNVERIFIED against ACNC]**). AIS text is same-filing (use t−1).

**Verdict.** A strong alternative to CCEW, with cleaner structured y (staff, volunteers and government revenue for all AIS filers). Use it if CC scraping is infeasible.

### 2.5 SEC EDGAR 10-K (Item 1 business description plus XBRL financials)

**Access.** sec.gov full-index, Financial Statement Data Sets (quarterly zips) and companyfacts JSON. **Blocked.** Public domain **[UNVERIFIED]**.

**Population.** About 6–7k 10-K filers per year **[UNVERIFIED]**. These are the largest firms, so in a real business survey they are mostly take-all units with direct reporting, which leaves little design-based room.

**y.** Revenue, R&D expense (XBRL `ResearchAndDevelopmentExpense` **[UNVERIFIED tag usage rates]**) and employees.

**Leakage.**
- **Headcount is reported in the Item 1 prose ("Human Capital"/"Employees")**, and `dei:EntityNumberOfEmployees` is a voluntary cover-page tag that most filers leave empty (Coral-Bricks-AI/reef `extract_item1_human_capital_headcount/SKILL.md`; Verdenroz/finance-query `employee_count.rs`).
- So the same-year Item 1 states y for employees. Even the t−1 Item 1 states y_{t−1}, which acts as a lagged admin value.

**Multimodal.** Yes (tables and charts in HTML/PDF exhibits).

**NSO fit.** Large-unit imputation for BERD-type surveys (note 02) rather than estimation for a sample.

**Verdict.** Not primary. Possibly a side study on R&D imputation for large units.

### 2.6 Inside Airbnb (listing text, photos, price, occupancy and revenue estimates)

**Access.** Per-city snapshots: `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz`, `neighbourhoods.geojson`. **Blocked.**

**Licence.** Third-party notes disagree: CC BY 4.0 (arieg88/str-agent DATA_NOTES.md) against CC0 1.0 (Cijan540/airbnb-pricing-intelligence data_dictionary.md). The **photos are hosts' copyrighted images on Airbnb's CDN**, and the dataset licence does not cover them **[UNVERIFIED but likely]**.

**y.** `price` (host-set, concurrent with the text and photos, so not "produced before y"). `estimated_occupancy_l365d` and `estimated_revenue_l365d` appear in newer snapshots. Inside Airbnb has **"No booking data"** (arieg88), so these y are necessarily *model-derived*. **Clarification:** arieg88's line "Occupancy is estimated from reviews" describes that project's *own* estimate, which it shows next to Inside Airbnb's columns as a cross-check. That Inside Airbnb's own columns are review-based, and that review-count covariates nearly reproduce them, is **[UNVERIFIED]**.

**Text and images.** Description, amenities and `picture_url`: genuine images.

**Problems.** No admin frame (no tax-based size), units churn between snapshots, and a city is not an NSO population.

**Verdict.** It is the only easy native-image population, but it fails (a) and (b). Use it only as a sandbox demo of image embeddings, never as evidence of variance reduction.

### 2.7 NIH RePORTER / ExPORTER and NSF awards

**Access.**
- NIH ExPORTER project and abstract CSVs (projects `RePORTER_PRJ_C_FY[YEAR]_new.csv`, abstracts `RePORTER_PRJABS_C_FY[YEAR]_new.csv`). `TOTAL_COST`, `ABSTRACT_TEXT`, `PROJECT_TERMS` and `ORG_NAME` are key columns of the *merged* project-plus-abstract frame (per the akulesa/NIH_RePORTER README), not of the abstracts file alone.
- NSF yearly zips of per-award XML via `nsf.gov/awardsearch/download?DownloadFileName=YYYY&All=true`, with fields `AbstractNarration` and `AwardAmount` ("amount awarded to date") (macks22/nsf-award-data `docs/nsf-xml-schema-details.md`).
- Both blocked. Public domain **[UNVERIFIED]**. OpenAlex S3 is reachable but has no award amounts.

**Timing.** The abstract is written at application, before the award, which is good.

**Weaknesses.**
- Award size is driven by the mechanism or activity code (R01 vs R21 etc.), the institute and the duration. These are frame variables that dominate.
- `AwardAmount` is cumulative "to date", which builds in measurement drift.
- The population of *awards* is selected, not a frame of establishments.
- The NSO fit is weak; the closest analogues are NCSES HERD and federal-funds surveys.

**Verdict.** A secondary sanity check at best.

### 2.8 USAspending (award transactions)

Award descriptions are short, coded, program-generated text **[UNVERIFIED]**. The units are awards or transactions with amendments (the same amendment-row duplication pathology documented for Canadian federal G&C data in GovAlta KNOWN-DATA-ISSUES F-2/F-11). y is the obligation amount, recorded in the same record. **Not suitable.**

### 2.9 SBA PPP loan-level data

About 11–12M loans **[UNVERIFIED]**. Fields: NAICS, ZIP, business type, jobs reported, loan amount. There is **no descriptive text beyond the business name**, and loan amount is a formula of payroll **[UNVERIFIED]**. Borrowers are a selected population. **Not suitable**, except to show that *name-only* embeddings are a universally available but weak auxiliary.

### 2.10 Area-level SAE: US counties (ACS/SAIPE as y; Wikipedia, satellite and POI auxiliaries)

**Truth problem.** SAIPE county estimates are model-based, and ACS estimates are survey estimates with sampling error. Neither is a pseudo-census truth. A design-based area simulation needs a **census or admin truth**: decennial counts, or IRS SOI county income tabulations, which come from full return files **[UNVERIFIED access]**. All these are on blocked hosts.

**Auxiliaries.**
- **Wikipedia** county articles quote census population and income figures, which leak y, and they are edited post hoc. Sheehan et al. (2019) used geolocated Wikipedia embeddings (via note 02).
- **AlphaEarth** annual 64-dimensional embeddings, 2017 onward (arXiv 2507.22291, via note 02). The GCS bucket is **reachable**. (Confirmed: a 2017 tile header reports 64 samples per pixel on an 8192×8192 grid, and the bucket holds years 2017–2025.)
- **Sentinel-2 L2A COGs** are reachable.
- **Overture Places** (POI names, categories, websites) are reachable; licence **[UNVERIFIED]**.

**Lessons (via note 02).** Validate on a pseudo-census, not against direct estimates (Corral, Henderson & Segovia 2025). Non-sampled areas do worse (Newhouse 2023).

**Verdict.** A good *second-stage* SAE arm with imagery and no Cohere needed, but not the POC primary. The y truth must be imported by the user.

### 2.11 Others considered briefly (not verified this session)

- **UK Companies House accounts** (iXBRL and PDF; average-employees note; SIC codes). Strong business-register analogue, but micro-entity accounts are thin and narrative text is scarce **[UNVERIFIED]**. Blocked.
- **NCES IPEDS** (institutions' mission statements plus enrollment and finance). About 6k units, too small, but a federal statistical survey **[UNVERIFIED]**.
- **New Zealand Charities Register** and **OSCR Scotland**: further annual-return populations with activity text and performance reports **[UNVERIFIED]**.

---

## 3. Ranking

Scores run from 0 (fails) to 3 (fully meets). **Reach** = usable from this sandbox today. **MM** = genuine PDF or image auxiliary.

| Dataset | (a) y for all units | (b) pre-y text | (c) frame vars | (d) size and skew | (e) licence | (f) panel and domains | Reach | NSO fit | MM | Leakage-control ease | **Total /30** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **IRS 990 (GivingTuesday)** | 3 | 3 (t−1 return) | 3 (BMF) | 3 (≈349k, CV 18) | 2 (public records; GT terms unverified) | 3 | **3** | 3 | 0 | 3 | **26** |
| **CRA T3010** | 3 | 3 (per-FPE programs, EN/FR) | 2 (category, designation, province; no NAICS) | 2 (≈84k) | 3 (OGL-C) | 3 | 0 | **3** | 0 | 2 (backfilled names) | **21** |
| **UK CCEW** | 2 (Part B y only for large units) | 2 (objects ok; activities is a snapshot) | 2 | 3 | 2–3 (OGL) | 2 (5-period history) | 0 | 2 | **3** (accounts PDFs, scans) | 1 (snapshot text; PDFs need t−1) | **19** |
| **ACNC** | 3 | 2 (AIS text same-filing; use t−1) | 2 | 2 (≈53k) | 1–2 (2024 licence unclear) | 3 | 0 | 2 | 2 (financial reports, access unverified) | 2 | **19** |
| County SAE (imagery + census or admin truth) | 2 (truth must be census or admin) | 3 (imagery precedes y) | 3 | 1 (3k areas) | 2 | 2 | 1 (imagery only) | 2 | 3 (image) | 2 (Wikipedia leaks) | **21** |
| SEC EDGAR | 2 | 1 (Item 1 states headcount) | 1 | 1 (~6–7k large firms) | 3 | 3 | 0 | 1 | 2 | 1 | **15** |
| NIH/NSF awards | 3 | 3 | 2 | 2 | 3 | 1 | 0 | 1 | 0 | 2 | **17** |
| Inside Airbnb | 1 (y model-derived) | 0 (concurrent with price) | 0 | 2 | 1 (photos) | 1 | 0 | 1 | 3 | 1 | **10** |
| USAspending | 2 | 0 | 1 | 2 | 3 | 1 | 0 | 0 | 0 | 1 | **10** |
| SBA PPP | 2 | 0 (name only) | 2 | 3 | 3 | 0 | 0 | 1 | 0 | 2 | **13** |

**Recommendation.**
- **PRIMARY: IRS 990.**
- **R1 (Canadian): CRA T3010.** It reruns the same benchmark with bilingual text and government-revenue-by-level y.
- **R2 (multimodal): UK CCEW**, with t−1 accounts PDFs embedded as page images by Cohere embed-v4 against text extracted from the same PDFs and against governing-document charitable objects (current snapshot; amendable). Fall back to ACNC if scraping is infeasible.
- **Optional SAE arm:** county imagery embeddings, but only with a census or admin truth.

---

## 4. Primary dataset: construction recipe (IRS 990)

### 4.1 Real-world framing (what the simulation pretends to be)

A national statistical office runs an **annual survey of tax-exempt organisations, reference year TY2023**. The survey is designed on **30 June 2024** with the information available then:
- the IRS BMF as the frame;
- the prior-year (TY2022) returns as lagged administrative data;
- prior-year descriptive text as the candidate auxiliary.

It collects y from a sample. Because every TY2023 return is later observed, the full TY2023 Form 990 file plays the role of the **pseudo-census truth**. This mirrors a StatCan design in which the Business Register (BR) and CRA data are the frame and lagged admin data, BN or T3010 text is the auxiliary, and the survey collects y that is not on the BR.

### 4.2 Timeline

| Object | Source | Date rule |
|---|---|---|
| Frame F (design time) | Default: `EfileData/BMF/IRS_BMF_2023_12_12_raw.csv`. It predates almost every TY2023 filing, and the equality test gives 0.62%. Sensitivity run: `IRS_BMF_2024_4_11_raw.csv`, the latest snapshot before a 30 Jun 2024 cutoff. It may already carry TY2023 values for the about 12k units that filed in 2024Q1. (Confirmed: its `REVENUE_AMT` equals TY2023 revenue for 3.6% of units, about 12.1k, against 0.62% in Dec-2023.) That is realistic for an NSO, but it must be reported through the equality test. | Snapshot date ≤ cutoff. Assert that the BMF TAX_PERIOD maps to tax year ≤ 2022 (default) and quantify the share that does not (sensitivity). |
| Lagged admin x_{t−1} | TY2022 index plus XML, **first** return per EIN with `ReturnTs` ≤ cutoff | Later amendments are excluded from the frame. |
| Text z_{t−1} | Same TY2022 return as x_{t−1} | ≤ cutoff (91.0% coverage at 30 Jun 2024; run 31 Mar 2024 as a stress case: 76.7%) |
| Truth y_t | TY2023 index plus XML (index built 2026-08-25), **last** return per EIN (amended supersedes original) | Complete by about late 2025 |

### 4.3 Population U (TY2023) and exclusions

Apply in this order and log the count dropped at each step:
1. FormType = 990 (full form). Exclude 990-EZ, 990-N, 990-PF and 990-T. 990-EZ filers are out of scope for Part I lines 5–6 and Part VIII 1e granularity; run them as a separate stratum later if wanted.
2. Exclude `GroupReturnForAffiliates = True` (589 returns, 5.5% of revenue).
3. One unit per EIN.
   - If an EIN has more than one TaxPeriod in TY2023, keep the latest full-year period (and flag it).
   - If it has both a 990 and a 990-EZ (359 EINs), keep the 990.
   - Among same-period returns, keep the latest `ReturnTs` for truth. Record whether that return is amended and the original-versus-amended y difference.
4. Geography: take state from the **BMF** (the index `LegalDomicileState` is missing for 17,139 units). Keep the 50 states plus DC (331k). Report territories and foreign units separately.
5. Optionally restrict to 501(c)(3) (255k) for the POC, to make "program text" homogeneous; keep subsection as a frame variable otherwise.
6. Do **not** drop negative or zero revenue. Report them, and test estimator robustness (1,500 negative, 3,683 zero).
7. FinalReturn units stay in the population (they existed in TY2023). InitialReturn units without a TY2022 return form the **births stratum (S3)**.

**POC scale options** (Form 990 non-group TY2023 counts):

| Option | Units | Note |
|---|---|---|
| NY alone | 24,939 | Dev and debug |
| MN+WI+IA | 19,516 | |
| WA+OR | 12,936 | |
| NY+PA+OH+MA+MI+NC+VA+NJ | **99,232** | Recommended: realistic N, SAE domains, about 10 GB of XML for two years |
| National | ≈349k | About 36 GB of XML for two years at 51 KB per file; prefer the monthly TEOS zips over per-file GETs |

### 4.4 Frame variables (the conventional comparators)

- From the BMF: `NTEE_CD` (major group, plus "missing" as its own level: 30% missing), `SUBSECTION`, `STATE`, `ZIP` (to county via a crosswalk), ruling year, `FOUNDATION`, `ASSET_CD`, `INCOME_CD`, `ASSET_AMT`, `INCOME_AMT`, `REVENUE_AMT`, `AFFILIATION`, `GROUP`.
- From the TY2022 return (scenario S2 only): lagged total revenue, expenses, assets, and the lagged value of the target y.
- Do **not** use `Website` from the index. It is a defect for 2022v5.0 schema returns (note 02); use the XML `WebsiteAddressTxt`. (Checked on the 418 TY2022 sample returns: the index `Website` is non-blank for 41%, against 90% for XML `WebsiteAddressTxt`, or 67% excluding "N/A". That the defect is specific to 2022v5.0 is not established: 98% of TY2022 returns are 2022v5.0, the 7 v4.1 returns show the same gap, and TY2023 2023v6.0 returns are also low at 43%.)

### 4.5 Text arms (all from TY2022, the t−1 return)

| Arm | Content | Coverage | Purpose |
|---|---|---|---|
| T0 | Organization name (BMF) | 100% of the frame, births included | What every register has. Also the **memorization probe** (§5, item 18). |
| T1 | Mission: Part I line 1 plus Part III line 1 (deduplicated), with "See Schedule O" pointers resolved | t−1 filers | Closest to a BN/SS-4 activity description |
| T2 | T1 plus Part III 4a–4e program descriptions (whitelisted XPaths only) | t−1 filers | Main arm |
| T2-nodigits | T2 with all digits and "$" amounts masked | | Leakage-through-numbers ablation (program text has ≥2-digit numbers in 20–68% of cases) |
| T3 | T2 plus Schedule O | | Ablation only (governance boilerplate; "$" amounts in 3–25% of Schedule O proper, or 5–53% if all-schedule `ExplanationTxt` is included) |
| T-cheap | TF-IDF or LSA of T2 | | The question is whether *generic embeddings* beat *cheap text features* |

Embedding models: local MiniLM (training data predates TY2023) and, once allowlisted, Cohere embed-v4 (text mode). For units with no t−1 return, the text arms are missing. Handle this with a missingness indicator and a separate stratum, and **never impute from y_t**.

### 4.6 Target variables (headline vs control)

Indicative out-of-sample R² for log1p(y_t). Source: 10-fold ridge on n=418 (volunteers: n=307 non-blank), which is unweighted and stratified by y_t revenue, so treat the values as directional only. (Re-run: all values reproduce.) Frame = BMF income, assets, age, NTEE major and subsection.

| y_t | R² frame only (S1) | R² frame + lagged y (S2) | Role |
|---|---|---|---|
| Government grants (Part VIII 1e) | 0.12 | 0.54 | **Headline** (the most residual variance left after lagged y) |
| Contributions and grants (Part I 8) | 0.25 | 0.66 | Headline |
| Volunteers (Part I 6) | ≈0 | 0.87 | S1 headline, but define it as "reported volunteers" with item non-response handled explicitly. Blanks are 25–53% for small units, and the values heap. |
| Employees (Part I 5) | 0.34 | 0.97 | S1 only (S2 leaves about 3% headroom) |
| Program service revenue | 0.24 | 0.92 | S1 |
| Salaries and benefits | 0.32 | 0.96 | S1 |
| Total revenue | 0.49 | 0.76 (raw-scale correlation with the lag 0.991) | **Negative control** |
| Binary: any government grant; any employees; foreign activities; grants to individuals | n/a | n/a | Proportion targets, also usable for nonresponse-propensity tests |

### 4.7 Scenarios and uses

- **S1, y not on the register.** Frame = BMF only. Text competes with NTEE × state × size class.
- **S2, register has lagged y.** The baselines are GREG and ratio estimators on lagged y. Text must add over these.
- **S3, births and no t−1 return** (3.7% of units by the final index, 9% at a 30 Jun 2024 cutoff). Only T0 is available.

Uses:
- **Stratification:** text clusters within NTEE major × size class, or text-imputed NTEE where NTEE is missing.
- **Nonresponse adjustment:** simulated response models whose propensity depends on size and NTEE, *and* on a latent factor correlated with the text. Also run a version where it is **not** correlated, so that text has no business helping.
- **Calibration and model-assisted estimation:** model calibration or GREG on cross-fitted ŷ(text); the model is fitted on the sample only, within each replicate.
- **SAE:** state × NTEE-major domains (1,394 domains; 412 have fewer than 30 units). Report sampled and non-sampled domains separately.

### 4.8 Multimodal for the 990 (honest limit)

The public 990 data are XML, and no page images exist. The only multimodal route is website screenshots or HTML for the units with a URL (31–95% by size class; XML field, excluding "N/A"). That needs crawling outside the sandbox (Common Crawl is blocked here). Treat it as an **est-time** auxiliary with size-graded missingness. Do not render 990 XML to images just to "use" the image model: that is artificial and does not match real-world practice.

---

## 5. Red-team checklist (notebook tests)

Each test should print a number and a pass/fail rule. Values observed this session are given where available.

**Population and units**
1. **One row per EIN.** Log the drops by reason (form type, group return, duplicates, multiple periods, territory). Observed in TY2023: 8,369 EINs with more than one return; 1,258 with more than one TaxPeriod; 359 with both a 990 and a 990-EZ.
2. **Amendments.** Share of truth values taken from amended returns, and the distribution of |y_amended − y_original| / y_original. Fail if more than 1% of total revenue changes.
3. **Group returns excluded.** Report their revenue share (observed 5.5%) and the number of subordinate EINs on the BMF that therefore have no own return.
4. **Reference periods.** FYE-month mix (Dec 59%, Jun 25%). Flag short or long periods. Run a sensitivity analysis restricted to Dec FYE.
5. **Geography.** Index state missing (17,139). Assert agreement between BMF state and XML state above 98%, otherwise investigate. (Observed in verification: the index `LegalDomicileState` matches Dec-2023 BMF `STATE` for only 94.5% of units where both exist, so this check fails as written. Part of the gap may be legitimate, because legal domicile is not the same as a mailing address **[UNVERIFIED field semantics]**.)

**Timing and leakage**

6. **Frame snapshot date.** Assert that the BMF snapshot is ≤ cutoff. Test that the share with `REVENUE_AMT == y_t` is ≈0. Observed: 0.62% for Dec-2023 (pass) against 18.3% for the 2026 snapshot (fail).
7. **t−1 returns submitted ≤ cutoff.** Report coverage by size class (91.0% overall at 30 Jun 2024).
8. **Same-year text is never used.** Assert that every embedded text's ObjectId belongs to a TY2022 return.
9. **Numbers in text.** Share of texts containing the exact y_t, y_{t−1} or "$" amounts (observed under 1% for exact values). Compare the T2 and T2-nodigits results; a large gap means numeric leakage.
10. **XPath whitelist.** Assert that no `OtherExpensesGrp/Desc` (expense labels) or numeric siblings (`ExpenseAmt`, `GrantAmt`, `RevenueAmt`) enter the text.
11. **Text staleness.** Share of texts identical between t−1 and t (observed: mission 70–93%, program 38–86%). Reported so that readers interpret the gains as between-unit, not change.
12. **Cross-fitting leakage via templates.** Detect near-duplicate texts across EINs (chapters and affiliates; about 5% of TY2023 990 names match chapter patterns such as PTA, VFW, Lions, Rotary, Habitat, YMCA). Fold assignment must keep near-duplicate clusters together (group k-fold), otherwise the ŷ are optimistic.
13. **Models fitted only on sample data.** No ŷ model may see population y_t. Rerun it in every replicate, or fit it on an *independent* earlier-year sample (for example TY2021→TY2022) and freeze it.

**Coverage**

14. **Text missingness pattern.** Missing rate by size class, NTEE, state and births. Test whether it is correlated with y_t (it is: linkage runs from 92% to 99% by revenue). The estimator must handle it explicitly (stratum or indicator).
15. **Frame completeness.** NTEE missing 30% (60% for rulings up to 1990; 15% for 501(c)(3)). The conventional stratifier must treat "missing" as a level. Include an arm that imputes NTEE from text (fit on units with a known NTEE).

**y quality**

16. **y quality.**
    - Negative and zero values; outliers (top 1,000 units = 49.5% of revenue).
    - The PY column at t versus CY at t−1 (observed 98.3% exact).
    - Index versus XML (observed 100%).
    - Volunteers: blank versus zero (25–53% blank for small units), and heaping at multiples of 10, 50 and 100 (31%, 15%, 7%).
    - Cross-check against the IRS SOI extract (`23eoextract990.xlsx`; note that it is by *processing year*, not tax year).
17. **Schema versions.** TY2022 is 98% 2022v5.0; TY2023 mixes 2023v4.0, v5.0, v5.1 and v6.0. Assert that the text and y XPaths parse identically across versions (count nulls by version).

**Model and memorization**

18. **Embedding memorization probe.**
    - Predict log(y_t / y_{t−1}) (the change, which the text should *not* know) from T0 name embeddings and from T2.
    - Compare with the same models applied to **shuffled names** and to a **pre-2022 embedding model**.
    - If Cohere embed-v4 predicts change materially better than MiniLM or TF-IDF, suspect contamination from web pages that publish 990 financials **[UNVERIFIED mechanism; treat as a test]**.
19. **Placebo y.** Use a y that the text cannot know (for example the last digit of revenue, or a random permutation within strata). The text arms must show no gain.

**Simulation hygiene**

20. **Monte Carlo error.** Report the MC SE of relative bias, RRMSE and CI coverage for each arm. Use at least 1,000 replicates for proportions, and more for heavily skewed totals.
21. **Domains.** SAE domains with fewer than 30 population units are flagged. Report sampled and non-sampled domains separately.
22. **Negative control.** Total revenue in S2: the text gain should be ≈0. If it is not, look for leakage first.

---

## 6. Replication recipes (short)

**R1: CRA T3010.**
- Download `IDENT`, `NEW_ONGOING_PROGRAMS`, `FINANCIAL_SECTION_A_B_AND_C`, `FINANCIAL D_AND _SCHEDULE_6` and `SCHEDULE_3_COMPENSATION` for two consecutive years (for example FPE 2021 as t−1 and FPE 2022 as t) from dataset `2dc3c94d-f98c-4144-98c8-3c7da3d844d5`.
- U = BN-FPE filings in year t with designation C (charitable organisations); foundations are optional.
- y: 4700; 4540, 4550 and 4560 (government revenue by level); 5100; 300 and 370 (staff); 390.
- Text: t−1 program descriptions (EN/FR) plus program-area descriptions.
- Frame: category, sub-category, designation, province, and lagged 4700.
- Red team: Section D versus Schedule 6 (structural missingness for revenue < $100k), the GovAlta arithmetic-violation flags (12.8% of BN-years), the backfilled names (do not use name embeddings as frame-time), and form-version field changes.

**R2: UK CCEW (multimodal).**
- U = main charities (`linked_charity_number = 0`) with Part A for financial periods t−1 and t.
- y_t: `total_gross_income`, `income_from_government_grants` and `…_contracts`, the government-funding flags, `count_volunteers`, and Part B `count_employees` for the larger-charity subset.
- Auxiliaries:
  - (i) `charitable_objects` (governing-document text; a current snapshot that can be amended, see §2.3);
  - (ii) the t−1 accounts PDF: page images for Cohere embed-v4, against the PDF text layer for a text model, against OCR text;
  - (iii) `charity_activities` **only** from an extract archived before t.
- Frame: classification codes, area of operation, registration date, lagged income.
- Red team: the snapshot timing of activities text, PDF availability by size, scans versus text-layer PDFs, trustee names in the PDFs (privacy), and the Part B threshold.

---

## 7. Real-world mapping: which NSO data sources these mimic

| Simulation component | 990 testbed | Real NSO source (Canada) | Real NSO source (US) |
|---|---|---|---|
| Frame | IRS BMF (Dec 2023) | Business Register with CRA BN, NAICS and size | Census Business Register; IRS BMF for tax-exempt units |
| Lagged admin y | TY2022 990 financials | T2/GIFI, T3010, PD7A payroll (note 02) | IRS SOI EO extracts |
| Frame-time text | TY2022 mission and program text | BN registration "major business activity" text (confidential); T3010 program descriptions (public) | SS-4 line 16–17 text (note 02); Form 1023 descriptions **[UNVERIFIED access]** |
| Name-only text | BMF name | BR legal and operating names | BR names |
| Est-time text or images | Websites from `WebsiteAddressTxt` | Enterprise websites (StatCan web-scraping policy, note 02) | Same |
| y not on the register | Volunteers, government grants by type, program mix | Volunteering and non-profit satellite-account inputs **[UNVERIFIED program names]** | Census SAS for NAICS 813 **[UNVERIFIED]** |

---

## 8. Open issues to resolve with the user

1. Allowlist `open.canada.ca` (T3010) and the CCEW blob host plus the register host, or download them outside the sandbox. Cohere needs `api.cohere.com` plus `COHERE_API_KEY`.
2. Confirm the GivingTuesday data-lake terms of use and the IRS public-record status for redistributing derived tables.
3. Decide POC scale: the 8-state population (99k) or national (349k), and Form 990 only or 990 plus 990-EZ.
4. Decide how to treat volunteers: drop them, use them only in S1, or model item non-response.
5. Check whether archived CCEW extracts exist (so that pre-t `charity_activities` text can be used).

---

## 9. Sources and artefacts

**Primary data (accessed this session).**
- GivingTuesday 990 data lake bucket listing and files: https://gt990datalake-rawdata.s3.amazonaws.com/ (prefixes `EfileData/{BMF,XmlFiles,XmlZips,Extracts,Postcards,Revocations,…}`, `Indices/990xmls/yearly/`).
- Local copies: `/tmp/claude-0/data/raw990/idx_{2021,2022,2023}.csv`, `bmf_2026_5_28.csv`, `23eofinextractdoc.xlsx`; downloaded `IRS_BMF_2023_12_12_raw.csv`.
- AlphaEarth bucket: https://storage.googleapis.com/alphaearth_foundations/ (`satellite_embedding/v1/annual/…`). Sentinel-2 COGs: https://sentinel-cogs.s3.us-west-2.amazonaws.com/. Overture: https://overturemaps-us-west-2.s3.amazonaws.com/release/.

**Secondary documentation (GitHub raw files fetched this session; publisher pages were unreachable).**
- CCEW field definitions: https://raw.githubusercontent.com/drkane/charity-lookups/HEAD/ccew-fields.md
- CCEW extract URLs and sizes: https://raw.githubusercontent.com/matthewgall/charitylens/HEAD/SEEDING.md ; https://raw.githubusercontent.com/tompickup23/tompickup.co.uk/HEAD/scripts/observatory/fetch_charities.py
- CCEW accounts-PDF access and scans: https://raw.githubusercontent.com/sspencespence/ftbuild-bank-well-church-switch/HEAD/church_charity_extractor.py ; https://raw.githubusercontent.com/sspencespence/ftbuild-bank-well-church-switch/HEAD/.claude/skills/extract-pdf-accounts/SKILL.md
- CCEW charitable objects used for classification: https://raw.githubusercontent.com/charity-classification/ukcat/HEAD/src/ukcat/fetch_charities.py
- CRA *Open Data Dictionary v2.0* (mirror): https://raw.githubusercontent.com/GovAlta/agency-26-hackathon/HEAD/CRA/docs/guides-forms/OPEN-DATA-DICTIONARY-V2.0%20ENG.md
- T3010 schema, coverage and data issues: https://raw.githubusercontent.com/GovAlta/agency-26-hackathon/HEAD/CRA/docs/DATA_DICTIONARY.md ; https://raw.githubusercontent.com/GovAlta/agency-26-hackathon/HEAD/KNOWN-DATA-ISSUES.md ; https://raw.githubusercontent.com/GovAlta/agency-26-hackathon/HEAD/CRA/README.md ; https://raw.githubusercontent.com/GovAlta/agency-26-hackathon/HEAD/ATTRIBUTIONS.md ; https://raw.githubusercontent.com/GovAlta/agency-26-hackathon/HEAD/.local-db/README.md
- ACNC register and AIS sources, licences and fields: https://raw.githubusercontent.com/jaketracey/opax/HEAD/docs/DATA-TAX-CHARITY.md ; https://raw.githubusercontent.com/cam874/charityscanner/HEAD/database/importer.js ; https://raw.githubusercontent.com/ben-gy/au-charities/HEAD/pipeline/aggregate.mjs
- Inside Airbnb notes: https://raw.githubusercontent.com/arieg88/str-agent/HEAD/DATA_NOTES.md ; https://raw.githubusercontent.com/Cijan540/airbnb-pricing-intelligence/HEAD/docs/data_dictionary.md
- NIH ExPORTER: https://raw.githubusercontent.com/akulesa/NIH_RePORTER/HEAD/README.md
- NSF award XML: https://raw.githubusercontent.com/macks22/nsf-award-data/HEAD/docs/nsf-xml-schema-details.md
- EDGAR employees: https://raw.githubusercontent.com/Coral-Bricks-AI/reef/HEAD/alphacumen/skills/extract_item1_human_capital_headcount/SKILL.md ; https://raw.githubusercontent.com/Verdenroz/finance-query/HEAD/src/adapters/edgar/filings/employee_count.rs
- 990 variable concordance: https://raw.githubusercontent.com/Nonprofit-Open-Data-Collective/irs-efile-master-concordance-file/master/concordance.csv ; https://raw.githubusercontent.com/Nonprofit-Open-Data-Collective/irs-990-efiler-database/master/README.md

**Literature (via note 02, found by search earlier in this session; bibliographic details re-confirmed in verification from GitHub-hosted records).**
- Ma, J. (2021). "Automated Coding Using Machine Learning and Remapping the U.S. Nonprofit Sector: A Guide and Benchmark." *NVSQ* 50(3): 662–687. DOI 10.1177/0899764020968153.
- Corral, P., Henderson, H. & Segovia, S. (2025). "Poverty mapping in the age of machine learning." *J. Dev. Econ.* 172: 103377. DOI 10.1016/j.jdeveco.2024.103377.
- Newhouse, D. (2023). "Small Area Estimation of Poverty and Wealth Using Geospatial Data: What Have We Learned So Far?" WB PRWP 10512. DOI 10.1596/1813-9450-10512.
- Sheehan, E. et al. (2019). "Predicting Economic Development using Geolocated Wikipedia Articles." KDD 2019. DOI 10.1145/3292500.3330784; arXiv 1905.01627.
- "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data." arXiv 2507.22291.
- Rolf, E. et al. (2021). "A generalizable and accessible approach to machine learning with global satellite imagery." *Nat. Commun.* 12: 4392. DOI 10.1038/s41467-021-24638-z.

**Scripts and intermediate files (session scratchpad; copy into `src/` if they need to persist).**
- `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/scripts/profile_idx.py`: index profiling (form types, DocStatus, duplicates, skew)
- `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/scripts/panel.py`: TY2022→TY2023 linkage and lag correlations
- `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/scripts/bmf_frame.py`: BMF timing, NTEE, leakage equality test
- `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/scripts/timing.py`: design-cutoff coverage, FYE mix, concentration
- `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/scripts/sample_pairs.py`, `parse_pairs.py`, `headroom.py`: the 418-pair XML sample, text and y red-team statistics, R² headroom
- Data: `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/pairs420.csv`, `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/pairs_parsed.pkl`, XMLs under `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/xml/{t,p}/`
- Reachability log: `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/ds03/reach.txt`

---

## Verification log

Adversarial pass, 2026-10-23. **Method and limits.** WebSearch was unavailable (the shared 200-call budget was exhausted), and WebFetch resolved only github.com: arxiv.org, doi.org, crossref, nature.com, sagepub, semanticscholar, docs.cohere.com and developers.google.com all failed with `ENOTFOUND`. Checks therefore used:
- independent re-computation on the local index, BMF and XML files;
- fresh `curl` bucket listings, plus a newly downloaded `IRS_BMF_2024_4_11_raw.csv`;
- re-fetches of all 23 cited raw.githubusercontent.com files, which all returned HTTP 200;
- GitHub code search for bibliographic records.

Verifier scripts are at `/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/verify03/scripts/` (`v_index.py`, `v_index2.py`, `v_bmf.py`, `v_xml.py`, `v_xml2.py`), and re-fetched sources are under `…/verify03/gh/`.

**Outcome codes.** C = confirmed. C* = confirmed, with a clarification added in place. X = corrected in place. F = flagged (not established, or the sources conflict).

### Reachability and bucket structure (§1, §2.1)

| # | Claim | Outcome |
|---|---|---|
| 1 | GT data lake, AlphaEarth, Sentinel COGs, Overture, Landsat index, OpenAlex S3 and NOAA return 200 | C (re-tested; the Landsat range request returned 206) |
| 2 | Old IRS AWS index returns 404; nccsdata, naip and commoncrawl S3 return 403 | C |
| 3 | Government, Cohere, HF, Wikipedia, OpenAlex API, NSF and Inside Airbnb hosts return 000 | C (11 hosts re-tested) |
| 4 | AlphaEarth `satellite_embedding/v1/annual/2017/10N/…`; Overture `release/2026-10-21.0/`; Sentinel `sentinel-s2-l2a-cogs/` | C |
| 5 | AlphaEarth embeddings are 64-dimensional, 2017 onward | C (TIFF tag 277 = 64; years 2017–2025 listed) |
| 6 | Yearly index naming, ~300 MB per year, column list | C |
| 7 | Per-return XML averages ~51 KB (838 files = 43 MB) | C (43.3 MB) |
| 8 | XmlZips `{YYYY}_TEOS_XML_{MM}{A-D}`, 56 MB–1.4 GB | C* (letters run to H; older `CT{n}` and `download990xml_2020_{n}` names) |
| 9 | 30 BMF snapshots, 2023-08-16 to 2026-05-28, ~320–340 MB | C* (size range corrected: 317–339 MB plus one of 219 MB) |
| 10 | `IRS_BMF_2024_4_11` is the last snapshot before 30 Jun 2024 | C (the next is 2024-07-25) |
| 11 | SOI extracts are "TY18–23", 308–381 MB, documentation `{YY}eofinextractdoc.xlsx` | **X**: labelled by processing year; documentation sits in `Extracts/Documentation/` |
| 12 | Postcards are pipe-delimited, ~250–280 MB; revocations exist | C* (246–282 MB) |
| 13 | 990-N postcards carry a name and sometimes a URL | C (the website/email field is populated for some rows) |

### Index-derived population facts (§0, §2.1, §4.3, §4.7)

| # | Claim | Outcome |
|---|---|---|
| 14 | TY2023 and TY2022 row and form-type counts | C (exact) |
| 15 | ≈349k units (348,907–348,997) | C |
| 16 | 331,124 in 50 states + DC; 17,139 missing state; 734 other; 255,248 501(c)(3) | C (within 0.05%, depending on dedup) |
| 17 | Top 10 / 100 / 1,000 / 1% revenue shares; CV 18.3; 1,500 negative and 3,683 zero revenue | C (5.67 / 19.6 / 49.5 / 67.4–67.5%) |
| 18 | Top 1% hold 68–69% in TY2021–23 | **X**: 66.8–67.4% on the defined population; 68.1–68.9% only on all 990 rows |
| 19 | 562,743 rows; 554,007 EINs; 8,369 multi-return EINs (max 7); 40% amended; 1,258 with more than one TaxPeriod; 359 filed both forms | C (exact) |
| 20 | 589 group returns hold 5.5% of revenue | C (5.46%) |
| 21 | DocStatus counts (~8.5k / 22k / 6.6k / 1.1k) | C* (single-flag values over all form types) |
| 22 | `SubmittedOn` years run 1900–8202 | C* (across the TY2021–23 indices; TY2023 alone 1924–7202) |
| 23 | 96.3% linked (322,482 / 13,507 / 12,918); linkage by revenue class; unlinked 34% InitialReturn and 38% YearFormed ≥ 2022 | C (exact) |
| 24 | Cutoff coverage 66.3 / 76.7 / 91.0 / 94.9 / 95.6% | C (66.3 / 76.7 / 91.0 / 94.9 / 95.6) |
| 25 | TY2023 arrivals by quarter | C (exact) |
| 26 | FYE mix | C (Dec 59.2%, Jun 24.9%, Sep 5.2%, Aug 2.5%, Mar 2.0%, May 1.9%) |
| 27 | Lag correlations: raw 0.9913 / 0.9952 / 0.9941, log 0.805 / 0.900 / 0.857, 18% median change, 0.68% carry-forward | C |
| 28 | Schema versions (TY2022 98% 2022v5.0; TY2023 mix) | C (97.6%) |
| 29 | POC state counts (24,939 / 19,516 / 12,936 / 99,232) | C (±15 units) |
| 30 | ~5% chapter-type names | C (5.3%) |
| 31 | XML volume ≈10 GB (8 states) and ≈36 GB (national) | C (arithmetic) |
| 32 | S3 is 3.7% (final) and 9% (30 Jun 2024) | C |
| 33 | Text exists for 26.5% of BMF entities | C* (May-2026 BMF; 28.6% on Dec-2023) |

### BMF frame (§2.1, §4.2, §5)

| # | Claim | Outcome |
|---|---|---|
| 34 | Dec-2023 BMF has 1,852,262 rows and holds 97.2% of filers | C |
| 35 | TAX_PERIOD counts 290k / 32k / 11.6k / 12.8k missing | C |
| 36 | `REVENUE_AMT == y_t`: 0.62% (Dec-2023) vs 18.3% (May-2026); 2026 TAX_PERIOD mostly 2024–25 | C |
| 37 | Dec-2023 BMF vs TY2023 revenue correlation: raw 0.986, log 0.668 | C |
| 38 | NTEE missing 30.1%; by ruling year; by subsection | C (≤1990 re-computed 60.3% vs 60.1%; others exact) |
| 39 | 1,394 domains, median 68, 412 under 30 | C* (these include territories; 50 states + DC give 1,324 / 76 / 348) |
| 40 | Apr-2024 BMF already carries TY2023 values for ~12k units | C (3.6%, ≈12.1k) |
| 41 | Red-team item 5 threshold (state agreement > 98%) | C* (new observation: index vs BMF state agreement is 94.5%) |

### XML sample, n=418 (§0, §2.1, §4.5, §4.6)

| # | Claim | Outcome |
|---|---|---|
| 42 | Element names and XPaths, including `IRS990/Desc` beside `ExpenseAmt`/`GrantAmt`/`RevenueAmt` and the `OtherExpensesGrp/Desc` pitfall | C |
| 43 | 418 of 420 pairs parsed (60 per class, seed 11) | C |
| 44 | Red-team table: employees, volunteers blank, gov. grants, mission and program identity, "$", digits | C (exact) |
| 45 | "Schedule O has '$'" column; "35–53% over $5M"; T3 "5–53%" | **X**: the parser read `ExplanationTxt` from every schedule. Schedule O proper is 0.03–0.25 (0.17–0.25 over $5M) |
| 46 | Website-in-XML column | C* (excludes "N/A"; raw element non-blank in 90%) |
| 47 | Exact-value leakage 0.2 / 0.0 / 0.2 / 0.7%; t−1 0.2–1.0% | C |
| 48 | PY(t) = CY(t−1) for 98.3% (98.5% within 1%); index = XML for 100% | C |
| 49 | Volunteer blanks 25–53% vs 3%; heaping 31 / 15 / 7%; median 30, p90 455; employees 5% | C |
| 50 | Lag table (identity shares, log1p correlations) | C* (blank–blank pairs inflate identity: non-response for volunteers, persistent zeros for government grants) |
| 51 | Median words: Schedule O 144 | **X**: 117 for Schedule O proper (mission 30 and program 29 confirmed) |
| 52 | 8.4% of missions have ≤ 5 words | C |
| 53 | 7.2% mention volunteers; 21.8% mention government/grant/contract | C* (21.8% uses a wider regex; 18.4% for the stated terms) |
| 54 | R² headroom table (S1 / S2) | C (headroom.py re-run reproduces every value; volunteers n=307) |
| 55 | Prior-year text present on 100% of linked returns | C* (sample covers 990→990 links only) |
| 56 | Index `Website` is a 2022v5.0 defect | **F**: under-reporting confirmed (41% vs 90%); version-specificity not established |

### Concordance and SOI documentation

| # | Claim | Outcome |
|---|---|---|
| 57 | NODC names `F9_01_ACT_GVRN_EMPL_TOT`, `F9_01_ACT_GVRN_VOL_TOT`, `F9_08_REV_CONTR_GOVT_GRANT`; volunteers "required = FALSE" | C* (all three are FALSE; concordance versions run only to 2016) |
| 58 | 990-EZ mission is in `PrimaryExemptPurposeTxt`; 990-EZ lacks Part I lines 5–6 | C (concordance; it has no 990-EZ XPath for the employee or volunteer variables) |
| 59 | SOI `noemplyeesw3cnt` is the W-3 employee count | C (23eofinextractdoc: "Number of employees", Part V-2a) |

### Secondary documentation (re-fetched)

| # | Claim | Outcome |
|---|---|---|
| 60 | All 23 cited GitHub raw URLs resolve | C (HTTP 200) |
| 61 | CCEW field definitions (`charity_activities` wording, `latest_income`, `linked_charity_number = 0`, Part A government fields, `count_volunteers` "not obliged", £60k salary bands, Part B `count_employees`, consolidated flag) | C |
| 62 | CCEW zip sizes and row count (~250 MB / ~395k; Part B ~200 MB; trustees ~90 MB) | C (charitylens) |
| 63 | Accounts PDFs are keyed by organisation number (~66% differ), use a redirect link and include scans; 0.5 s spacing; trustee names | C (sspencespence) |
| 64 | Part B threshold of £500k "per charitylens/sspencespence" | **X**: attributed to sspencespence only |
| 65 | No plain employee count in Part A | C (tompickup23) |
| 66 | Charitable objects are "registration-time" and "dated" | **X**: a current snapshot that can be amended, with no field date |
| 67 | ukcat uses `charitable_objects` | C |
| 68 | CCEW licence version; ~170k charities | C* (OGL v3.0 and ~170k from third-party sources; still unverified against CC) |
| 69 | CRA dataset `2dc3c94d-…`, CKAN `datastore_search`, OGL-Canada | C |
| 70 | CRA ODD v2.0 layout: file names; 2,500-char description "either in English or French and no translation is provided"; 60-char program areas; lines 4540–5100; 300/370/390; years 2013–2023 | C |
| 71 | ~84k records a year; ~48,500 Dec-31 filers; 71,954 in 2024 | C |
| 72 | Section D below $100k; Schedule 3 iff field 3400; 247 sub-categories | C |
| 73 | GovAlta C-1 (54,010 / 30,856 / 12.8% of 240,714 / 24,960 / 13,504 / 6,697), C-2 (1,075), C-8 (v24→v27), F-2 / F-11 | C |
| 74 | C-7: 1.4% of BNs show any name change | **F**: KNOWN-DATA-ISSUES says 1.4%, but the same repo's README says 0.86% |
| 75 | WEBURL exact-duplicate rows | C (350 rows, `CRA/docs/DATA_DICTIONARY.md`) |
| 76 | ACNC register ID, 66,399 rows, weekly, 15 MB, 607 withheld; AIS 2013–2024 (CSV from 2019); 53,697 / 53,198 / 52,744 rows; reporting groups 296–328; verified 2026-09-29 | C |
| 77 | ACNC licences (register CC BY 3.0 AU; AIS 2.5 / 3.0 / 4.0; 2024 "notspecified"; "personal, non-commercial") | C |
| 78 | ACNC fields and the government-revenue definition | C (opax, charityscanner) |
| 79 | Basic religious charities give no financial information | C* (opax: 8,449 rows; still unverified against ACNC) |
| 80 | Inside Airbnb licence conflict (CC BY 4.0 vs CC0); file names; `estimated_*_l365d` columns | C |
| 81 | "Occupancy is estimated from reviews" read as Inside Airbnb's method | **X**: the line describes arieg88's own estimate; Inside Airbnb's method is [UNVERIFIED] |
| 82 | NIH ExPORTER file names and columns | C* (the columns belong to the merged project-plus-abstract frame) |
| 83 | NSF download URL, `AbstractNarration`, `AwardAmount` "to date" | C |
| 84 | EDGAR: headcount in Item 1 prose; `dei:EntityNumberOfEmployees` voluntary and mostly empty | C (third-party) |

### Literature (§2.1, §2.10, §9)

| # | Claim | Outcome |
|---|---|---|
| 85 | Ma (2021) NVSQ, DOI 10.1177/0899764020968153; NTEE-from-text classifier | C (ma-ji/npo_classifier README; bib record gives 50(3):662–687) |
| 86 | Corral, Henderson & Segovia (2025) *J. Dev. Econ.* 172; validate on a pseudo-census | C (bib: 103377, DOI 10.1016/j.jdeveco.2024.103377; WB metadata abstract: pseudo-census design-based simulations) |
| 87 | Newhouse (2023) PRWP 10512, DOI 10.1596/1813-9450-10512; non-sampled areas do worse | C (WB metadata: WPS10512, 2023-06-28; abstract: "Estimates for sampled areas are significantly more accurate than those for non-sampled areas") |
| 88 | Sheehan et al. (2019) KDD, arXiv 1905.01627 | C (KDD 2019 DOI 10.1145/3292500.3330784; ermongroup repo) |
| 89 | AlphaEarth Foundations, arXiv 2507.22291 | C (title confirmed; 64-D from data and opengeos) |
| 90 | Rolf et al. (2021) *Nat. Commun.* 12:4392, DOI 10.1038/s41467-021-24638-z | C (CEGA bib record) |
| 91 | Cohere embed-v4 takes page-image (image) input | C (third-party code only: v2 API `image_url` content; Cohere docs unreachable) |

### Internal arithmetic

| # | Claim | Outcome |
|---|---|---|
| 92 | §3 ranking totals | C (every row sums correctly; ranges are scored at their lower bound) |

### Not checkable here, left as [UNVERIFIED] in the text

- IRS public-inspection status and GivingTuesday terms of use.
- CCEW £10k/£25k thresholds and the Part B threshold, against CC itself.
- SEC 10-K filer count, PPP loan count, IPEDS size, and Overture licence.
- Companies House, NZ and OSCR details.
- SS-4 lines 16–17, Form 1023 access and StatCan program names (from note 02).
- Whether ACNC financial reports are available in bulk.
- Whether `charitable_objects` are rarely amended.
