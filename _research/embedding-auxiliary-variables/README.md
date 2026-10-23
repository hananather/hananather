# Embeddings as auxiliary variables: a design-based benchmark

**Claim under test:** "Text embeddings of business descriptions, websites or area profiles can serve as covariates
for stratification, nonresponse adjustment, calibration and small area estimation. This is variance reduction,
not productivity."

**Approach.** Take a real finite population where every unit's survey value is known (IRS Form 990 non-profits).
Rebuild exactly what a statistical agency would hold on the design date: a register from the previous year with
codes, lagged administrative amounts and free-text descriptions, restricted to information filed by then. Redraw a
production-style survey thousands of times. Then compare text-assisted estimators against the strongest text-free
alternatives under a **pre-registered protocol** with validity gates, negative, positive and leakage controls, and
explicit pass/fail rules.

Start with `FINDINGS.md` (results and decision), then `notes/benchmark-protocol.md` (the design).

## Testbed

| Element | Choice | Real-world analogue |
|---|---|---|
| Population | New England full Form 990 filers in TY2022 and TY2023, N = 20,626 (development / proof of concept). Eight held-out US Census divisions (285k units) are reserved for confirmation. | National non-profit survey population |
| Frame (design date D = 2024-06-30) | IRS BMF snapshot 2023-08-16 (all codes); TY2022 return **only if filed by D**, else BMF amounts, TY2021 text, or name | Business register + prior-year tax data, as known at design time |
| Register tiers | F0 codes only · F1 + lagged revenue / expenses / assets / employees · **F1p + payroll (primary)** · F2 + full prior return · F2o (unmasked oracle) | BR-only frames; BR + tax summary; BR + PD7A payroll; charity returns (T3010) |
| Survey targets | TY2023: 10 headline variables not on the F1p register (gov. grants, contributions, program revenue, fundraising, officer comp., volunteers, and 4 yes/no items) + 4 register-type controls | Survey-only content of a non-profit survey |
| Text | TY2022 mission + program descriptions (median 71 words) | Registration "nature of business" text; charity program descriptions |
| Nonresponse | TY2023 return filed > 300 days after FYE (real behaviour: 61% of take-some, 13% of take-all units respond) | Late / non-returned questionnaires |
| Previous wave | TY2021 → TY2022, used to select the embedding and to train frozen models | Last cycle's survey |

Data source: GivingTuesday 990 data lake, public bucket `s3://gt990datalake-rawdata`.

## Layout

| Path | What |
|---|---|
| `FINDINGS.md` | Results, verdict, real-world interpretation, next steps |
| `notes/benchmark-protocol.md` | Pre-registered protocol v1.0, from a judge-panel synthesis of three independent drafts (`notes/design/`) |
| `prereg/deviations.md` | Every departure from the protocol, logged before the affected results were read |
| `notes/research/01–06` | Methods literature; applied precedents and NSO data sources; dataset scouting and red team; embedding models and Cohere; 990 parsing spec; the skeptic's case. Each is citation-verified. |
| `notes/review/` | Adversarial code and design review (18 confirmed findings, all fixed) |
| `notes/early-findings.md`, `notes/working-log.md` | Pilot findings and the decision log |
| `notebooks/01_data_and_red_team` | Population construction and about 45 data-quality, leakage and timing checks (pass/flag/fail) |
| `notebooks/02_benchmark_results` | Pilot v1 Monte Carlo (pre-protocol; kept for the record; known defects listed in protocol §0.3) |
| `notebooks/03_protocol_stage2_ne_verdict` | **Stage 2A: primary pairs P1/P2, gates G1–G9, attribution factorial, planted-signal power** |
| `notebooks/04_protocol_nonresponse_and_sae` | Stage 3 (late-filer nonresponse with take-all imputation) and Stage 4 (small area estimation) |
| `notebooks/05_protocol_stage2b_weak_register` | Stage 2B: the weak register F0 (codes and BMF amounts only) on its own design; S2 and gap closure S2b |
| `notebooks/06_protocol_stage2f_stratification` | Stage 2F: text at the design stage. Text-coded NTEE for missing codes (S4) and prediction strata (S5) |
| `src/` | Data: `fetch990`, `parse990_min`, `parse990` (an independent second parser, cross-checked), `parse_all`, `build_population`. Embeddings: `embed` (cached, hashed; TF-IDF/LSA, MiniLM, Cohere, random). Estimation: `survey`, `frame`, `mc`, `protocol`, `frozen`, `sae`, `analysis`. Runners: `stage2_ne` (2A), `stage2b_ne` (2B), `stage2f_ne` (2F), `stage34_ne` (3, 4). Tests: `test_harness` (11 tests, all pass). |
| `results/` | Small CSV/JSON outputs |

The raw XML and derived parquet files live outside the repository (`EMBAUX_DATA`, default `/tmp/claude-0/data`). The scripts in `src/` rebuild them from the public bucket. A full rebuild plus the New England runs takes a few CPU hours.

## Running the Cohere arm (needs user action)

1. Add `api.cohere.com` to the environment's allowed network domains.
2. Add a production `COHERE_API_KEY` as an environment secret. A trial key's 1,000 calls/month cannot complete one pass.
3. Embed once and cache: `embed.build(cache, "cohere_v4_full", "cohere", texts, "text_full", model="embed-v4.0", input_type="classification", dim=1536)`.
4. Add it as a secondary ladder rung. Under the protocol, an API model can never replace the pre-registered E*.

The Form 990 data contain no PDFs or images. The protocol's multimodal arm uses UK Charity Commission accounts PDFs or archived website screenshots, each compared against structured extraction of the numbers on the page.
