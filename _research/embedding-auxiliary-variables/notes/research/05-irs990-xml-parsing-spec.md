# 05 — IRS Form 990 e-file XML: tested parsing specification

Date: 2026-10-23. Scope: full Form 990 (`ReturnData/IRS990`) public e-file XML from the GivingTuesday 990 data lake.
Deliverable: `src/parse990.py` (754 lines, needs only lxml; `tokenizers` is optional and used only by the self-test).

**Bottom line**
- Every element path we need is **unchanged across all 16 schema versions tested, 2019v5.0 to 2023v6.0**. Parsing is not where this project's risk lies. Three things are: missing-vs-zero semantics, frame selection, and the numbers inside the text.
- Text coverage:
  - Mission text is ~100% filled. Part III line 4a is filled for 99.3%.
  - Line 4b is filled for 30%, 4c for 21%, 4d for 12%.
  - A usable website is present for only 65% of returns: 21% of filers write "N/A" or "NONE".
- Leakage risk is concentrated in the **program accomplishment text (Part III 4a–4d)**:
  - 37% of these texts contain a number.
  - 14% contain a count of 1,000 or more.
  - Mentions scale with organisation size: money mentions rise from 2.3% (<$100k revenue) to 20.3% (>$100M); counts of 1,000+ from 5% to 51%.
  - Word count alone correlates with log revenue at Spearman 0.32–0.36.
- Mission text is nearly number-free (8%; 1% have a count of 1,000+). Year over year it is identical for 88% of organisations, so a lagged (t−1) mission text is realistic as a frame variable.
- Download cost, TY2022+TY2023, one return per EIN, group returns excluded:
  - US: 690,947 returns, 28.4 GB raw XML (≈5.2 GB gzipped), ≈77 min at the measured 149 files/s.
  - A mid-sized state (WI or MO): ≈13k returns, ≈0.55 GB raw (≈0.10 GB gzipped), under 2 min.

---

## 1. Sources and test corpora

**Data source.** All paths are under `https://gt990datalake-rawdata.s3.amazonaws.com/`. The bucket is public and reachable from the sandbox by curl.
- `Indices/990xmls/yearly/{TY}_efiledata_xmls_created_on_{DATE}.csv`
  - One file per tax year, re-issued as snapshots.
  - Sizes for the 2026-08-25 snapshot: TY2021 298.7 MB, TY2022 304.0 MB, TY2023 310.1 MB.
  - Columns: `BuildTs, DAF, DateSigned, DocStatus, EIN, FileSha256, FileSizeBytes, FormType, GrossReceipts, GroupAffiliatesIncluded, GroupExemptionNumber, GroupReturnForAffiliates, IndexedOn, LegalDomicileCountry, LegalDomicileState, ObjectId, OrgType, OrganizationName, ReturnTs, ReturnVersion, SubmittedOn, TaxPeriod, TaxPeriodBeginDate, TaxPeriodEndDate, TaxStatus, TaxYear, TotalAssetsBkEOY, TotalExpensesCY, TotalLiabilitiesBkEOY, TotalNetAssetsBkEOY, TotalRevenueCY, URL, Website, YearFormed, ZipFile`.
  - `FileSizeBytes` is the uncompressed XML size, so download volume can be estimated without downloading.
- `EfileData/XmlFiles/{ObjectId}_public.xml`: one return per file.

**Index snapshots grow as late filers arrive.** The TY2022 index was 199.6 MB on 2025-05-30, 303.1 MB on 2025-10-08 and 304.0 MB on 2026-06-04. The TY2024 index is still growing (32 MB on 2025-05-30, 267.6 MB on 2026-08-25).
- As of the 2026-08-25 snapshot, **TY2022+TY2023 is the most recent essentially complete pair of years**.
- An NSO using this source would see a 1–2 year publication lag. `SubmittedOn` lets you simulate a "who has filed by date d" response process.

**Form 990 rows in the index:**

| TY | 990 rows | unique EINs | ReturnVersion mix |
|---|---|---|---|
| 2021 | 336,521 | 330,745 | 2021v4.2 169,548; v4.0 114,740; v4.1 52,233 |
| 2022 | 349,091 | 342,603 | 2022v5.0 340,580; v4.1 7,893; v4.0 618 |
| 2023 | 355,712 | 349,575 | 2023v5.1 132,346; v6.0 112,337; v5.0 102,178; v4.0 8,851 |

TY2019 (2019v5.0/5.1/5.2) and TY2020 (2020v4.0/4.1/4.2) rows came from 8 MB HTTP Range slices (head and middle) of those years' index files.

**Test corpora** (all under `/tmp/claude-0/data/xml_samples/`):

| corpus | n | design |
|---|---|---|
| stratified core | 45 | TY2021/22: every ReturnVersion × 5 revenue classes (<100k, 100k–1M, 1M–10M, 10M–100M, >100M); plus a group return, an amended return, a final return, a 501(c)(4), a 501(c)(6) and one 2.4 MB file. Plus one file per 2019/2020/2023 version. Manifest: `sample_manifest.csv`. |
| by-version coverage | 2,400 | 150 random full 990s × 16 versions, 2019v5.0 to 2023v6.0 (`coverage/`, 95 MB; manifest `coverage/coverage_manifest.csv`) |
| bulk TY2022 | 5,000 | random draw (seed 11) from the 341,956-file US TY2022 download in `/tmp/claude-0/data/xml/2022/` |
| year pairs | 3,000 EINs | New England EINs with both TY2022 and TY2023 returns (`/tmp/claude-0/data/xml/{2022,2023}`) |
| negative controls | 3 | one each of 990-EZ, 990-PF and 990-T (`non990/`); all must return None |

**Cross-references used to verify element-to-line mappings:**
- IRSx 0.5.1 metadata (`irsx/metadata/variables.csv`, 3,687 xpaths with form line, IRS type and version range). Taken from the PyPI wheel; read as data, not installed or run.
- Nonprofit Open Data Collective master concordance (`concordance.csv`, 6,864 xpaths).
- Neither source covers 2019+ versions reliably:
  - The concordance's `versions` lists stop at 2016v3.0 for nearly all fields. Only 50 xpaths, all new header elements, carry 2020–2023 versions.
  - IRSx's version ranges are coarse; for example, `NoDonorRestrictionNetAssetsGrp` is listed as "2019–2019" yet is present in every 2020–2023 version.
- **The 16-version empirical presence matrix is therefore the primary evidence for version stability** (`coverage/version_tag_matrix.csv`).

## 2. Document-level rules (verified)

1. **Root and namespace.**
   - The root is `{http://www.irs.gov/efile}Return`, carrying the `returnVersion` attribute (e.g., `2022v5.0`).
   - Children are `ReturnHeader` and `ReturnData`. `ReturnData` holds `IRS990` plus zero or more `IRS990Schedule{A..R}`, and occasionally `AffiliateListing`.
   - Files begin with a UTF-8 BOM. lxml `fromstring(bytes)` handles it.
   - The parser accepts namespace-less XML too (tested).
2. **Full 990 detection.** Use the presence of `ReturnData/IRS990`; `ReturnHeader/ReturnTypeCd` = `990` agrees. 990-EZ, 990-PF and 990-T return None (tested).
3. **Use direct children of `IRS990` only.** These local names are reused inside groups:
   - `Desc`: top level is Part III 4a; also inside ProgSrvcAccomActy2Grp, ProgramServiceRevenueGrp and others.
   - `ExpenseAmt`.
   - `FundraisingAmt`: top level is Part VIII 1c; inside `*Grp` it is Part IX column D.
   - `TotalAmt`.
   - A descendant search such as `//Desc` silently returns the wrong line.
4. **Part III 4a is not grouped.** `ActivityCd`, `ExpenseAmt`, `GrantAmt`, `RevenueAmt` and `Desc` are bare children of `IRS990`. 4b is `ProgSrvcAccomActy2Grp`, 4c is `ProgSrvcAccomActy3Grp`, and 4d is the repeating `ProgSrvcAccomActyOtherGrp`. This holds in all 16 versions (`Desc` present in 100% of returns in every version; no other "Acty" group exists).
5. **Booleans.**
   - `BooleanType` values are `true`/`false` or `1`/`0`, depending on the filer's software. Example counts for `LobbyingActivitiesInd` in the 2,400-file set: false 1,093; 0 737; 1 52; true 22.
   - `CheckboxType` is `X` when checked and the element is absent otherwise. This applies to `Organization501c3Ind`, `Organization501cInd`, `AmendedReturnInd`, `FinalReturnInd`, `TypeOfOrganization*Ind` and `MethodOfAccounting*Ind`.
   - Map all of these to True/False/None. For checkboxes, None means unchecked.
6. **501(c) type.** 501(c)(3) is its own checkbox. Any other subsection is `<Organization501cInd organization501cTypeTxt="6">X</…>`. Observed type values: 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 19, 25, 2. The parser outputs `subsection` as `03`, `06`, `4947a1` or `527`.
7. **Amounts** are integers with no decimals (0 of 2,400 files had any). Negatives occur legitimately, e.g., `CYRevenuesLessExpensesAmt` (830 of 2,400 files), `TotalRevenueColumnAmt` and `BOYAmt`/`EOYAmt`.
8. **Missing vs zero (critical).**
   - Blank amount lines are usually omitted, but some filer software writes explicit zeros. For returns filed under schema 2022v4.0, 102 of 130 `MembershipDuesAmt` elements were literal `0`, versus 6 of 36 for 2022v5.0. The pattern repeats for GovernmentGrantsAmt, FundraisingAmt and FederatedCampaignsAmt, and the 2022v4.0 files also always write `TotalOtherProgSrvc*Amt`.
   - So **element presence encodes the filer's software, not the organisation**. Never use missingness indicators of Part VIII/IX lines as covariates. Use `parse_return(..., zero_fill=True)`, which zero-fills the 59 amount fields in `ZERO_IF_MISSING`.
   - Do **not** zero-fill:
     - prior-year (`*_py`) columns, which are genuinely absent for new organisations;
     - `TotalVolunteersCnt`, which is missing for 27% of returns, and blank ≠ 0 there;
     - balance-sheet EOY items.
9. **Text caps (observed, consistent with schema length limits).**
   - `ActivityOrMissionDesc` and `MissionDesc` never exceed exactly 1,000 characters. 0.36% and 0.46% of them sit at 995 characters or more, i.e. they were truncated, and filers continue them in Schedule O ("CONT'D ON SCH O").
   - Part III `Desc` maxes out at exactly 9,000 characters.
10. **Header.**
    - `ReturnHeader/TaxYr` matched the index `TaxYear` in 2,400 of 2,400 returns.
    - Header blocks that differ by version are irrelevant to us: `FilingSecurityInformation`/IP data appears only in 2019v5.0/5.1; `TrustedCustomerGrp`, `SigningOfficerGrp` and `IRSResponsiblePrtyInfoCurrInd` appear only from 2020.
    - Personal data (signing officer name, SSN element, preparer PTIN) is in the header of the public files. **The parser deliberately extracts none of it.** It keeps only `PreparerFirmEIN`, a business identifier, which is useful for red-teaming preparer boilerplate.
11. **Security.** The parser uses `resolve_entities=False, no_network=True, load_dtd=False`. An XXE test document left the entity unresolved (`&x;`).
12. **Malformed XML.** The parser retries in lxml recover mode and sets `xml_recovered=True`. 0 of 7,445 real files needed it; a file truncated at 50% was recovered and flagged. Empty input returns None.

## 3. Field specification

All paths are relative to `Return/ReturnData/IRS990` unless marked `HDR:`, which means relative to `Return/ReturnHeader`.

Coverage column: share of returns with a non-missing value, for the 16-version stratified set (n=2,445) / bulk TY2022 (n=5,000). Where a second pair follows, it is the share True or nonzero, in the same order.

### 3.1 Header and identification
| output | path | line | coverage | notes |
|---|---|---|---|---|
| `ein` | HDR:`Filer/EIN` | D | 1.000 / 1.000 | 9-digit string with leading zeros; keep it as a string |
| `tax_year` | HDR:`TaxYr` | A | 1.000 / 1.000 | int |
| `tax_period_begin`, `tax_period_end`, `tax_period` | HDR:`TaxPeriodBeginDt`, `TaxPeriodEndDt` | A | 1.000 | `tax_period` = YYYYMM of the end date |
| `return_ts` | HDR:`ReturnTs` | — | 1.000 | for deduplication: keep the latest per EIN×TY |
| `org_name` | HDR:`Filer/BusinessName/BusinessNameLine1Txt` + `Line2Txt` | C | 1.000 | Line 2 present in ≈29% of the core sample |
| `addr_state`, `addr_city`, `zip5`, `addr_foreign_country` | HDR:`Filer/USAddress/{StateAbbreviationCd,CityNm,ZIPCd}`, `Filer/ForeignAddress/CountryCd` | E | 0.999 | preparer address uses the same tag names under `PreparerFirmGrp`, so do not search descendants |
| `preparer_firm_ein` | HDR:`PreparerFirmGrp/PreparerFirmEIN` | — | 0.794 / 0.879 | absent if self-prepared |
| `return_version` | `Return/@returnVersion` | — | 1.000 | |
| `website_raw` → `website` | `WebsiteAddressTxt` | J | 0.644 / 0.658 after placeholder removal | the element exists in ≈85%. **20.9% of returns say N/A, NONE or NA.** `website` is normalised (lowercase, scheme/`www.` stripped; emails and dotless strings dropped) |
| `formation_year` | `FormationYr` | L | 0.925 / 0.940 | |
| `legal_domicile_state` | `LegalDomicileStateCd` | M | 0.948 / 0.951 | **5.0% missing**; differs from the address state for 5.6% (DE 3,165, DC 1,392) |
| `group_return` | `GroupReturnForAffiliatesInd` | H(a) | 1.000 (True 0.2%) | BooleanType |
| `group_exemption_num` | `GroupExemptionNum` | H(c) | 0.031 | |
| `amended`, `final_return`, `initial_return`, `address_change`, `name_change`, `application_pending` | `AmendedReturnInd`, `FinalReturnInd`, `InitialReturnInd`, `AddressChangeInd`, `NameChangeInd`, `ApplicationPendingInd` | B | checked: amended 2.1/1.9%, final 0.9/0.4% | CheckboxType; the index `DocStatus` is derived from these |
| `is_501c3`, `is_501c_other`, `is_4947a1`, `is_527`, `subsection` | `Organization501c3Ind`, `Organization501cInd/@organization501cTypeTxt`, `Organization4947a1NotPFInd`, `Organization527Ind` | I | c3 0.783 / 0.772; other 501(c) 0.216 / 0.227 | |
| `form_corp/trust/assoc/other` | `TypeOfOrganization{Corp,Trust,Assoc,Other}Ind` | K | corp 0.88–0.90 | |
| `gross_receipts` | `GrossReceiptsAmt` | G | 1.000 | |

### 3.2 Text fields (embedding candidates) and their leakage risk
| output | path | line | coverage | number leakage (share of non-empty texts; bulk TY2022) |
|---|---|---|---|---|
| `mission_p1` | `ActivityOrMissionDesc` | I-1 | 1.000 / 1.000 | any number 7.9%; money 0.2%; count ≥1k 1.1% → **low** |
| `mission_p3` | `MissionDesc` | III-1 | 0.989 / 0.992 | 8.3%; 0.2%; 1.1% → **low** |
| `prog4a_desc` | `Desc` (direct child) | III-4a | 0.993 / 0.993 | 33.4%; 3.2%; 11.2% → **high** |
| `prog4b_desc` (+ `_expense/_grants/_revenue`) | `ProgSrvcAccomActy2Grp/{Desc,ExpenseAmt,GrantAmt,RevenueAmt}` | III-4b | 0.276 / 0.304 | 35.2%; 4.3%; 11.0% |
| `prog4c_desc` (+ amounts) | `ProgSrvcAccomActy3Grp/…` | III-4c | 0.200 / 0.209 | 33.3%; 4.8%; 10.5% |
| `prog4d_desc`, `n_prog4d` | `ProgSrvcAccomActyOtherGrp/Desc` (repeating) | III-4d | 0.119 / 0.118 | 23.9%; 4.1%; 6.8% |
| `prog_rev_desc`, `prog_rev_business_codes`, `n_prog_rev_lines` | `ProgramServiceRevenueGrp/{Desc,BusinessCd}` (repeating) | VIII-2a–e | 0.587 / 0.605 | 1.2% of texts have numbers, **but the field's presence implies program revenue > 0**: it reveals the outcome by existing |
| `sched_o_all`, `n_sched_o` | `ReturnData/IRS990ScheduleO/SupplementalInformationDetail/{FormAndLineReferenceDesc,ExplanationTxt}` | Sch O | 1.000 / 0.998 | 83.8%; 9.4%; 30.0% → **very high**. Mostly governance boilerplate: 8,508 of 12,647 references in the 2,400-file set are to Part VI |
| `sched_o_program` | Schedule O entries whose reference parses to Part III, or to Part I line 1 | — | 0.132 / 0.147 | 44.9%; 6.5%; 14.6% |
| `mission_resolved` | p1 ∪ p3 ∪ Schedule O entries for (I-1, III-1), case-insensitively deduplicated, pointers dropped | — | 0.998 / 0.994 | 10.4%; 0.4%; 1.7% → **recommended main text** |
| `program_desc_all` | 4a ∪ 4b ∪ 4c ∪ all 4d | — | 0.990 / 0.987 | 37.3%; 4.6%; 13.6% |
| `text_profile` | mission_resolved ∪ program_desc_all ∪ sched_o_program | — | 1.000 / 0.998 | 41.8%; 5.5%; 14.9% |
| `mission_p{1,3}_is_pointer` | flag | — | True 1.5% / 1.3% (bulk) | "SEE SCHEDULE O", "SEE PART III, LINE 1", "SEE ATTACHED STATEMENT" |

How Schedule O references are matched:
- References are free text, e.g., "FORM 990, PAGE 2, PART III, LINE 4D", "PART III LINE 4A", "FORM 990 PART I SECTION 1 LINE 1".
- `classify_sched_o_ref` extracts (Part roman numeral, Line + letter).
- About 10% of entries carry no "PART" token and stay unclassified.

### 3.3 Part I summary (CY and PY)
Paths: `CY…`/`PY…` children of IRS990. The CY coverage is 1.000 in both corpora unless stated.

| output (cy / py) | tag (CY…/PY…) | line | PY coverage (strat / bulk) |
|---|---|---|---|
| `contrib_*` | `ContributionsGrantsAmt` | I-8 | 0.867 / 0.875 |
| `prog_rev_*` | `ProgramServiceRevenueAmt` | I-9 | 0.750 / 0.738 |
| `invest_inc_*`, `other_rev_*` | `InvestmentIncomeAmt`, `OtherRevenueAmt` | I-10, I-11 | 0.80 / 0.72 |
| `revenue_*` | `TotalRevenueAmt` | I-12 | 0.955 / 0.956 |
| `grants_paid_*`, `member_benefits_*` | `GrantsAndSimilarPaidAmt`, `BenefitsPaidToMembersAmt` | I-13, I-14 | 0.57 / 0.46 |
| `salaries_*` | `SalariesCompEmpBnftPaidAmt` | I-15 | 0.773 / 0.770 |
| `prof_fundraising_*` | `TotalProfFndrsngExpnsAmt` | I-16a | 0.46 |
| `fundraising_exp_cy` | `CYTotalFundraisingExpenseAmt` (CY only) | I-16b | — |
| `other_exp_*`, `expenses_*`, `rev_less_exp_*` | `OtherExpensesAmt`, `TotalExpensesAmt`, `RevenuesLessExpensesAmt` | I-17, I-18, I-19 | expenses_py 0.955 / 0.953 |
| `assets_boy/eoy`, `liabilities_boy/eoy`, `net_assets_boy/eoy` | `TotalAssetsBOYAmt`/`EOYAmt`, `TotalLiabilitiesBOYAmt`/`EOYAmt`, `NetAssetsOrFundBalancesBOYAmt`/`EOYAmt` | I-20 to I-22 | BOY: 0.978 / 0.977 assets; 0.82 liabilities |
| `employees_p1` | `TotalEmployeeCnt` | I-5 | 1.000 (nonzero 55.0% / 57.8%) |
| `volunteers` | `TotalVolunteersCnt` | I-6 | **0.732 / 0.729**; blank ≠ 0; an estimate |
| `voting_members`, `independent_members` | `VotingMembersGoverningBodyCnt`, `VotingMembersIndependentCnt` | I-3, I-4 | 1.000 |
| `ubi_gross`, `ubi_net` | `TotalGrossUBIAmt`, `NetUnrelatedBusTxblIncmAmt` | I-7a, I-7b | 1.000 / 0.60 |

### 3.4 Part III amounts, Part V
| output | path | line | coverage |
|---|---|---|---|
| `prog4a_expense/grants/revenue/code` | `ExpenseAmt`, `GrantAmt`, `RevenueAmt`, `ActivityCd` (direct children) | III-4a | 0.854 / 0.843; 0.335; 0.553; code 0.001 (essentially unused) |
| `prog4d_expense/grants/revenue` | `TotalOtherProgSrvc{Expense,Grant,Revenue}Amt` (falls back to the sum of the 4d groups) | III-4d | 0.21 / 0.15 |
| `prog_exp_total_p3` | `TotalProgramServiceExpensesAmt` | III-4e | 0.880 / 0.861 |
| `new_program`, `significant_change`, `sched_o_part3` | `SignificantNewProgramSrvcInd`, `SignificantChangeInd`, `InfoInScheduleOPartIIIInd` | III-2, III-3, heading | True ≈1% / 1% / 28–31% |
| `employees_w3` | `EmployeeCnt` | V-2a | 1.000; equals I-5 in 100% of returns |
| `employment_tax_filed` | `EmploymentTaxReturnsFiledInd` | V-2b | 0.59–0.62 |

### 3.5 Part IV checklist and other booleans
Shown as present / True, stratified then bulk. All are BooleanType unless marked.

| output | tag | line | coverage |
|---|---|---|---|
| `c3_described` | `DescribedInSection501c3Ind` | IV-1 | 1.0 / 0.78 |
| `sched_b_required` | `ScheduleBRequiredInd` | IV-2 | 1.0 / 0.48–0.50 |
| `political` | `PoliticalCampaignActyInd` | IV-3 | 1.0 / 0.009 |
| `lobbying` | `LobbyingActivitiesInd` | IV-4 | **0.79 / 0.78**, True 0.031 / 0.038; asked of 501(c)(3)s only |
| `proxy_tax` | `SubjectToProxyTaxInd` | IV-5 | 0.96–0.97 |
| `donor_advised_fund` | `DonorAdvisedFundInd` | IV-6 | 1.0 / 0.007–0.009 |
| `endowments` | `DonorRstrOrQuasiEndowmentsInd` | IV-10 | 1.0 / 0.11 |
| `school` | `SchoolOperatingInd` | IV-13 | 1.0 / 0.05 |
| `foreign_office`, `foreign_activities` | `ForeignOfficeInd`, `ForeignActivitiesInd` | IV-14a, 14b | 1.0 / 0.016–0.023 and 0.035–0.045 |
| `foreign_grants_orgs`, `foreign_grants_indiv` | `MoreThan5000KToOrgInd`, `MoreThan5000KToIndividualsInd` | IV-15, IV-16 | 1.0 |
| `professional_fundraising` | `ProfessionalFundraisingInd` | IV-17 | 1.0 / 0.023–0.028 |
| `fundraising_events` | `FundraisingActivitiesInd` | IV-18 | 1.0 / 0.19 |
| `gaming` | `GamingActivitiesInd` | IV-19 | 1.0 / 0.03 |
| `hospital` | `OperateHospitalInd` | IV-20a | 1.0 / 0.005–0.007 |
| `hospital_audited_fs` | `AuditedFinancialStmtAttInd` | IV-20b | asked of hospitals only |
| `grants_orgs_us`, `grants_indiv_us` | `GrantsToOrganizationsInd`, `GrantsToIndividualsInd` | IV-21, IV-22 | 1.0 / 0.13–0.15 and 0.10–0.11 |
| `tax_exempt_bonds` | `TaxExemptBondsInd` | IV-24a | 1.0 |
| `terminated` | `TerminateOperationsInd` | IV-31 | 1.0 |
| `related_entity`, `related_org_ctrl` | `RelatedEntityInd`, `RelatedOrganizationCtrlEntInd` | IV-33, IV-34 | 1.0 |
| `local_chapters` | `LocalChaptersInd` | VI-10a | 1.0 |
| `conflict_policy`, `whistleblower_policy` | `ConflictOfInterestPolicyInd`, `WhistleblowerPolicyInd` | VI-12a, VI-13 | 1.0 |
| `acct_cash`, `acct_accrual` | `MethodOfAccounting{Cash,Accrual}Ind` | XII-1 | checkbox |
| `fs_compiled_reviewed`, `fs_audited` | `AccountantCompileOrReviewInd`, `FSAuditedInd` | XII-2a, 2b | fs_audited True 0.34 / 0.38 |
| `single_audit_required` | `FederalGrantAuditRequiredInd` | XII-3a | 0.89–0.90 |

The `schedules` field lists the attached schedules (e.g., `A,B,D,O`). Schedule O was attached to 100% of the 2,400 coverage-set returns.

### 3.6 Part VIII (revenue)
Use `zero_fill=True` before analysis.

| output | path | line | element present (strat / bulk) |
|---|---|---|---|
| `federated_campaigns` | `FederatedCampaignsAmt` | VIII-1a | 0.137 / 0.084 |
| `membership_dues` | `MembershipDuesAmt` | VIII-1b | 0.249 / 0.202 |
| `fundraising_events_contrib` | `FundraisingAmt` (direct child) | VIII-1c | 0.231 / 0.186 |
| `related_org_contrib` | `RelatedOrganizationsAmt` | VIII-1d | 0.142 / 0.093 |
| `gov_grants` | `GovernmentGrantsAmt` | VIII-1e | 0.385 / 0.347 (nonzero 30.6% / 30.7%) |
| `other_contrib` | `AllOtherContributionsAmt` | VIII-1f | 0.715 / 0.696 |
| `noncash_contrib` | `NoncashContributionsAmt` | VIII-1g (included in 1a–1f) | 0.224 / 0.176 |
| `total_contrib_p8` | `TotalContributionsAmt` | VIII-1h | 0.859 / 0.850 |
| program revenue lines | `ProgramServiceRevenueGrp/{Desc,BusinessCd,TotalRevenueColumnAmt,…}` | VIII-2a–e | 0.595 / 0.613 have at least one line |
| `prog_rev_other_p8` | `TotalOthProgramServiceRevGrp/TotalRevenueColumnAmt` | VIII-2f | |
| `total_prog_rev_p8` | `TotalProgramServiceRevenueAmt` | VIII-2g | |
| `invest_income_p8` | `InvestmentIncomeGrp/TotalRevenueColumnAmt` | VIII-3 | |
| `fundraising_gross_income`, `fundraising_direct_exp`, `net_fundraising_p8` | `FundraisingGrossIncomeAmt`, `FundraisingDirectExpensesAmt`, `NetIncmFromFundraisingEvtGrp/TotalRevenueColumnAmt` | VIII-8a, 8b, 8c | |
| `gaming_gross_income`, `net_gaming_p8` | `GamingGrossIncomeAmt`, `NetIncomeFromGamingGrp/TotalRevenueColumnAmt` | VIII-9a, 9c | |
| `inventory_gross_sales` | `GrossSalesOfInventoryAmt` | VIII-10a | |
| `revenue_p8`, `revenue_related_p8`, `revenue_ubi_p8`, `revenue_excluded_p8` | `TotalRevenueGrp/{TotalRevenueColumnAmt,RelatedOrExemptFuncIncomeAmt,UnrelatedBusinessRevenueAmt,ExclusionAmt}` | VIII-12 A–D | 1.000 (column A) |

IRSx's description of `MembershipDuesAmt` is wrong ("Outside fundraising or commercial co-ventures"); its line number, 1b, is right.

### 3.7 Part IX (functional expenses) and Part X
| output | path | line | coverage |
|---|---|---|---|
| `func_exp_total`, `func_exp_program`, `func_exp_mgmt`, `func_exp_fundraising` | `TotalFunctionalExpensesGrp/{TotalAmt,ProgramServicesAmt,ManagementAndGeneralAmt,FundraisingAmt}` | IX-25 A–D | A 1.000; B–D ≈0.95. Columns B–D are mandatory only for 501(c)(3), 501(c)(4) and 4947(a)(1) filers |
| `officer_comp` | `CompCurrentOfcrDirectorsGrp/TotalAmt` | IX-5 | 0.551 / 0.557 (nonzero 0.43–0.45) |
| `other_salaries`, `pension_contrib`, `other_benefits`, `payroll_taxes` | `OtherSalariesAndWagesGrp`, `PensionPlanContributionsGrp`, `OtherEmployeeBenefitsGrp`, `PayrollTaxesGrp` (each `/TotalAmt`) | IX-7 to IX-10 | 0.38–0.67 |
| `grants_domestic_orgs`, `grants_domestic_indiv`, `grants_foreign` | `GrantsToDomesticOrgsGrp`, `GrantsToDomesticIndividualsGrp`, `ForeignGrantsGrp` (each `/TotalAmt`) | IX-1 to IX-3 | |
| `fees_prof_fundraising`, `advertising`, `occupancy`, `travel`, `depreciation` | `FeesForServicesProfFundraising`, `AdvertisingGrp`, `OccupancyGrp`, `TravelGrp`, `DepreciationDepletionGrp` (each `/TotalAmt`) | IX-11e, 12, 16, 17, 22 | |
| `cash_eoy`, `savings_eoy`, `land_bldg_net_eoy`, `invest_public_eoy` | `CashNonInterestBearingGrp`, `SavingsAndTempCashInvstGrp`, `LandBldgEquipBasisNetGrp`, `InvestmentsPubTradedSecGrp` (each `/EOYAmt`) | X-1, 2, 10c, 11 | |
| `assets_eoy_p10`, `liabilities_eoy_p10`, `net_assets_eoy_p10` | `TotalAssetsGrp`, `TotalLiabilitiesGrp`, `TotalNetAssetsFundBalanceGrp` (each `/EOYAmt`) | X-16, 26, 32 | 1.000 |
| `n_indiv_comp_gt100k`, `n_contractors_gt100k`, `comp_reportable_org` | `IndivRcvdGreaterThan100KCnt`, `CntrctRcvdGreaterThan100KCnt`, `TotalReportableCompFromOrgAmt` | VII-A-2, VII-B-2, VII-A-1d | 0.71 / 0.58 / 0.75 |

## 4. Version quirks (16-version presence matrix, 150 returns per version)

- **No target field changes path or disappears between 2019v5.0 and 2023v6.0.**
  - The only tags with structural zeros by version are header security and identity elements: `FilingSecurityInformation`/`IPAddress` only in 2019v5.0/5.1; `TrustedCustomerGrp`, `SigningOfficerGrp`, `SSN` and `IRSResponsiblePrtyInfoCurrInd` from 2020 onward.
  - All other flagged tags are rare lines, with presence of 0–15% in some versions.
- **Filer-software fingerprints show up as version differences.** TY2022 returns under 2022v4.0 (618 in the index; 150 sampled):
  - write explicit zeros on blank lines (`FederatedCampaignsAmt` present 87% vs 1–17% in the other 15 versions);
  - write mission text in mixed case (ALL-CAPS share 7.7%, vs 76% for 2022v5.0);
  - always include `InvestmentExpenseAmt`.
  - Treat `return_version` as a nuisance or batch variable, not a covariate.
- From IRSx metadata (not re-verified here beyond presence): the FASB 117 → ASU 2016-14 net-asset lines (`NoDonorRestrictionNetAssetsGrp`/`DonorRestrictionNetAssetsGrp` replacing unrestricted/temporarily/permanently restricted) came in with the 2018–2019 schemas; `SubjToTaxRmnrtnExPrchtPymtInd` (Part V-15) and `SubjectToExcsTaxNetInvstIncInd` (V-16) start in 2018. All of these are present throughout 2019–2023 in our data.
- 2023 has four concurrent versions (4.0, 5.0, 5.1, 6.0) for one tax year. Version reflects when the return was filed, not the tax year.

## 5. Text properties relevant to embeddings (bulk TY2022, n=5,000)

| field | present | words p10 / p50 / p90 / p99 / max | MiniLM WordPiece p50 / p90 | share >256 tok | ALL-CAPS | pointer |
|---|---|---|---|---|---|---|
| mission_p1 | 1.000 | 5 / 18 / 46 / 124 / 184 | 24 / 59 | 0.000 | 0.764 | 0.010 |
| mission_p3 | 0.992 | 6 / 23 / 55 / 129 / 184 | 29 / 70 | 0.000 | 0.764 | 0.011 |
| prog4a_desc | 0.993 | 7 / 25 / 105 / 460 / 1,345 | 33 / 135 | 0.032 | 0.760 | 0.005 |
| program_desc_all | 0.987 | 9 / 33 / 196 / 777 / 3,216 | 42 / 246 | 0.095 | 0.760 | — |
| sched_o_program | 0.147 | 5 / 27 / 211 / 1,309 / 3,510 | 35 / 280 | 0.107 | 0.787 | — |
| sched_o_all | 0.998 | 21 / 99 / 366 / 1,196 / 6,688 | 123 / 461 | 0.241 | 0.710 | — |
| mission_resolved | 0.994 | 7 / 29 / 80 / 178 / 2,577 | 37 / 101 | 0.008 | 0.761 | — |
| text_profile | 0.998 | 21 / 71 / 273 / 874 / 3,670 | 88 / 342 | 0.149 | 0.744 | — |

The stratified 16-version set gives near-identical figures; see `selftest_output.txt`.

- **Truncation.**
  - The bundled MiniLM `tokenizer.json` ships with truncation and padding fixed at 128. `src/embed.py` correctly overrides truncation to 256.
  - Even at 256, 9.5% of program texts and 14.9% of `text_profile` are cut off. Mission text never is.
  - Cohere embed-v4's longer context would remove this; its exact limit is [UNVERIFIED in this session].
- **Case.**
  - About 76% of texts are ALL CAPS. The share depends only weakly on size: 0.76 / 0.73 / 0.79 / 0.87 / 0.81 across the five revenue classes.
  - MiniLM's tokenizer lowercases (`do_lower_case: true`), so case doesn't matter there. For cased API models, normalise case or test case-folded input.
- **Boilerplate.** 3.5% of `mission_p1` values are exact duplicates across EINs. They are mostly pointers ("SEE SCHEDULE O" ×56 in 5,000) or generic ("EDUCATION").
- **Preparers.** 88% of returns have a paid preparer firm. 17% come from firms with at least 5 clients in the same 5,000-return sample, so preparer-written boilerplate can form clusters in embedding space.

**Size gradient (outcome leakage through text), program_desc_all by revenue class:**

| revenue class | money mention | count ≥1,000 | any number | mean words |
|---|---|---|---|---|
| <100k | 0.023 | 0.051 | 0.199 | 35 |
| 100k–1M | 0.031 | 0.086 | 0.305 | 60 |
| 1M–10M | 0.058 | 0.161 | 0.452 | 109 |
| 10M–100M | 0.126 | 0.332 | 0.591 | 188 |
| >100M | 0.203 | 0.507 | 0.580 | 258 |

Spearman correlation of word count with log10 revenue:
- `mission_p1`: 0.067
- `program_desc_all`: 0.317
- `text_profile`: 0.355
- `n_programs`: 0.174
- `n_sched_o`: 0.330

So **text length alone is a size proxy**, and an embedding of program text partly encodes verbosity and quoted counts.

**Year-to-year stability** (3,000 New England EINs, TY2022 → TY2023; texts upper-cased and whitespace-collapsed before comparing):

| field | identical | only numbers changed | text changed |
|---|---|---|---|
| mission_p1 | 0.882 | 0.002 | 0.117 |
| mission_p3 | 0.876 | 0.002 | 0.122 |
| program_desc_all | 0.675 | 0.027 | 0.298 |
| text_profile | 0.622 | 0.025 | 0.353 |
| website | 0.972 | 0.000 | 0.028 |

## 6. Accounting identities (data-quality checks), bulk TY2022 n=5,000

- Identities that always held, which suggests the e-file system enforces them (an inference): I-12 = VIII-12A, I-18 = IX-25A, I-8 = VIII-1h, I-9 = VIII-2g, I-20 = X-16, I-22 = X-33, I-5 = V-2a, VIII-1h = Σ(1a..1f). All passed in 100% of returns, in both corpora.
- Identities that are evidently not enforced:
  - III-4e = IX-25B: 96.5% pass (evaluable for c3/c4 filers only, 81.3% of returns).
  - IX-25 A = B+C+D: 98.4%.
  - III-4e = 4a+4b+4c+4d: 99.8%.

**Implication:** the consistent "cross-part" totals are exactly equal, so they add no independent information. Pick one definition per target (e.g., y = `revenue_cy` = `revenue_p8`). Expect a 2–4% filer-error rate on the functional-expense split.

## 7. Module: `src/parse990.py`

- `parse_return(xml_bytes, zero_fill=False) -> dict | None`
  - Returns 200 fields: clean snake_case names, Python `int`/`bool`/`str`, or None.
  - Returns None for non-990 documents, empty input or unparseable bytes.
  - `zero_fill=True` sets the 59 Part I-CY/III/VIII/IX amount fields in `ZERO_IF_MISSING` to 0 when absent.
- Helpers:
  - `clean_text`: NFKC normalisation, removes control and zero-width characters, collapses whitespace; the placeholders NONE, N/A, NA, -, ., 0 become None.
  - `mask_numbers`: replaces numbers with `[AMT]`, `[PCT]` and `[NUM]`; keeps years, `501(c)(3)`, "section 170" and list enumerators.
  - `numeric_profile` (number-mention counts), `is_sched_o_pointer`, `classify_sched_o_ref`, `normalize_website`, `check_identities`.
  - All are None- and NaN-safe.
- **Self-test:** `python3 -I src/parse990.py --selftest DIR [--limit N --seed S]` recursively parses `*.xml` and `*.xml.gz` under DIR and prints:
  - file, error and version counts;
  - per-field coverage (non-missing, and True/nonzero share);
  - text length tables (word counts; MiniLM WordPiece counts if `tokenizers` and the MiniLM `tokenizer.json` are available);
  - ALL-CAPS and pointer rates;
  - number-leakage table;
  - identity pass rates.
  - Exit code 0 means no errors and no record missing EIN, tax year or revenue.
- **Results:**
  - `xml_samples/` (2,448 files): 2,445 parsed, 3 non-990 controls → None, 0 errors, exit 0, ≈5 ms per file.
  - Bulk 5,000: 0 errors, exit 0, 5.7–6.8 ms per file.
  - Outputs are saved in `/tmp/claude-0/data/xml_samples/selftest_output.txt` and `selftest_bulk2022_5000.txt`.
- **Independent cross-validation** against `src/parse990_min.py` (different traversal code) on 66 shared fields × 2,445 returns: 0 disagreements apart from deliberate placeholder handling. Those cases: `website` 511 ("N/A" etc.), `mission_p3` 19 ("NONE"), `prog4a_desc` 17 ("N/A").
- **Robustness tests** (`/tmp/claude-0/scripts/robustness.py`):
  - BOM, namespace-less XML, a file truncated at 50% (recovered and flagged), empty bytes (None), an XXE attempt (entity not resolved);
  - 990-EZ, 990-PF and 990-T (None);
  - the CLI on `.gz` input.
- **Throughput:** about 5–7 ms per return single-threaded with lxml, so ≈65–80 CPU-minutes for 690k returns. The work is embarrassingly parallel.

## 8. Download size and time estimates

From index `FileSizeBytes` (uncompressed XML).

**Per file (TY2022 full 990):**
- Mean 41.3 KB, median 33.1 KB, p10 23.0 KB, p90 57.5 KB, p99 181 KB, max 57.5 MB (a group return).
- Median by revenue class: <100k 26 KB; 100k–1M 31 KB; 1M–10M 40 KB; 10M–100M 63 KB; >100M 127 KB (mean 232 KB). The growth comes from attached schedules.
- gzip ratio 5.48 (3,000-file sample): stored TY2022 US files average ≈7.4 KB.

| scope (2 consecutive TYs) | selection | returns | raw XML | ≈gzip |
|---|---|---|---|---|
| **US, TY2022+TY2023** | latest per EIN, non-group | 690,947 | 28.35 GB | ≈5.2 GB |
| US, TY2022+TY2023 | all 990 rows (incl. superseded/amended) | 704,803 | 29.01 GB | ≈5.3 GB |
| US, TY2021+TY2022 | latest per EIN, non-group | 672,051 | 27.64 GB | ≈5.0 GB |
| **WI, TY2022+TY2023** (rank 19 of 62 jurisdictions by filer count) | latest per EIN, non-group, by legal domicile | 13,074 | 0.557 GB | ≈0.10 GB |
| MO, TY2022+TY2023 (rank 20) | same | 12,949 | 0.543 GB | ≈0.10 GB |
| MN / CO, TY2022+TY2023 | same | 16,702 / 14,078 | 0.689 / 0.567 GB | ≈0.13 / 0.10 GB |
| New England (6 states), TY2022+TY2023 | same | 43,778 | 1.979 GB | ≈0.36 GB |

- Measured on the US TY2022 download: 341,956 files = 2.54 GB gzipped.
- **Time.**
  - Measured: 149 files/s (48 threads, sandbox egress) over the middle 80% of the US TY2022 run, and 2,400 files in 13 s with 16 curl processes.
  - US two-year pull ≈ 77 minutes; one mid-sized state ≈ 1.5 minutes. Add about 0.9 GB for three yearly index CSVs.
- **State selection caveat.**
  - Selecting a state by index `LegalDomicileState` (as `fetch990.py` does) misses the 5.0% of returns with that field blank, and it uses legal domicile rather than location.
  - TY2022 counts by legal domicile, by BMF (2023-08) mailing state, and the union: WI 6,466 / 6,548 / 7,010; MO 6,387 / 6,729 / 7,306; MN 8,247 / 8,518 / 8,943.
  - **Select by BMF `STATE` ∪ legal domicile.** That is about 8% more units and matches how a statistical office would define a regional frame. 97.8% of TY2022 990 EINs are in the 2023-08 BMF.

## 9. Red-team findings and concrete recommendations for the benchmark

1. **Bug in `build_population.py`.** `has_website = website_t1.notna()` counts "N/A", "NONE" and "NA", which appear in 20.9% of returns, as having a website. Use `parse990.normalize_website`, or the parser's `website` field.
2. **Mislabelled field in `parse990_min.py`.** `audited_fs` reads `AuditedFinancialStmtAttInd`. That is Part IV-20b, hospitals only (0.7% True), not audited financial statements in general, which are `FSAuditedInd`, Part XII-2b (37.6% True). It is not used downstream yet; rename it or drop it.
3. **Frame selection** in `fetch990.py` (see §8). For the New England population this drops about 5% of units, and not at random (missingness varies by version and software: 9.5% under 2022v4.1).
4. **Missing ≠ zero** for Part VIII/IX lines (§2.8). Always zero-fill, and never feed missingness flags to a model. They identify the filer's software.
5. **Leakage controls the protocol should include:**
   - Use **lagged text (t−1) as the frame variable** and the t return for y. Mission text is identical year over year for 88%; program text for 67.5%.
   - Run the number-masked ablation (`mask_numbers`) on program text and `text_profile`.
   - Add a **length-only control covariate** (log word count, number of programs, number of Schedule O entries), so verbosity isn't credited to "semantics".
   - Exclude `prog_rev_desc`, `n_prog_rev_lines` and `sched_o_all` from embeddable inputs. They encode the outcome by their presence, or carry 84% numeric boilerplate.
   - Use a random-vector negative control (already in `embed.py`).
6. **Recommended text inputs, from least to most leakage:** `mission_resolved` → `program_desc_all` (masked) → `text_profile` (masked). Report results for each separately.
7. **Duplicates.** Keep the latest `ReturnTs` per EIN×TaxYear. That removes 6,488 superseded or amended rows in TY2022. Group returns (0.2%) should be excluded or treated as clusters; their subordinate organisations file no individual return, which is a coverage gap shared with any 990-based frame.
8. **Timeliness realism.** Model publication lag explicitly. The TY2022 index grew 52% between the 2025-05 and 2025-10 snapshots. `SubmittedOn` supports a "filed-by-date" response mechanism, i.e. realistic nonresponse.
9. **Multimodal (PDF) angle.** The e-file is already structured text. Rendering it to page images for Cohere's image embeddings would put the financial tables, the y variables themselves, into the image. That is maximal leakage. Use images only for genuinely external documents (e.g., website screenshots, annual reports), and mask or crop any figures.

## 10. Reproducibility (files written this session)

**Project files**
- Module: `/home/user/hananather/_research/embedding-auxiliary-variables/src/parse990.py`
- This report: `/home/user/hananather/_research/embedding-auxiliary-variables/notes/research/05-irs990-xml-parsing-spec.md`

**Scripts** (`/tmp/claude-0/scripts/`)

| script | purpose |
|---|---|
| `idx_profile.py` | index profiles |
| `select_samples.py` | stratified sample selection |
| `select_coverage.py` | by-version coverage sample |
| `inventory.py` | tag inventory |
| `version_matrix.py`, `target_presence.py` | 16-version presence matrix |
| `value_vocab.py` | boolean vocabularies and explicit zeros |
| `scho_scan.py` | Schedule O references |
| `inspect_fail.py` | identity failures |
| `crosscheck_parsers.py` | cross-validation against parse990_min |
| `robustness.py` | robustness tests |
| `size_estimates.py`, `state_check.py` | download sizing and frame selection |
| `text_analyses.py`, `style_fingerprint.py` | text properties |

**Data** (`/tmp/claude-0/data/`)
- Samples: `xml_samples/` (45 core + `coverage/` 2,400 + `non990/` 3), with `selftest_output.txt` and `selftest_bulk2022_5000.txt`.
- Reference metadata: `refs/irsx/{variables,line_numbers,groups,descriptions}.csv` and `refs/concordance/concordance.csv`.
- Index slices: `raw990_slices/idx_{2019,2020}_{head,mid}.csv`.

## Sources (all accessed in this session)

WebSearch was unavailable: the shared per-turn budget was exhausted. The sources below were obtained by direct download.
- GivingTuesday 990 Data Lake, public S3 bucket `gt990datalake-rawdata`: https://gt990datalake-rawdata.s3.amazonaws.com/ (yearly index listing under `Indices/990xmls/yearly/`; returns under `EfileData/XmlFiles/`).
- Fenton, J. IRSx ("990-xml-reader"), version 0.5.1, PyPI, uploaded 2026-09-22: https://pypi.org/project/irsx/ (wheel `irsx-0.5.1-py3-none-any.whl`; metadata `irsx/metadata/variables.csv`; source https://github.com/jsfenfen/990-xml-reader). Used for xpath ↔ form-line ↔ IRS type mapping.
- Nonprofit Open Data Collective, IRS e-file master concordance file: https://raw.githubusercontent.com/Nonprofit-Open-Data-Collective/irs-efile-master-concordance-file/master/concordance.csv (repository https://github.com/Nonprofit-Open-Data-Collective/irs-efile-master-concordance-file). Used for variable names and line mapping; its version lists mostly end at 2016v3.0.
- IRS e-file business rules enforcing the cross-part identities in §6: [UNVERIFIED]; inferred from 100% pass rates.
- Schema maxLength of ShortExplanationType = 1,000 and ExplanationType = 9,000: [UNVERIFIED as schema facts]; observed as exact empirical maxima.
- Cohere embed-v4 context length: [UNVERIFIED].
