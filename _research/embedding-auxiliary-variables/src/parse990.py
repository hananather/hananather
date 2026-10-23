"""parse990 -- parser for full IRS Form 990 e-file (MeF) XML returns.

Scope: ReturnData/IRS990 (full Form 990) in the public XML published by the IRS and mirrored in the
GivingTuesday 990 data lake (s3://gt990datalake-rawdata/EfileData/XmlFiles/{ObjectId}_public.xml).
Tested on schema versions 2019v5.0-5.2, 2020v4.0-4.2, 2021v4.0-4.2, 2022v4.0/4.1/5.0, 2023v4.0/5.0/5.1/6.0.
Every element path used here exists unchanged across all of those versions (see notes/research/05-*.md).

Public API
    parse_return(xml_bytes, zero_fill=False) -> dict | None
        None if the document is not a full Form 990 (990-EZ, 990-PF, 990-T ...) or is not a Return.
        Missing elements -> None.  zero_fill=True turns missing *amount* lines of Parts VIII/IX into 0
        (on the paper form a blank amount line means zero; e-file software differs on whether it emits
        an explicit <...Amt>0</...Amt> or omits the element, so 'missing' is NOT informative there).
    clean_text(s)            NFKC, drop control chars, collapse whitespace, strip, placeholders -> None
    mask_numbers(s)          replace money amounts / numbers with [AMT] / [NUM] (leakage control)
    numeric_profile(s)       counts of money mentions, numbers, percents, years in a text
    check_identities(rec)    accounting identities between Parts I, III, VIII, IX, X (data-quality checks)

Design notes
  * Element paths are explicit, namespace-qualified ElementPath lookups of *direct children* of
    IRS990. Several local names are reused inside groups (Desc, ExpenseAmt, FundraisingAmt, TotalAmt),
    so descendant searches (//Desc) would silently pick the wrong line.
  * Part III line 4a is NOT wrapped in a group: its ActivityCd/ExpenseAmt/GrantAmt/RevenueAmt/Desc are
    direct children of IRS990. 4b and 4c are ProgSrvcAccomActy2Grp / ProgSrvcAccomActy3Grp; 4d is the
    repeating ProgSrvcAccomActyOtherGrp; 4e is TotalProgramServiceExpensesAmt.
  * Booleans: BooleanType is written as 'true'/'false' OR '1'/'0' depending on the filer's software;
    CheckboxType is 'X' when checked and absent otherwise. Both map to True/False/None here
    (checkbox: True or None -> use `or False` downstream if you need a 0/1 flag).
  * The parser never resolves external entities or DTDs (untrusted input).

Usage
    python3 -I parse990.py FILE.xml[.gz] ...           print parsed records as JSON
    python3 -I parse990.py --selftest DIR [--limit N]  parse every *.xml / *.xml.gz under DIR and print
                                                       field coverage, text-length, leakage and
                                                       identity-check tables
"""
from __future__ import annotations

import re
import unicodedata

from lxml import etree

EFILE_NS = "http://www.irs.gov/efile"

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=True,
                          remove_comments=True)
_PARSER_RECOVER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                                  huge_tree=True, remove_comments=True, recover=True)

# --------------------------------------------------------------------------------------------------
# Field maps.  Keys are output names; values are paths relative to IRS990 (or ReturnHeader), written
# with bare local names and qualified with the document namespace at parse time.
# --------------------------------------------------------------------------------------------------

HEADER_TEXT = {
    "return_type": "ReturnTypeCd",
    "ein": "Filer/EIN",
    "name_line1": "Filer/BusinessName/BusinessNameLine1Txt",
    "name_line2": "Filer/BusinessName/BusinessNameLine2Txt",
    "name_control": "Filer/BusinessNameControlTxt",
    "addr_city": "Filer/USAddress/CityNm",
    "addr_state": "Filer/USAddress/StateAbbreviationCd",
    "addr_zip": "Filer/USAddress/ZIPCd",
    "addr_foreign_country": "Filer/ForeignAddress/CountryCd",
    "tax_period_begin": "TaxPeriodBeginDt",
    "tax_period_end": "TaxPeriodEndDt",
    "return_ts": "ReturnTs",
    "preparer_firm_ein": "PreparerFirmGrp/PreparerFirmEIN",  # paid preparer firm (absent if self-prepared)
}
HEADER_INT = {"tax_year": "TaxYr"}

F990_TEXT = {
    # Header block of page 1 (items C-M)
    "dba_name": "DoingBusinessAsName/BusinessNameLine1Txt",
    "website_raw": "WebsiteAddressTxt",                       # item J
    "legal_domicile_state": "LegalDomicileStateCd",          # item M
    "legal_domicile_country": "LegalDomicileCountryCd",
    "group_exemption_num": "GroupExemptionNum",              # item H(c)
    # Text: Part I line 1, Part III line 1
    "mission_p1": "ActivityOrMissionDesc",                   # Part I line 1 (ShortExplanationType)
    "mission_p3": "MissionDesc",                             # Part III line 1
}

F990_INT = {
    "formation_year": "FormationYr",                         # item L
    "gross_receipts": "GrossReceiptsAmt",                    # item G
    # Part I summary
    "voting_members": "VotingMembersGoverningBodyCnt",       # I-3
    "independent_members": "VotingMembersIndependentCnt",    # I-4
    "employees_p1": "TotalEmployeeCnt",                      # I-5
    "volunteers": "TotalVolunteersCnt",                      # I-6 (blank != 0; often omitted)
    "ubi_gross": "TotalGrossUBIAmt",                         # I-7a
    "ubi_net": "NetUnrelatedBusTxblIncmAmt",                 # I-7b
    "contrib_py": "PYContributionsGrantsAmt",                # I-8
    "contrib_cy": "CYContributionsGrantsAmt",
    "prog_rev_py": "PYProgramServiceRevenueAmt",             # I-9
    "prog_rev_cy": "CYProgramServiceRevenueAmt",
    "invest_inc_py": "PYInvestmentIncomeAmt",                # I-10
    "invest_inc_cy": "CYInvestmentIncomeAmt",
    "other_rev_py": "PYOtherRevenueAmt",                     # I-11
    "other_rev_cy": "CYOtherRevenueAmt",
    "revenue_py": "PYTotalRevenueAmt",                       # I-12
    "revenue_cy": "CYTotalRevenueAmt",
    "grants_paid_py": "PYGrantsAndSimilarPaidAmt",           # I-13
    "grants_paid_cy": "CYGrantsAndSimilarPaidAmt",
    "member_benefits_py": "PYBenefitsPaidToMembersAmt",      # I-14
    "member_benefits_cy": "CYBenefitsPaidToMembersAmt",
    "salaries_py": "PYSalariesCompEmpBnftPaidAmt",           # I-15
    "salaries_cy": "CYSalariesCompEmpBnftPaidAmt",
    "prof_fundraising_py": "PYTotalProfFndrsngExpnsAmt",     # I-16a
    "prof_fundraising_cy": "CYTotalProfFndrsngExpnsAmt",
    "fundraising_exp_cy": "CYTotalFundraisingExpenseAmt",    # I-16b (CY only)
    "other_exp_py": "PYOtherExpensesAmt",                    # I-17
    "other_exp_cy": "CYOtherExpensesAmt",
    "expenses_py": "PYTotalExpensesAmt",                     # I-18
    "expenses_cy": "CYTotalExpensesAmt",
    "rev_less_exp_py": "PYRevenuesLessExpensesAmt",          # I-19
    "rev_less_exp_cy": "CYRevenuesLessExpensesAmt",
    "assets_boy": "TotalAssetsBOYAmt",                       # I-20
    "assets_eoy": "TotalAssetsEOYAmt",
    "liabilities_boy": "TotalLiabilitiesBOYAmt",             # I-21
    "liabilities_eoy": "TotalLiabilitiesEOYAmt",
    "net_assets_boy": "NetAssetsOrFundBalancesBOYAmt",       # I-22
    "net_assets_eoy": "NetAssetsOrFundBalancesEOYAmt",
    # Part III line 4a (ungrouped direct children of IRS990) and 4d/4e totals
    "prog4a_code": "ActivityCd",
    "prog4a_expense": "ExpenseAmt",
    "prog4a_grants": "GrantAmt",
    "prog4a_revenue": "RevenueAmt",
    "prog4d_expense": "TotalOtherProgSrvcExpenseAmt",
    "prog4d_grants": "TotalOtherProgSrvcGrantAmt",
    "prog4d_revenue": "TotalOtherProgSrvcRevenueAmt",
    "prog_exp_total_p3": "TotalProgramServiceExpensesAmt",   # III-4e
    # Part V
    "employees_w3": "EmployeeCnt",                           # V-2a (W-3 employee count)
    # Part VII
    "n_indiv_comp_gt100k": "IndivRcvdGreaterThan100KCnt",    # VII-A-2
    "n_contractors_gt100k": "CntrctRcvdGreaterThan100KCnt",  # VII-B-2
    "comp_reportable_org": "TotalReportableCompFromOrgAmt",  # VII-A-1d (D)
    # Part VIII line 1 (contributions) and 2 (program service revenue)
    "federated_campaigns": "FederatedCampaignsAmt",          # VIII-1a
    "membership_dues": "MembershipDuesAmt",                  # VIII-1b
    "fundraising_events_contrib": "FundraisingAmt",          # VIII-1c (NOT Part IX col D)
    "related_org_contrib": "RelatedOrganizationsAmt",        # VIII-1d
    "gov_grants": "GovernmentGrantsAmt",                     # VIII-1e
    "other_contrib": "AllOtherContributionsAmt",             # VIII-1f
    "noncash_contrib": "NoncashContributionsAmt",            # VIII-1g (included in 1a-1f)
    "total_contrib_p8": "TotalContributionsAmt",             # VIII-1h
    "total_prog_rev_p8": "TotalProgramServiceRevenueAmt",    # VIII-2g
    "fundraising_gross_income": "FundraisingGrossIncomeAmt", # VIII-8a
    "fundraising_direct_exp": "FundraisingDirectExpensesAmt",  # VIII-8b
    "gaming_gross_income": "GamingGrossIncomeAmt",           # VIII-9a
    "inventory_gross_sales": "GrossSalesOfInventoryAmt",     # VIII-10a
}

F990_GROUP_INT = {  # (group, child) under IRS990
    "prog_rev_other_p8": ("TotalOthProgramServiceRevGrp", "TotalRevenueColumnAmt"),   # VIII-2f
    "invest_income_p8": ("InvestmentIncomeGrp", "TotalRevenueColumnAmt"),             # VIII-3
    "net_fundraising_p8": ("NetIncmFromFundraisingEvtGrp", "TotalRevenueColumnAmt"),  # VIII-8c
    "net_gaming_p8": ("NetIncomeFromGamingGrp", "TotalRevenueColumnAmt"),             # VIII-9c
    "revenue_p8": ("TotalRevenueGrp", "TotalRevenueColumnAmt"),                       # VIII-12A
    "revenue_related_p8": ("TotalRevenueGrp", "RelatedOrExemptFuncIncomeAmt"),        # VIII-12B
    "revenue_ubi_p8": ("TotalRevenueGrp", "UnrelatedBusinessRevenueAmt"),             # VIII-12C
    "revenue_excluded_p8": ("TotalRevenueGrp", "ExclusionAmt"),                       # VIII-12D
    # Part IX statement of functional expenses
    "grants_domestic_orgs": ("GrantsToDomesticOrgsGrp", "TotalAmt"),                  # IX-1
    "grants_domestic_indiv": ("GrantsToDomesticIndividualsGrp", "TotalAmt"),          # IX-2
    "grants_foreign": ("ForeignGrantsGrp", "TotalAmt"),                               # IX-3
    "officer_comp": ("CompCurrentOfcrDirectorsGrp", "TotalAmt"),                      # IX-5
    "other_salaries": ("OtherSalariesAndWagesGrp", "TotalAmt"),                       # IX-7
    "pension_contrib": ("PensionPlanContributionsGrp", "TotalAmt"),                   # IX-8
    "other_benefits": ("OtherEmployeeBenefitsGrp", "TotalAmt"),                       # IX-9
    "payroll_taxes": ("PayrollTaxesGrp", "TotalAmt"),                                 # IX-10
    "fees_prof_fundraising": ("FeesForServicesProfFundraising", "TotalAmt"),          # IX-11e
    "advertising": ("AdvertisingGrp", "TotalAmt"),                                    # IX-12
    "occupancy": ("OccupancyGrp", "TotalAmt"),                                        # IX-16
    "travel": ("TravelGrp", "TotalAmt"),                                              # IX-17
    "depreciation": ("DepreciationDepletionGrp", "TotalAmt"),                         # IX-22
    "func_exp_total": ("TotalFunctionalExpensesGrp", "TotalAmt"),                     # IX-25A
    "func_exp_program": ("TotalFunctionalExpensesGrp", "ProgramServicesAmt"),         # IX-25B
    "func_exp_mgmt": ("TotalFunctionalExpensesGrp", "ManagementAndGeneralAmt"),       # IX-25C
    "func_exp_fundraising": ("TotalFunctionalExpensesGrp", "FundraisingAmt"),         # IX-25D
    # Part X balance sheet (EOY column)
    "cash_eoy": ("CashNonInterestBearingGrp", "EOYAmt"),                              # X-1
    "savings_eoy": ("SavingsAndTempCashInvstGrp", "EOYAmt"),                          # X-2
    "land_bldg_net_eoy": ("LandBldgEquipBasisNetGrp", "EOYAmt"),                      # X-10c
    "invest_public_eoy": ("InvestmentsPubTradedSecGrp", "EOYAmt"),                    # X-11
    "assets_eoy_p10": ("TotalAssetsGrp", "EOYAmt"),                                   # X-16
    "liabilities_eoy_p10": ("TotalLiabilitiesGrp", "EOYAmt"),                         # X-26
    "net_assets_eoy_p10": ("TotalNetAssetsFundBalanceGrp", "EOYAmt"),                 # X-32/33
}

F990_BOOL = {
    # item B checkboxes
    "address_change": "AddressChangeInd",
    "name_change": "NameChangeInd",
    "initial_return": "InitialReturnInd",
    "final_return": "FinalReturnInd",
    "amended": "AmendedReturnInd",
    "application_pending": "ApplicationPendingInd",
    # item H / I / K
    "group_return": "GroupReturnForAffiliatesInd",           # H(a)
    "all_affiliates_included": "AllAffiliatesIncludedInd",   # H(b)
    "is_501c3": "Organization501c3Ind",                      # I (checkbox)
    "is_501c_other": "Organization501cInd",                  # I (checkbox, type in attribute)
    "is_4947a1": "Organization4947a1NotPFInd",
    "is_527": "Organization527Ind",
    "form_corp": "TypeOfOrganizationCorpInd",                # K
    "form_trust": "TypeOfOrganizationTrustInd",
    "form_assoc": "TypeOfOrganizationAssocInd",
    "form_other": "TypeOfOrganizationOtherInd",
    # Part III
    "new_program": "SignificantNewProgramSrvcInd",           # III-2
    "significant_change": "SignificantChangeInd",            # III-3
    "sched_o_part3": "InfoInScheduleOPartIIIInd",            # Part III heading checkbox
    # Part IV checklist
    "c3_described": "DescribedInSection501c3Ind",            # IV-1
    "sched_b_required": "ScheduleBRequiredInd",              # IV-2
    "political": "PoliticalCampaignActyInd",                 # IV-3
    "lobbying": "LobbyingActivitiesInd",                     # IV-4 (501(c)(3) only; absent otherwise)
    "proxy_tax": "SubjectToProxyTaxInd",                     # IV-5
    "donor_advised_fund": "DonorAdvisedFundInd",             # IV-6
    "conservation_easements": "ConservationEasementsInd",    # IV-7
    "collections_art": "CollectionsOfArtInd",                # IV-8
    "credit_counseling": "CreditCounselingInd",              # IV-9
    "endowments": "DonorRstrOrQuasiEndowmentsInd",           # IV-10
    "school": "SchoolOperatingInd",                          # IV-13
    "foreign_office": "ForeignOfficeInd",                    # IV-14a
    "foreign_activities": "ForeignActivitiesInd",            # IV-14b
    "foreign_grants_orgs": "MoreThan5000KToOrgInd",          # IV-15
    "foreign_grants_indiv": "MoreThan5000KToIndividualsInd", # IV-16
    "professional_fundraising": "ProfessionalFundraisingInd",  # IV-17
    "fundraising_events": "FundraisingActivitiesInd",        # IV-18
    "gaming": "GamingActivitiesInd",                         # IV-19
    "hospital": "OperateHospitalInd",                        # IV-20a
    "hospital_audited_fs": "AuditedFinancialStmtAttInd",     # IV-20b
    "grants_orgs_us": "GrantsToOrganizationsInd",            # IV-21
    "grants_indiv_us": "GrantsToIndividualsInd",             # IV-22
    "tax_exempt_bonds": "TaxExemptBondsInd",                 # IV-24a
    "terminated": "TerminateOperationsInd",                  # IV-31
    "related_entity": "RelatedEntityInd",                    # IV-33
    "related_org_ctrl": "RelatedOrganizationCtrlEntInd",     # IV-34
    # Part V / VI / XII
    "employment_tax_filed": "EmploymentTaxReturnsFiledInd",  # V-2b
    "local_chapters": "LocalChaptersInd",                    # VI-10a
    "conflict_policy": "ConflictOfInterestPolicyInd",        # VI-12a
    "whistleblower_policy": "WhistleblowerPolicyInd",        # VI-13
    "acct_cash": "MethodOfAccountingCashInd",                # XII-1
    "acct_accrual": "MethodOfAccountingAccrualInd",
    "fs_compiled_reviewed": "AccountantCompileOrReviewInd",  # XII-2a
    "fs_audited": "FSAuditedInd",                            # XII-2b
    "single_audit_required": "FederalGrantAuditRequiredInd", # XII-3a
}

# Amount lines whose absence means "blank line" (= 0) on a complete full Form 990.
ZERO_IF_MISSING = (
    {k for k in F990_INT if k.endswith(("_cy", "_contrib", "_campaigns", "_dues", "_grants",
                                         "_p8", "_income", "_exp", "_sales"))
     and not k.endswith("_py")}
    | {k for k, (g, _) in F990_GROUP_INT.items() if not k.endswith("_eoy") and not k.endswith("_p10")}
    | {"gross_receipts", "prog4a_expense", "prog4a_grants", "prog4a_revenue",
       "prog4d_expense", "prog4d_grants", "prog4d_revenue", "prog_exp_total_p3"}
)

TEXT_FIELDS = ("mission_p1", "mission_p3", "prog4a_desc", "prog4b_desc", "prog4c_desc",
               "prog4d_desc", "program_desc_all", "prog_rev_desc", "sched_o_program",
               "sched_o_all", "mission_resolved", "text_profile")

# --------------------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ﻿]")
_PLACEHOLDERS = {"", "NONE", "N/A", "NA", "N.A.", "-", "--", ".", "0", "NOT APPLICABLE", "NO",
                 "NULL", "X", "NONE.", "N/A."}
_XREF = re.compile(r"\bSCH(?:EDULE|ED|D|\.)?\s*O\b|\bPART\s+[IVX]+\b|\bATTACH(?:ED|MENT)?\b|\bSTATEMENT\b|\bSTMT\b",
                   re.I)
_XREF_VOCAB = re.compile(
    r"\b(?:PLEASE|SEE|REFER(?:RED)?|TO|CONT(?:INUED|'?D)?|DESCRIBED|AS|ON|IN|THE|AND|OF|FOR|AT|IS|ARE|"
    r"FORM|990|PAGE|SCH(?:EDULE|ED|D)?|O|PART|[IVX]+|LINES?|\d+[A-H]?|ATTACH(?:ED|MENT)?|STATEMENT|STMT|"
    r"SUPPLEMENTAL|INFORMATION|BELOW|ABOVE)\b", re.I)


def clean_text(s: str | None) -> str | None:
    """Unicode-normalize (NFKC), drop control/zero-width chars, collapse whitespace, strip.
    Placeholder answers ('NONE', 'N/A', '-', '0', ...) become None."""
    if not isinstance(s, str):  # None / NaN-safe
        return None
    s = unicodedata.normalize("NFKC", s)
    s = _CTRL.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if s.upper() in _PLACEHOLDERS:
        return None
    return s


def is_sched_o_pointer(s: str | None, max_len: int = 80) -> bool:
    """True when a short text is only a cross-reference ('SEE SCHEDULE O', 'SEE PART III, LINE 1',
    'SEE ATTACHED STATEMENT'): it contains a reference and < 2 content words remain after removing
    reference vocabulary. 'Provide housing; see Sch O' is NOT a pointer (it has content)."""
    if not isinstance(s, str) or not s or len(s) > max_len or not _XREF.search(s):
        return False
    rest = _XREF_VOCAB.sub(" ", s)
    return len(re.findall(r"[A-Za-z]{2,}", rest)) < 2


def join_unique(parts, sep=" | ") -> str | None:
    """Join non-empty strings, dropping case-insensitive duplicates and pure Schedule-O pointers."""
    seen, out = set(), []
    for p in parts:
        if not isinstance(p, str) or not p or is_sched_o_pointer(p):
            continue
        k = p.upper()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return sep.join(out) if out else None


_MONEY = re.compile(
    r"(?:\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|mm|[mbk])\b)?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion)\b)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s*dollars\b)", re.I)
_PCT = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent\b)", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


_LEGAL = re.compile(r"\b50[1-9]\s*\(?\s*[a-z]{1,2}\s*\)?\s*\(?\s*\d{0,2}\s*\)?"
                    r"|\b(?:section|sec\.?|irc|code|chapter|title)\s*\d+[\w()]*"
                    r"|\b4947\s*\(?a\)?\s*\(?1?\)?", re.I)
_ENUM = re.compile(r"(?:(?<=\s)|^|(?<=[.;:]))#?\d{1,2}[.)](?=\s|[A-Za-z])")


def numeric_profile(s: str | None) -> dict:
    """Counts of money amounts, percents, 4-digit years and other numbers (counts) in a text.
    Legal references (501(c)(3), 'section 170') and list enumerators ('1.', '2)') are removed first.
    'n_count_like' excludes money, percents and years; numbers >= 1000 are counted separately."""
    if not isinstance(s, str) or not s:  # None / NaN-safe
        return {"n_money": 0, "n_pct": 0, "n_year": 0, "n_count_like": 0, "n_big_count": 0}
    s = _ENUM.sub(" ", _LEGAL.sub(" ", s))
    rest = _MONEY.sub(" ", s)
    n_money = len(_MONEY.findall(s))
    n_pct = len(_PCT.findall(rest))
    rest = _PCT.sub(" ", rest)
    n_year = len(_YEAR.findall(rest))
    rest = _YEAR.sub(" ", rest)
    nums = _NUM.findall(rest)
    big = 0
    for x in nums:
        try:
            big += float(x.replace(",", "")) >= 1000
        except ValueError:
            pass
    return {"n_money": n_money, "n_pct": n_pct, "n_year": n_year, "n_count_like": len(nums),
            "n_big_count": big}


def mask_numbers(s: str | None, keep_years: bool = True) -> str | None:
    """Replace money amounts with [AMT], percents with [PCT] and other numbers with [NUM]."""
    if not isinstance(s, str):  # None / NaN-safe
        return None
    s = _MONEY.sub("[AMT]", s)
    s = _PCT.sub("[PCT]", s)
    keep = [_LEGAL.pattern, _ENUM.pattern] + ([r"\b(?:19|20)\d{2}\b"] if keep_years else [])
    parts = re.split("(" + "|".join(f"(?:{k})" for k in keep) + ")", s, flags=re.I)
    return "".join(p if i % 2 else _NUM.sub("[NUM]", p) for i, p in enumerate(parts) if p is not None)


def normalize_website(s: str | None) -> str | None:
    """Lower-case host+path without scheme/'www.'/trailing slash; None for placeholders/no dot."""
    s = clean_text(s)
    if not s:
        return None
    t = s.lower().strip(" ./")
    t = re.sub(r"^(?:https?:?/*)", "", t)
    t = re.sub(r"^www\d?\.", "", t)
    t = t.split()[0] if t else t
    t = t.rstrip("/")
    if "." not in t or "@" in t or t in {"n/a", "none", "na"}:
        return None
    return t


# --------------------------------------------------------------------------------------------------
# XML helpers
# --------------------------------------------------------------------------------------------------

def _qualify(path: str, ns: str) -> str:
    if not ns:
        return path
    return "/".join(f"{{{ns}}}{p}" for p in path.split("/"))


def _text(el) -> str | None:
    if el is None:
        return None
    return clean_text("".join(el.itertext()))


def _int(el) -> int | None:
    if el is None:
        return None
    t = (el.text or "").strip().replace(",", "")
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        try:
            return int(round(float(t)))
        except ValueError:
            return None


def _bool(el) -> bool | None:
    if el is None:
        return None
    t = (el.text or "").strip().lower()
    if t in ("x", "1", "true", "yes"):
        return True
    if t in ("0", "false", "no"):
        return False
    return None


_REF_PART = re.compile(r"PART\s*([IVX]+)\b")
_REF_LINE = re.compile(r"\bL(?:INE|N)?S?\s*(\d+)\s*([A-H])?\b")


def classify_sched_o_ref(ref: str | None) -> tuple[str | None, str | None]:
    """('III', '4D') from free-text references such as 'FORM 990, PAGE 2, PART III, LINE 4D'."""
    if not ref:
        return None, None
    r = re.sub(r"[^A-Z0-9 ]", " ", ref.upper())
    r = _WS.sub(" ", r)
    m = _REF_PART.search(r)
    part = m.group(1) if m else None
    tail = r[m.end():] if m else r
    m2 = _REF_LINE.search(tail)
    line = (m2.group(1) + (m2.group(2) or "")) if m2 else None
    return part, line


# --------------------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------------------

def parse_return(xml_bytes: bytes, zero_fill: bool = False) -> dict | None:
    """Parse one e-file return. Returns a flat dict, or None if it is not a full Form 990."""
    if not xml_bytes or not xml_bytes.strip():
        return None
    recovered = False
    try:
        root = etree.fromstring(xml_bytes, parser=_PARSER)
    except etree.XMLSyntaxError:
        try:
            root = etree.fromstring(xml_bytes, parser=_PARSER_RECOVER)
        except etree.XMLSyntaxError:
            return None
        recovered = True
        if root is None:
            return None
    q = etree.QName(root)
    if q.localname != "Return":
        return None
    ns = q.namespace or ""
    Q = lambda p: _qualify(p, ns)  # noqa: E731
    hdr = root.find(Q("ReturnHeader"))
    f = root.find(Q("ReturnData/IRS990"))
    if hdr is None or f is None:
        return None

    r: dict = {"return_version": root.get("returnVersion"), "xml_recovered": recovered}
    for k, p in HEADER_TEXT.items():
        r[k] = _text(hdr.find(Q(p)))
    for k, p in HEADER_INT.items():
        r[k] = _int(hdr.find(Q(p)))
    r["org_name"] = " ".join(x for x in (r.pop("name_line1"), r.pop("name_line2")) if x) or None
    r["zip5"] = r["addr_zip"][:5] if r.get("addr_zip") else None
    tpe = r.get("tax_period_end")
    r["tax_period"] = tpe[:7].replace("-", "") if tpe else None

    for k, p in F990_TEXT.items():
        r[k] = _text(f.find(Q(p)))
    for k, p in F990_INT.items():
        r[k] = _int(f.find(Q(p)))
    for k, (g, c) in F990_GROUP_INT.items():
        r[k] = _int(f.find(Q(f"{g}/{c}")))
    for k, p in F990_BOOL.items():
        r[k] = _bool(f.find(Q(p)))

    r["website"] = normalize_website(r["website_raw"])
    # 501(c) type: 501(c)(3) is its own checkbox; other subsections carry organization501cTypeTxt
    c_el = f.find(Q("Organization501cInd"))
    c_type = c_el.get("organization501cTypeTxt") if c_el is not None else None
    if r["is_501c3"]:
        r["subsection"] = "03"
    elif c_type:
        r["subsection"] = c_type.strip().zfill(2)
    elif r["is_4947a1"]:
        r["subsection"] = "4947a1"
    elif r["is_527"]:
        r["subsection"] = "527"
    else:
        r["subsection"] = None

    # ---------------- Part III program service accomplishments ----------------
    r["prog4a_desc"] = _text(f.find(Q("Desc")))
    for tag, key in (("ProgSrvcAccomActy2Grp", "prog4b"), ("ProgSrvcAccomActy3Grp", "prog4c")):
        g = f.find(Q(tag))
        r[f"{key}_desc"] = _text(g.find(Q("Desc"))) if g is not None else None
        r[f"{key}_expense"] = _int(g.find(Q("ExpenseAmt"))) if g is not None else None
        r[f"{key}_grants"] = _int(g.find(Q("GrantAmt"))) if g is not None else None
        r[f"{key}_revenue"] = _int(g.find(Q("RevenueAmt"))) if g is not None else None
    others = f.findall(Q("ProgSrvcAccomActyOtherGrp"))
    r["n_prog4d"] = len(others)
    r["prog4d_desc"] = join_unique([_text(g.find(Q("Desc"))) for g in others])
    if r["prog4d_expense"] is None and others:  # some filers omit the 4d total
        vals = [_int(g.find(Q("ExpenseAmt"))) for g in others]
        r["prog4d_expense"] = sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None
    r["n_programs"] = (sum(r[f"prog4{x}_desc"] is not None for x in "abc") + len(others))

    # ---------------- Part VIII line 2 program service revenue lines ----------------
    psr = f.findall(Q("ProgramServiceRevenueGrp"))
    r["n_prog_rev_lines"] = len(psr)
    r["prog_rev_desc"] = join_unique([_text(g.find(Q("Desc"))) for g in psr])
    codes = [_text(g.find(Q("BusinessCd"))) for g in psr]
    r["prog_rev_business_codes"] = ";".join(c for c in codes if c) or None

    # ---------------- Schedule O ----------------
    so = root.find(Q("ReturnData/IRS990ScheduleO"))
    so_all, so_prog, so_mission = [], [], []
    n_so = 0
    if so is not None:
        for d in so.findall(Q("SupplementalInformationDetail")):
            ref = _text(d.find(Q("FormAndLineReferenceDesc")))
            txt = _text(d.find(Q("ExplanationTxt")))
            if not txt:
                continue
            n_so += 1
            so_all.append(txt)
            part, line = classify_sched_o_ref(ref)
            if part == "III" or (part == "I" and line == "1"):
                so_prog.append(txt)
                if line == "1":
                    so_mission.append(txt)
    r["n_sched_o"] = n_so
    r["sched_o_all"] = join_unique(so_all)
    r["sched_o_program"] = join_unique(so_prog)

    # ---------------- composite texts for embedding ----------------
    r["mission_p1_is_pointer"] = is_sched_o_pointer(r["mission_p1"])
    r["mission_p3_is_pointer"] = is_sched_o_pointer(r["mission_p3"])
    r["mission_resolved"] = join_unique([r["mission_p1"], r["mission_p3"], *so_mission])
    prog_parts = [r["prog4a_desc"], r["prog4b_desc"], r["prog4c_desc"],
                  *[_text(g.find(Q("Desc"))) for g in others]]
    r["program_desc_all"] = join_unique(prog_parts)
    # "text_profile": the organisation's self-description (mission + programs + Part I/III Sched O)
    r["text_profile"] = join_unique([r["mission_p1"], r["mission_p3"], *prog_parts, *so_prog])

    # ---------------- schedules attached (presence only) ----------------
    rd = root.find(Q("ReturnData"))
    scheds = sorted(etree.QName(c).localname.replace("IRS990Schedule", "")
                    for c in rd if isinstance(c.tag, str) and etree.QName(c).localname.startswith("IRS990Schedule"))
    r["schedules"] = ",".join(scheds) or None

    if zero_fill:
        for k in ZERO_IF_MISSING:
            if r.get(k) is None:
                r[k] = 0
    return r


# --------------------------------------------------------------------------------------------------
# Data-quality checks
# --------------------------------------------------------------------------------------------------
IDENTITIES = {
    "I12_eq_VIII12": ("revenue_cy", "revenue_p8"),
    "I18_eq_IX25": ("expenses_cy", "func_exp_total"),
    "I8_eq_VIII1h": ("contrib_cy", "total_contrib_p8"),
    "I9_eq_VIII2g": ("prog_rev_cy", "total_prog_rev_p8"),
    "III4e_eq_IX25B": ("prog_exp_total_p3", "func_exp_program"),
    "I20_eq_X16": ("assets_eoy", "assets_eoy_p10"),
    "I22_eq_X33": ("net_assets_eoy", "net_assets_eoy_p10"),
    "I5_eq_V2a": ("employees_p1", "employees_w3"),
}


def check_identities(rec: dict, tol: int = 1) -> dict:
    """Return {check_name: True/False/None}; None when a side is missing (after zero-filling
    amount lines). Also checks IX-25A = B+C+D (501(c)(3)/(c)(4) only, where columns B-D are
    mandatory), III-4e = 4a+4b+4c+4d and VIII-1h = sum(1a..1f)."""
    z = lambda k: rec.get(k) if rec.get(k) is not None else (0 if k in ZERO_IF_MISSING else None)  # noqa: E731
    out = {}
    cols_bcd_required = rec.get("subsection") in ("03", "04", "4947a1")
    for name, (a, b) in IDENTITIES.items():
        if name == "III4e_eq_IX25B" and not cols_bcd_required:
            out[name] = None  # Part IX column (B) is optional for other subsections
            continue
        va, vb = z(a), z(b)
        out[name] = None if va is None or vb is None else abs(va - vb) <= tol
    parts = [z(k) for k in ("func_exp_program", "func_exp_mgmt", "func_exp_fundraising")]
    tot = z("func_exp_total")
    out["IX25_A_eq_BCD"] = (None if tot is None or not cols_bcd_required
                            else abs(tot - sum(p or 0 for p in parts)) <= tol)
    p3 = [z(k) for k in ("prog4a_expense", "prog4b_expense", "prog4c_expense", "prog4d_expense")]
    out["III4e_eq_4a_4d"] = (None if rec.get("prog_exp_total_p3") is None
                             else abs(rec["prog_exp_total_p3"] - sum(v or 0 for v in p3)) <= tol)
    l1 = [z(k) for k in ("federated_campaigns", "membership_dues", "fundraising_events_contrib",
                         "related_org_contrib", "gov_grants", "other_contrib")]
    out["VIII1h_eq_sum1a_1f"] = abs((z("total_contrib_p8") or 0) - sum(v or 0 for v in l1)) <= tol
    return out


# --------------------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------------------

def _read(path: str) -> bytes:
    import gzip
    if path.endswith(".gz"):
        with gzip.open(path) as fh:
            return fh.read()
    with open(path, "rb") as fh:
        return fh.read()


def _quantiles(xs, qs=(0.1, 0.5, 0.9, 0.99)):
    if not xs:
        return [None] * len(qs)
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs]


def selftest(directory: str, limit: int | None = None, seed: int = 0, tokenizer_json: str | None = None) -> int:
    import collections, glob, os, random, time
    files = sorted(glob.glob(os.path.join(directory, "**", "*.xml"), recursive=True)
                   + glob.glob(os.path.join(directory, "**", "*.xml.gz"), recursive=True))
    if limit and len(files) > limit:
        files = sorted(random.Random(seed).sample(files, limit))
    tok = None
    tj = tokenizer_json or os.environ.get("MINILM_TOKENIZER", "/tmp/claude-0/models/minilm/onnx/tokenizer.json")
    try:
        from tokenizers import Tokenizer
        if os.path.exists(tj):
            tok = Tokenizer.from_file(tj)
            tok.no_truncation()
            tok.no_padding()
    except Exception:
        tok = None
    recs, n_none, errors, versions = [], 0, [], collections.Counter()
    t0 = time.time()
    for p in files:
        try:
            rec = parse_return(_read(p))
        except Exception as e:  # report and continue
            errors.append((p, f"{type(e).__name__}: {e}"))
            continue
        if rec is None:
            n_none += 1
            continue
        recs.append(rec)
        versions[rec["return_version"]] += 1
    dt = time.time() - t0
    n = len(recs)
    print(f"files={len(files)} parsed_990={n} not_990={n_none} errors={len(errors)} "
          f"seconds={dt:.1f} ({1000 * dt / max(1, len(files)):.2f} ms/file)")
    for p, e in errors[:10]:
        print("  ERROR", p, e)
    print("versions:", dict(sorted(versions.items())))
    if not n:
        return 1
    print(f"xml_recovered (malformed XML parsed in recover mode): {sum(r['xml_recovered'] for r in recs)}")

    # -------- coverage table --------
    keys = list(recs[0].keys())
    print("\n=== Coverage: share of returns with a non-missing value ===")
    print(f"{'field':<28}{'non-missing':>12}{'  true/nonzero':>15}")
    for k in keys:
        vals = [r.get(k) for r in recs]
        nm = sum(v is not None for v in vals)
        if all(isinstance(v, bool) or v is None for v in vals):
            extra = f"{sum(v is True for v in vals) / n:>14.3f}"
        elif all(isinstance(v, int) and not isinstance(v, bool) or v is None for v in vals):
            extra = f"{sum(bool(v) for v in vals if v is not None) / n:>14.3f}"
        else:
            extra = ""
        print(f"{k:<28}{nm / n:>12.3f} {extra}")

    # -------- text statistics --------
    print("\n=== Text fields: length (whitespace words / MiniLM WordPiece tokens) ===")
    hdr = f"{'field':<20}{'present':>8}{'w_p10':>7}{'w_p50':>7}{'w_p90':>7}{'w_p99':>7}{'w_max':>7}{'chmax':>7}"
    hdr += f"{'tok_p50':>8}{'tok_p90':>8}{'>256tok':>8}" if tok else ""
    hdr += f"{'ALLCAPS':>8}{'ptrSchO':>8}"
    print(hdr)
    for k in TEXT_FIELDS:
        ts = [r.get(k) for r in recs if r.get(k)]
        if not ts:
            print(f"{k:<20}{0:>8.3f}")
            continue
        w = [len(t.split()) for t in ts]
        line = f"{k:<20}{len(ts) / n:>8.3f}" + "".join(f"{x:>7}" for x in _quantiles(w)) + f"{max(w):>7}"
        line += f"{max(len(t) for t in ts):>7}"
        if tok:
            enc = tok.encode_batch(ts)
            tl = [len(e.ids) for e in enc]
            p50, p90 = _quantiles(tl, (0.5, 0.9))
            line += f"{p50:>8}{p90:>8}{sum(x > 256 for x in tl) / len(tl):>8.3f}"
        caps = sum(sum(c.isupper() for c in t) >= 0.9 * max(1, sum(c.isalpha() for c in t)) for t in ts)
        ptr = sum(is_sched_o_pointer(t) for t in ts)
        line += f"{caps / len(ts):>8.3f}{ptr / len(ts):>8.3f}"
        print(line)

    # -------- numeric leakage in text --------
    print("\n=== Numbers inside text fields (share of non-empty texts mentioning ...) ===")
    print(f"{'field':<20}{'money':>8}{'pct':>8}{'year':>8}{'count':>8}{'count>=1k':>10}{'any':>8}")
    for k in TEXT_FIELDS:
        ts = [r.get(k) for r in recs if r.get(k)]
        if not ts:
            continue
        prof = [numeric_profile(t) for t in ts]
        sh = lambda key: sum(p[key] > 0 for p in prof) / len(ts)  # noqa: E731
        anyn = sum(any(v > 0 for v in p.values()) for p in prof) / len(ts)
        print(f"{k:<20}{sh('n_money'):>8.3f}{sh('n_pct'):>8.3f}{sh('n_year'):>8.3f}"
              f"{sh('n_count_like'):>8.3f}{sh('n_big_count'):>10.3f}{anyn:>8.3f}")

    # -------- accounting identities --------
    print("\n=== Accounting identities (pass rate among returns where both sides exist) ===")
    chk = [check_identities(r) for r in recs]
    for name in chk[0]:
        vals = [c[name] for c in chk if c[name] is not None]
        print(f"{name:<22} evaluable={len(vals) / n:6.3f} pass={sum(vals) / max(1, len(vals)):6.3f}")

    # -------- sanity assertions on the test set --------
    bad = [r for r in recs if not r.get("ein") or r.get("tax_year") is None or r.get("revenue_cy") is None]
    print(f"\nrecords missing EIN/tax_year/revenue_cy: {len(bad)}")
    return 0 if not errors and not bad else 2


if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--selftest", metavar="DIR")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero-fill", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.selftest, a.limit, a.seed))
    for p in a.paths:
        print(json.dumps(parse_return(_read(p), zero_fill=a.zero_fill), indent=1, ensure_ascii=False))
