"""Minimal, explicit-path parser for full Form 990 e-file XML (schema 2013v3+).

Independent of src/parse990.py on purpose: the two parsers are cross-checked as a data-quality test.
"""
from __future__ import annotations

import re

from lxml import etree

_WS = re.compile(r"\s+")


def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _children(el):
    out = {}
    for c in el:
        out.setdefault(_local(c.tag), []).append(c)
    return out


def _txt(el):
    if el is None:
        return None
    t = "".join(el.itertext())
    t = _WS.sub(" ", t).strip()
    return t or None


def _num(el):
    t = _txt(el)
    if t is None:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _bool(el):
    t = _txt(el)
    if t is None:
        return None
    return t.lower() in ("true", "1", "x")


def _first(ch, name):
    v = ch.get(name)
    return v[0] if v else None


def _find(el, path):
    """path of local names separated by '/'; first match at each level."""
    cur = el
    for p in path.split("/"):
        if cur is None:
            return None
        cur = next((c for c in cur if _local(c.tag) == p), None)
    return cur


NUM_FIELDS = {
    # Part I summary (current and prior year columns)
    "employees_p1": "TotalEmployeeCnt", "volunteers": "TotalVolunteersCnt",
    "contrib_cy": "CYContributionsGrantsAmt", "contrib_py": "PYContributionsGrantsAmt",
    "prog_rev_cy": "CYProgramServiceRevenueAmt", "prog_rev_py": "PYProgramServiceRevenueAmt",
    "invest_inc_cy": "CYInvestmentIncomeAmt", "other_rev_cy": "CYOtherRevenueAmt",
    "revenue_cy": "CYTotalRevenueAmt", "revenue_py": "PYTotalRevenueAmt",
    "grants_paid_cy": "CYGrantsAndSimilarPaidAmt", "salaries_cy": "CYSalariesCompEmpBnftPaidAmt",
    "salaries_py": "PYSalariesCompEmpBnftPaidAmt", "fundraising_exp_cy": "CYTotalFundraisingExpenseAmt",
    "prof_fundraising_cy": "CYTotalProfFndrsngExpnsAmt",
    "expenses_cy": "CYTotalExpensesAmt", "expenses_py": "PYTotalExpensesAmt",
    "assets_boy": "TotalAssetsBOYAmt", "assets_eoy": "TotalAssetsEOYAmt",
    "net_assets_eoy": "NetAssetsOrFundBalancesEOYAmt",
    "gross_receipts": "GrossReceiptsAmt",
    # Part VIII
    "gov_grants": "GovernmentGrantsAmt", "federated_campaigns": "FederatedCampaignsAmt",
    "membership_dues": "MembershipDuesAmt", "fundraising_events_contrib": "FundraisingAmt",
    "related_org_contrib": "RelatedOrganizationsAmt", "other_contrib": "AllOtherContributionsAmt",
    "noncash_contrib": "NoncashContributionsAmt", "total_contrib_p8": "TotalContributionsAmt",
    "total_prog_rev_p8": "TotalProgramServiceRevenueAmt",
    # Part V
    "employees_w3": "EmployeeCnt",
    # Part III totals
    "prog_exp_total_p3": "TotalProgramServiceExpensesAmt",
    "formation_year": "FormationYr",
}
BOOL_FIELDS = {
    "lobbying": "LobbyingActivitiesInd", "political": "PoliticalCampaignActyInd",
    "foreign_activities": "ForeignActivitiesInd", "foreign_office": "ForeignOfficeInd",
    "hospital": "OperateHospitalInd", "school": "SchoolOperatingInd",
    "grants_to_orgs_us": "GrantsToOrganizationsInd", "grants_to_indiv_us": "GrantsToIndividualsInd",
    "fundraising_activities": "FundraisingActivitiesInd", "gaming": "GamingActivitiesInd",
    "audited_fs": "AuditedFinancialStmtAttInd", "group_return": "GroupReturnForAffiliatesInd",
    "c3": "Organization501c3Ind", "new_program": "SignificantNewProgramSrvcInd",
    "significant_change": "SignificantChangeInd", "donor_advised": "DonorAdvisedFundInd",
}
GROUP_SUM = {  # Part IX columns and officer compensation
    "func_exp_total": ("TotalFunctionalExpensesGrp", "TotalAmt"),
    "func_exp_program": ("TotalFunctionalExpensesGrp", "ProgramServicesAmt"),
    "func_exp_mgmt": ("TotalFunctionalExpensesGrp", "ManagementAndGeneralAmt"),
    "func_exp_fundraising": ("TotalFunctionalExpensesGrp", "FundraisingAmt"),
    "officer_comp": ("CompCurrentOfcrDirectorsGrp", "TotalAmt"),
}


def parse_return(xml_bytes: bytes) -> dict | None:
    root = etree.fromstring(xml_bytes, parser=etree.XMLParser(recover=True, huge_tree=True))
    hdr = _find(root, "ReturnHeader")
    f990 = _find(root, "ReturnData/IRS990")
    if hdr is None or f990 is None:
        return None
    r = {
        "ein": _txt(_find(hdr, "Filer/EIN")),
        "tax_year": _num(_find(hdr, "TaxYr")),
        "tax_period_end": _txt(_find(hdr, "TaxPeriodEndDt")),
        "return_ts": _txt(_find(hdr, "ReturnTs")),
        "name": _txt(_find(hdr, "Filer/BusinessName/BusinessNameLine1Txt")),
        "state_addr": _txt(_find(hdr, "Filer/USAddress/StateAbbreviationCd")),
        "zip": _txt(_find(hdr, "Filer/USAddress/ZIPCd")),
        "return_version": root.get("returnVersion"),
    }
    ch = _children(f990)
    for k, tag in NUM_FIELDS.items():
        r[k] = _num(_first(ch, tag))
    for k, tag in BOOL_FIELDS.items():
        r[k] = _bool(_first(ch, tag))
    for k, (g, t) in GROUP_SUM.items():
        ge = _first(ch, g)
        r[k] = _num(_find(ge, t)) if ge is not None else None
    r["amended"] = _bool(_first(ch, "AmendedReturnInd"))
    r["final_return"] = _bool(_first(ch, "FinalReturnInd"))
    r["website"] = _txt(_first(ch, "WebsiteAddressTxt"))
    r["legal_domicile_state"] = _txt(_first(ch, "LegalDomicileStateCd"))
    r["mission_p1"] = _txt(_first(ch, "ActivityOrMissionDesc"))
    r["mission_p3"] = _txt(_first(ch, "MissionDesc"))
    c3 = _first(ch, "Organization501cInd")
    r["subsection_txt"] = c3.get("organization501cTypeTxt") if c3 is not None else ("3" if r["c3"] else None)
    # Part III line 4a lives directly under IRS990; 4b/4c/4d are groups
    progs = []
    if _first(ch, "Desc") is not None or _first(ch, "ExpenseAmt") is not None:
        progs.append((_txt(_first(ch, "Desc")), _num(_first(ch, "ExpenseAmt"))))
    for g in ("ProgSrvcAccomActy2Grp", "ProgSrvcAccomActy3Grp", "ProgSrvcAccomActyOtherGrp"):
        for ge in ch.get(g, []):
            progs.append((_txt(_find(ge, "Desc")), _num(_find(ge, "ExpenseAmt"))))
    r["n_programs"] = len(progs)
    r["program_desc"] = " || ".join(d for d, _ in progs if d) or None
    r["program_desc_4a"] = progs[0][0] if progs else None
    return r


if __name__ == "__main__":
    import gzip, json, sys
    for p in sys.argv[1:]:
        with (gzip.open(p) if p.endswith(".gz") else open(p, "rb")) as f:
            print(json.dumps(parse_return(f.read()), indent=1)[:3000])
