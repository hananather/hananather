"""Download Form 990 e-file XMLs for a set of states and tax years from the
GivingTuesday 990 data lake (public S3 bucket gt990datalake-rawdata).

One return per (EIN, TaxYear): the most recently submitted, non-group return.
Files are stored gzip-compressed as {out}/{year}/{EIN}.xml.gz with a manifest CSV.
"""
import argparse, concurrent.futures as cf, gzip, os, sys, time, urllib.request
import pandas as pd

INDEX_URL = "https://gt990datalake-rawdata.s3.amazonaws.com/Indices/990xmls/yearly/{year}_efiledata_xmls_created_on_2026-08-25.csv"


def select_returns(index_csv, states):
    cols = ["EIN", "FormType", "TaxYear", "TaxPeriod", "LegalDomicileState", "OrganizationName",
            "ReturnTs", "SubmittedOn", "DocStatus", "GroupReturnForAffiliates", "ObjectId",
            "URL", "Website", "YearFormed", "ReturnVersion", "TotalRevenueCY", "TotalExpensesCY",
            "TotalAssetsBkEOY", "FileSizeBytes"]
    d = pd.read_csv(index_csv, dtype=str, usecols=cols, low_memory=False)
    d = d[(d.FormType == "990") & (d.GroupReturnForAffiliates != "True")]
    if states:
        d = d[d.LegalDomicileState.isin(states)]
    d = d.sort_values(["EIN", "ReturnTs"]).drop_duplicates("EIN", keep="last")
    return d


def fetch(url, dest, retries=4):
    if os.path.exists(dest):
        return dest, "cached"
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                b = r.read()
            tmp = dest + ".tmp"
            with gzip.open(tmp, "wb", compresslevel=1) as f:
                f.write(b)
            os.replace(tmp, dest)
            return dest, "ok"
        except Exception as e:  # network hiccup: exponential backoff
            err = e
            time.sleep(2 ** (k + 1))
    return dest, f"fail:{err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--states", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tag", default="", help="suffix for manifest file names")
    a = ap.parse_args()
    for y in a.years:
        sel = select_returns(os.path.join(a.index_dir, f"idx_{y}.csv"), a.states)
        os.makedirs(os.path.join(a.out, str(y)), exist_ok=True)
        sel.to_csv(os.path.join(a.out, f"manifest_{y}{a.tag}.csv"), index=False)
        print(y, "returns:", len(sel), flush=True)
        jobs = [(u, os.path.join(a.out, str(y), f"{e}.xml.gz")) for u, e in zip(sel.URL, sel.EIN)]
        status = {}
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for i, (dest, s) in enumerate(ex.map(lambda j: fetch(*j), jobs)):
                status[s.split(":")[0]] = status.get(s.split(":")[0], 0) + 1
                if i % 2000 == 0:
                    print(y, i, status, flush=True)
        print(y, "done", status, flush=True)


if __name__ == "__main__":
    sys.exit(main())
