"""Parse every downloaded return listed in a manifest into one parquet file per tax year."""
import argparse, gzip, multiprocessing as mp, os, sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse990_min import parse_return  # noqa: E402


def _one(path):
    try:
        with gzip.open(path) as f:
            r = parse_return(f.read())
        return r
    except Exception as e:  # keep going; failures are counted and reported
        return {"_error": f"{type(e).__name__}: {e}", "_path": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest, dtype=str)
    paths = [os.path.join(a.xml_dir, str(a.year), f"{e}.xml.gz") for e in man.EIN]
    present = [p for p in paths if os.path.exists(p)]
    print(f"manifest {len(paths)}  present {len(present)}", flush=True)
    with mp.Pool(a.procs) as pool:
        rows = pool.map(_one, present, chunksize=200)
    errs = [r for r in rows if r is None or "_error" in r]
    ok = [r for r in rows if r is not None and "_error" not in r]
    print(f"parsed {len(ok)}  failed/none {len(errs)}", flush=True)
    df = pd.DataFrame(ok)
    keep = ["EIN", "SubmittedOn", "ReturnTs", "TaxPeriod", "DocStatus", "Website", "YearFormed",
            "TotalRevenueCY", "TotalExpensesCY", "TotalAssetsBkEOY", "LegalDomicileState"]
    m = man[keep].rename(columns={"EIN": "ein", "ReturnTs": "idx_return_ts", "Website": "idx_website",
                                  "LegalDomicileState": "idx_state"})
    df = df.merge(m, on="ein", how="left")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print("wrote", a.out, df.shape, flush=True)


if __name__ == "__main__":
    main()
