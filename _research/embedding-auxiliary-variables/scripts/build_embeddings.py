"""Build every embedding the NE runs use, directly from the v3 populations (no splicing).

The cloud session built MiniLM by splicing changed rows through v1 -> v2 -> v3 and refit LSA on v3 text; a
direct build is equivalent up to floating point. LSA is deterministic (fixed seed in embed.embed_tfidf_lsa).

Usage: python3 -I scripts/build_embeddings.py        (EMBAUX_DATA, MINILM_DIR from the environment)
"""
import os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd

from embed import _save, build, lsa_fit_transform, text_hash

DATA = os.environ.get("EMBAUX_DATA", os.path.expanduser("~/embaux-data"))
mask = lambda t: re.sub(r"\$?\d[\d,\.]*", "#", t or "")
cur = pd.read_parquet(f"{DATA}/pop/ne_pop_v3.parquet")
prev = pd.read_parquet(f"{DATA}/pop/ne_wave2022_pop_v3.parquet")

# current wave (E* = lsa256_full; ladder L4 = minilm_full; numbers-masked control)
for name, be, texts, field, kw in [
        ("lsa256_full", "tfidf_lsa", cur.text_full.tolist(), "text_full (v3)", {"dim": 256}),
        ("lsa256_mission", "tfidf_lsa", cur.text_mission.tolist(), "text_mission (v3)", {"dim": 256}),
        ("lsa256_full_nonum", "tfidf_lsa", [mask(t) for t in cur.text_full], "text_full numbers masked (v3)", {"dim": 256}),
        ("minilm_full", "minilm", cur.text_full.tolist(), "text_full (v3)", {"threads": os.cpu_count()})]:
    X, meta = build(f"{DATA}/emb/ne_v3", name, be, texts, field, **kw)
    print("ne_v3", name, X.shape, flush=True)

# previous wave: Stage-1 selection of E* (LSA and MiniLM, mission and full text)
for name, be, field in [("lsa256_full", "tfidf_lsa", "text_full"), ("lsa256_mission", "tfidf_lsa", "text_mission"),
                        ("minilm_full", "minilm", "text_full"), ("minilm_mission", "minilm", "text_mission")]:
    kw = {"dim": 256} if be == "tfidf_lsa" else {"threads": os.cpu_count()}
    X, meta = build(f"{DATA}/emb/ne_wave2022_v3", name, be, prev[field].tolist(), field + " (previous wave v3)", **kw)
    print("ne_wave2022_v3", name, X.shape, flush=True)

# frozen heads: LSA fitted on the previous wave's text, applied to the current wave's
Ep, Ec = lsa_fit_transform(prev.text_full.tolist(), cur.text_full.tolist())
_save(f"{DATA}/emb/ne_frozen_v3", "lsa256_full_prev", Ep, {"field": "prev wave v3 (fit)", "input_sha256": text_hash(prev.text_full.tolist())})
_save(f"{DATA}/emb/ne_frozen_v3", "lsa256_full_curr", Ec, {"field": "curr wave v3 (transform)", "input_sha256": text_hash(cur.text_full.tolist())})
print("ne_frozen_v3", Ep.shape, Ec.shape)
