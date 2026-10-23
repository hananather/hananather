"""Embedding backends with a reproducible on-disk cache.

Every embedding matrix is stored as  {cache}/{name}.npy  plus  {name}.json  holding the model id,
backend, input field, dimension, creation date and a SHA-256 of the exact input texts, so that
results can be reproduced even if an API model is later deprecated.

Backends
  tfidf_lsa      TF-IDF (1-2 grams) + truncated SVD; the classical text baseline
  minilm         all-MiniLM-L6-v2 (ONNX, 384-d), mean pooled + L2 normalised; local / on-prem
  cohere         Cohere embed-v4.0 via API (text, or images for PDF pages / screenshots).
                 Needs COHERE_API_KEY and network access to api.cohere.com.
  random         Gaussian noise of a given dimension: negative control
"""
from __future__ import annotations

import datetime as dt, hashlib, json, os, time

import numpy as np

MINILM_DIR = os.environ.get("MINILM_DIR", "/tmp/claude-0/models/minilm/onnx")


def text_hash(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update((t or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _save(cache, name, X, meta):
    os.makedirs(cache, exist_ok=True)
    np.save(os.path.join(cache, f"{name}.npy"), X.astype(np.float32))
    with open(os.path.join(cache, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=1)


def load(cache, name):
    X = np.load(os.path.join(cache, f"{name}.npy"))
    with open(os.path.join(cache, f"{name}.json")) as f:
        return X, json.load(f)


def embed_tfidf_lsa(texts, dim=256, seed=0):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                         strip_accents="unicode", lowercase=True)
    M = tf.fit_transform([t or "" for t in texts])
    Z = TruncatedSVD(dim, random_state=seed).fit_transform(M)
    return normalize(Z), {"vocab_size": len(tf.vocabulary_)}


def embed_minilm(texts, batch=64, max_len=256, threads=None):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MINILM_DIR, "tokenizer.json"))
    tok.enable_truncation(max_len)
    tok.enable_padding()
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(os.path.join(MINILM_DIR, "model.onnx"), so, providers=["CPUExecutionProvider"])
    # sort by length so that padding is minimal
    order = np.argsort([len(t or "") for t in texts])
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = tok.encode_batch([(texts[j] or "").lower() or "[empty]" for j in idx])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        m = np.array([e.attention_mask for e in enc], dtype=np.int64)
        h = sess.run(None, {"input_ids": ids, "attention_mask": m, "token_type_ids": np.zeros_like(ids)})[0]
        v = (h * m[..., None]).sum(1) / m.sum(1, keepdims=True)
        out[idx] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return out, {"max_len": max_len, "pooling": "mean", "lowercased": True}


def embed_cohere(texts=None, images=None, model="embed-v4.0", input_type="classification",
                 dim=1024, batch=96, sleep=0.0):
    """Text: list of strings. Images: list of data-URI strings ('data:image/png;base64,...').
    Requires `pip install cohere` and COHERE_API_KEY."""
    import cohere
    co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    items = texts if texts is not None else images
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        kw = dict(model=model, input_type=input_type, embedding_types=["float"], output_dimension=dim)
        r = co.embed(texts=chunk, **kw) if texts is not None else co.embed(images=chunk, **kw)
        out.append(np.asarray(r.embeddings.float_, dtype=np.float32))
        if sleep:
            time.sleep(sleep)
    X = np.vstack(out)
    return X / np.linalg.norm(X, axis=1, keepdims=True), {"input_type": input_type, "output_dimension": dim}


def embed_random(n, dim=384, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True), {"seed": seed}


def build(cache, name, backend, texts, field, **kw):
    path = os.path.join(cache, f"{name}.npy")
    h = text_hash(texts)
    if os.path.exists(path):
        X, meta = load(cache, name)
        if meta.get("input_sha256") == h:
            return X, meta
    t0 = time.time()
    if backend == "tfidf_lsa":
        X, extra = embed_tfidf_lsa(texts, **kw); model = "tfidf-1-2gram+svd"
    elif backend == "minilm":
        X, extra = embed_minilm(texts, **kw); model = "sentence-transformers/all-MiniLM-L6-v2 (onnx, chroma build)"
    elif backend == "cohere":
        X, extra = embed_cohere(texts=texts, **kw); model = kw.get("model", "embed-v4.0")
    elif backend == "random":
        X, extra = embed_random(len(texts), **kw); model = "gaussian-noise"
    else:
        raise ValueError(backend)
    meta = {"name": name, "backend": backend, "model": model, "field": field, "n": len(texts),
            "dim": int(X.shape[1]), "input_sha256": h, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0, 1), **extra}
    _save(cache, name, X, meta)
    return X, meta
