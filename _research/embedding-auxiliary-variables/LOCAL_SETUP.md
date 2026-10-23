# Moving this project to a MacBook

## Why moving locally helps
- **Persistent storage.** In the cloud container, the derived data (`/tmp/claude-0/data`, about 11 GB) disappears when the session ends. Only the repository survives.
- **No network sandbox.** Locally, `api.cohere.com`, Hugging Face, Common Crawl and `open.canada.ca` are reachable. That unblocks the Cohere and Qwen embedding rungs, the website-text arm, and the Canadian T3010 replication (user actions U1, U2, U4, U7).
- **More cores.** An Apple-silicon laptop with 8–12 cores runs the Monte Carlo 2–3× faster than the 4-core container. The national runs (8 held-out divisions, 285k units) are the step that needs it.

## When to move
At a stage boundary, with everything committed. That is now true: the NE proof of concept, the gate diagnosis and the nonresponse follow-up are all pushed to `claude/embedding-auxiliary-benchmark`.

The best cut-over point is **before the national build**. It is the largest CPU job, and its intermediate data should live on persistent storage. The remaining cloud time is best spent on analysis and writing, which needs no stored data.

## Setup (about 15 minutes of work, then unattended)

```bash
git clone <your repo> && cd <repo>/_research/embedding-auxiliary-variables
git checkout claude/embedding-auxiliary-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EMBAUX_DATA=~/embaux-data            # anywhere persistent; about 3 GB for NE
export MINILM_DIR=$EMBAUX_DATA/models/minilm/onnx
export WORKERS=8                            # performance cores minus one or two
bash scripts/reproduce_ne.sh data pop emb stage1     # downloads + builds, about 1 h
bash scripts/reproduce_ne.sh s2a s2b s2f s34 notebooks
```

Steps can be run one at a time. `scripts/reproduce_ne.sh` lists the expected times.

## What to expect
- **Exactness.** Every sample, fold and estimator stream is seeded, so the results match the committed ones up to BLAS floating-point differences.
- **Embeddings.** The cloud built MiniLM by splicing changed rows across frame versions; locally it is built directly. LSA (E*) is deterministic.
- **Stage 1.** It reruns on the v3 previous wave. The cloud used v2, which has the same frame and text and differs only in response timing. It should select the same E*: LSA full text, K = 16.
- **Notebook 01** (data red team) also reads the older v1/v2 populations and the 2026 BMF (for the leak contrast). Its committed outputs stand. To rerun it, build those inputs with the commands in `notes/working-log.md`.
- **Moving it out of this repo (optional).** The project lives under `_research/` in the website repository; the `_` prefix keeps Quarto from rendering it. For long-running local work, a dedicated repository would keep data scripts and large notebooks out of the site's history.
