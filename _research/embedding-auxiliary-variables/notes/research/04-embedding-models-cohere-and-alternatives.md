# 04: Embedding models for the auxiliary-variable benchmark (Cohere embed-v4, embed-v5 and alternatives)

Written 2026-10-23. Sibling notes: `01-methods-literature.md` (estimators), `02-applied-precedents-and-real-data-sources.md` (data sources).

## 0. How the evidence was collected (read this first)

- WebSearch hit the session's search limit before this task started, and WebFetch failed with DNS errors on every host (docs.cohere.com, cohere.com, docs.aws.amazon.com, arxiv.org). `docs.cohere.com` and `cohere.com` also return 403 to curl from the sandbox.
- Cohere facts therefore come from the **source of docs.cohere.com**, the public repo `cohere-ai/cohere-developer-experience` (Fern `.mdx` pages plus `cohere-openapi.yaml`). I read it through raw.githubusercontent.com on 2026-10-23. Each fact is cited by its published URL and slug. This is first-party text, but the published site may render it slightly differently.
- Python SDK facts come from the wheels on PyPI: `cohere==7.3.0`, `voyageai==0.5.0`, `openai==3.37.0` and `mteb==2.26.20`, which carries model metadata and citation strings.
- Benchmark numbers in §3 are **computed by me** from the raw JSON in `embeddings-benchmark/results` (git commit `0acf39c`, 2026-10-22). They are not copied from a leaderboard.
- Google prices come from `cloud.google.com/vertex-ai/generative-ai/pricing`, which curl could reach. Some prices and regions come from secondary sources (the LiteLLM price table, AWS and Oracle sample repos). Those are marked **(secondary)**.
- Anything I could not confirm in this session is marked **[UNVERIFIED]**. Paper citations taken from mteb's citation strings are marked "(citation metadata from mteb; paper not fetched)".

---

## 1. Bottom line

1. **Cohere now has two generations.** `embed-v4.0` was released 2025-04-15. `embed-v5.0-pro` and `embed-v5.0-fast` were released 2026-09-30, and the docs' code examples now default to v5. Both generations take **text, images, and mixed text+image inputs (PDF pages as images) in one vector**. Both have a 128k context and Matryoshka dimensions: v4 offers {256, 512, 1024, 1536 (default)}; v5 offers {256, 512, 768, 1024, 1536, 2048 (default)}. Output types are float, int8, uint8, binary, ubinary and base64.
2. **Which `input_type` to use for covariates.** For covariates, use `input_type="classification"` as the primary setting, `clustering` when building strata or clusters, and `search_document` only as a sensitivity check. Fix the choice before seeing outcomes and never mix values within a population.
3. **Embedding the population is cheap; images cost more and run slower.**
   - Text: about 300k × 60 tokens = 18M tokens, or **about $2.16 per pass** with v4 at $0.12 per 1M tokens.
   - Page images: **about $60–$450 per pass** for 300k images, at $0.47 per 1M image tokens. The range depends on image tokens per page, which a 100-image pilot will measure.
   - Time is the binding constraint, not money. Production rate limits are 2,000 text inputs/min (2.5 h per text pass) and 400 images/min (12.5 h per image pass).
   - A trial key cannot do this. It is capped at 1,000 API calls per month, while one text pass needs 3,125 calls, and it allows only 5 images/min.
4. **Data residency is the real blocker, not price.**
   - The Cohere SaaS API, Azure pay-as-you-go hubs (US regions plus Sweden Central) and Bedrock offer no Canadian processing. From ca-central-1, Bedrock reaches Embed v4 only through the `global.` cross-region profile, which may process data in any commercial AWS region (secondary source, checked 2026-10-11).
   - Paths an NSO could use with confidential register text: a Cohere **private deployment** (Embed v4 runs as a one-GPU container), **Model Vault** (single-tenant; region by agreement; an Encrypted, confidential-computing tier in beta), or **open weights**.
5. **Benchmark position.**
   - On the 29 RTEB retrieval tasks it shares with other models, embed-v4.0 scores **71.6** (mean NDCG@10). That is strong but not top: Qwen3-Embedding-8B, open, scores 75.4; gemini-embedding-001 76.0; voyage-4-large 79.8; NVIDIA Nemotron-3-Embed-8B, open, 81.9.
   - int8 costs nothing (71.7) and binary costs 2.6 points (69.0).
   - **No public classification or clustering results exist for embed-v4** in the MTEB results repo. Those are the task families closest to covariate use, so the benchmark will have to produce that evidence itself.
6. **The open-weight alternatives are close enough that they must be in the ladder.**
   - Qwen3-Embedding (Apache-2.0, 0.6B/4B/8B, text) and Qwen3-VL-Embedding (Apache-2.0, 2B/8B, text, images and mixed) are the on-prem analogues of Cohere text and Cohere image.
   - Jina v4 and v5 are CC-BY-NC-4.0 and should be treated as not usable for production statistics.
   - ColPali-style multi-vector models and CLIP/SigLIP are the wrong tools for single-vector covariates of document pages.
7. **Reproducibility.**
   - Cohere's lifecycle runs Active → Legacy → Deprecated → Shutdown. Embed v2 was shut down on 2026-04-04, so assume v4 has a lifetime of a few years, not permanence.
   - Embed the finite population **once** and cache it, content-addressed by (provider, model id, input_type, output_dimension, embedding type, preprocessing version, input hash). Re-embed a canary set to detect drift.
   - Never call the API inside the Monte Carlo loop.

---

## 2. Cohere embed fact sheet

### 2.1 Model ids, modalities, limits

| Item | `embed-v4.0` | `embed-v5.0-pro` / `embed-v5.0-fast` | Source |
|---|---|---|---|
| Released | 2025-04-15 | 2026-09-30 | Cohere changelog [C7], [C8] |
| Modalities | Text, images, mixed texts/images ("i.e. PDFs") | same | [C1], [C11] |
| Context | 128k tokens | 128k | [C1] |
| Output dims (Matryoshka) | 256, 512, 1024, **1536 (default)** | 256, 512, 768, 1024, 1536, **2048 (default)** | [C1], [C8] |
| Similarity | cosine, dot product, Euclidean | same | [C1] |
| Embedding types | `float`, `int8`, `uint8`, `binary`, `ubinary`, `base64`. Binary is packed 8 bits per byte, so 1536 dims become 192 bytes. | v5 announcement lists float, int8 and binary | [C2], [C10], [C8] |
| `input_type` enum | `search_document`, `search_query`, `classification`, `clustering`, `image` (required for v3 and later) | same; `image` with v5 "defaults to `search_document`" | [C10], [C2] |
| Texts per call | max 96 (`texts`) | 96 | [C10] |
| Inputs per call (`inputs`, mixed) | max 96 | 96 | [C10] |
| Images per call (`images`) | "no limit on the number of images per call"; combined size ≤ **20 MB** | docs say up to 96 images | [C10], [C2] |
| Image formats | png, jpeg, webp, gif, sent as base64 data URI | same | [C2], [C10] |
| Image resizing (with `inputs`) | images > 2,458,624 px (1568²) are downsampled; images < 3,136 px (56²) are upsampled; payload ≤ 20 MB | stated for v5 in the current page | [C2] |
| Other request params | `max_tokens` (per-input truncation), `truncate` = `NONE`/`START`/`END` (default END), `priority` 0–999 (default 0 is highest) | same | [C10] |
| Billing metadata returned | `meta.billed_units.{input_tokens, image_tokens, images}`, `meta.api_version`, response `id` | same | [C10] |
| v5 specifics | — | Pro and Fast **share one embedding space**. Cohere recommends indexing with Pro and querying with Fast. | [C8] |

Fused input shape: `inputs=[{"content": [{"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]` returns one vector per input [C2].

### 2.2 Python SDK (cohere 7.3.0, `ClientV2.embed`)

The verified signature from `cohere/v2/client.py` is `embed(*, model, input_type, texts=None, images=None, inputs=None, max_tokens=None, output_dimension=None, embedding_types=None, truncate=None, priority=None)`. It returns `EmbedByTypeResponse`; float vectors are at `.embeddings.float_`, with the attribute aliased to the JSON key `float`.

```python
import cohere, base64
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])   # needs api.cohere.com allow-listed

# text covariates
r = co.embed(model="embed-v4.0", input_type="classification",
             texts=batch_of_up_to_96, output_dimension=1536,
             embedding_types=["float", "int8"], truncate="NONE")
E = r.embeddings.float_; log(r.id, r.meta.api_version, r.meta.billed_units)

# page image (or website screenshot)
url = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
r = co.embed(model="embed-v4.0", input_type="classification",
             inputs=[{"content": [{"type": "image_url", "image_url": {"url": url}}]}],
             output_dimension=1536, embedding_types=["float"])

# fused text + image -> one vector
r = co.embed(model="embed-v4.0", input_type="classification",
             inputs=[{"content": [{"type": "text", "text": mission},
                                  {"type": "image_url", "image_url": {"url": url}}]}],
             output_dimension=1536, embedding_types=["float"])
```

SDK details that matter:
- The v1 `cohere.Client.embed` auto-batches texts into groups of 96 over a 64-thread pool. That can exceed the 2,000 inputs/min limit. `ClientV2` does not batch, so throttle yourself.
- There are `cohere.BedrockClient` / `BedrockClientV2` and `cohere.OciClient` / `OciClientV2` wrappers; install the OCI one with `pip install cohere[oci]` [C12], [C14].
- Use `truncate="NONE"` so over-long inputs fail loudly instead of being silently cut.

### 2.3 Rate limits, batch jobs, pricing

- **Rate limits** [C5]:

  | Endpoint | Trial | Production |
  |---|---|---|
  | Embed (text) | 2,000 inputs/min | 2,000 inputs/min |
  | Embed (images) | 5 inputs/min | 400 inputs/min |
  | EmbedJob | 5 req/min | 50 req/min |

  Trial keys are "limited to 1,000 API calls a month".
- **Embed Jobs** (`/v1/embed-jobs`, `co.embed_jobs.create(dataset_id=..., input_type=..., model=..., embedding_types=..., truncate=...)`) are meant for 100K+ texts [C4]. The inputs are uploaded as a Cohere **Dataset** (CSV or JSONL with a `text` field), and **datasets are kept on Cohere for 30 days** unless deleted [C17]. That is a confidentiality issue.
  - The docs contradict each other on model support. The Embed Jobs page says the API is "only compatible with our embed v3.0 models", yet it also describes v4 output dimensions, and the models table lists Embed Jobs for `embed-v4.0` [C4], [C11]. Test before relying on it.
  - Embed Jobs take text only.
  - A newer `/v2/batches` API exists in the OpenAPI spec; its examples are for chat [C10].
- **Prices**: cohere.com/pricing was unreachable, so these are from LiteLLM's table, which cites cohere.com/pricing **(secondary)**:

  | Model | Text, per 1M tokens | Images, per 1M image tokens |
  |---|---|---|
  | `embed-v4.0` | **$0.12** | **$0.47** |
  | `embed-v5.0-pro` | $0.12 | $0.40 (Azure listing) |
  | `embed-v5.0-fast` | $0.08 | $0.40 (Azure listing) |

  Bedrock (`cohere.embed-v4:0`) and OCI (`cohere.embed-v4.0`) are also listed at $0.12 per 1M text tokens (secondary).
- **Tokens per image are not documented in what I could read.** The resize limits (56² minimum, 1568² maximum, ratio exactly 28²) suggest about one token per 28×28 patch, so at most 3,136 tokens per image. A third-party note says "tokens about pixels/784" **[UNVERIFIED]**. The response reports `billed_units.image_tokens` [C10], so measure it in a pilot.
- **Dedicated capacity** (Model Vault Standard, priced per instance) [C16]:

  | Model and tier | Hourly | Monthly | Annual |
  |---|---|---|---|
  | Embed 4, Small | $4.00 | $2,500 | $25,000 |
  | Embed 4, Medium | $5.00 | $3,250 | $32,500 |
  | Embed 5 Pro or Fast, Small | $3.00 | $2,000 | $20,000 |
  | Embed 5 Pro or Fast, Medium | $5.00 | $3,250 | $32,500 |

  Vaults have no rate limits. Encrypted Vault is a beta, "free of charge to design partners".

### 2.4 Where it runs: platforms, regions, data residency

| Route | Model id | Canadian processing? | Notes |
|---|---|---|---|
| Cohere SaaS API (`api.cohere.com/v2/embed`) | `embed-v4.0`, `embed-v5.0-*` | Not documented **[UNVERIFIED]** | Simplest. Datasets/Embed Jobs retained 30 days [C17]. |
| AWS Bedrock | `cohere.embed-v4:0`; profiles `us.cohere.embed-v4:0` and `global.cohere.embed-v4:0` (secondary: AWS samples, LiteLLM) | **No.** From ca-central-1, only `global.` exists, and it "may process a request outside Canada… any commercial AWS Region" [S2]. `ca.` profiles exist for few models. | Cohere's own platform table still says "(Coming Soon)" for v4 on Bedrock and OCI [C11]. That table is stale. v5 is not on Bedrock [C11]. |
| AWS SageMaker (Marketplace) | "Unique per deployment" [C11] | Possibly, if the listing supports ca-central-1 **[UNVERIFIED]** | You get your own endpoint. v4 and v5 are both listed [C7], [C8]. |
| Azure AI / Microsoft Foundry | `cohere-embed-v-4-plan`, `Cohere-Embed-V5-Pro`, `Cohere-Embed-V5-Fast` [C11] | **No** for pay-as-you-go. Hubs must be in East US, East US 2, North Central US, South Central US, Sweden Central, West US or West US 3 [C13]. | Two routes: `v1/embeddings` (Azure schema) and `v1/embed` (Cohere schema) [C13]. |
| Oracle OCI Generative AI | `cohere.embed-v4.0` (seen at `us-chicago-1` in Oracle devrel samples, secondary) | Unknown **[UNVERIFIED]** | v3 has separate `-image-` model ids on OCI [C11]. |
| Cohere private deployment (on-prem or VPC) | container image; `docker run --gpus=1` for "embed-v4" [C15] | **Yes**, in your own environment | "The data never leaves your environment, and the model can be fully network-isolated" [C15]. Needs a commercial agreement. Frozen weights give the best reproducibility. |
| Model Vault (Standard or Encrypted) | vault endpoint | "Region selection supports data-residency requirements; contact Cohere to confirm the regions" [C16] | Single-tenant. Zero Data Retention option. Encrypted tier uses TEEs (Intel TDX / AMD SEV-SNP plus NVIDIA confidential-computing GPUs) with remote attestation. |

For StatCan, the confidential production path is a private deployment or a Model Vault with Canadian residency confirmed in writing. The public-data proof of concept (IRS 990, public text) can use the SaaS API.

### 2.5 Versioning and deprecation (reproducibility for official statistics)

- **Lifecycle terms** [C6]:
  - *Active*.
  - *Legacy*: no more updates.
  - *Deprecated*: "no longer available to new customers but remain available to existing users until retirement". An existing user is anyone who used the model within 90 days before the announcement. A shutdown date is assigned.
  - *Shutdown*: requests fail.
- Notice is given "via email and in our documentation along with blog posts" [C6].
- **Track record**: `embed-english-v2.0`, `embed-english-light-v2.0` and `embed-multilingual-v2.0` were retired on **2026-04-04**, and requests now fail [C9]. Cohere's v2 representation models date from roughly 2022–2023 per the changelog index **[UNVERIFIED exact dates]**.
- There is **no documented guarantee that outputs are bit-identical over time** for a given model id. Deterministic serving, and whether outputs depend on batch composition, are also undocumented. Test both (§7).
- **Inconsistencies to keep on file**:
  - mteb metadata gives `release_date="2024-12-01"` for embed-v4.0, while Cohere's changelog says 2025-04-15.
  - LiteLLM lists `deprecation_date: 2027-01-23` for the Azure Embed v5 listings **[UNVERIFIED; looks like a marketplace placeholder]**.
- **Recommendation.** Treat a model change like a classification revision.
  - Pin `embed-v4.0` (never an alias).
  - Archive the vectors.
  - On any model change, run a bridging study: re-embed, re-estimate, and report the shift in estimates.
  - Run v5-pro beside v4 now. The v4→v5 shift *is* that bridging experiment.

### 2.6 Choosing `input_type` for covariates

- Cohere's guidance: "When using embedding for `classification` and `clustering` tasks, you can set `input_type` to either 'classification' or 'clustering'". The enum descriptions are "embeddings passed through a text classifier" and "embeddings run through a clustering algorithm" [C2], [C10].
- Mapping to survey uses:
  - **Model-assisted, GREG, calibration, nonresponse and SAE covariates.** These all mean "a prediction function of e", so use `classification`. That is the pre-registered primary.
  - **Building strata or post-strata by k-means or HDBSCAN.** Use `clustering`.
  - **Sensitivity check.** `search_document`, the most heavily trained regime.
- Images through `inputs` must use one of the four text input types [C2]. Use the same value as for text, so fused and separate vectors stay comparable.
- The value is an analyst degree of freedom (the "forking paths" problem). Fix it in the protocol, report all three, and choose estimators only on the design-based criteria in §5.

---

## 3. Benchmark position (computed, with caveats)

**Method.** I sparse-cloned `embeddings-benchmark/results` at commit `0acf39c` (2026-10-22). For each model I took the revision folder with the most tasks, and from each task file the `test` split's `main_score`, averaged over its subsets. The task lists come from `mteb==2.26.20`. These are my aggregates, not the official leaderboard, but they use the same files.

**Caveat.** These are retrieval and general benchmarks. A covariate's value depends on how well a regularized model can predict y from e (the population R², §5). MTEB *classification* is the closest proxy. Retrieval NDCG is a weak proxy.

**(a) RTEB retrieval tasks.** 29 tasks shared by embed-v4.0 and Qwen3-Embedding-8B (finance, health, legal, code, multilingual; several are private RTEB sets). Main score is NDCG@10.

| Model | Mean | Open weights? |
|---|---|---|
| nvidia Nemotron-3-Embed-8B | 81.9 | yes (openmdw-1.1) |
| voyage-4-large | 79.8 | no |
| voyage-3-large | 78.1 | no |
| gemini-embedding-001 | 76.0 | no |
| Qwen3-Embedding-8B | 75.4 | yes (Apache-2.0) |
| **Cohere embed-v4.0, int8** | **71.7** | no |
| **Cohere embed-v4.0, float** | **71.6** | no |
| jina-embeddings-v4 | 71.0 | yes (CC-BY-NC) |
| **Cohere embed-v4.0, binary** | **69.0** | no |
| text-embedding-3-large | 65.6 | no |
| snowflake-arctic-embed-l-v2.0 | 61.5 | yes |
| multilingual-e5-large-instruct | 60.9 | yes |
| Cohere embed-multilingual-v3.0 | 59.4 | no |
| gte-multilingual-base | 59.2 | yes |
| BGE-M3 | 58.9 | yes |
| text-embedding-3-small | 58.7 | no |
| Cohere embed-english-v3.0 | 49.2 | no |
| all-MiniLM-L6-v2 | 38.9 | yes |
| static-retrieval-mrl-en-v1 | 34.7 | yes |
| potion-base-8M (static) | 25.7 | yes |

**(b) MTEB(eng, v2), 41 tasks.** Mean(Task) and the Classification and Clustering means:

| Model | Mean(Task) | Classification | Clustering |
|---|---|---|---|
| Qwen3-Embedding-8B | 75.2 | 90.4 | 58.6 |
| Qwen3-Embedding-4B | 74.6 | 89.8 | 57.5 |
| gemini-embedding-001 | 73.3 | 90.0 | 59.4 |
| Qwen3-Embedding-0.6B | 70.5 | 84.6 | 54.1 |
| embeddinggemma-300m | 69.7 | 87.6 | 56.6 |
| embeddinggemma-2 | 67.5 | 88.8 | 51.7 |
| multilingual-e5-large-instruct | 65.6 | 75.7 | 49.9 |
| snowflake-arctic-l-v2.0 | 63.6 | 73.4 | 44.4 |
| granite-embedding-english-r2 | 62.8 | 70.7 | 47.2 |
| all-MiniLM-L6-v2 | 59.0 | 69.3 | 44.9 |
| potion-retrieval-32M | 53.9 | 67.0 | 41.3 |
| static-retrieval-mrl-en-v1 | 51.3 | 64.5 | 32.2 |

**Cohere v4/v5, OpenAI 3-large, Voyage, BGE-M3 and Jina v4 have no complete MTEB(eng, v2) run in the repo.** That gap is exactly the classification and clustering evidence this project needs.

**(c) ViDoRe v2, visual document retrieval.** 4 tasks; main score NDCG@5.

| Model | Mean |
|---|---|
| nemotron-colembed-vl-8b-v2 | 65.2 |
| gemini-embedding-2 | 62.4 |
| colnomic-embed-multimodal-7b | 60.2 |
| colqwen2.5-v0.2 | 60.1 |
| jina-embeddings-v4 | 58.2 |
| voyage-multimodal-3 | 54.9 |
| colpali-v1.3 | 54.7 |
| SigLIP-so400m-384 | 33.0 |
| jina-clip-v2 | 26.9 |

**Cohere embed-v4.0 has no ViDoRe result in the repo.**

**(d) Self-reported (Qwen3-VL-Embedding README) [Q2].**
- MMEB-V2 overall: Qwen3-VL-Embedding-8B 77.8, 2B 73.2. VisDoc overall: 82.4 and 79.2.
- MMTEB Mean(Task): 8B 67.9 vs Qwen3-Embedding-8B 70.6, Gemini Embedding 68.4, Cohere-embed-multilingual-v3.0 61.1, text-embedding-3-large 58.9.
- Cohere's Embed 4 blog claims (blog unreachable) are **[UNVERIFIED]**.

**Takeaways.**
- On public evidence, Cohere v4 is a strong second-tier frontier model for text retrieval. Open models of 4–8B parameters match or beat it.
- Its distinctive feature is **one API that handles text, page images and fused inputs in a single vector space with a 128k context**. The open challenger for that is Qwen3-VL-Embedding.
- int8 is free, so store int8 plus float.

---

## 4. Alternatives

Quality tiers are relative and based on §3:
- **A**: frontier, at or near the top of RTEB or MTEB.
- **B**: strong general-purpose model.
- **C**: baseline or small model.

"EN/FR" matters because Canadian registers are bilingual.

### 4.1 API models (external processing; fine for public-data proofs of concept, not for confidential frames)

| Model | Dims | Context | Modality | Price | Tier | Notes / source |
|---|---|---|---|---|---|---|
| Cohere `embed-v4.0` | 1536 (256–1536 MRL) | 128k | text, image, mixed | $0.12 per 1M text tokens; $0.47 per 1M image tokens (sec.) | A−/B+ | RTEB 71.6; §2 |
| Cohere `embed-v5.0-pro` / `-fast` | 2048 (256–2048 MRL) | 128k | text, image, mixed | $0.12 / $0.08 per 1M text tokens; $0.40 per 1M image tokens (sec.) | not yet benchmarked | Pro and Fast share one space [C8] |
| OpenAI `text-embedding-3-large` | 3072 (`dimensions` param) | 8192 tokens per input; ≤ 2048 inputs and ≤ 300k tokens per request | text | $0.13 per 1M; batch $0.065 (sec.) | B+ | OpenAPI spec [O1]; RTEB 65.6 |
| Google `gemini-embedding-001` | 3072 (MRL) | 2048 | text | $0.15 per 1M online, $0.12 batch | A | Vertex pricing [G1]; MTEB(eng, v2) 73.3 |
| Google `gemini-embedding-2` | 3072 / 1536 / 768 | 8192 | text, image, video, audio, PDF | $0.20 per 1M text (batch $0.10); $0.45 per 1M image tokens (batch $0.225) | A (ViDoRe v2 62.4) | Pricing page still says "Preview" [G1]; mteb gives GA 2026-04-22 |
| Google Vertex `multimodalembedding@001` | 1408 (sec.: 128/256/512 also) | short text | text, image, video | $0.0001 per image | C for documents | [G1]; text reportedly English-only and 32 tokens (sec.) **[UNVERIFIED]** |
| Voyage `voyage-4-large` / `-4` / `-4-lite` | 1024 (2048 option) | 32k | text | $0.12 / $0.06 / $0.02 per 1M (sec.) | A (RTEB 79.8) | SDK `voyageai.Client.embed(texts, model, input_type, output_dtype, output_dimension)` [V1] |
| Voyage `voyage-multimodal-3` / `-3.5` | 1024 (3.5: 256–2048, sec.) | 32k | text, image (3.5: plus video) | $0.12 per 1M text (sec.); pixel-based image price (sec.) **[UNVERIFIED]** | B+ (ViDoRe v2 54.9) | `Client.multimodal_embed(inputs, model, input_type, ...)` [V1] |
| Amazon `nova-2-multimodal-embeddings` | 3072 | 8192 | text, image, audio, video | **[UNVERIFIED]** | unknown | mteb metadata, 2025-10-28 |
| Amazon `titan-embed-text-v2:0` | 1024 | 8192 | text | $0.02 per 1M (sec.) | C/B− | **Callable in-region in ca-central-1** [S2]. Useful as the "Canadian-resident API" control. |

### 4.2 Open-weight / on-prem models (what an NSO can run inside its own enclave)

Parameters, dims, context, license and release dates are from mteb 2.26.20 `ModelMeta`. Scores are from §3.

| Model | Params | Dims (MRL) | Context | License | Modality | EN/FR | Tier | Role / caveat |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B / 4B / 8B | 0.60 / 4.0 / 7.6B | 1024 / 2560 / 4096 (MRL yes) | 32k | Apache-2.0 | text | yes (100+ languages) | B+ / A / A | **Primary open text rung.** Accepts instructions: "using instructions… typically yields an improvement of 1% to 5%" [Q1]. A task instruction such as "Represent this nonprofit's mission for predicting its size" is a free extra treatment. |
| Qwen3-VL-Embedding-2B / 8B (2026-01) | 2.1 / 8.1B | 2048 / 4096 (MRL) | 32k | Apache-2.0 | text, image, screenshot, video, mixed | yes (33 languages incl. FR) | A (self-reported) | **Open counterpart of Cohere image and fused input.** Single vector from the EOS token [Q2]. |
| NVIDIA Nemotron-3-Embed-1B / 8B; Nemotron-3.5-Embed-1B / 8B (2026-10-20) | 1.1 / 8.0B; 1.6 / 8.4B | 2048 / 4096 | 32k | openmdw-1.1 | 3: text; 3.5: text+image | multilingual | A (3-8B RTEB 81.9) | Strongest open text model on RTEB here. 3.5 is days old with no checked results. Check license terms **[UNVERIFIED]**. |
| BGE-M3 | 568M | 1024 | 8192 | MIT | text (dense, sparse and multi-vector) | yes | B | Classic multilingual baseline; RTEB 58.9 |
| multilingual-e5-large-instruct | 560M | 1024 | 514 | MIT | text | yes | B | MTEB(eng, v2) 65.6; short context |
| gte-multilingual-base | 305M | 768 | 8192 | Apache-2.0 | text | yes | B− | Cheap multilingual |
| granite-embedding-311m-multilingual-r2 (2026-04) | 312M | 128–768 (MRL) | 8192 | Apache-2.0 | text | yes | B (unchecked) | Enterprise-friendly provenance **[UNVERIFIED]** |
| embeddinggemma-300m; embeddinggemma-2 (2026-10-06) | 308M; 271M | 768; {768, 512, 256, 128} | 2048; 8192 | Gemma license | text; text+image+audio+video | yes | B+ | MTEB(eng, v2) 69.7 / 67.5. Read the Gemma terms. |
| voyage-4-nano | 346M | 2048 (2048 / 1024 / 512 / 256) | 32k | Apache-2.0 | text | multilingual (sec.) | B? | Shares the Voyage-4 family; runs locally through `voyageai` [V1] |
| nomic-embed-text-v1.5 | 137M | 768 (MRL) | 8192 | Apache-2.0 | text | EN | B− | Fully open training data and code (Nussbaum et al. 2024, arXiv:2402.01613) |
| all-MiniLM-L6-v2 | 22.7M | 384 | 256 | Apache-2.0 | text | EN only | C | **Already in sandbox** (ONNX). Measured **about 35 texts/s on 4 vCPU**, so 300k texts take about 2.4 h. |
| model2vec potion-base-8M; static-retrieval-mrl-en-v1 | 7.6M; 31M | 256; 1024 | ∞ (static) | MIT; Apache-2.0 | text | EN | C− | "Static" rung; MTEB(eng, v2) about 51–54 |
| jina-embeddings-v4; jina-v5-omni | 3.9B; 1.0–1.6B | 2048; ≤1024 | 32k | **CC-BY-NC-4.0** | text+image; omni | yes | A−/B+ | **License excludes production use**; benchmark only, if at all |
| ColPali v1.3, ColQwen2.5 v0.2, colnomic-embed-multimodal-7b, nemotron-colembed | 2.9–8.7B | **multi-vector**: 128 dims per patch or token | 16k–262k | Apache-2.0 adapters; base-model licenses vary **[UNVERIFIED]** | page images | partly | A for retrieval | **Not a single vector.** It must be mean-pooled to use as a covariate, which throws away what it was trained for. Use only as a sensitivity rung. |
| CLIP ViT-L/14, SigLIP / SigLIP2 so400m | 0.43B; 0.88–1.1B | 768; 1152 | 77; 64 text tokens | MIT / Apache-2.0 | natural images + short text | partly | C for documents (ViDoRe v2 SigLIP 33.0) | Fine for photos (storefronts, satellite tiles). Wrong for PDF pages. |

**Sandbox reality.**
- Weights for every model above except MiniLM come from huggingface.co, which is blocked here; so are ModelScope, Ollama and the qdrant-fastembed GCS bucket.
- Running the open rungs needs `huggingface.co` allow-listed, or weights copied in. The 4–8B and VL models also need a GPU.
- Only TF-IDF/LSA, GloVe or spaCy vectors (from GitHub releases) and MiniLM are runnable now.

---

## 5. The embedding ladder to benchmark

The scientific question is not "is Cohere good?". It is: **once the frame's conventional variables are in the model, how much more design-based efficiency does unstructured text or document information add, and how much of that needs a frontier proprietary multimodal model?**

The ladder breaks the total gain into increments. Each rung changes one thing. All rungs share:
- the same input text (a fixed template with no financial numbers),
- the same frame variables,
- the same downstream learners (cross-fitted ridge with CV-chosen penalty as primary; gradient boosting as sensitivity),
- the same estimators and sample designs (`01-methods-literature.md`).

| Rung | Representation | Increment it isolates | Runs where | Cost (300k units) |
|---|---|---|---|---|
| **R0** | Frame only: NTEE major group × state × asset/income code (plus BMF `INCOME_AMT`/`ASSET_AMT` where realistic). HT, post-stratified, GREG and calibration. | The baseline every embedding must beat. If the frame has lagged size measures, R0 is already strong, and that is the honest comparison. | anywhere | 0 |
| **R0-neg** | (i) Gaussian random vectors with the same dimension. (ii) Real embeddings **permuted across units**. (iii) Embedding of an empty string or name only. | **Negative controls.** The machinery should show a gain of about 0 and keep CI coverage. If it shows a gain, the pipeline is overfitting or leaking. | anywhere | 0 |
| **R1** | Hand-built text features: length, presence of mission text, about 30 keyword flags (hospital, university, church, foundation, housing…). | What a methodologist could do without ML. | anywhere | 0 |
| **R2** | TF-IDF (unigrams and bigrams, min_df ≥ 5) → TruncatedSVD (LSA) at k ∈ {64, 256} | Lexical signal; deterministic and transparent | in the enclave | CPU, minutes |
| **R3** | Static word vectors averaged (GloVe 300d or spaCy `en_core_web_lg`; model2vec potion if available) | Distributional semantics without context | in the enclave | CPU, minutes |
| **R4** | all-MiniLM-L6-v2 (384d) | Small contextual model, the cheapest "real embedding" | in the enclave (in sandbox now) | about 2.4 h CPU |
| **R5** | Strong open text model: Qwen3-Embedding-0.6B (primary; also 8B), BGE-M3 or mE5-large-instruct for EN/FR. Run Qwen with and without a task instruction. | Model scale and quality, with nothing leaving the enclave. **The key control for the Cohere question.** | GPU in the enclave | GPU hours |
| **R6** | **Cohere embed-v4.0, text**. Primary `classification`, 1536 float. Factors: `input_type` ∈ {classification, clustering, search_document}; dims ∈ {256, 512, 1024, 1536}; float vs int8. | The proprietary-API increment. The input_type and dims factors measure sensitivity to analyst choices. | external API (public data only) | about $2.16 per pass |
| **R6b** | Cohere embed-v5.0-pro, text (2048 and 1536) | Vendor version change; the bridging-study analogue | external | about $2.16 |
| **R7** | **Cohere embed-v4.0, image** of the document page or website screenshot; **R7f** fused `inputs` (text + image → one vector) | Whether layout and visual content add anything beyond the text | external | about $60–$450 per pass |
| **R7-open** | Qwen3-VL-Embedding-2B or 8B on the same images (fused too); optionally a mean-pooled ColQwen as sensitivity | Whether the multimodal increment needs Cohere | GPU in the enclave | GPU hours |
| **R8** | Ceiling: (a) a small model fine-tuned on a *disjoint* earlier year (labels allowed); (b) the oracle line from the population R² (below). | How far the off-the-shelf models are from what labels could achieve | GPU | — |

**What to report for each rung:**
1. **Census diagnostic.** Population-level, K-fold cross-fitted R² of y on (frame, embedding). Under GREG with a working model, the approximate variance is ∝ S²_y(1 − R²)/n. The predicted relative efficiency versus the frame-only GREG is RE ≈ (1 − R²_frame)/(1 − R²_frame+emb). This first-order approximation **ignores model-fitting variance at small n**. That is why the Monte Carlo is needed, and why the gap between predicted and realized RE at n = 200, 500, 2,000 is itself a result.
2. **Monte Carlo metrics**: relative bias, variance, relative RMSE, RE versus R0, CI coverage and length (see `01-methods-literature.md` §9).
3. **Share of the R0→best gap closed by each rung.** This is the headline decomposition: "x% of the achievable gain is already in TF-IDF; Cohere adds y points over Qwen-0.6B".
4. **Matched-dimension comparisons.** Project every embedding to the same k (64 and 256) with PCA fitted on the population. This separates "more dimensions" from "better semantics" and avoids rewarding models only for having 1536–4096 dimensions when n is about 500.

**Design rules that keep the ladder honest.**
- **No leakage through the document.** A 990 page image or full-return text contains the revenue figure, so its embedding is a noisy measure of y, not a frame auxiliary.
  - Use text and images that would exist on the frame **before** collection: the prior-year filing, the website, register descriptions.
  - Or redact the numeric fields.
  - Then compare against R0 *including the same lagged numbers*. If the image only re-reads last year's revenue, it adds nothing an NSO lacks.
  - Report one run with numbers masked and one with numbers visible. The difference measures how far the model reads the numbers in the image.
- **Missing text.** Units with no mission statement or website get a fixed placeholder plus a missingness indicator. Report coverage by stratum, because the text-poor units are usually the small ones.
- **One setting per population.** Never mix input_type, dims, provider (Cohere API vs Bedrock vs Azure) or model revision within a population matrix.
- **Pre-registration.** Fix the primary rung settings (Cohere `classification`, 1536, float; Qwen3-0.6B without instruction) before looking at Monte Carlo results. Report everything else as sensitivity.

---

## 6. Reproducibility protocol (cache once, simulate many)

The finite population is fixed, so **embed all N units once per configuration**. Monte Carlo replicates only resample rows of a cached matrix. API cost and drift risk therefore do not scale with the number of replicates.

**Cache key** (content-addressed):
```python
key = sha256(json.dumps({
  "provider": "cohere-api",          # or bedrock / azure / local
  "model": "embed-v4.0", "sdk": "cohere==7.3.0", "api_version": r.meta.api_version.version,
  "input_type": "classification", "output_dimension": 1536, "embedding_type": "float",
  "truncate": "NONE", "max_tokens": None,
  "preproc": "text_template_v1+NFC+ws-collapse",           # or "render:pypdfium2==X,dpi=150,page=0,maxpx=2458624,png"
  "input_sha256": sha256(normalized_text_or_png_bytes)
}, sort_keys=True).encode()).hexdigest()
```

**Store** one Parquet file per configuration with columns `unit_id, input_sha256, vector (float32 or int8), response_id, billed_input_tokens, billed_image_tokens, ts_utc`. Add a manifest JSON with the config, row counts, a sha256 of the vector matrix, the git commit of the code, and the date.

Open-weight models are pinned by HF revision hash (e.g. Qwen3-Embedding-0.6B `b22da495…`, BGE-M3 `5617a9f6…`, MiniLM `8b3219a9…` from mteb metadata), plus library versions, dtype, device and a container image digest.

**Tests to run before the full campaign** (each costs cents):
1. **Repeatability.** Embed 200 texts twice and compare max |Δ| and minimum cosine. Expect about 1.0. Report it either way.
2. **Batch invariance.** Embed a text alone and inside a batch of 96, and compare.
3. **Matryoshka equivalence.** Compare the API `output_dimension=256` against `float[:256]` re-normalized. If they match within tolerance, embed once at full dimension and truncate locally. That saves 75% of the dimension-factor cost.
4. **int8 vs float.** Measure the downstream R² difference; RTEB suggests it is negligible.
5. **Image tokens.** Record `billed_units.image_tokens` on 100 rendered pages at two DPIs. This fixes the cost model.
6. **Canary drift.** Re-embed a frozen 1,000-item canary set (text and images) at the start and end of the campaign, then monthly. Alarm if cosine to the cached vectors falls below 0.999 (threshold to calibrate from test 1).

**Notebook rule.** The analysis notebook reads only cached matrices plus manifests and asserts the manifest hashes. Embedding code lives in a separate, logged script.

---

## 7. Cost and time estimate (Cohere, about 300k units)

| Job | Volume | Price basis | Cost | Wall time (production key) |
|---|---|---|---|---|
| Text, 1 pass (v4, one input_type, 1536) | 300k × 60 = 18.0M tokens; 3,125 calls | $0.12 per 1M tokens | **$2.16** | 150 min at 2,000 inputs/min |
| Text, full factorial (3 input_types × 4 dims) | 12 passes | same | $25.92 (or **$6.48** if the Matryoshka test passes: 3 passes) | 30 h (or 7.5 h) |
| Text, v5-pro and v5-fast | 1 pass each | $0.12 / $0.08 | $2.16 / $1.44 | 150 min each |
| Page images, 1 pass (v4) | 300k images × T image tokens | $0.47 per 1M image tokens (sec.) | T = 432 (512×662 px): **$61**; T = 1,193 (850×1100, about 100 DPI letter): **$168**; T = 2,683 (150 DPI letter): **$378**; T = 3,136 (cap): **$442**. T ≈ px/784 is **[UNVERIFIED]**; measure it. | 750 min = 12.5 h at 400 images/min |
| Fused text + image | same plus about 60 text tokens per unit | — | image cost + $2.16 | 12.5 h |
| Images, v5 | same | $0.40 per 1M (sec.) | $52–$376 | 12.5 h |
| Pilot (all tests in §6) | about 2,000 texts + 200 images | — | **< $0.50** | < 15 min |
| Storage per configuration | 300k × 1536 | float32 / int8 | 1.84 GB / 0.46 GB (256-d float: 0.31 GB) | — |

**What the user has to set up.**
- A production key. A trial key's 1,000 calls/month and 5 images/min make the full campaign impossible.
- `api.cohere.com` allow-listed, plus `COHERE_API_KEY`.
- Optionally `huggingface.co` and a GPU for the open rungs.

**For comparison (text, 18M tokens).**

| Model | Cost |
|---|---|
| OpenAI 3-large | $2.34 (batch $1.17) |
| gemini-embedding-001 | $2.70 (batch $2.16) |
| gemini-embedding-2 | $3.60 (batch $1.80) |
| voyage-4-large | $2.16 |
| Vertex multimodalembedding@001 (images) | $30 for 300k images |

Every option is negligible next to analyst time. **Choose the model on governance and evidence, not on API price.**

---

## 8. Red-team notes specific to the model layer

1. **Benchmark-transfer risk.** All public evidence for Cohere v4 is retrieval. Covariate value is linear decodability of y. The ladder has to show whether retrieval rank predicts covariate rank. A plausible failure: MiniLM or TF-IDF captures most of the gain, because size and sector signals are lexical ("hospital", "university").
2. **Leakage risk is highest for the image rung** (§5): page images contain numbers.
3. **Provider equivalence is untested.** `embed-v4.0` on the Cohere API, `cohere.embed-v4:0` on Bedrock and the Azure deployment may not be byte-identical. Pick one provider per study.
4. **Residency.** In practice, none of the managed clouds keep Embed v4 processing in Canada (§2.4). Only private deployment, a Model Vault with a written region guarantee, or open weights fit confidential frames. Make "open-weight rung within X% of Cohere" a decision criterion.
5. **Licensing.** Jina v4 and v5 (CC-BY-NC) and possibly some ColPali bases should not go into a production recommendation. Gemma and openmdw terms need legal review **[UNVERIFIED]**.
6. **Language.** Cohere v4 and v5, Qwen3, BGE-M3, mE5 and Granite-multilingual cover French. MiniLM, nomic-v1.5 and the static English models do not. For Canadian registers, add a French-text subset check.
7. **Docs inconsistencies.** v4 support in Embed Jobs, the stale Bedrock and OCI table, and the v4 release date (§2) show why the protocol records what the API actually returned (`api_version`, billed units), not what the docs say.

---

## 9. Sources

Cohere first-party. All were read from `github.com/cohere-ai/cohere-developer-experience` (main branch, 2026-10-23) and are cited by published URL.

- [C1] Cohere. "Cohere's Embed Models (Details and Application)". https://docs.cohere.com/docs/cohere-embed (fern/pages/models/embed.mdx)
- [C2] Cohere. "Introduction to Embeddings at Cohere". https://docs.cohere.com/docs/embeddings (fern/pages/text-embeddings/embeddings.mdx)
- [C3] Cohere. "Unlocking the Power of Multimodal Embeddings". https://docs.cohere.com/v2/docs/multimodal-embeddings
- [C4] Cohere. "Batch Embedding Jobs with the Embed API". https://docs.cohere.com/v2/docs/embed-jobs-api
- [C5] Cohere. "Different Types of API Keys and Rate Limits". https://docs.cohere.com/docs/rate-limits
- [C6] Cohere. "Deprecations". https://docs.cohere.com/docs/deprecations
- [C7] Cohere changelog. "Announcing Embed Multimodal v4" (2025-04-15). https://docs.cohere.com/changelog/embed-multimodal-v4
- [C8] Cohere changelog. "Announcing Cohere's Embed 5 Models" (2026-09-30). https://docs.cohere.com/changelog/embed-v5
- [C9] Cohere changelog. "Retirement of Embed v2.0 and Aya Expanse / Vision 8B" (2026-04-04). https://docs.cohere.com/changelog/2026-04-04-embed-v2-aya-8b-retirement
- [C10] Cohere OpenAPI spec, `cohere-openapi.yaml`: `/v2/embed` request body, `EmbedInputType`, `EmbeddingType`, `EmbedInput`, `ApiMeta.billed_units`, `/v2/batches`. https://github.com/cohere-ai/cohere-developer-experience/blob/main/cohere-openapi.yaml
- [C11] Cohere. "An Overview of Cohere's Models": platform model-id table. https://docs.cohere.com/docs/models
- [C12] Cohere. "Cohere Models on Amazon Bedrock". https://docs.cohere.com/v2/docs/amazon-bedrock
- [C13] Cohere. "Cohere on Microsoft Azure": models list and pay-as-you-go hub regions. https://docs.cohere.com/v2/docs/cohere-on-microsoft-azure
- [C14] Cohere. "Cohere on Oracle Cloud Infrastructure (OCI)". https://docs.cohere.com/docs/oracle-cloud-infrastructure-oci
- [C15] Cohere. "Private Deployment Overview": https://docs.cohere.com/docs/private-deployment-overview. "Model Deployment" (single container): https://docs.cohere.com/v2/docs/single-container-on-private-clouds
- [C16] Cohere. Model Vault overview: https://docs.cohere.com/docs/model-vault. Standard pricing: https://docs.cohere.com/docs/model-vault/standard/pricing. Encrypted supported models, compliance and pricing: https://docs.cohere.com/docs/model-vault/encrypted/supported-models, …/compliance, …/pricing
- [C17] Cohere. "Datasets" (30-day retention). https://docs.cohere.com/docs/datasets
- [C18] `cohere` Python SDK 7.3.0 (PyPI wheel; `cohere/v2/client.py`, `cohere/client.py`, `types/embed_by_type_response_embeddings.py`). https://pypi.org/project/cohere/

Benchmarks and model metadata:
- [M1] `mteb` 2.26.20 (PyPI wheel; `mteb/models/model_implementations/*.py` ModelMeta; `benchmarks/benchmarks.py` task lists). https://pypi.org/project/mteb/
- [M2] embeddings-benchmark/results, commit 0acf39c2bd00210e746b163fdcf215954f281af0 (2026-10-22). https://github.com/embeddings-benchmark/results
- [M3] Muennighoff et al. (2022). "MTEB: Massive Text Embedding Benchmark". arXiv:2210.07316 (citation metadata from mteb; paper not fetched)
- [M4] Enevoldsen et al. (2025). "MMTEB: Massive Multilingual Text Embedding Benchmark". arXiv:2502.13595, doi:10.48550/arXiv.2502.13595 (citation metadata from mteb)
- [M5] Xiao, Chung, Kerboua et al. (2025). "MIEB: Massive Image Embedding Benchmark". ICCV 2025, pp. 22187–22198 (citation metadata from mteb)
- [M6] Macé, Loison, Faysse (2025). "ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval". arXiv:2505.17166. Loison et al. (2026). "ViDoRe V3…". arXiv:2601.08620 (citation metadata from mteb)

Models:
- [Q1] QwenLM. Qwen3-Embedding README: https://github.com/QwenLM/Qwen3-Embedding. Zhang et al. (2025), "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models", arXiv:2506.05176 (citation metadata from mteb)
- [Q2] QwenLM. Qwen3-VL-Embedding README (model specs, MMEB-V2 and MMTEB tables): https://github.com/QwenLM/Qwen3-VL-Embedding. Tech report arXiv:2601.04720 (citation metadata from mteb)
- Chen et al. (2024). "BGE M3-Embedding…". arXiv:2402.03216. Wang et al. (2024). "Multilingual E5 Text Embeddings: A Technical Report". arXiv:2402.05672. Faysse et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models". arXiv:2407.01449. Nussbaum et al. (2024). "Nomic Embed…". arXiv:2402.01613. Günther et al. (2025). "jina-embeddings-v4…". arXiv:2506.18902. "Gemini Embedding…" (2025), arXiv:2503.07891. "Gemini Embedding 2…" (2026), arXiv:2605.27295. SigLIP arXiv:2303.15343. SigLIP 2 arXiv:2502.14786. CLIP arXiv:2103.00020. (All citation metadata from mteb; papers not fetched.)
- [O1] OpenAI OpenAPI spec, `CreateEmbeddingRequest`. https://github.com/openai/openai-openapi (master/openapi.yaml)
- [V1] `voyageai` Python SDK 0.5.0 (PyPI; `client.py`, `local/model_registry.py`). https://pypi.org/project/voyageai/
- [G1] Google Cloud. "Vertex AI generative AI pricing", embedding sections (fetched 2026-10-23). https://cloud.google.com/vertex-ai/generative-ai/pricing

Secondary (use with care):
- [S1] BerriAI/litellm, `model_prices_and_context_window.json` (main, 2026-10-23); per-model `source` fields cite cohere.com/pricing, aws.amazon.com/bedrock/pricing, Oracle and Azure retail price APIs. https://github.com/BerriAI/litellm
- [S2] AWS Solutions Library. "Canada (Central) Region Model Support", accelerated-intelligent-document-processing-on-aws (live checks 2026-10). https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws/blob/main/docs/canada-region-model-support.md
- [S3] aws-samples/Cohere-on-AWS, `demo-pdf-processing/pdf_processor_cohere.py` (`us.cohere.embed-v4:0`, us-west-2). https://github.com/aws-samples/Cohere-on-AWS
- [S4] oracle-devrel/technology-engineering, `ai/generative-ai-service/image-embed-and-search/README.md` (`cohere.embed-v4.0`, us-chicago-1). https://github.com/oracle-devrel/technology-engineering
