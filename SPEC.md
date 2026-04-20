# LegalMind — Schema, Ingestion Pipeline & Tech Stack

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Frontend | React + Next.js + Tailwind | Two-panel layout; SSR for demo responsiveness |
| API | Python FastAPI (single service) | One process, two routers (`/research`, `/draft`). Shares BGE embedding model in memory. Go gateway removed. |
| Vector DB | Actian Vector DB | Hybrid SQL + vector in one query operation. Core technical differentiator — SQL predicates applied at query time, not post-retrieval. |
| Embeddings | BAAI/bge-large-en-v1.5 (local) | 768-dim. Outperforms all-MiniLM-L6-v2 on legal retrieval. Zero API cost, no rate limits, fully offline. |
| LLM — research | Claude Sonnet | Query-time only for "Why this matters for your case" field (~50 tokens per result). Everything else pre-generated. |
| LLM — drafting | Claude Sonnet (streaming) | Assembles grounded demand letter from retrieved chunks + templates. Citation hallucination blocked at prompt + post-validation level. |
| LLM — ingestion | Claude Haiku (batch) | Generates `ai_summary`, `key_concepts`, `outcome`, `is_primary_holding` at ingestion time. Fast and cheap for 5,000+ docs. |
| Chunking | LlamaIndex SemanticSplitterNodeParser | Semantic splits on legal opinion structure — not fixed token count. Threshold: 90th percentile breakpoint. Size guard: 80–600 tokens. |
| Data sources | CourtListener API + HuggingFace pile-of-law + US Code XML + CA Code XML | Real, citable, free. CourtListener provides `negative_treatment` field (source of truth for `is_good_law`). |

---

## Data Schema

Four tables. Three active collections + one reference table. All vector columns are `VECTOR(768)` matching BGE-large output dimensions.

---

### `case_summaries`

One row per case. Searched in Stage 1 of dual-RAG. Contains AI-generated summary and key concepts produced at ingestion.

| Field | Type | Source | Purpose |
|---|---|---|---|
| `case_id` | VARCHAR(64) PK | CourtListener opinion ID or HF hash | Links to all chunks in `case_law` |
| `case_name` | VARCHAR(512) | API metadata | Display; denormalized to avoid join at query time |
| `citation` | VARCHAR(128) | API metadata | Display; SQL filter |
| `court` | VARCHAR(128) | API `court_id` | Display; SQL filter |
| `court_tier` | INTEGER | Computed: SCOTUS=3, Circuit=2, District/State=1 | Re-ranking signal; binding vs persuasive badge |
| `jurisdiction` | VARCHAR(32) | Computed from `court_id` | "federal" or "CA" — SQL filter |
| `practice_area` | VARCHAR(64) | Keyword classifier at ingestion | employment / tort / family / criminal — SQL filter |
| `year` | INTEGER | Parsed from `cluster.date_filed` | SQL filter; recency boost |
| `is_good_law` | BOOLEAN | CourtListener `negative_treatment` (inverted) | SQL WHERE filter; UI badge. Real data — not a default. pile-of-law docs flagged separately. |
| `outcome` | VARCHAR(32) | Haiku-extracted at ingestion | "plaintiff_won" / "defendant_won" / "mixed" — filterable in UI; re-ranking boost for demand letters |
| `ai_summary` | VARCHAR(1500) | Haiku-generated at ingestion | 4-sentence summary: facts, legal question, holding, significance. This field is embedded for Stage 1 search. |
| `key_concepts` | JSON array | Haiku-extracted at ingestion | ["wrongful termination", "FLSA §215", "retaliation"]. Rendered as Key Reasoning bullets in result card. |
| `summary_embedding` | VECTOR(768) | BGE-large on `ai_summary` | Stage 1 searches this column |

---

### `case_law`

One row per semantic chunk of a case opinion. Searched in Stage 2 of dual-RAG, constrained to `case_id IN (top-20 from Stage 1)`.

| Field | Type | Source | Purpose |
|---|---|---|---|
| `id` | VARCHAR(64) PK | Chunk-level ID (case_id + chunk index) | Primary key |
| `case_id` | VARCHAR(64) FK | Foreign key → `case_summaries.case_id` | Load-bearing for Stage 2 WHERE clause |
| `case_name` | VARCHAR(512) | Denormalized | Display |
| `citation` | VARCHAR(128) | Denormalized | Display; injected into draft |
| `court` | VARCHAR(128) | Denormalized | Display |
| `court_tier` | INTEGER | Denormalized | Re-ranking |
| `jurisdiction` | VARCHAR(32) | Denormalized | Label in draft |
| `practice_area` | VARCHAR(64) | Denormalized | Template scoping in drafting |
| `year` | INTEGER | Denormalized | Recency signal |
| `is_good_law` | BOOLEAN | Denormalized | Blocks chunk from draft context if FALSE |
| `is_primary_holding` | BOOLEAN | Haiku-tagged at ingestion | Which chunk contains the ratio decidendi. Boosted in re-ranking. Used verbatim in result card HOLDING field. |
| `holding_text` | VARCHAR(2000) | Semantic chunk (80–600 tokens) | What is embedded; returned in search; injected as `[CASE-XXX]` context in draft prompt |
| `full_cite_str` | VARCHAR(256) | Formatted at ingestion | "Smith v. Jones, 547 F.3d 410 (9th Cir. 2019)". Injected verbatim as citation string in draft. |
| `embedding` | VECTOR(768) | BGE-large on `holding_text` | Stage 2 cosine similarity search |

**Chunking parameters:** SemanticSplitterNodeParser, breakpoint_percentile_threshold=90, buffer_size=1. Size guard: merge chunks < 80 tokens with next; halve chunks > 600 tokens. Average ~2.5 chunks per opinion.

---

### `statutes`

One row per statute section. No chunking — sections are already atomic legal units.

| Field | Type | Source | Purpose |
|---|---|---|---|
| `id` | VARCHAR(64) PK | Computed: jurisdiction + code + section number | Primary key |
| `title` | VARCHAR(512) | XML header | Display in result card |
| `code` | VARCHAR(128) | e.g. "42 U.S.C. § 1983" / "CA Labor Code § 1102.5" | Display; injected as `[STATUTE-XXX]` in draft |
| `jurisdiction` | VARCHAR(32) | "federal" or "CA" | SQL filter |
| `practice_area` | VARCHAR(64) | Manually tagged at ingestion | SQL filter; template scoping |
| `section_text` | VARCHAR(2000) | Full section text from XML parse | Embedded for search; injected into draft prompt |
| `last_amended` | INTEGER | Year from XML metadata | Display; recency signal |
| `embedding` | VECTOR(768) | BGE-large on `section_text` | Cosine similarity search (same query as case_law) |

**Sources:** US Code XML (uscode.house.gov) — Title 29 (FLSA, FMLA), Title 42 (§1983, ADA, Title VII), Title 28. California Codes (leginfo.legislature.ca.gov) — Labor Code §§ 98–132, 1100–1199; Family Code §§ 3000–4000; Penal Code §§ 832–847; Civil Code §§ 1714–1725.

---

### `clause_templates`

One row per demand letter clause variant. The drafting engine's structural skeleton. Manually curated.

| Field | Type | Source | Purpose |
|---|---|---|---|
| `id` | VARCHAR(64) PK | Manual slug e.g. "emp-fed-intro-aggressive" | Primary key; attorney can pin a specific template |
| `clause_name` | VARCHAR(256) | Manual | Debug and admin UI |
| `doc_type` | VARCHAR(64) | "demand_letter" | Primary SQL filter in template retrieval |
| `section` | VARCHAR(64) | "intro" / "facts" / "legal_standard" / "argument" / "damages" / "demand" | Controls which section of the draft this template populates |
| `practice_area` | VARCHAR(64) | employment / tort / family / criminal | SQL filter — matched to user's matter |
| `jurisdiction` | VARCHAR(32) | "federal" / "CA" / "both" | SQL filter |
| `tone` | VARCHAR(32) | "aggressive" / "moderate" / "conservative" | Default: moderate. Future UI control. |
| `clause_text` | VARCHAR(3000) | Manually written prose with {placeholder} slots | Embedded for semantic retrieval; injected as structural skeleton into LLM prompt |
| `embedding` | VECTOR(768) | BGE-large on `clause_text` | Semantic match when SQL filters return multiple candidates |

**Volume:** 5 clause types × 4 practice areas × 2 jurisdictions × 3 tones ≈ 120 base variants + ~170 sub-clauses (damages paragraphs, statutory citations, boilerplate) = ~290 rows.

---

### Corpus Summary

| Collection | Source docs | Indexed rows | Vector column |
|---|---|---|---|
| `case_summaries` | 3,600 cases | 3,600 | VECTOR(768) on `ai_summary` |
| `case_law` | 3,600 cases | ~9,000 chunks | VECTOR(768) on `holding_text` |
| `statutes` | 500 sections | 500 | VECTOR(768) on `section_text` |
| `clause_templates` | Manual | ~290 | VECTOR(768) on `clause_text` |
| **Total** | **5,390 source docs** | **~13,390 rows** | |

---

## Ingestion Pipeline

Four isolated stages. Each stage writes output to disk before the next begins. A failure in any stage never requires re-running a prior stage. Every stage is independently resumable.

**Rule:** Fetch ≠ Process ≠ Embed ≠ Insert. Never embed or insert during the fetch stage.

---

### Stage 1 — Fetch Raw (2–3 hours)

Write all raw data to disk. Process nothing.

**CourtListener API**
- Endpoint: `courtlistener.com/api/rest/v4/opinions/`
- Filters: `court_id=scotus,ca9,cacd,cand,caed,casd,cal,calctapp`
- Pagination: cursor-based. Save last cursor to `state.json` after every page.
- Output: `raw/cl/page_{n}.json` (20 opinions per page)
- Resilience: resume from saved cursor on failure. Free account = 50 req/sec.

**HuggingFace pile-of-law**
- Stream only — never download full dataset (256GB)
- Subsets: `ca_court_opinions` (1,500 docs), `courtlistener_opinions` (500 docs), `us_bills` (200 docs)
- Filter to 4 practice areas on-the-fly during streaming
- Output: `raw/pol/{subset}.jsonl` (one doc per line)
- Resilience: write each matching doc immediately. Re-streaming is fast since filtering is aggressive.

**Statutes**
- Federal: download XML from uscode.house.gov. Static files. No failure risk.
- California: download from leginfo.legislature.ca.gov.
- Output: `raw/statutes/federal/` and `raw/statutes/ca/`

---

### Stage 2 — Parse + Chunk + LLM Tag (4–6 hours, run overnight)

For each raw opinion:

1. Strip case headers, docket numbers, page breaks, and procedural boilerplate
2. Run SemanticSplitterNodeParser (threshold=90, buffer_size=1)
3. Apply size guard: merge chunks < 80 tokens; halve chunks > 600 tokens
4. Send primary holding chunk(s) to Claude Haiku with structured prompt requesting JSON output: `ai_summary` (4 sentences), `key_concepts` (array), `outcome` (enum), `primary_holding_chunk_index` (integer)
5. Write all chunks to `processed/case_law.jsonl`
6. Write one summary row to `processed/case_summaries.jsonl`

For statutes: parse XML by `<section>` tag. Extract section number, title, text. Manually tag `practice_area`. Write to `processed/statutes.jsonl`.

**Resilience:** Track processed opinion IDs in SQLite `processing_state` table. Check before processing each opinion — skip if already done. Safe to kill and resume. Haiku calls batched 10 at a time.

**Edge case:** Opinions under 300 tokens skip Haiku. Construct `ai_summary` deterministically from case_name + court + year + first 200 tokens of text.

---

### Stage 3 — Embed (6–8 hours CPU / ~45 min GPU, run overnight)

For `case_law.jsonl`: embed `holding_text` field in batches of 64. Append embedding to each row. Write to `embedded/case_law.jsonl`.

For `case_summaries.jsonl`: embed `ai_summary` field. Write to `embedded/case_summaries.jsonl`.

For `statutes.jsonl`: embed `section_text` field. Write to `embedded/statutes.jsonl`.

For `clause_templates`: embed `clause_text`. Write to `embedded/clause_templates.jsonl`.

**Resilience:** Write to disk every 500 rows. Track last embedded row index in `state.json`. On restart: skip rows where embedding already exists in output file.

**Note:** If no GPU, start Stage 3 concurrently with Stage 2 as processed JSONL rows appear. Do not wait for Stage 2 to complete before beginning Stage 3.

---

### Stage 4 — Insert into Actian (30–60 minutes)

Read all `embedded/*.jsonl` files. Bulk INSERT in batches of 200 rows using `executemany`. Each batch wrapped in a transaction. On batch failure: rollback that batch only, log failed IDs, continue with next batch.

**Idempotency:** `ON CONFLICT DO NOTHING` on primary key. Already-inserted rows silently skipped. Safe to re-run entire stage.

**Validation:** After insertion, run 30 ground-truth Q&A pairs against the live database. Pass threshold: correct source document in top-3 results for ≥ 27/30 queries. If below threshold, check chunking quality and metadata tagging before Day 2 feature work.

---

### Ingestion Timeline

| Stage | Duration | When |
|---|---|---|
| Stage 1 — fetch | 2–3 hrs | Start night before hackathon (Day 0) |
| Stage 2 — parse + chunk + tag | 4–6 hrs | Start Day 1 morning; run overnight |
| Stage 3 — embed | 6–8 hrs (CPU) / ~45 min (GPU) | Start concurrently with Stage 2 |
| Stage 4 — insert | 30–60 min | Day 1 afternoon after overnight stages complete |
| Validation | 30 min | Day 1 afternoon, immediately after insert |

---

### is_good_law Population

CourtListener returns a `negative_treatment` field on each opinion cluster. At ingestion: if any negative treatment exists → `is_good_law = FALSE`. This is real data sourced from CourtListener's citation tracking — not a default value.

pile-of-law opinions do not carry this field. All pile-of-law cases are set `is_good_law = TRUE` with a UI disclaimer: *"Good law status unverified — confirm before filing."*