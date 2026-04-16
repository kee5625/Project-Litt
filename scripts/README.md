# LegalMind — Data Creation & Ingestion Strategy
**Optimized for 3-Day Hackathon Build | Aligned with Baseline Resources**

---

## Executive Summary

**Goal:** Ingest ~580 documents (cases, statutes, templates) into Actian Vector DB with high-quality embeddings and clean metadata in <30 minutes by Day 1 evening.

**Strategy:** Hybrid chunking (semantic) + baseline embeddings (sentence-transformers for cost/speed) + lightweight metadata extraction + batch embedding + Actian bulk insert.

**Key insight:** Start small and clean, not big and messy. Quality corpus matters more than quantity for demo credibility.

---

## Data Sources & Target Corpus

### Recommended Sources (Pre-Evaluated for Hackathon)

| Source | Format | Access | Target Count | Why |
|---|---|---|---|---|
| **HuggingFace `pile-of-law` dataset** | Parquet/JSON | Direct download | 300 cases | Pre-chunked legal opinions, free, no API limits, includes metadata (court, year, cite) |
| **CourtListener API** | REST + bulk export | Free tier adequate | 150 state cases | Real California cases with structured metadata; rate limits manageable for 150 docs |
| **Public domain US Code** | Text/XML | govinfo.gov or uscode.house.gov | 80 statutes | FLSA, ADA, Title VII, CA Labor Code — download once, parse locally |
| **Clause templates** | Curated or template libraries | Craft manually | 50 templates | Motion to dismiss boilerplate, demand letter structure — write from memory or use legal template libraries (FormAssembly, LawDepot) |
| **Total** | | | **~580 docs** | Indexable in <30 min; representative enough for 3 demo queries |

**Why NOT CourtListener alone?**
- Rate limits: 10 req/s with backoff = slow for 500+ docs
- Fallback: Use `pile-of-law` as primary, supplement with CourtListener for state cases

---

## Phase 1: Chunking Strategy (Day 1 Morning)

### Step 1.1: Establish Ground Truth (30 min)

Create 30 representative Q&A pairs reflecting your demo queries:

```
Q: "What's the standard for summary judgment in employment cases, 9th Circuit?"
A: Anderson v. Liberty Lobby (477 U.S. 242) + SCOTUS Circuit precedent cases

Q: "ADA reasonable accommodation standard for remote work"
A: 42 U.S.C. § 12112 + case law on interactive process

Q: "Wrongful termination California workers comp"
A: CA Labor Code § 132a + state employment law cases 2015+
```

Use these to validate chunking quality later.

### Step 1.2: Choose Chunking Method

**Recommendation: Hybrid Semantic Chunking** (balances quality + speed)

**Why:** Legal documents have variable structure (headings, citations, quoted passages). Semantic chunking respects paragraph boundaries while splitting long sections intelligently.

#### Option A: Use Docling (Recommended for Legal Docs)
```python
# Install once
pip install docling

from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("opinion.pdf")

# Docling extracts structure, metadata, citations automatically
# Output: DoclingDocument with semantic blocks
```

**Pros:**
- Built for document parsing (PDFs, Word, structured text)
- Respects legal document structure (headings, footnotes)
- Extracts metadata like courts, dates during parsing
- Handles citations natively

**Cons:**
- Overkill for plain-text `pile-of-law` data

#### Option B: LlamaIndex Semantic Chunking (Lightweight)
```python
from llama_index.core.node_parser import SemanticSplitNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

parser = SemanticSplitNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model,
)

nodes = parser.get_nodes_from_documents(docs)
```

**Pros:**
- Lightweight, no heavy ML libraries
- Works with plain text directly
- Breakpoint logic: splits when embedding similarity drops (= new semantic block)

**Cons:**
- Requires embedding calls upfront (slow)

### Step 1.3: Configuration for Legal Corpus

```python
# Semantic chunking config for legal text
CHUNK_CONFIG = {
    "chunk_size": 500,           # tokens; legal holdings are ~400-600 tokens
    "chunk_overlap": 100,        # keep citations + context bridges
    "breakpoint_percentile": 95, # aggressive splitting = smaller, cleaner chunks
    "min_chunk_size": 150,       # discard stubs
}

# Expected result:
# - 300 case opinions → ~600-900 chunks (2-3 per opinion holding)
# - 80 statutes → ~150-200 chunks (section-by-section)
# - 50 templates → ~50 chunks (as-is, pre-formatted)
# TOTAL: ~800-1150 chunks → embed all, insert ~580 into Actian
```

---

## Phase 2: Metadata Extraction (Day 1 Morning → Early Afternoon)

### Step 2.1: Structured Extraction Pipeline

For each source, extract metadata BEFORE chunking:

```python
import json
import re
from dataclasses import dataclass

@dataclass
class CaseMetadata:
    id: str                    # unique identifier
    case_name: str            # "Smith v. Jones"
    citation: str             # "123 F.3d 456"
    court: str                # "9th Cir."
    jurisdiction: str         # "CA" or "federal"
    practice_area: str        # "employment", "family", "IP"
    year: int                 # 2019
    is_good_law: bool         # default True; manually curate later
    holding_text: str         # the 500-token chunk being embedded
    full_cite_str: str        # "Smith v. Jones, 123 F.3d 456 (9th Cir. 2019)"
    source_url: str           # CourtListener ID or pile-of-law ref

# Extraction for each source type:

# A. Pile-of-law (structured)
def parse_pile_of_law(json_record):
    return CaseMetadata(
        id=json_record["opinion_id"],
        case_name=json_record["case_name"],
        citation=json_record["citation"],
        court=classify_court(json_record["court"]),  # normalize
        jurisdiction=extract_jurisdiction(json_record["court"]),
        practice_area=classify_practice_area(json_record["text"]),  # keyword or embedding
        year=json_record["year"],
        is_good_law=True,  # assume true; flag overruled cases later
        holding_text=extract_holding(json_record["text"]),  # key sentence(s)
        full_cite_str=f"{json_record['case_name']}, {json_record['citation']} ({json_record['court']} {json_record['year']})",
        source_url=json_record["url"]
    )

# B. CourtListener (API)
def parse_courtlistener_opinion(opinion_dict):
    return CaseMetadata(
        id=opinion_dict["id"],
        case_name=opinion_dict["case_name"],
        citation=opinion_dict["citations"][0]["cite"] if opinion_dict["citations"] else "unpublished",
        court=opinion_dict["court"],
        jurisdiction="CA" if "California" in opinion_dict["court"] else "federal",
        practice_area=infer_practice_area_from_docket(opinion_dict["docket"]),
        year=opinion_dict["date_filed"].year,
        is_good_law=check_good_law_status(opinion_dict["id"]),  # call CourtListener API
        holding_text=extract_holding(opinion_dict["plain_text"]),
        full_cite_str=format_full_citation(opinion_dict),
        source_url=opinion_dict["absolute_url"]
    )

# C. Statutes (text files)
def parse_statute(statute_text, statute_code):
    sections = re.split(r'§ (\d+[A-Za-z]?)', statute_text)
    results = []
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            results.append(StatuteMetadata(
                id=f"statute_{statute_code}_{sections[i]}",
                title=statute_code,
                code=f"{statute_code} § {sections[i]}",
                jurisdiction="federal" if statute_code.startswith("42 USC") else "CA",
                practice_area=infer_from_code(statute_code),
                section_text=sections[i+1][:2000],
                last_amended=2024,  # update from official source
            ))
    return results
```

### Step 2.2: Practice Area Classification

Use **keyword matching** (fast) + **embedding-based fallback** (accurate):

```python
PRACTICE_AREA_KEYWORDS = {
    "employment": ["wrongful termination", "discrimination", "harassment", "wage", "hour", "FLSA", "ADA", "Title VII"],
    "family": ["divorce", "custody", "child support", "alimony", "domestic violence"],
    "IP": ["patent", "trademark", "copyright", "infringement"],
    "criminal": ["conviction", "sentencing", "plea", "trial", "DUI"],
}

def classify_practice_area(text):
    text_lower = text.lower()
    scores = {}
    for area, keywords in PRACTICE_AREA_KEYWORDS.items():
        scores[area] = sum(text_lower.count(kw) for kw in keywords)
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    
    # Fallback: embedding-based
    embedding = embed_model.embed(text[:1000])
    return classify_via_embedding(embedding)
```

---

## Phase 3: Embedding Strategy (Day 1 Afternoon)

### Step 3.1: Choose Embedding Model

**Recommendation: `sentence-transformers/all-MiniLM-L6-v2`**

| Criterion | all-MiniLM-L6-v2 | text-embedding-3-small | BERT (large) |
|---|---|---|---|
| **Dimension** | 384 | 1536 | 768 |
| **Speed** | ⚡⚡⚡ Fast | ⚡ Slower (API) | ⚡⚡ Medium |
| **Cost** | Free (local) | $0.02/1M tokens | Free (local) |
| **Legal domain** | Good (trained on diverse text) | Excellent (OpenAI, general) | Good (domain-agnostic) |
| **Latency** | <5ms per doc | 100-500ms + network | ~20ms per doc |
| **Hackathon fit** | ✅ Best (local, fast, free) | ❌ Costs money, slower | ⚠️ Medium option |

**Why not text-embedding-3-small for hackathon?**
- No API budget (unless provided)
- Network latency ruins demo speed
- Local embedding means instant re-queries, no rate limits

**Deploy locally:**
```bash
pip install sentence-transformers torch

python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print(model.encode(['test'])[0].shape)"
# Output: (384,)  ← 384-dim embeddings, ready to insert into Actian
```

### Step 3.2: Batch Embedding Pipeline

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_batch(texts, batch_size=32):
    """
    Embed list of texts in batches.
    Returns: list of (384-dim) numpy arrays
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings

# Usage:
all_chunks = [chunk.text for chunk in parsed_docs]
chunk_metadata = [chunk.metadata for chunk in parsed_docs]

embeddings = embed_batch(all_chunks, batch_size=64)

# Verify shape
assert embeddings.shape == (len(all_chunks), 384)

# Save to parquet for bulk insert
import pandas as pd
df = pd.DataFrame({
    'chunk_text': all_chunks,
    'embedding': [list(e) for e in embeddings],
    **{k: [m[k] for m in chunk_metadata] for k in chunk_metadata[0].keys()}
})
df.to_parquet('chunks_embedded.parquet')
```

### Step 3.3: Speed Optimization

For 580 documents (~800-1000 chunks):

```
Embedding time: 1000 chunks × 384-dim @ 64 batch_size
= ~3-5 seconds on CPU (GPU: <1s)
Total pipeline: <15 minutes (extract + chunk + embed)
```

---

## Phase 4: Actian Vector DB Insertion (Day 1 Late Afternoon)

### Step 4.1: Prepare Bulk Insert Data

```python
import psycopg2
from pgvector.psycopg2 import register_vector

# Connect to Actian Vector DB
conn = psycopg2.connect(
    dbname="legalmind_demo",
    user="postgres",
    password="...",
    host="localhost",
    port=5432
)
register_vector(conn)

# Bulk insert prepared DataFrame
df_cases = pd.DataFrame([
    {
        'id': metadata['id'],
        'case_name': metadata['case_name'],
        'citation': metadata['citation'],
        'court': metadata['court'],
        'jurisdiction': metadata['jurisdiction'],
        'practice_area': metadata['practice_area'],
        'year': metadata['year'],
        'is_good_law': metadata['is_good_law'],
        'holding_text': chunk_text,
        'full_cite_str': metadata['full_cite_str'],
        'embedding': embedding,
    }
    for chunk_text, metadata, embedding in zip(all_chunks, chunk_metadata, embeddings)
])

# Execute bulk insert
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://postgres:...@localhost/legalmind_demo')

df_cases.to_sql('case_law', engine, if_exists='append', index=False)
print(f"Inserted {len(df_cases)} case law chunks")
```

### Step 4.2: Verify Hybrid Queries Work

```python
# Test query from Day 1 gate
query = "employment wrongful termination california"
query_embedding = model.encode(query)

# Hybrid query
result = conn.execute("""
    SELECT
        case_name, citation, court, year, holding_text,
        embedding <-> %s::vector AS distance
    FROM case_law
    WHERE
        practice_area = 'employment'
        AND jurisdiction = 'CA'
        AND year >= 2015
        AND is_good_law = TRUE
    ORDER BY distance
    LIMIT 5
""", (query_embedding.tolist(),))

for row in result.fetchall():
    print(f"{row[0]} ({row[1]}): {row[5]:.4f}")
```

---

## Data Quality Checkpoints

### Checkpoint 1: Chunking Quality (Day 1 Morning)

```python
# Validate ground truth Q&As can be retrieved
ground_truth_qa = [
    ("summary judgment employment", "Anderson v. Liberty Lobby"),
    ("ADA reasonable accommodation", "42 U.S.C. § 12112"),
    ("wrongful termination california", "~5 CA employment law cases"),
]

for query, expected_result in ground_truth_qa:
    query_vec = model.encode(query)
    results = hybrid_search(query_vec, practice_area="employment")
    found = any(expected_result in r['case_name'] or expected_result in r['citation'] for r in results)
    print(f"✓ '{query}' retrieves '{expected_result}': {found}")
```

### Checkpoint 2: Metadata Consistency (Day 1 Afternoon)

```python
# Check for missing/malformed metadata
df_cases = pd.read_sql("SELECT * FROM case_law LIMIT 100", engine)

issues = []
for col in ['case_name', 'citation', 'court', 'jurisdiction', 'practice_area', 'year']:
    nulls = df_cases[col].isna().sum()
    if nulls > 0:
        issues.append(f"  {col}: {nulls} nulls")
    
    # Check for obvious malforms
    if col == 'year' and (df_cases['year'] < 1950 or df_cases['year'] > 2025).any():
        issues.append(f"  {col}: out-of-range years found")

if not issues:
    print("✓ Metadata validation passed")
else:
    print("✗ Issues found:")
    for issue in issues:
        print(issue)
```

### Checkpoint 3: Embedding Quality (Day 1 Evening)

```python
# Cosine similarity sanity check
from sklearn.metrics.pairwise import cosine_similarity

# Embed two variations of same query
q1 = model.encode("what is the standard for summary judgment")
q2 = model.encode("summary judgment standard requirement")

sim = cosine_similarity([q1], [q2])[0][0]
assert sim > 0.8, f"Similar queries should have cosine_sim > 0.8, got {sim}"

# Embed different topics
q3 = model.encode("how to cure a ham")
sim_diff = cosine_similarity([q1], [q3])[0][0]
assert sim_diff < 0.5, f"Different topics should have cosine_sim < 0.5, got {sim_diff}"

print("✓ Embedding quality OK")
```

---

## Timeline & Ownership

| Phase | Task | Owner | Duration | Day 1 Time Block |
|---|---|---|---|---|
| 1.1 | Ground truth Q&A creation + chunking setup | P2 (RAG lead) | 30 min | 9:00-9:30 AM |
| 1.2 | Download pile-of-law + CourtListener API queries | P1 (data) | 20 min | 9:00-9:20 AM |
| 1.3 | Metadata extraction pipeline (code) | P1 | 45 min | 9:30 AM-10:15 AM |
| 2.x | Run extraction on all sources | P1 | 30 min | 10:30 AM-11:00 AM |
| 3.1 | Embedding model setup (sentence-transformers) | P2 | 15 min | 11:00-11:15 AM |
| 3.2 | Batch embed all chunks | P1 | 10 min | 1:00-1:10 PM (after lunch) |
| 4.1 | Actian schema + bulk insert | P1 | 15 min | 1:15-1:30 PM |
| Check | Validate hybrid queries work (Day 1 gate) | Both | 15 min | 2:00-2:15 PM |
| **Total** | | | **2.5 hours** | 9 AM - 2:30 PM |

**By Day 1 evening:** 580 documents embedded, indexed, queried successfully.

---

## Fallback Strategies

| Issue | Plan A (Recommended) | Plan B (Fallback) |
|---|---|---|
| **Pile-of-law download slow** | Use mirrors (HF cdn) or pre-cache locally | Use CourtListener API only (slower but works) |
| **CourtListener rate limits** | Download bulk export once, re-use | Pre-seed with 150 cached opinions from prior HF snapshot |
| **Embedding API costs** | Use free local sentence-transformers | Use smaller all-MiniLM (68M params) for ultra-fast edge embed |
| **Actian bulk insert fails** | Use psycopg2 connection + parameterized INSERT loop | CSV import + native Actian COPY command |
| **Metadata extraction errors** | Log problematic docs, skip them, continue | Use regex + heuristics, accept 80% accuracy for demo |

---

## Key Files to Create

```
/project/data/
├── ingest_config.yaml          # URLs, batch sizes, model names
├── chunking.py                 # Docling or LlamaIndex wrapper
├── metadata_extractor.py       # Source-specific parsers
├── embedding_pipeline.py       # sentence-transformers batch encoder
├── actian_loader.py            # pgvector bulk insert
├── ground_truth.jsonl          # 30 Q&A pairs for validation
├── raw/
│   ├── pile-of-law/            # downloaded parquet files
│   ├── courtlistener/          # API responses cached as JSON
│   ├── statutes/               # govinfo.gov text files
│   └── templates/              # manually curated clause samples
└── processed/
    ├── chunks_metadata.csv     # 800+ rows: chunk + metadata
    └── chunks_embedded.parquet # 800+ rows: chunk + 384-dim embedding
```

---

## Success Metrics (Day 1 Evening Gate)

- [ ] 500+ documents downloaded
- [ ] 800-1000 chunks with metadata extracted
- [ ] All chunks embedded (384-dim, no nulls)
- [ ] Bulk insert into Actian successful
- [ ] 3 demo queries return results in <2s each
- [ ] Ground truth Q&A validation: ≥27/30 queries return expected results

---

## Implemented Script Entry Point

The scripts folder now includes an executable ingestion pipeline that pulls CourtListener and pile-of-law data, applies strict quality filtering, writes uncited records to quarantine CSV, and produces feasibility manifests.

### Files Added

```
scripts/
├── run_ingestion.py
├── requirements.txt
├── .env.example
└── ingest/
    ├── __init__.py
    ├── config.py
    ├── models.py
    ├── utils.py
    ├── courtlistener.py
    ├── pile_of_law.py
    ├── quality.py
    └── exporters.py
```

### Quick Start

```bash
cd scripts
pip install -r requirements.txt
copy .env.example .env
python run_ingestion.py --dry-run --sources all --project-root ..
```

### Full Run

```bash
python run_ingestion.py --sources all --project-root ..
```

### Output Layout

```
scripts/output/
├── raw/
│   ├── courtlistener/
│   └── pile_of_law/
├── processed/
│   ├── courtlistener_<run_id>_accepted.csv
│   ├── courtlistener_<run_id>_quarantine.csv
│   ├── pile_of_law_<run_id>_accepted.csv
│   ├── pile_of_law_<run_id>_quarantine.csv
│   ├── merged_<run_id>_accepted.csv
│   └── merged_<run_id>_quarantine.csv
└── manifests/
    └── ingestion_manifest_<run_id>.json
```

### Feasibility Controls Implemented

- REST-first CourtListener fetching with retry/backoff and bulk fallback parsing.
- Strict filtering for missing text and short opinions.
- Missing citation routing to quarantine CSV.
- Near-duplicate suppression.
- Soft domain balancing with adaptive recency weighting.
- Runtime and output-size budget checks in manifest (`<60 min`, `<15 GB`).