# Assignment Report: RAG-based Talent Matching System

**Authors:** Abdullah Iqbal, Anushe Ali
**Course:** Information Retrieval | **Date:** April 2026
**Production URL:** [http://3.144.109.78](http://3.144.109.78/) (Amazon EC2)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AMAZON EC2 INSTANCE                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Docker Container                              │  │
│  │                                                                   │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │  │
│  │  │  Streamlit   │───▶│  Retrieval   │───▶│   LLM Generation    │  │  │
│  │  │  Frontend    │    │   Engine     │    │  (HF / Groq)        │  │  │
│  │  │  (app.py)    │◀───│              │◀───│                     │  │  │
│  │  └──────┬───────┘    └──────┬───────┘    └─────────────────────┘  │  │
│  │         │                   │                                     │  │
│  │         ▼                   ▼                                     │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │  │
│  │  │  Add Resume  │    │   BM25       │    │  CrossEncoder       │  │  │
│  │  │  Add Job     │    │   (Local)    │    │  Re-ranker          │  │  │
│  │  │  (Live CRUD) │    │              │    │  (ms-marco-MiniLM)  │  │  │
│  │  └──────────────┘    └──────────────┘    └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┬──────────────────┘
                      │                               │
                      ▼                               ▼
               ┌──────────────┐                ┌──────────────┐
               │   Pinecone   │                │  HuggingFace │
               │  Vector DB   │                │  / Groq API  │
               │              │                │              │
               │ vend-candidates│               │ Llama-3.1-8B │
               │ vend-jobs     │               │  -Instruct   │
               └──────────────┘                └──────────────┘
```

### Retrieval Pipeline Flow

```
    User Query
        │
        ▼
┌───────────────┐     ┌───────────────┐
│  Semantic      │     │    BM25       │
│  Search        │     │   Search      │
│  (Pinecone)    │     │   (Local)     │
└───────┬───────┘     └───────┬───────┘
        │                     │
        └──────────┬──────────┘
                   ▼
          ┌────────────────┐
          │  RRF Fusion    │
          │  (k=60)        │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │  CrossEncoder  │
          │  Re-ranking    │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │  LLM Answer    │
          │  Generation    │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │  LLM-as-Judge  │
          │  Evaluation    │
          └────────────────┘
```

---

## 2. Data & Platform Details

| Aspect | Detail |
|--------|--------|
| **Domain** | HR Talent Matching (Resumes + Job Descriptions) |
| **Resume Dataset** | 3,500 resumes across 36 categories (Kaggle) |
| **Job Dataset** | 2,277 job postings across 15 titles (Kaggle) |
| **Vector DB** | Pinecone (Cloud, Starter Plan) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` (HF primary, Groq fallback) |
| **Deployment** | Amazon EC2 with Docker Compose |
| **UI Framework** | Streamlit |

**Live Features:** Users can add resumes and job postings directly from the UI — data is saved, embedded, and indexed in real-time without restart.

---

## 3. Retrieval Strategies & Models

### 3.1 Three Retrieval Variants Implemented

1. **Semantic Only** — Pinecone cosine similarity search using MiniLM-L6-v2 embeddings
2. **Hybrid + RRF** — BM25 (lexical) + Semantic (dense) fused via Reciprocal Rank Fusion (k=60)
3. **Hybrid + RRF + CrossEncoder** — Above + cross-encoder re-ranking for precision

### 3.2 Chunking Strategy Comparison

| Strategy | Description |
|----------|-------------|
| **Fixed Chunking** | Constant character windows (512 chars) with 50-char overlap |
| **Recursive Chunking** | Paragraph/sentence-aware splitting with max chunk size |
| **No Chunking (Full-doc)** | Each resume/job treated as a single document (selected approach) |

Full-document embedding was chosen because resumes and job descriptions are short enough to fit within the model's token limit, preserving full context.

### 3.3 LLM-as-a-Judge Evaluation

- **Faithfulness:** Extract claims from LLM answer → verify each claim against retrieved context
- **Relevancy:** Generate 3 alternate questions from answer → compute cosine similarity with original query

---

## 4. Performance Results

### 4.1 Fixed Test Set (15 Queries)

| Metric | Value |
|--------|-------|
| Avg Faithfulness | 36.7% |
| Avg Relevancy | 27.1% |
| Avg Retrieval Time | 1.168s |
| Avg Generation Time | 1.393s |
| Avg End-to-End Time | 7.673s |

### 4.2 Retrieval Ablation

| Strategy | Faithfulness | Relevancy | Retrieval (s) | Total (s) |
|----------|:-----------:|:---------:|:-------------:|:---------:|
| Semantic Only | 33.3% | 4.5% | 0.316 | 5.096 |
| **Hybrid + RRF** | **50.0%** | 4.5% | 0.260 | 4.854 |
| Hybrid + RRF + CE | 16.7% | 35.9% | 0.977 | 7.613 |

### 4.3 Chunking Ablation

| Chunking | Chunks | Faithfulness | Relevancy | Total (s) |
|----------|:------:|:-----------:|:---------:|:---------:|
| **Fixed** | 1,045 | **75.0%** | 2.9% | 8.162 |
| Recursive | 984 | 25.0% | 22.0% | 8.903 |

**Selected Pipeline:** Hybrid + RRF — best combined Faithfulness + Relevancy with lowest latency.

### 4.4 Claim Verification Examples

**Example 1** — *"React frontend developer with TypeScript"* (Candidate mode)
- Faithfulness: **100%** | Relevancy: 35.6%
- Supported claim: *"Shakeel has 4 years of experience as a React Native developer"* — verified against Experience section

**Example 2** — *"DevOps engineer with Docker and Kubernetes"* (Candidate mode)
- Faithfulness: 0% | Relevancy: **63.4%**
- Unsupported claim: *"M. Zakaria Nazir has extensive experience with Docker, Kubernetes"* — context did not contain these skills (hallucination detected)

**Example 3** — *"Python backend developer with FastAPI and PostgreSQL, 3+ years"* (Candidate mode)
- Faithfulness: 0% | Relevancy: 36.8%
- Unsupported claim: *"He has listed Python as his primary skill"* — context listed multiple skills without ranking

---

## 5. Reproducibility

```bash
# 1. Clone & configure .env (Pinecone + HuggingFace + Groq keys)
# 2. Build and run
docker compose build && docker compose up -d

# 3. Ingest data to Pinecone
docker compose exec app python -m pipeline.ingest

# 4. Run evaluation + ablation
docker compose exec app python -m pipeline.experiments

# 5. Access the application
# Local:      http://localhost:8501
# Production: http://3.144.109.78
```

### Project Structure

```
app.py                  Streamlit UI (Search + Add Resume + Add Job)
pipeline/
  serializer.py         Text serialization for resumes & jobs
  ingest.py             Embed & upsert to Pinecone
  retrieval.py          Hybrid BM25 + Semantic + RRF + CrossEncoder
  generation.py         LLM generation (HF primary, Groq fallback)
  evaluation.py         LLM-as-a-Judge (Faithfulness + Relevancy)
  experiments.py        Automated evaluation & ablation study
data/
  resumes_dataset.jsonl 3,500 resumes (36 categories)
  job_title_des.csv     2,277 job postings (15 titles)
reports/                Generated evaluation artifacts
```

### References

- Pinecone Documentation | Sentence Transformers | Streamlit
- HuggingFace Inference API | Groq API | rank-bm25
- Cross-Encoder: ms-marco-MiniLM-L-6-v2 (Reimers & Gurevych, 2019)

---
*Generated artifacts: `reports/experiment_results.json`, `reports/ablation_summary.csv`*
