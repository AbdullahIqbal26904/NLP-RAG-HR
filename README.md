# RAG Talent Matching Project

A Retrieval-Augmented Generation (RAG) system for talent matching between resumes and job descriptions. It embeds pre-structured resume and job data into separate Pinecone vector indexes, and provides a Streamlit UI with hybrid BM25 + semantic search, RRF fusion, cross-encoder re-ranking, and LLM-generated answers.

## Dataset

Two publicly available Kaggle datasets:

- **Resumes:** `data/resumes_dataset.jsonl` — 3,500 resumes across 36 categories (Java Developer, Python Developer, DevOps, Data Science, etc.)
- **Job Descriptions:** `data/job_title_des.csv` — 2,277 job postings across 15 unique job titles (Backend Developer, Full Stack Developer, Machine Learning, etc.)

Each resume and each job description gets its own vector embedding in Pinecone. The system supports:
- **Recruiter mode:** Input a job requirement, retrieve matching candidate resumes
- **Candidate mode:** Input your profile/skills, retrieve matching job postings
- **Add Resume:** Candidates can submit their resume via the UI — it is saved to the JSONL file, embedded, and upserted to Pinecone instantly
- **Add Job:** Recruiters can post a new job description via the UI — it is saved to the CSV file, embedded, and upserted to Pinecone instantly

## Architecture

```
data/resumes_dataset.jsonl  +  data/job_title_des.csv
        |                              |
        v                              v
  [Serialization]  (pipeline/serializer.py)
        |                              |
        v                              v
  [Embedding: all-MiniLM-L6-v2]  (pipeline/ingest.py)
        |                              |
        v                              v
  Pinecone: vend-candidates      Pinecone: vend-jobs
        |                              |
        +----------+-------------------+
                   |
                   v
         [Hybrid Retrieval]  (pipeline/retrieval.py)
           - BM25 (local) + Semantic (Pinecone)
           - RRF Fusion
           - CrossEncoder Re-ranking (ms-marco-MiniLM-L-6-v2)
                   |
                   v
         [LLM Generation]  (pipeline/generation.py)
           - Primary: HuggingFace Inference API
           - Fallback: Groq API
                   |
                   v
         [LLM-as-a-Judge Evaluation]  (pipeline/evaluation.py)
           - Faithfulness: claim extraction + verification
           - Relevancy: alternate question generation + cosine similarity
                   |
                   v
         [Streamlit Dashboard]  (app.py)
           - Search tab: generated answer, full original data, retrieval scores
           - Add Resume tab: submit resume -> saved + embedded + indexed live
           - Add Job tab: post job -> saved + embedded + indexed live
           - Faithfulness & Relevancy scores
           - Ablation study sidebar
```

## Live Data Submission

Users can add new resumes and job postings directly from the Streamlit UI without restarting the application.

**Add Resume (Candidates):**
- Fill in: Name, Email, Phone, Location, Job Category, Skills, Education, Summary, Experience
- On submit: saved to `data/resumes_dataset.jsonl`, embedded with all-MiniLM-L6-v2, upserted to Pinecone, BM25 index rebuilt
- Immediately searchable by recruiters

**Add Job (Recruiters):**
- Fill in: Job Title, Job Description
- On submit: saved to `data/job_title_des.csv`, embedded with all-MiniLM-L6-v2, upserted to Pinecone, BM25 index rebuilt
- Immediately searchable by candidates

User-submitted entries use IDs prefixed with `USER_` (resumes) and `USERJOB_` (jobs) to distinguish them from the original Kaggle data.

## Quick Start (Docker)

### 1. Configure Environment

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_CANDIDATE_INDEX=vend-candidates
PINECONE_JOB_INDEX=vend-jobs
HF_API_TOKEN=your_huggingface_token
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
GROQ_API_KEY=your_groq_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RETRIEVE=20
TOP_K_RERANK=5
```

### 2. Build and Run

```bash
docker compose build
docker compose up -d
```

App is available at: **http://localhost:8501**

### 3. Verify Data

```bash
docker compose exec app python run_etl.py
```

### 4. Ingest Data to Pinecone

Embeds all 3,500 resumes + 2,277 jobs and upserts to Pinecone:

```bash
docker compose exec app python -m pipeline.ingest
```

### 5. Run Automated Evaluation + Ablation Study

```bash
docker compose exec app python -m pipeline.experiments
```

Outputs:
- `reports/experiment_results.json`
- `reports/ablation_summary.csv`

### 6. Generate Submission Report

```bash
docker compose exec app python -m pipeline.report_builder
```

Output:
- `reports/submission_report.md`

### Stop

```bash
docker compose down
```

## Applying Code Changes

Since `app.py`, `pipeline/`, and `data/` are volume-mounted, most code changes are picked up automatically. Just restart the container:

```bash
docker compose restart app
```

If you change `requirements.txt` or `Dockerfile`, rebuild:

```bash
docker compose build app
docker compose up -d
```

## Local Development (without Docker)

```bash
pip install -r requirements.txt
python run_etl.py                      # verify datasets
python -m pipeline.ingest              # embed & upsert to Pinecone
python -m pipeline.experiments         # evaluation + ablation
python -m pipeline.report_builder      # generate report
streamlit run app.py                   # launch UI
```

## Project Structure

```
app.py                      Streamlit web interface
run_etl.py                  Data verification script
pipeline/
  serializer.py             Text serialization for resumes & jobs
  ingest.py                 Embed & upsert to Pinecone
  retrieval.py              Hybrid BM25 + Semantic + RRF + CrossEncoder
  generation.py             LLM answer generation (HF primary, Groq fallback)
  evaluation.py             LLM-as-a-Judge (Faithfulness + Relevancy)
  experiments.py            Automated evaluation & ablation study
  report_builder.py         Generate submission report from results
data/
  resumes_dataset.jsonl     3,500 resumes (Kaggle)
  job_title_des.csv         2,277 job descriptions (Kaggle)
reports/                    Generated evaluation reports
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PINECONE_API_KEY` | Pinecone vector database |
| `PINECONE_CANDIDATE_INDEX` | Pinecone index name for resumes (default: `vend-candidates`) |
| `PINECONE_JOB_INDEX` | Pinecone index name for jobs (default: `vend-jobs`) |
| `HF_API_TOKEN` | HuggingFace Inference API token (primary LLM) |
| `HF_MODEL_ID` | HuggingFace model for generation (default: `meta-llama/Llama-3.1-8B-Instruct`) |
| `GROQ_API_KEY` | Groq API key (fallback LLM) |
| `EMBEDDING_MODEL` | Sentence transformer model (default: `all-MiniLM-L6-v2`) |
| `TOP_K_RETRIEVE` | Number of candidates for initial retrieval (default: `20`) |
| `TOP_K_RERANK` | Number of results after re-ranking (default: `5`) |

## Technical Stack

| Component | Choice |
|-----------|--------|
| Vector DB | Pinecone (Free Starter) |
| Embeddings | all-MiniLM-L6-v2 (pre-computed) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM Generation | HuggingFace Inference API (primary), Groq (fallback) |
| Retrieval | Hybrid BM25 + Semantic with RRF + CrossEncoder |
| Evaluation | LLM-as-a-Judge (Faithfulness + Relevancy) |
| UI | Streamlit |
| Hosting | Docker / HuggingFace Spaces |
