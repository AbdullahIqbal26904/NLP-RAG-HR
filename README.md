# RAG Talent Matching Project

A Retrieval-Augmented Generation (RAG) system for talent matching. It extracts structured data from resume PDFs using an LLM-powered ETL pipeline, embeds them into a Pinecone vector database, and provides a Streamlit UI for hybrid semantic + BM25 search with cross-encoder re-ranking and LLM-generated answers.

## Dataset

Resume PDFs sourced from a publicly available Kaggle dataset:
**[Resume Data (PDF) - Kaggle](https://www.kaggle.com/datasets/hadikp/resume-data-pdf)**

- ~2,485 PDF resumes across 24 job categories (Accountant, Engineer, Healthcare, IT, Sales, etc.)
- 72 resumes processed (3 per category) for this assignment

## ETL Pipeline (`etl/`)

The ETL pipeline extracts structured candidate data from raw PDF resumes in four stages:

```
PDF File
  |
  v
[1. Text Extraction]  (etl/text_extractor.py)
    - Uses pdfplumber with layout-aware column detection
    - Handles multi-column resumes, sidebars, headers/footers
    - Also supports DOCX via python-docx
  |
  v
[2. Text Cleaning]  (etl/text_clean.py)
    - 11-step normalization pipeline
    - Fixes: Unicode, broken words, spaced characters (S K I L L S),
      page markers, link artifacts, whitespace
  |
  v
[3. LLM Structured Extraction]  (etl/resume_extractor.py)
    - BAML-defined function with strict JSON schema enforcement
    - Uses Groq Llama 3.3 70B (free tier, 30 RPM)
    - Retry logic with exponential backoff (3 attempts)
    - Extracts: personal info, work experience, education, skills,
      certifications, projects, languages, achievements
  |
  v
[4. Pydantic Validation]  (etl/validated_models.py)
    - Date normalization (YYYY-MM format)
    - Cross-field consistency (current employer <-> end_date)
    - CGPA range validation
    - Skill deduplication
    - City extraction fallback from address/institute name
  |
  v
Output: data/candidates.json
```

### Running the ETL Pipeline

```bash
# Run with Docker
docker compose --profile etl build etl
docker compose --profile etl run --rm etl

# Or locally
python run_etl.py
```

The pipeline saves results incrementally to `data/candidates.json` and supports resuming from where it left off.

### BAML Configuration

LLM client config is in `etl/baml_src/clients.baml`. To switch providers, edit this file and regenerate:

```bash
cd etl && npx @boundaryml/baml generate
```

## 1. Start the Application (Docker)

Build and run:

```bash
docker compose build
docker compose up -d
```

Open the app at:

```text
http://localhost:8501
```

Stop:

```bash
docker compose down
```

## 2. Ingest Data to Pinecone

This clears existing embeddings and re-ingests from `data/candidates.json`:

```bash
docker compose exec -T app python -m pipeline.ingest
```

## 3. Run Automated Evaluation + Ablation

```bash
docker compose exec -T app python -m pipeline.experiments
```

Outputs:
- reports/experiment_results.json
- reports/ablation_summary.csv

## 4. Generate Submission Report

```bash
docker compose exec -T app python -m pipeline.report_builder
```

Output:
- reports/submission_report.md

## 5. Optional Local (non-Docker)

```bash
source .venv/bin/activate
python run_etl.py
python -m pipeline.ingest
python -m pipeline.experiments
python -m pipeline.report_builder
streamlit run app.py
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | LLM for resume extraction (ETL) |
| `PINECONE_API_KEY` | Vector database |
| `PINECONE_CANDIDATE_INDEX` | Pinecone index name for candidates |
| `HF_API_TOKEN` | HuggingFace for generation model |
| `HF_MODEL_ID` | Generation LLM model ID |
| `EMBEDDING_MODEL` | Sentence transformer model |
