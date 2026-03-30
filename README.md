# RAG Talent Matching Project

## Confidentiality Notice
This dataset was provided by VentureDive for our Final Year Project. We have signed an MOU and cannot share the raw dataset publicly.

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

Health check:

```bash
curl http://localhost:8501/_stcore/health
```

Stop:

```bash
docker compose down
```

## 2. Ingest Data to Pinecone

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
python -m pipeline.ingest
python -m pipeline.experiments
python -m pipeline.report_builder
streamlit run app.py
```
