FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install CPU-only PyTorch first (much smaller than full torch), then the rest
RUN python -m pip install --upgrade pip && \
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt

COPY app.py ./
COPY run_etl.py ./
COPY pipeline ./pipeline
COPY data/resumes_dataset.jsonl ./data/resumes_dataset.jsonl
COPY data/job_title_des.csv ./data/job_title_des.csv
COPY README.md ./README.md

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
