"""
Ingest: Load CSV/JSONL -> Serialize -> Embed -> Upsert to Pinecone.
Run: python -m pipeline.ingest
"""
import csv
import json
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from pipeline.serializer import serialize_candidate, serialize_job

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CANDIDATE_INDEX = os.getenv("PINECONE_CANDIDATE_INDEX", "vend-candidates")
JOB_INDEX = os.getenv("PINECONE_JOB_INDEX", "vend-jobs")
BATCH_SIZE = 50


def load_resumes(path: str = "data/resumes_dataset.jsonl") -> list[dict]:
    """Load resumes from JSONL file."""
    resumes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                resumes.append(json.loads(line))
    print(f"Loaded {len(resumes)} resumes from {path}")
    return resumes


def load_jobs(path: str = "data/job_title_des.csv") -> list[dict]:
    """Load job descriptions from CSV file."""
    jobs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row["job_id"] = f"JOB_{i:05d}"
            jobs.append(row)
    print(f"Loaded {len(jobs)} jobs from {path}")
    return jobs


def get_or_create_index(pc: Pinecone, index_name: str, dimension: int):
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index: {index_name}")
    else:
        print(f"Index exists: {index_name}")
    return pc.Index(index_name)


def build_candidate_metadata(record: dict, text: str) -> dict:
    skills_raw = record.get("Skills") or ""
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
    return {
        "type": "candidate",
        "resume_id": record.get("ResumeID") or "",
        "name": record.get("Name") or "",
        "category": record.get("Category") or "",
        "location": record.get("Location") or "",
        "skills": skills[:20],
        "education": (record.get("Education") or "")[:500],
        "text": text[:4000],
    }


def build_job_metadata(record: dict, text: str) -> dict:
    return {
        "type": "job",
        "job_id": record.get("job_id") or "",
        "job_title": record.get("Job Title") or "",
        "text": text[:4000],
    }


def embed_and_upsert(index, records, serialize_fn, id_field, metadata_fn, model):
    texts = [serialize_fn(r) for r in records]
    print(f"Embedding {len(texts)} records...")

    embeddings = model.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )

    vectors = []
    for record, embedding, text in zip(records, embeddings, texts):
        record_id = record.get(id_field)
        vectors.append({
            "id": str(record_id),
            "values": embedding.tolist(),
            "metadata": metadata_fn(record, text),
        })

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // BATCH_SIZE + 1}/{(len(vectors) - 1) // BATCH_SIZE + 1}")

    print(f"Done - {len(vectors)} vectors upserted")


def clear_index(pc: Pinecone, index_name: str):
    """Delete all vectors from an existing index."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        print(f"Cleared all vectors from index: {index_name}")
    else:
        print(f"Index {index_name} does not exist, nothing to clear")


def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    model = SentenceTransformer(EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()

    # Clear existing embeddings
    print("\n--- Clearing existing embeddings ---")
    clear_index(pc, CANDIDATE_INDEX)
    clear_index(pc, JOB_INDEX)

    # Ingest resumes
    resumes = load_resumes()
    print(f"\n--- Ingesting {len(resumes)} resumes ---")
    cand_index = get_or_create_index(pc, CANDIDATE_INDEX, dimension)
    embed_and_upsert(
        cand_index, resumes, serialize_candidate, "ResumeID",
        build_candidate_metadata, model,
    )

    # Ingest jobs
    jobs = load_jobs()
    print(f"\n--- Ingesting {len(jobs)} jobs ---")
    job_index = get_or_create_index(pc, JOB_INDEX, dimension)
    embed_and_upsert(
        job_index, jobs, serialize_job, "job_id",
        build_job_metadata, model,
    )

    print("\nIngestion complete")


if __name__ == "__main__":
    main()
