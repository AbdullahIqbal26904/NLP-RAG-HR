"""
Ingest: Load JSON -> Serialize -> Embed -> Upsert to Pinecone.
Run: python -m pipeline.ingest
"""
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
    skills = [s.get('skill_name') or '' for s in (record.get('skills') or [])]
    return {
        "type": "candidate",
        "candidate_id": record.get('candidate_id'),
        "name": f"{record.get('first_name') or ''} {record.get('last_name') or ''}".strip(),
        "current_role": record.get('current_role') or '',
        "industry": record.get('associated_industry') or '',
        "years_experience": record.get('years_experience') or 0,
        "city": record.get('city') or '',
        "country": record.get('country') or '',
        "skills": skills[:20],
        "text": text[:4000],
    }


def build_job_metadata(record: dict, text: str) -> dict:
    required_skills = [
        s.get('skill_name') or ''
        for s in (record.get('skills') or [])
        if s.get('is_required')
    ]
    return {
        "type": "job",
        "job_id": record.get('job_id'),
        "job_title": record.get('job_title') or '',
        "industry": record.get('industry') or '',
        "employment_type": record.get('employment_type') or '',
        "min_years_experience": record.get('min_years_experience') or 0,
        "max_years_experience": record.get('max_years_experience') or 0,
        "required_skills": required_skills[:20],
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


def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    model = SentenceTransformer(EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()

    with open("data/candidates.json", encoding="utf-8") as f:
        candidates = json.load(f)
    with open("data/jobs.json", encoding="utf-8") as f:
        jobs = json.load(f)

    print(f"\n--- Ingesting {len(candidates)} candidates ---")
    cand_index = get_or_create_index(pc, CANDIDATE_INDEX, dimension)
    embed_and_upsert(cand_index, candidates, serialize_candidate, "candidate_id", build_candidate_metadata, model)

    print(f"\n--- Ingesting {len(jobs)} jobs ---")
    job_index = get_or_create_index(pc, JOB_INDEX, dimension)
    embed_and_upsert(job_index, jobs, serialize_job, "job_id", build_job_metadata, model)

    print("\nIngestion complete")


if __name__ == "__main__":
    main()
