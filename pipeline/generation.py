"""LLM generation via HuggingFace Inference API."""
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


def _get_client() -> InferenceClient:
    return InferenceClient(
        model=os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
        token=os.getenv("HF_API_TOKEN"),
    )


def build_recruiter_prompt(query: str, results: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"Candidate {i}: {r.get('metadata', {}).get('name', '')}\n{r.get('text', '')[:600]}"
        for i, r in enumerate(results, 1)
    )
    return f"""You are an expert recruiter assistant.

JOB REQUIREMENT:
{query}

RETRIEVED CANDIDATE PROFILES:
{context}

Based only on the profiles above:
1. Summarize the top 3 matching candidates
2. Explain why each matches the requirement
3. Note any skill gaps

Do not fabricate information not present in the profiles."""


def build_candidate_prompt(query: str, results: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"Job {i}: {r.get('metadata', {}).get('job_title', '')}\n{r.get('text', '')[:600]}"
        for i, r in enumerate(results, 1)
    )
    return f"""You are a career advisor.

CANDIDATE PROFILE:
{query}

RETRIEVED JOB POSTINGS:
{context}

Based only on the jobs above:
1. Recommend the top 3 most suitable jobs and why
2. How well the candidate's skills match each role
3. Any upskilling suggestions

Do not fabricate information not present in the job postings."""


def generate_answer(prompt: str, max_tokens: int = 512) -> str:
    client = _get_client()
    response = client.text_generation(
        f"[INST] {prompt} [/INST]",
        max_new_tokens=max_tokens,
        temperature=0.3,
        repetition_penalty=1.1,
        do_sample=True,
    )
    return response.strip()
