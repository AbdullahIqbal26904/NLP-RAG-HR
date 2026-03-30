"""LLM generation via HuggingFace Inference API."""
import json
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests

load_dotenv()


def _get_client() -> InferenceClient:
    return InferenceClient(
        model=os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
        token=os.getenv("HF_API_TOKEN"),
    )


def _extract_generated_text(payload) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("generated_text"), str):
            return payload["generated_text"]
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                text = choice.get("text")
                if isinstance(text, str):
                    return text
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
            return first["generated_text"]
    return str(payload)


def _router_text_generation(
    prompt: str,
    max_tokens: int,
    temperature: float,
    repetition_penalty: float,
    do_sample: bool,
) -> str:
    model_id = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise ValueError("HF_API_TOKEN is not set")

    url = "https://router.huggingface.co/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HF router request failed: {response.status_code} {response.text}")

    try:
        parsed = response.json()
    except json.JSONDecodeError:
        return response.text.strip()

    return _extract_generated_text(parsed).strip()


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
    formatted_prompt = f"[INST] {prompt} [/INST]"
    client = _get_client()
    try:
        response = client.text_generation(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=0.3,
            repetition_penalty=1.1,
            do_sample=True,
        )
        return response.strip()
    except Exception:
        return _router_text_generation(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            repetition_penalty=1.1,
            do_sample=True,
        )
