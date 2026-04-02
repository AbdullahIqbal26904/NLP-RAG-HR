"""LLM-as-a-Judge: Faithfulness + Relevancy evaluation."""
import json
import os
import re

import numpy as np
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


class RAGEvaluator:
    def __init__(self):
        self.embedder = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.max_claims = int(os.getenv("EVAL_MAX_CLAIMS", "10"))

    def evaluate(self, query: str, answer: str, context: str) -> dict:
        faith = self._faithfulness(answer, context)
        rel = self._relevancy(query, answer)
        return {
            "faithfulness_score": faith["score"],
            "relevancy_score": rel["score"],
            "claims": faith["claims"],
            "verified_claims": faith["verified"],
            "generated_questions": rel["questions"],
            "similarity_scores": rel["similarities"],
        }

    def _faithfulness(self, answer: str, context: str) -> dict:
        # Step 1: Extract claims
        raw = self._call_llm(
            f"""Extract all factual claims from this answer as a JSON array of strings.
Return ONLY the JSON array, nothing else.

ANSWER: {answer}""",
            max_tokens=300,
        )
        try:
            claims = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
        except Exception:
            claims = [answer[:200]]

        # Step 2: Verify each claim
        verified = []
        for claim in claims[: self.max_claims]:
            raw2 = self._call_llm(
                f"""Is this claim supported by the context? Reply with JSON only: {{"supported": true or false, "reason": "one sentence"}}

CLAIM: {claim}
CONTEXT: {context[:800]}""",
                max_tokens=100,
            )
            try:
                result = json.loads(raw2[raw2.find("{"):raw2.rfind("}") + 1])
                verified.append({
                    "claim": claim,
                    "supported": bool(result.get("supported")),
                    "reason": result.get("reason", ""),
                })
            except Exception:
                verified.append({"claim": claim, "supported": False, "reason": "parse error"})

        score = sum(1 for v in verified if v["supported"]) / len(verified) if verified else 0.0
        return {"score": score, "claims": claims, "verified": verified}

    def _relevancy(self, query: str, answer: str) -> dict:
        # Step 1: Generate 3 questions from answer
        raw = self._call_llm(
            f"""Generate exactly 3 questions that this answer is responding to.
Return ONLY a JSON array of 3 strings.

ANSWER: {answer}""",
            max_tokens=150,
        )
        try:
            questions = json.loads(raw[raw.find("["):raw.rfind("]") + 1])[:3]
        except Exception:
            questions = [query]
        while len(questions) < 3:
            questions.append(query)

        # Step 2: Cosine similarity
        q_emb = self.embedder.encode(query, normalize_embeddings=True)
        q_embs = self.embedder.encode(questions, normalize_embeddings=True)
        similarities = [float(np.dot(q_emb, e)) for e in q_embs]

        return {
            "score": float(np.mean(similarities)),
            "questions": questions,
            "similarities": similarities,
        }

    def _call_llm(self, prompt: str, max_tokens: int = 256) -> str:
        # Primary: HuggingFace (free)
        try:
            return self._hf_generate(prompt, max_tokens=max_tokens)
        except Exception:
            pass
        # Fallback 1: Groq
        try:
            return self._groq_generate(prompt, max_tokens=max_tokens)
        except Exception:
            return self._heuristic_llm_response(prompt)

    @staticmethod
    def _hf_generate(prompt: str, max_tokens: int = 256) -> str:
        token = os.getenv("HF_API_TOKEN")
        model_id = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
        if not token:
            raise ValueError("HF_API_TOKEN is not set")

        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HF request failed: {response.status_code} {response.text}")

        parsed = response.json()
        choices = parsed.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return str(parsed)

    @staticmethod
    def _groq_generate(prompt: str, max_tokens: int = 256) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Groq request failed: {response.status_code} {response.text}")

        parsed = response.json()
        choices = parsed.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return str(parsed)

    def _heuristic_llm_response(self, prompt: str) -> str:
        lower = prompt.lower()

        if "extract all factual claims" in lower:
            answer = prompt.split("ANSWER:", 1)[-1].strip()
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", answer)
                if s.strip()
            ]
            claims = sentences[: max(1, min(self.max_claims, 3))] or [answer[:220]]
            return json.dumps(claims)

        if "is this claim supported by the context" in lower:
            claim = ""
            context = ""
            if "CLAIM:" in prompt:
                claim = prompt.split("CLAIM:", 1)[1].split("CONTEXT:", 1)[0].strip()
            if "CONTEXT:" in prompt:
                context = prompt.split("CONTEXT:", 1)[1].strip()

            claim_terms = {
                t
                for t in re.findall(r"[a-zA-Z0-9_]+", claim.lower())
                if len(t) > 3
            }
            ctx_terms = {
                t
                for t in re.findall(r"[a-zA-Z0-9_]+", context.lower())
                if len(t) > 3
            }
            overlap = len(claim_terms & ctx_terms)
            supported = overlap >= 2
            reason = (
                "Lexical overlap found between claim and context."
                if supported
                else "Insufficient lexical overlap between claim and context."
            )
            return json.dumps({"supported": supported, "reason": reason})

        if "generate exactly 3 questions" in lower:
            answer = prompt.split("ANSWER:", 1)[-1].strip()
            keywords = [
                t
                for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", answer)
                if len(t) > 4
            ]
            topic = " ".join(keywords[:3]) if keywords else "the provided answer"
            questions = [
                f"What are the main points about {topic}?",
                f"Which skills or qualifications are emphasized in {topic}?",
                f"What gaps or next steps are suggested in {topic}?",
            ]
            return json.dumps(questions)

        return "[]"


# Fixed test set - queries aligned with resume/job dataset categories
TEST_QUERIES_RECRUITER = [
    "Looking for a Java developer with Spring Boot and microservices experience",
    "Need a Python developer with machine learning and data science skills",
    "Seeking a DevOps engineer with AWS, Docker, and Kubernetes experience",
    "Looking for a backend developer with Node.js and database management skills",
    "Need a full-stack developer with React and Django experience",
    "Seeking a database administrator with SQL and performance tuning skills",
    "Looking for a cybersecurity analyst with network security experience",
    "Need a cloud engineer with Azure or AWS certification",
    "Seeking a blockchain developer with smart contract development skills",
    "Looking for a data scientist with deep learning and NLP experience",
]

TEST_QUERIES_CANDIDATE = [
    "I have 5 years Java development experience with Spring Boot and AWS, looking for senior roles",
    "Python developer with 3 years in machine learning and data analysis seeking ML engineer positions",
    "DevOps professional with Docker, Kubernetes, and CI/CD pipeline expertise looking for opportunities",
    "Full-stack developer with React, Node.js, and PostgreSQL skills seeking remote positions",
    "Database administrator with 4 years Oracle and MySQL experience looking for DBA roles",
]
