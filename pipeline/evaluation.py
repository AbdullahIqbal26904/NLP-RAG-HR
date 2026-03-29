"""LLM-as-a-Judge: Faithfulness + Relevancy evaluation."""
import json
import os

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

load_dotenv()


class RAGEvaluator:
    def __init__(self):
        self.llm = InferenceClient(
            model=os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
            token=os.getenv("HF_API_TOKEN"),
        )
        self.embedder = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )

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
        for claim in claims[:10]:
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
        return self.llm.text_generation(
            f"[INST] {prompt} [/INST]",
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
        ).strip()


# Fixed test set
TEST_QUERIES_RECRUITER = [
    "Looking for a Python backend developer with FastAPI and PostgreSQL, 3+ years",
    "Need a React frontend developer with TypeScript experience",
    "Seeking a DevOps engineer with Docker and Kubernetes skills",
    "Looking for a data scientist with machine learning and Python expertise",
    "Need a full stack developer with Node.js and React, 2+ years",
    "Seeking a mobile developer with Flutter or React Native",
    "Looking for a project manager with Agile and Scrum certification",
    "Need a cloud architect with AWS or Azure, 5+ years",
    "Seeking a UI/UX designer with Figma and user research skills",
    "Looking for a cybersecurity analyst with penetration testing experience",
]

TEST_QUERIES_CANDIDATE = [
    "I have 4 years Python experience, Django and FastAPI, looking for backend roles",
    "React developer with 3 years experience, TypeScript and Next.js",
    "DevOps engineer, 2 years with Docker, CI/CD, AWS",
    "Data scientist with scikit-learn, pandas, 2 years ML experience",
    "Full stack developer, Node.js backend, React frontend, 3 years total",
]
