"""
Hybrid retrieval: BM25 + Semantic -> RRF -> CrossEncoder re-ranking.
"""
import json
import os
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from pipeline.serializer import serialize_candidate, serialize_job

load_dotenv()

SearchMode = Literal["candidate", "job"]


class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.candidate_index = pc.Index(os.getenv("PINECONE_CANDIDATE_INDEX", "vend-candidates"))
        self.job_index = pc.Index(os.getenv("PINECONE_JOB_INDEX", "vend-jobs"))

        with open("data/candidates.json", encoding="utf-8") as f:
            candidates = json.load(f)
        with open("data/jobs.json", encoding="utf-8") as f:
            jobs = json.load(f)

        self._candidate_ids = [str(c["candidate_id"]) for c in candidates]
        self._job_ids = [str(j["job_id"]) for j in jobs]
        self._candidate_texts = [serialize_candidate(c) for c in candidates]
        self._job_texts = [serialize_job(j) for j in jobs]

        self._bm25_candidates = BM25Okapi([t.lower().split() for t in self._candidate_texts])
        self._bm25_jobs = BM25Okapi([t.lower().split() for t in self._job_texts])

        self.top_k = int(os.getenv("TOP_K_RETRIEVE", 20))
        self.top_k_final = int(os.getenv("TOP_K_RERANK", 5))

    def retrieve(self, query: str, mode: SearchMode, strategy: str = "hybrid_rrf_ce") -> list[dict]:
        query_vector = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        semantic = self._semantic_search(query_vector, mode)
        bm25 = self._bm25_search(query, mode)

        if strategy == "semantic_only":
            ranked = sorted(semantic, key=lambda x: x.get("score", 0.0), reverse=True)[: self.top_k_final]
            for i, doc in enumerate(ranked, 1):
                doc.setdefault("semantic_score", doc.get("score", 0.0))
                doc.setdefault("bm25_score", 0.0)
                doc.setdefault("rrf_score", 0.0)
                doc.setdefault("cross_encoder_score", 0.0)
                doc["final_rank"] = i
            return ranked

        fused = self._rrf_fusion(semantic, bm25)

        if strategy == "hybrid_rrf":
            ranked = fused[: self.top_k_final]
            for i, doc in enumerate(ranked, 1):
                doc.setdefault("cross_encoder_score", 0.0)
                doc["final_rank"] = i
            return ranked

        if strategy == "hybrid_rrf_ce":
            reranked = self._cross_encoder_rerank(query, fused)
            return reranked[:self.top_k_final]

        raise ValueError(
            f"Unknown strategy '{strategy}'. Use one of: semantic_only, hybrid_rrf, hybrid_rrf_ce"
        )

    def _semantic_search(self, query_vector: list, mode: SearchMode) -> list[dict]:
        index = self.candidate_index if mode == "candidate" else self.job_index
        response = index.query(vector=query_vector, top_k=self.top_k, include_metadata=True)
        return [
            {
                "id": m["id"],
                "score": float(m["score"]),
                "metadata": m.get("metadata", {}),
                "text": m.get("metadata", {}).get("text", ""),
            }
            for m in response.get("matches", [])
        ]

    def _bm25_search(self, query: str, mode: SearchMode) -> list[dict]:
        tokens = query.lower().split()
        if mode == "candidate":
            scores = self._bm25_candidates.get_scores(tokens)
            ids, texts = self._candidate_ids, self._candidate_texts
        else:
            scores = self._bm25_jobs.get_scores(tokens)
            ids, texts = self._job_ids, self._job_texts

        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [
            {"id": ids[i], "score": float(scores[i]), "text": texts[i], "metadata": {}}
            for i in top_indices if scores[i] > 0
        ]

    def _rrf_fusion(self, semantic: list[dict], bm25: list[dict], k: int = 60) -> list[dict]:
        rrf: dict[str, float] = {}
        sem_map, bm25_map = {}, {}

        for rank, r in enumerate(semantic):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1 / (k + rank + 1)
            sem_map[r["id"]] = r

        for rank, r in enumerate(bm25):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1 / (k + rank + 1)
            bm25_map[r["id"]] = r

        fused = []
        for doc_id, rrf_score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
            s = sem_map.get(doc_id, {})
            b = bm25_map.get(doc_id, {})
            fused.append({
                "id": doc_id,
                "text": s.get("text") or b.get("text") or "",
                "metadata": s.get("metadata") or b.get("metadata") or {},
                "semantic_score": s.get("score", 0.0),
                "bm25_score": b.get("score", 0.0),
                "rrf_score": rrf_score,
            })

        return fused[:self.top_k]

    def _cross_encoder_rerank(self, query: str, docs: list[dict]) -> list[dict]:
        if not docs:
            return []
        pairs = [(query, d["text"][:512]) for d in docs]
        scores = self.cross_encoder.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["cross_encoder_score"] = float(score)
        reranked = sorted(docs, key=lambda x: x["cross_encoder_score"], reverse=True)
        for rank, doc in enumerate(reranked):
            doc["final_rank"] = rank + 1
        return reranked
