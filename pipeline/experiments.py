"""Run automated evaluation, ablation studies, and latency benchmarks.

Usage:
    python -m pipeline.experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from pipeline.evaluation import (
    RAGEvaluator,
    TEST_QUERIES_CANDIDATE,
    TEST_QUERIES_RECRUITER,
)
from pipeline.generation import (
    build_candidate_prompt,
    build_recruiter_prompt,
    generate_answer,
)
from pipeline.retrieval import HybridRetriever
from pipeline.serializer import serialize_candidate

load_dotenv()


def _avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _now_iso() -> str:
    import datetime

    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _summarize_rows(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "avg_faithfulness": _avg([r["faithfulness"] for r in rows]),
        "avg_relevancy": _avg([r["relevancy"] for r in rows]),
        "avg_retrieval_s": _avg([r["retrieval_s"] for r in rows]),
        "avg_generation_s": _avg([r["generation_s"] for r in rows]),
        "avg_evaluation_s": _avg([r["evaluation_s"] for r in rows]),
        "avg_total_s": _avg([r["total_s"] for r in rows]),
    }


def _run_query(
    query: str,
    mode: str,
    strategy: str,
    retriever: HybridRetriever,
    evaluator: RAGEvaluator,
    gen_tokens: int,
    keep_details: bool = False,
) -> dict:
    t0 = time.perf_counter()
    results = retriever.retrieve(query, mode=mode, strategy=strategy)
    t1 = time.perf_counter()

    if mode == "candidate":
        prompt = build_recruiter_prompt(query, results)
    else:
        prompt = build_candidate_prompt(query, results)

    answer = generate_answer(prompt, max_tokens=gen_tokens)
    t2 = time.perf_counter()

    context_text = "\n\n".join(r.get("text", "")[:500] for r in results)
    ev = evaluator.evaluate(query, answer, context_text)
    t3 = time.perf_counter()

    row = {
        "query": query,
        "mode": mode,
        "strategy": strategy,
        "retrieval_count": len(results),
        "faithfulness": float(ev.get("faithfulness_score", 0.0)),
        "relevancy": float(ev.get("relevancy_score", 0.0)),
        "retrieval_s": t1 - t0,
        "generation_s": t2 - t1,
        "evaluation_s": t3 - t2,
        "total_s": t3 - t0,
    }

    if keep_details:
        row["answer"] = answer
        row["claims"] = ev.get("claims", [])
        row["verified_claims"] = ev.get("verified_claims", [])
        row["generated_questions"] = ev.get("generated_questions", [])
        row["similarity_scores"] = ev.get("similarity_scores", [])

    return row


def run_fixed_set(
    retriever: HybridRetriever,
    evaluator: RAGEvaluator,
    gen_tokens: int,
    max_queries: int,
) -> dict:
    rows: list[dict] = []

    recruiter_qs = TEST_QUERIES_RECRUITER[:max_queries]
    candidate_qs = TEST_QUERIES_CANDIDATE[:max(1, min(max_queries // 2, len(TEST_QUERIES_CANDIDATE)))]

    for q in recruiter_qs:
        rows.append(
            _run_query(
                q,
                mode="candidate",
                strategy="hybrid_rrf_ce",
                retriever=retriever,
                evaluator=evaluator,
                gen_tokens=gen_tokens,
                keep_details=True,
            )
        )

    for q in candidate_qs:
        rows.append(
            _run_query(
                q,
                mode="job",
                strategy="hybrid_rrf_ce",
                retriever=retriever,
                evaluator=evaluator,
                gen_tokens=gen_tokens,
                keep_details=True,
            )
        )

    examples = rows[:3]
    return {
        "summary": _summarize_rows(rows),
        "rows": rows,
        "example_verifications": examples,
    }


def run_retrieval_ablation(
    retriever: HybridRetriever,
    evaluator: RAGEvaluator,
    gen_tokens: int,
    max_queries: int,
) -> dict:
    rows: list[dict] = []
    queries = TEST_QUERIES_RECRUITER[:max_queries]
    strategies = ["semantic_only", "hybrid_rrf", "hybrid_rrf_ce"]

    for strategy in strategies:
        for q in queries:
            rows.append(
                _run_query(
                    q,
                    mode="candidate",
                    strategy=strategy,
                    retriever=retriever,
                    evaluator=evaluator,
                    gen_tokens=gen_tokens,
                    keep_details=False,
                )
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["strategy"]].append(r)

    by_strategy = {
        s: _summarize_rows(grouped[s]) for s in ["semantic_only", "hybrid_rrf", "hybrid_rrf_ce"]
    }

    return {
        "summary_by_strategy": by_strategy,
        "rows": rows,
    }


def _fixed_chunks(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    out = []
    for i in range(0, len(text), step):
        c = text[i : i + chunk_size].strip()
        if c:
            out.append(c)
    return out


def _recursive_chunks(text: str, chunk_size: int = 700) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: list[str] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if not sentences:
            sentences = [para]
        for s in sentences:
            if len(s) <= chunk_size:
                units.append(s)
            else:
                units.extend(_fixed_chunks(s, chunk_size=chunk_size, overlap=80))

    chunks: list[str] = []
    current = ""
    for u in units:
        candidate = f"{current} {u}".strip() if current else u
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = u
    if current:
        chunks.append(current)

    return chunks


class LocalChunkRetriever:
    def __init__(self, chunks: list[dict], retriever: HybridRetriever):
        self._chunks = chunks
        self._texts = [c["text"] for c in chunks]
        self._ids = [c["id"] for c in chunks]
        self._meta = [
            {
                "name": c.get("name", ""),
                "source_id": c.get("source_id", ""),
                "chunking": c.get("chunking", ""),
            }
            for c in chunks
        ]
        self._embedder = retriever.embedding_model
        self._cross_encoder = retriever.cross_encoder
        self._top_k = retriever.top_k
        self._top_k_final = retriever.top_k_final

        self._embeddings = self._embedder.encode(
            self._texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._bm25 = BM25Okapi([t.lower().split() for t in self._texts])

    def retrieve(self, query: str, strategy: str = "hybrid_rrf_ce") -> list[dict]:
        q = self._embedder.encode(query, normalize_embeddings=True)
        sem_scores = np.dot(self._embeddings, q)
        sem_idx = np.argsort(sem_scores)[::-1][: self._top_k]

        semantic = [
            {
                "id": self._ids[i],
                "score": float(sem_scores[i]),
                "metadata": self._meta[i],
                "text": self._texts[i],
            }
            for i in sem_idx
        ]

        if strategy == "semantic_only":
            ranked = semantic[: self._top_k_final]
            for rank, d in enumerate(ranked, 1):
                d["semantic_score"] = d.get("score", 0.0)
                d["bm25_score"] = 0.0
                d["rrf_score"] = 0.0
                d["cross_encoder_score"] = 0.0
                d["final_rank"] = rank
            return ranked

        bm_scores = self._bm25.get_scores(query.lower().split())
        bm_idx = np.argsort(bm_scores)[::-1][: self._top_k]
        bm25 = [
            {
                "id": self._ids[i],
                "score": float(bm_scores[i]),
                "metadata": self._meta[i],
                "text": self._texts[i],
            }
            for i in bm_idx
            if bm_scores[i] > 0
        ]

        fused = self._rrf(semantic, bm25)
        if strategy == "hybrid_rrf":
            out = fused[: self._top_k_final]
            for rank, d in enumerate(out, 1):
                d["cross_encoder_score"] = 0.0
                d["final_rank"] = rank
            return out

        pairs = [(query, d["text"][:512]) for d in fused]
        ce_scores = self._cross_encoder.predict(pairs)
        for d, s in zip(fused, ce_scores):
            d["cross_encoder_score"] = float(s)
        out = sorted(fused, key=lambda x: x["cross_encoder_score"], reverse=True)[: self._top_k_final]
        for rank, d in enumerate(out, 1):
            d["final_rank"] = rank
        return out

    @staticmethod
    def _rrf(semantic: list[dict], bm25: list[dict], k: int = 60) -> list[dict]:
        rrf: dict[str, float] = {}
        sem_map = {d["id"]: d for d in semantic}
        bm_map = {d["id"]: d for d in bm25}

        for rank, d in enumerate(semantic):
            rrf[d["id"]] = rrf.get(d["id"], 0.0) + 1.0 / (k + rank + 1)

        for rank, d in enumerate(bm25):
            rrf[d["id"]] = rrf.get(d["id"], 0.0) + 1.0 / (k + rank + 1)

        fused: list[dict] = []
        for doc_id, score in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True):
            s = sem_map.get(doc_id, {})
            b = bm_map.get(doc_id, {})
            fused.append(
                {
                    "id": doc_id,
                    "text": s.get("text") or b.get("text") or "",
                    "metadata": s.get("metadata") or b.get("metadata") or {},
                    "semantic_score": s.get("score", 0.0),
                    "bm25_score": b.get("score", 0.0),
                    "rrf_score": score,
                }
            )
        return fused


def _build_chunk_dataset(chunking: str) -> list[dict]:
    """Build chunked dataset from resumes for chunking ablation."""
    resumes = []
    with open("data/resumes_dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                resumes.append(json.loads(line))

    chunks: list[dict] = []
    for r in resumes:
        rid = str(r.get("ResumeID", ""))
        name = r.get("Name") or rid
        text = serialize_candidate(r)

        if chunking == "fixed":
            pieces = _fixed_chunks(text)
        elif chunking == "recursive":
            pieces = _recursive_chunks(text)
        else:
            raise ValueError(f"Unknown chunking method: {chunking}")

        for i, piece in enumerate(pieces, 1):
            chunks.append(
                {
                    "id": f"{rid}_{i}",
                    "source_id": rid,
                    "name": name,
                    "chunking": chunking,
                    "text": piece,
                }
            )

    return chunks


def run_chunking_ablation(
    retriever: HybridRetriever,
    evaluator: RAGEvaluator,
    gen_tokens: int,
    max_queries: int,
) -> dict:
    queries = TEST_QUERIES_RECRUITER[:max_queries]
    methods = ["fixed", "recursive"]
    rows: list[dict] = []

    for method in methods:
        chunk_dataset = _build_chunk_dataset(method)
        local = LocalChunkRetriever(chunk_dataset, retriever)

        for q in queries:
            t0 = time.perf_counter()
            results = local.retrieve(q, strategy="hybrid_rrf_ce")
            t1 = time.perf_counter()

            prompt = build_recruiter_prompt(q, results)
            answer = generate_answer(prompt, max_tokens=gen_tokens)
            t2 = time.perf_counter()

            context = "\n\n".join(r.get("text", "")[:500] for r in results)
            ev = evaluator.evaluate(q, answer, context)
            t3 = time.perf_counter()

            rows.append(
                {
                    "query": q,
                    "chunking": method,
                    "chunk_count": len(chunk_dataset),
                    "faithfulness": float(ev.get("faithfulness_score", 0.0)),
                    "relevancy": float(ev.get("relevancy_score", 0.0)),
                    "retrieval_s": t1 - t0,
                    "generation_s": t2 - t1,
                    "evaluation_s": t3 - t2,
                    "total_s": t3 - t0,
                }
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["chunking"]].append(r)

    by_method = {m: _summarize_rows(grouped[m]) for m in methods}
    by_method["fixed"]["chunk_count"] = grouped["fixed"][0]["chunk_count"] if grouped["fixed"] else 0
    by_method["recursive"]["chunk_count"] = grouped["recursive"][0]["chunk_count"] if grouped["recursive"] else 0

    return {
        "summary_by_chunking": by_method,
        "rows": rows,
    }


def _write_ablation_csv(path: Path, retrieval_ablation: dict, chunking_ablation: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Strategy",
                "Faithfulness",
                "Relevancy",
                "Avg Retrieval Time (s)",
                "Avg Total Time (s)",
            ],
        )
        writer.writeheader()

        strategy_labels = {
            "semantic_only": "Semantic only",
            "hybrid_rrf": "Hybrid + RRF",
            "hybrid_rrf_ce": "Hybrid + RRF + CE",
        }
        for key, summary in retrieval_ablation["summary_by_strategy"].items():
            writer.writerow(
                {
                    "Strategy": strategy_labels.get(key, key),
                    "Faithfulness": f"{summary['avg_faithfulness']:.4f}",
                    "Relevancy": f"{summary['avg_relevancy']:.4f}",
                    "Avg Retrieval Time (s)": f"{summary['avg_retrieval_s']:.3f}",
                    "Avg Total Time (s)": f"{summary['avg_total_s']:.3f}",
                }
            )

        for key, summary in chunking_ablation["summary_by_chunking"].items():
            writer.writerow(
                {
                    "Strategy": f"Chunking: {key.title()} (Hybrid + RRF + CE)",
                    "Faithfulness": f"{summary['avg_faithfulness']:.4f}",
                    "Relevancy": f"{summary['avg_relevancy']:.4f}",
                    "Avg Retrieval Time (s)": f"{summary['avg_retrieval_s']:.3f}",
                    "Avg Total Time (s)": f"{summary['avg_total_s']:.3f}",
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation, ablation, and benchmark suite")
    parser.add_argument("--baseline-queries", type=int, default=10)
    parser.add_argument("--retrieval-ablation-queries", type=int, default=5)
    parser.add_argument("--chunking-ablation-queries", type=int, default=3)
    parser.add_argument("--gen-max-tokens", type=int, default=220)
    parser.add_argument("--eval-max-claims", type=int, default=5)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    os.environ["EVAL_MAX_CLAIMS"] = str(args.eval_max_claims)

    print("Initializing retriever/evaluator...")
    retriever = HybridRetriever()
    evaluator = RAGEvaluator()

    print("Running fixed test-set evaluation...")
    baseline = run_fixed_set(
        retriever=retriever,
        evaluator=evaluator,
        gen_tokens=args.gen_max_tokens,
        max_queries=args.baseline_queries,
    )

    print("Running retrieval ablation...")
    retrieval_ablation = run_retrieval_ablation(
        retriever=retriever,
        evaluator=evaluator,
        gen_tokens=args.gen_max_tokens,
        max_queries=args.retrieval_ablation_queries,
    )

    print("Running chunking ablation...")
    chunking_ablation = run_chunking_ablation(
        retriever=retriever,
        evaluator=evaluator,
        gen_tokens=args.gen_max_tokens,
        max_queries=args.chunking_ablation_queries,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "run_at": _now_iso(),
        "config": {
            "hf_model_id": os.getenv("HF_MODEL_ID", ""),
            "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
            "top_k_retrieve": int(os.getenv("TOP_K_RETRIEVE", "20")),
            "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5")),
            "baseline_queries": args.baseline_queries,
            "retrieval_ablation_queries": args.retrieval_ablation_queries,
            "chunking_ablation_queries": args.chunking_ablation_queries,
            "gen_max_tokens": args.gen_max_tokens,
            "eval_max_claims": args.eval_max_claims,
        },
        "baseline_fixed_set": baseline,
        "ablation": {
            "retrieval": retrieval_ablation,
            "chunking": chunking_ablation,
        },
    }

    json_path = output_dir / "experiment_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = output_dir / "ablation_summary.csv"
    _write_ablation_csv(csv_path, retrieval_ablation, chunking_ablation)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")

    fixed = baseline["summary"]
    print(
        "Fixed-set averages -> "
        f"Faithfulness={fixed['avg_faithfulness']:.4f}, "
        f"Relevancy={fixed['avg_relevancy']:.4f}, "
        f"TotalLatency={fixed['avg_total_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
