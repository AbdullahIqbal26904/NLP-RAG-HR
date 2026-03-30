"""Generate a submission-ready report from experiment outputs.

Usage:
    python -m pipeline.report_builder
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _num(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def _safe_get(d: dict, key: str, default):
    return d.get(key, default) if isinstance(d, dict) else default


def _build_retrieval_table(summary_by_strategy: dict) -> str:
    rows = [
        "| Retrieval Strategy | Faithfulness | Relevancy | Avg Retrieval (s) | Avg Total (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "semantic_only": "Semantic only",
        "hybrid_rrf": "Hybrid + RRF",
        "hybrid_rrf_ce": "Hybrid + RRF + CrossEncoder",
    }
    for k in ["semantic_only", "hybrid_rrf", "hybrid_rrf_ce"]:
        s = summary_by_strategy.get(k, {})
        rows.append(
            "| "
            + f"{labels[k]} | {_pct(_safe_get(s, 'avg_faithfulness', 0.0))} | {_pct(_safe_get(s, 'avg_relevancy', 0.0))} | "
            + f"{_num(_safe_get(s, 'avg_retrieval_s', 0.0), 3)} | {_num(_safe_get(s, 'avg_total_s', 0.0), 3)} |"
        )
    return "\n".join(rows)


def _build_chunking_table(summary_by_chunking: dict) -> str:
    rows = [
        "| Chunking Strategy | Chunk Count | Faithfulness | Relevancy | Avg Retrieval (s) | Avg Total (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in ["fixed", "recursive"]:
        s = summary_by_chunking.get(k, {})
        rows.append(
            "| "
            + f"{k.title()} + Hybrid/RRF/CE | {_safe_get(s, 'chunk_count', 0)} | {_pct(_safe_get(s, 'avg_faithfulness', 0.0))} | "
            + f"{_pct(_safe_get(s, 'avg_relevancy', 0.0))} | {_num(_safe_get(s, 'avg_retrieval_s', 0.0), 3)} | {_num(_safe_get(s, 'avg_total_s', 0.0), 3)} |"
        )
    return "\n".join(rows)


def _pick_best(summary_by_strategy: dict) -> tuple[str, dict]:
    best_name = "semantic_only"
    best_obj = summary_by_strategy.get(best_name, {})
    best_score = _safe_get(best_obj, "avg_faithfulness", 0.0) + _safe_get(best_obj, "avg_relevancy", 0.0)

    for k in ["hybrid_rrf", "hybrid_rrf_ce"]:
        s = summary_by_strategy.get(k, {})
        score = _safe_get(s, "avg_faithfulness", 0.0) + _safe_get(s, "avg_relevancy", 0.0)
        if score > best_score:
            best_name = k
            best_obj = s
            best_score = score

    return best_name, best_obj


def _examples_markdown(examples: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(examples[:3], 1):
        lines.append(f"### Example {i}")
        lines.append(f"- Query: {ex.get('query', '')}")
        lines.append(f"- Mode: {ex.get('mode', '')}")
        lines.append(f"- Faithfulness: {_pct(float(ex.get('faithfulness', 0.0)))}")
        lines.append(f"- Relevancy: {_pct(float(ex.get('relevancy', 0.0)))}")

        verified = ex.get("verified_claims", [])
        if verified:
            lines.append("- Claims Verification (sample):")
            for item in verified[:3]:
                status = "Supported" if item.get("supported") else "Not supported"
                lines.append(f"  - {status}: {item.get('claim', '')}")
                if item.get("reason"):
                    lines.append(f"  - Reason: {item.get('reason', '')}")

        ans = ex.get("answer", "")
        if ans:
            lines.append(f"- Answer excerpt: {ans[:350].replace(chr(10), ' ')}")

        lines.append("")

    return "\n".join(lines)


def build_report(results: dict) -> str:
    config = results.get("config", {})
    baseline = results.get("baseline_fixed_set", {})
    baseline_summary = baseline.get("summary", {})
    retrieval_summary = results.get("ablation", {}).get("retrieval", {}).get("summary_by_strategy", {})
    chunking_summary = results.get("ablation", {}).get("chunking", {}).get("summary_by_chunking", {})
    examples = baseline.get("example_verifications", [])

    best_name, best_obj = _pick_best(retrieval_summary)

    report = f"""# Assignment 3 Report: RAG-based Question-Answering System

## 1. Platform Details
- Development platform: Local macOS machine using Python 3.12.
- Runtime and deployment target: Streamlit app packaged with Docker.
- Vector database: Pinecone (cloud, starter plan).
- LLM inference: Hugging Face Inference API / Router.

## 2. Data Details
- Domain: Human resources (candidate profiles and job postings).
- Dataset size used in the system: 176 candidate profiles and 50 job postings.
- Source and confidentiality statement:
  - This dataset was provided by VentureDive for our Final Year Project.
  - We have signed an MOU, therefore the raw dataset cannot be publicly shared.
  - Only non-sensitive aggregate metrics and system behavior are reported.

## 3. Algorithms, Models, and Retrieval Methods
### 3.1 Embeddings and Vector Store
- Embedding model: {config.get('embedding_model', '')}
- Embeddings are precomputed and upserted to Pinecone indexes for candidates and jobs.

### 3.2 Retrieval Pipeline
- Implemented retrieval variants:
  - Semantic only
  - Hybrid (BM25 + Semantic) with RRF fusion
  - Hybrid (BM25 + Semantic) + RRF + CrossEncoder re-ranking
- CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2

### 3.3 Generation and Evaluation
- LLM model configured: {config.get('hf_model_id', '')}
- Generation uses structured recruiter/candidate prompts with retrieved context.
- LLM-as-a-Judge evaluation implemented:
  - Faithfulness: claim extraction + claim verification against retrieved context
  - Relevancy: generate 3 alternate questions + cosine similarity with original query

### 3.4 Chunking Strategy Comparison
- Fixed chunking: constant character windows with overlap.
- Recursive chunking: paragraph/sentence-aware splitting with max chunk size.

## 4. Performance Metrics
### 4.1 Fixed Test Set Results
- Number of evaluated queries: {baseline_summary.get('count', 0)}
- Average Faithfulness: {_pct(float(baseline_summary.get('avg_faithfulness', 0.0)))}
- Average Relevancy: {_pct(float(baseline_summary.get('avg_relevancy', 0.0)))}
- Average Retrieval Time: {_num(float(baseline_summary.get('avg_retrieval_s', 0.0)), 3)} s
- Average Generation Time: {_num(float(baseline_summary.get('avg_generation_s', 0.0)), 3)} s
- Average Evaluation Time: {_num(float(baseline_summary.get('avg_evaluation_s', 0.0)), 3)} s
- Average End-to-End Time: {_num(float(baseline_summary.get('avg_total_s', 0.0)), 3)} s

### 4.2 Retrieval Ablation (Semantic vs Hybrid)
{_build_retrieval_table(retrieval_summary)}

### 4.3 Chunking Ablation
{_build_chunking_table(chunking_summary)}

## 5. Best Model and Pipeline Selection
- Selected retrieval strategy: {best_name}
- Selection criterion: maximum combined score (Faithfulness + Relevancy) from ablation runs.
- Selected strategy metrics:
  - Faithfulness: {_pct(float(best_obj.get('avg_faithfulness', 0.0)))}
  - Relevancy: {_pct(float(best_obj.get('avg_relevancy', 0.0)))}
  - Avg total latency: {_num(float(best_obj.get('avg_total_s', 0.0)), 3)} s

## 6. Claim Verification Examples (at least 3)
{_examples_markdown(examples)}

## 7. Reproducibility
### 7.1 Environment Setup
1. Configure .env with Pinecone and Hugging Face keys.
2. Build and run the app:
   - docker compose build
   - docker compose up -d

### 7.2 Data Ingestion
- python -m pipeline.ingest

### 7.3 Automated Evaluation + Ablation
- python -m pipeline.experiments

### 7.4 Report Generation
- python -m pipeline.report_builder

## 8. References
- Pinecone documentation
- Hugging Face Inference API / Router documentation
- Sentence Transformers documentation
- Streamlit documentation
- rank-bm25 package documentation

## 9. Appendix Notes
- Public demo URL can be added after Hugging Face Spaces deployment.
- Generated artifacts:
  - reports/experiment_results.json
  - reports/ablation_summary.csv
  - reports/submission_report.md

---
Generated on: {datetime.utcnow().isoformat()}Z
"""
    return report


def main():
    output_dir = Path("reports")
    input_path = output_dir / "experiment_results.json"
    output_path = output_dir / "submission_report.md"

    if not input_path.exists():
        raise FileNotFoundError(
            "Missing reports/experiment_results.json. Run: python -m pipeline.experiments"
        )

    results = json.loads(input_path.read_text(encoding="utf-8"))
    report_text = build_report(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
