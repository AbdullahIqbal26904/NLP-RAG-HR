"""Streamlit UI - RAG Talent Matching. Deployed on HuggingFace Spaces."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline.generation import (
    build_candidate_prompt,
    build_recruiter_prompt,
    generate_answer,
)

load_dotenv()

st.set_page_config(page_title="HR RAG", page_icon="", layout="wide")


@st.cache_resource
def load_retriever():
    from pipeline.retrieval import HybridRetriever

    return HybridRetriever()


@st.cache_resource
def load_evaluator():
    from pipeline.evaluation import RAGEvaluator

    return RAGEvaluator()


def load_ablation_table() -> pd.DataFrame:
    csv_path = Path("reports/ablation_summary.csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass

    return pd.DataFrame(
        {
            "Strategy": [
                "Semantic only",
                "Hybrid + RRF",
                "Hybrid + RRF + CE",
                "Chunking: Fixed (Hybrid + RRF + CE)",
                "Chunking: Recursive (Hybrid + RRF + CE)",
            ],
            "Faithfulness": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Relevancy": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


# Header
st.title("AI Talent Matching RAG")
st.caption("Hybrid BM25 + Semantic Search -> RRF -> CrossEncoder Re-ranking -> LLM Generation")

# Mode selector
mode = st.radio(
    "Search Mode",
    ["Recruiter - Find Candidates", "Candidate - Find Jobs"],
    horizontal=True,
)
is_recruiter = "Recruiter" in mode

placeholder = (
    "e.g. Looking for a Python backend developer with FastAPI, 3+ years"
    if is_recruiter
    else "e.g. I have 4 years Python experience with Django, looking for backend roles"
)
query = st.text_area("Query:", placeholder=placeholder, height=90)

col1, col2 = st.columns([1, 5])
with col1:
    search = st.button("Search", type="primary", use_container_width=True)
with col2:
    show_eval = st.checkbox("Show Faithfulness + Relevancy scores", value=False)

# Search
if search and query.strip():
    retriever = load_retriever()
    search_mode = "candidate" if is_recruiter else "job"

    with st.spinner("Retrieving and re-ranking..."):
        results = retriever.retrieve(query, mode=search_mode)

    if not results:
        st.warning("No results found.")
        st.stop()

    context_text = "\n\n".join(r.get("text", "")[:500] for r in results)

    with st.spinner("Generating answer with LLM..."):
        prompt = build_recruiter_prompt(query, results) if is_recruiter else build_candidate_prompt(query, results)
        answer = generate_answer(prompt)

    # Generated answer
    st.subheader("Generated Answer")
    st.markdown(answer)
    st.divider()

    # Retrieved context
    st.subheader(f"Retrieved Context - {len(results)} results")
    for i, r in enumerate(results, 1):
        orig = r.get("original", {})
        meta = r.get("metadata", {})

        # Use full original data for display, fall back to metadata
        if is_recruiter:
            title = orig.get("Name") or meta.get("name") or f"Result {i}"
            subtitle = orig.get("Category") or meta.get("category") or ""
        else:
            title = orig.get("Job Title") or meta.get("job_title") or f"Result {i}"
            subtitle = ""

        with st.expander(
            f"#{r.get('final_rank', i)} {title} | {subtitle} - "
            f"CE: {r.get('cross_encoder_score', 0):.3f} | "
            f"Sem: {r.get('semantic_score', 0):.3f} | "
            f"BM25: {r.get('bm25_score', 0):.2f} | "
            f"RRF: {r.get('rrf_score', 0):.4f}"
        ):
            # Scores row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CrossEncoder", f"{r.get('cross_encoder_score', 0):.3f}")
            c2.metric("Semantic", f"{r.get('semantic_score', 0):.3f}")
            c3.metric("BM25", f"{r.get('bm25_score', 0):.2f}")
            c4.metric("RRF", f"{r.get('rrf_score', 0):.4f}")

            if orig and is_recruiter:
                # Full resume details from original data
                st.markdown(f"**Name:** {orig.get('Name', 'N/A')}")
                st.markdown(f"**Category:** {orig.get('Category', 'N/A')}")
                st.markdown(f"**Location:** {orig.get('Location', 'N/A')}")
                st.markdown(f"**Email:** {orig.get('Email', 'N/A')}")
                st.markdown(f"**Skills:** {orig.get('Skills', 'N/A')}")
                st.markdown(f"**Education:** {orig.get('Education', 'N/A')}")
                summary = orig.get("Summary") or ""
                if summary:
                    st.markdown("**Summary:**")
                    st.text(summary[:1000])
                experience = orig.get("Experience") or ""
                if experience:
                    st.markdown("**Experience:**")
                    st.text(experience[:2000])

            elif orig and not is_recruiter:
                # Full job details from original data
                st.markdown(f"**Job Title:** {orig.get('Job Title', 'N/A')}")
                description = orig.get("Job Description") or ""
                if description:
                    st.markdown("**Job Description:**")
                    st.text(description)

            else:
                # Fallback to serialized text
                skills = meta.get("skills") or []
                if skills:
                    st.markdown(f"**Skills:** {', '.join(skills[:10])}")
                st.text(r.get("text", "")[:800])

    # Evaluation
    if show_eval:
        st.divider()
        st.subheader("LLM-as-a-Judge Evaluation")
        with st.spinner("Evaluating..."):
            evaluator = load_evaluator()
            ev = evaluator.evaluate(query, answer, context_text)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.metric("Faithfulness", f"{ev['faithfulness_score']:.1%}")
            st.markdown("**Claims:**")
            for v in ev.get("verified_claims", [])[:5]:
                icon = "supported" if v["supported"] else "not supported"
                st.markdown(f"[{icon}] {v['claim']}")
                if v.get("reason"):
                    st.caption(v["reason"])
        with ec2:
            st.metric("Relevancy", f"{ev['relevancy_score']:.1%}")
            st.markdown("**Generated Questions:**")
            for q, sim in zip(ev.get("generated_questions", []), ev.get("similarity_scores", [])):
                st.markdown(f"- {q}")
                st.caption(f"Similarity: {sim:.3f}")

# Sidebar: Ablation study
with st.sidebar:
    st.header("Ablation Study")
    ablation = load_ablation_table()
    st.dataframe(ablation, use_container_width=True, hide_index=True)
    st.divider()
    st.caption("Embedding: all-MiniLM-L6-v2")
    st.caption(f"LLM: {os.getenv('HF_MODEL_ID', 'N/A')}")
    st.caption("Re-ranker: ms-marco-MiniLM-L-6-v2")
    st.caption("Vector DB: Pinecone Serverless")
