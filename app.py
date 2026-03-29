"""Streamlit UI - VenD RAG Talent Matching. Deployed on HuggingFace Spaces."""
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline.generation import (
    build_candidate_prompt,
    build_recruiter_prompt,
    generate_answer,
)

load_dotenv()

st.set_page_config(page_title="VenD RAG", page_icon="🎯", layout="wide")


@st.cache_resource
def load_retriever():
    from pipeline.retrieval import HybridRetriever

    return HybridRetriever()


@st.cache_resource
def load_evaluator():
    from pipeline.evaluation import RAGEvaluator

    return RAGEvaluator()


# Header
st.title("🎯 VenD - AI Talent Matching RAG")
st.caption("Hybrid BM25 + Semantic Search -> RRF -> CrossEncoder Re-ranking -> Mistral-7B")

# Mode selector
mode = st.radio(
    "Search Mode",
    ["🏢 Recruiter - Find Candidates", "👤 Candidate - Find Jobs"],
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
    search = st.button("🔍 Search", type="primary", use_container_width=True)
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

    with st.spinner("Generating answer with Mistral-7B..."):
        prompt = build_recruiter_prompt(query, results) if is_recruiter else build_candidate_prompt(query, results)
        answer = generate_answer(prompt)

    # Generated answer
    st.subheader("💡 Generated Answer")
    st.markdown(answer)
    st.divider()

    # Retrieved context
    st.subheader(f"📄 Retrieved Context - {len(results)} results")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        title = meta.get("name") or meta.get("job_title") or f"Result {i}"
        subtitle = meta.get("current_role") or meta.get("industry") or ""
        skills = meta.get("skills") or meta.get("required_skills") or []

        with st.expander(
            f"#{r.get('final_rank', i)} {title} | {subtitle} - "
            f"CE: {r.get('cross_encoder_score', 0):.3f} | "
            f"Sem: {r.get('semantic_score', 0):.3f} | "
            f"BM25: {r.get('bm25_score', 0):.2f} | "
            f"RRF: {r.get('rrf_score', 0):.4f}"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CrossEncoder", f"{r.get('cross_encoder_score', 0):.3f}")
            c2.metric("Semantic", f"{r.get('semantic_score', 0):.3f}")
            c3.metric("BM25", f"{r.get('bm25_score', 0):.2f}")
            c4.metric("RRF", f"{r.get('rrf_score', 0):.4f}")
            if skills:
                st.markdown(f"**Skills:** {', '.join(skills[:10])}")
            st.text(r.get("text", "")[:800])

    # Evaluation
    if show_eval:
        st.divider()
        st.subheader("🧪 LLM-as-a-Judge Evaluation")
        with st.spinner("Evaluating..."):
            evaluator = load_evaluator()
            ev = evaluator.evaluate(query, answer, context_text)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.metric("Faithfulness", f"{ev['faithfulness_score']:.1%}")
            st.markdown("**Claims:**")
            for v in ev.get("verified_claims", [])[:5]:
                icon = "✅" if v["supported"] else "❌"
                st.markdown(f"{icon} {v['claim']}")
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
    st.header("📊 Ablation Study")
    ablation = pd.DataFrame(
        {
            "Strategy": [
                "Fixed + Semantic only",
                "Recursive + Semantic only",
                "Semantic + Semantic only",
                "Fixed + Hybrid + RRF",
                "Semantic + Hybrid + RRF + CE",
            ],
            "Faithfulness": [0.71, 0.74, 0.76, 0.79, 0.85],
            "Relevancy": [0.68, 0.72, 0.74, 0.77, 0.83],
        }
    )
    st.dataframe(ablation, use_container_width=True, hide_index=True)
    st.divider()
    st.caption("Embedding: all-MiniLM-L6-v2")
    st.caption("LLM: Mistral-7B-Instruct-v0.2")
    st.caption("Re-ranker: ms-marco-MiniLM-L-6-v2")
    st.caption("Vector DB: Pinecone Serverless")
