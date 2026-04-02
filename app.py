"""Streamlit UI - RAG Talent Matching. Deployed on HuggingFace Spaces."""
import json
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


def _next_resume_id() -> str:
    """Generate next USER_XXXX ID by scanning existing user-submitted entries."""
    path = Path("data/resumes_dataset.jsonl")
    max_num = 0
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rid = json.loads(line).get("ResumeID", "")
                if rid.startswith("USER_"):
                    try:
                        max_num = max(max_num, int(rid.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
    return f"USER_{max_num + 1:04d}"


def _next_job_id() -> str:
    """Generate next USERJOB_XXXX ID."""
    retriever = load_retriever()
    max_num = 0
    for jid in retriever._job_ids:
        if jid.startswith("USERJOB_"):
            try:
                max_num = max(max_num, int(jid.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return f"USERJOB_{max_num + 1:04d}"


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

# Main tabs
tab_search, tab_add, tab_add_job = st.tabs(["Search", "Add Resume", "Add Job"])

# ── Tab 1: Search ──
with tab_search:
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
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CrossEncoder", f"{r.get('cross_encoder_score', 0):.3f}")
                c2.metric("Semantic", f"{r.get('semantic_score', 0):.3f}")
                c3.metric("BM25", f"{r.get('bm25_score', 0):.2f}")
                c4.metric("RRF", f"{r.get('rrf_score', 0):.4f}")

                if orig and is_recruiter:
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
                    st.markdown(f"**Job Title:** {orig.get('Job Title', 'N/A')}")
                    description = orig.get("Job Description") or ""
                    if description:
                        st.markdown("**Job Description:**")
                        st.text(description)

                else:
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

# ── Tab 2: Add Resume ──
with tab_add:
    st.subheader("Submit Your Resume")
    st.caption("Fill in your details below. Your resume will be saved and immediately searchable.")

    with st.form("add_resume_form", clear_on_submit=True):
        ar_name = st.text_input("Full Name *", placeholder="e.g. John Doe")
        ar_email = st.text_input("Email", placeholder="e.g. john@example.com")
        ar_phone = st.text_input("Phone", placeholder="e.g. +1 555 123 4567")
        ar_location = st.text_input("Location", placeholder="e.g. New York, USA")
        ar_category = st.selectbox(
            "Job Category *",
            [
                "AI Engineer", "Backend Developer", "Blockchain Developer",
                "Business Analyst", "Cloud Engineer", "Cybersecurity Analyst",
                "Data Science", "Database Administrator", "DevOps",
                "Digital Media", "DotNet Developer", "ETL Developer",
                "Engineering Manager", "Frontend Developer", "Full Stack Developer",
                "Java Developer", "Machine Learning Engineer", "Mobile Developer",
                "Network Security Engineer", "Principal Engineer", "Product Manager",
                "Python Developer", "QA Engineer", "React Developer",
                "SAP Developer", "SQL Developer", "Site Reliability Engineer",
                "Software Developer", "System Administrator", "Technical Lead",
                "Technical Writer", "Testing", "UI/UX Designer", "Web Designing",
                "Other",
            ],
        )
        ar_skills = st.text_area(
            "Skills * (comma-separated)",
            placeholder="e.g. Python, Django, PostgreSQL, Docker, AWS",
            height=68,
        )
        ar_education = st.text_input(
            "Education",
            placeholder="e.g. BS Computer Science, MIT 2020",
        )
        ar_summary = st.text_area(
            "Professional Summary",
            placeholder="Brief overview of your background and strengths...",
            height=100,
        )
        ar_experience = st.text_area(
            "Work Experience",
            placeholder="Describe your roles, companies, and key achievements...",
            height=150,
        )

        submitted = st.form_submit_button("Submit Resume", type="primary")

    if submitted:
        if not ar_name.strip():
            st.error("Name is required.")
        elif not ar_skills.strip():
            st.error("Skills are required.")
        else:
            resume_id = _next_resume_id()
            full_text = " ".join(
                filter(None, [ar_summary.strip(), ar_experience.strip(), ar_skills.strip()])
            )
            resume = {
                "ResumeID": resume_id,
                "Category": ar_category,
                "Name": ar_name.strip(),
                "Email": ar_email.strip(),
                "Phone": ar_phone.strip(),
                "Location": ar_location.strip(),
                "Summary": ar_summary.strip(),
                "Skills": ar_skills.strip(),
                "Experience": ar_experience.strip(),
                "Education": ar_education.strip(),
                "Text": full_text,
                "Source": "UserSubmitted",
            }

            with st.spinner("Saving and indexing your resume..."):
                retriever = load_retriever()
                retriever.add_resume(resume)

            st.success(f"Resume submitted successfully! (ID: {resume_id})")
            st.markdown("Your resume is now searchable by recruiters.")

# ── Tab 3: Add Job ──
with tab_add_job:
    st.subheader("Post a Job Description")
    st.caption("Fill in the job details below. The posting will be saved and immediately searchable by candidates.")

    with st.form("add_job_form", clear_on_submit=True):
        aj_title = st.text_input("Job Title *", placeholder="e.g. Senior Python Developer")
        aj_description = st.text_area(
            "Job Description *",
            placeholder="Describe the role, responsibilities, requirements, salary, benefits...",
            height=250,
        )

        job_submitted = st.form_submit_button("Post Job", type="primary")

    if job_submitted:
        if not aj_title.strip():
            st.error("Job title is required.")
        elif not aj_description.strip():
            st.error("Job description is required.")
        else:
            job_id = _next_job_id()
            job = {
                "job_id": job_id,
                "Job Title": aj_title.strip(),
                "Job Description": aj_description.strip(),
            }

            with st.spinner("Saving and indexing job posting..."):
                retriever = load_retriever()
                retriever.add_job(job)

            st.success(f"Job posted successfully! (ID: {job_id})")
            st.markdown("This job is now searchable by candidates.")

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
