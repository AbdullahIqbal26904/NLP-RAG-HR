"""Generate the final assignment PDF report (max ~4 pages)."""
from fpdf import FPDF
import json
from pathlib import Path


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(
                0, 8,
                "Assignment Report - RAG-based Talent Matching System  |  Abdullah Iqbal, Anushe Ali",
                align="C",
            )
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(25, 60, 120)
        self.ln(3)
        self.cell(0, 8, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def sub_title(self, num, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, f"{num} {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.8, text)
        self.ln(1.5)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(4, 4.8, "-")
        self.multi_cell(0, 4.8, text)
        self.ln(0.5)

    def bold_inline(self, label, text):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(30, 30, 30)
        self.cell(self.get_string_width(label) + 1, 4.8, label)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4.8, text)
        self.ln(0.5)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, align="C", fill=True)
        self.ln()

    def table_row(self, cells, widths, fill=False):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        if fill:
            self.set_fill_color(240, 245, 255)
        for cell, w in zip(cells, widths):
            self.cell(w, 5.5, str(cell), border=1, align="C", fill=fill)
        self.ln()

    def draw_architecture_diagram(self):
        """Draw a compact system architecture diagram using PDF primitives."""
        start_y = self.get_y()
        pw = self.w - self.l_margin - self.r_margin
        cx = self.l_margin + pw / 2  # center x

        def box(x, y, w, h, label, fill_r=240, fill_g=245, fill_b=255, bold=False):
            self.set_fill_color(fill_r, fill_g, fill_b)
            self.set_draw_color(25, 60, 120)
            self.rect(x, y, w, h, "DF")
            self.set_xy(x, y + 1)
            self.set_font("Helvetica", "B" if bold else "", 7)
            self.set_text_color(25, 60, 120)
            self.cell(w, h - 2, label, align="C")

        def arrow_down(x, y1, y2):
            self.set_draw_color(80, 80, 80)
            self.line(x, y1, x, y2)
            self.line(x, y2, x - 2, y2 - 3)
            self.line(x, y2, x + 2, y2 - 3)

        def arrow_right(x1, y, x2):
            self.set_draw_color(80, 80, 80)
            self.line(x1, y, x2, y)
            self.line(x2, y, x2 - 3, y - 2)
            self.line(x2, y, x2 - 3, y + 2)

        bw = 42  # box width
        bh = 10  # box height
        gap = 5

        # Row 1: User Query
        y = start_y + 2
        box(cx - bw / 2, y, bw, bh, "User Query", 255, 230, 200, bold=True)
        arrow_down(cx, y + bh, y + bh + gap)

        # Row 2: Streamlit UI
        y += bh + gap
        box(cx - bw / 2, y, bw, bh, "Streamlit UI (app.py)", 220, 235, 255, bold=True)
        arrow_down(cx, y + bh, y + bh + gap)

        # Row 3: BM25 + Semantic (side by side)
        y += bh + gap
        bw2 = 38
        box(cx - bw2 - 4, y, bw2, bh, "BM25 Search (Local)")
        box(cx + 4, y, bw2, bh, "Semantic (Pinecone)")
        # arrows converging down
        arrow_down(cx - bw2 / 2 - 4, y + bh, y + bh + gap + 2)
        arrow_down(cx + bw2 / 2 + 4, y + bh, y + bh + gap + 2)

        # Row 4: RRF Fusion
        y += bh + gap + 2
        box(cx - bw / 2, y, bw, bh, "RRF Fusion (k=60)", 230, 255, 230)
        arrow_down(cx, y + bh, y + bh + gap)

        # Row 5: CrossEncoder
        y += bh + gap
        box(cx - bw / 2, y, bw, bh, "CrossEncoder Rerank", 230, 255, 230)
        arrow_down(cx, y + bh, y + bh + gap)

        # Row 6: LLM Generation
        y += bh + gap
        box(cx - bw / 2, y, bw, bh, "LLM Generation", 255, 240, 220, bold=True)

        # Side boxes: Pinecone DB (aligned with BM25/Semantic row) and HF/Groq API
        side_x_left = self.l_margin + 5
        sem_row_y = start_y + 2 + (bh + gap) * 2  # same Y as BM25/Semantic row
        sem_right_edge = cx + 4 + bw2               # right edge of Semantic box
        pine_x = sem_right_edge + 8                  # Pinecone box starts after a gap
        pine_w = bw2 - 4
        box(pine_x, sem_row_y, pine_w, bh, "Pinecone Vector DB", 255, 220, 220)
        box(side_x_left, start_y + 2 + (bh + gap) * 5, pine_w, bh, "HF / Groq API", 255, 220, 220)

        # Horizontal arrow from Pinecone DB to Semantic box (left-pointing)
        self.set_draw_color(80, 80, 80)
        arr_y = sem_row_y + bh / 2
        self.line(pine_x, arr_y, sem_right_edge, arr_y)
        # arrowhead pointing left (toward Semantic box)
        self.line(sem_right_edge, arr_y, sem_right_edge + 3, arr_y - 2)
        self.line(sem_right_edge, arr_y, sem_right_edge + 3, arr_y + 2)

        self.set_y(y + bh + 4)

    def draw_eval_diagram(self):
        """Draw compact evaluation pipeline diagram."""
        start_y = self.get_y()
        pw = self.w - self.l_margin - self.r_margin

        def box(x, y, w, h, label, fill_r=240, fill_g=245, fill_b=255):
            self.set_fill_color(fill_r, fill_g, fill_b)
            self.set_draw_color(25, 60, 120)
            self.rect(x, y, w, h, "DF")
            self.set_xy(x, y + 1)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(25, 60, 120)
            self.cell(w, h - 2, label, align="C")

        def arrow_down(x, y1, y2):
            self.set_draw_color(80, 80, 80)
            self.line(x, y1, x, y2)
            self.line(x, y2, x - 2, y2 - 3)
            self.line(x, y2, x + 2, y2 - 3)

        bw = 44
        bh = 9
        gap = 4
        half = pw / 4

        # Faithfulness side
        fx = self.l_margin + half - bw / 2
        y = start_y
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(25, 60, 120)
        self.set_xy(fx, y)
        self.cell(bw, 5, "Faithfulness", align="C")
        y += 6
        box(fx, y, bw, bh, "1. Extract Claims from Answer")
        arrow_down(fx + bw / 2, y + bh, y + bh + gap)
        y += bh + gap
        box(fx, y, bw, bh, "2. Verify vs Retrieved Context")
        arrow_down(fx + bw / 2, y + bh, y + bh + gap)
        y += bh + gap
        box(fx, y, bw, bh, "Score = % Supported Claims", 230, 255, 230)

        # Relevancy side
        rx = self.l_margin + 3 * half - bw / 2
        y = start_y
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(25, 60, 120)
        self.set_xy(rx, y)
        self.cell(bw, 5, "Relevancy", align="C")
        y += 6
        box(rx, y, bw, bh, "1. Generate 3 Alt. Questions")
        arrow_down(rx + bw / 2, y + bh, y + bh + gap)
        y += bh + gap
        box(rx, y, bw, bh, "2. Cosine Sim vs Original")
        arrow_down(rx + bw / 2, y + bh, y + bh + gap)
        y += bh + gap
        box(rx, y, bw, bh, "Score = Mean Similarity", 230, 255, 230)

        self.set_y(y + bh + 3)


def load_experiment_data():
    path = Path("reports/experiment_results.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pw = pdf.w - pdf.l_margin - pdf.r_margin

    exp = load_experiment_data()

    # ── Title Block (no separate title page — saves space) ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(0, 10, "Assignment Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "RAG-based Talent Matching System", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, "Abdullah Iqbal  |  Anushe Ali", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Course: NLP with Deep Learning  |  Due: 5th April 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(25, 60, 120)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Production URL: http://3.144.109.78 (Amazon EC2)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_draw_color(25, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)

    # ── Section 1: System Architecture ──
    pdf.section_title("1", "System Architecture")
    pdf.body_text(
        "Our system is a full-stack RAG application deployed on Amazon EC2. It combines hybrid "
        "retrieval (BM25 + semantic search), RRF fusion, CrossEncoder re-ranking, and LLM generation "
        "with automated LLM-as-a-Judge evaluation. The architecture diagram below shows the complete pipeline:"
    )
    pdf.draw_architecture_diagram()
    pdf.ln(1)

    # ── Section 2: Data & Platform ──
    pdf.section_title("2", "Data and Platform Details")
    cols = ["Aspect", "Detail"]
    ws = [pw * 0.30, pw * 0.70]
    pdf.table_header(cols, ws)
    rows = [
        ("Domain", "HR Talent Matching (Resumes + Job Descriptions)"),
        ("Resume Dataset", "3,500 resumes across 36 categories (Kaggle, JSONL)"),
        ("Job Dataset", "2,277 job postings across 15 titles (Kaggle, CSV)"),
        ("Vector DB", "Pinecone Serverless (4 indexes: EN + Urdu)"),
        ("Embedding (EN)", "all-MiniLM-L6-v2 (384-dim)"),
        ("Embedding (Urdu)", "paraphrase-multilingual-MiniLM-L12-v2 (384-dim)"),
        ("Re-ranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ("LLM", "Llama-3.1-8B-Instruct (HF primary, Groq fallback)"),
        ("Deployment", "Amazon EC2 + Docker Compose"),
        ("UI", "Streamlit (English + Urdu RTL support)"),
    ]
    for i, row in enumerate(rows):
        pdf.table_row(row, ws, fill=(i % 2 == 0))
    pdf.ln(2)

    pdf.body_text(
        "Live features: Users can add resumes and job postings from the UI. Data is saved, embedded, "
        "and indexed in Pinecone in real-time without restart. The Urdu track uses multilingual "
        "embeddings with native Nastaliq script support."
    )

    # ── Section 3: Retrieval Strategies & Chunking ──
    pdf.section_title("3", "Retrieval Strategies and Chunking")
    pdf.body_text(
        "We implemented three retrieval strategies, each building on the previous:"
    )
    pdf.bullet("Semantic Only: Cosine similarity search against Pinecone using MiniLM-L6-v2 embeddings.")
    pdf.bullet("Hybrid + RRF: BM25 (lexical, rank-bm25) + Semantic (dense) fused via Reciprocal Rank Fusion (k=60).")
    pdf.bullet("Hybrid + RRF + CrossEncoder: Above + cross-encoder pairwise re-ranking for final top-5.")
    pdf.ln(1)

    pdf.sub_title("3.1", "Chunking Comparison")
    pdf.bullet("Fixed Chunking: 700-char windows with 120-char overlap (1,045 chunks).")
    pdf.bullet("Recursive Chunking: Paragraph/sentence-aware splitting, max 700 chars (984 chunks).")
    pdf.bullet("Full-Document (selected): Each resume/job as a single embedding, preserving full context.")
    pdf.ln(1)

    # ── Section 4: Evaluation Protocol ──
    pdf.section_title("4", "Evaluation: LLM-as-a-Judge")
    pdf.body_text(
        "We evaluate on a fixed test set of 15 queries (10 recruiter + 5 candidate) using two automated metrics:"
    )
    pdf.draw_eval_diagram()
    pdf.ln(1)

    # ── Section 5: Results ──
    pdf.section_title("5", "Performance Results")

    pdf.sub_title("5.1", "Fixed Test Set (15 Queries)")
    if exp:
        s = exp["baseline_fixed_set"]["summary"]
        pdf.body_text(
            f"Avg Faithfulness: {s['avg_faithfulness']:.1%}  |  "
            f"Avg Relevancy: {s['avg_relevancy']:.1%}  |  "
            f"Avg Retrieval: {s['avg_retrieval_s']:.3f}s  |  "
            f"Avg End-to-End: {s['avg_total_s']:.3f}s"
        )
    else:
        pdf.body_text("Run: python -m pipeline.experiments to generate metrics.")

    pdf.sub_title("5.2", "Retrieval Ablation")
    cols_a = ["Strategy", "Faithfulness", "Relevancy", "Retrieval (s)", "Total (s)"]
    ws_a = [pw * 0.32, pw * 0.17, pw * 0.17, pw * 0.17, pw * 0.17]
    pdf.table_header(cols_a, ws_a)
    if exp:
        strat_labels = {
            "semantic_only": "Semantic Only",
            "hybrid_rrf": "Hybrid + RRF",
            "hybrid_rrf_ce": "Hybrid + RRF + CE",
        }
        ablation_rows = []
        for key in ["semantic_only", "hybrid_rrf", "hybrid_rrf_ce"]:
            sm = exp["ablation"]["retrieval"]["summary_by_strategy"][key]
            ablation_rows.append((
                strat_labels[key],
                f"{sm['avg_faithfulness']:.1%}",
                f"{sm['avg_relevancy']:.1%}",
                f"{sm['avg_retrieval_s']:.3f}",
                f"{sm['avg_total_s']:.3f}",
            ))
        for i, row in enumerate(ablation_rows):
            pdf.table_row(row, ws_a, fill=(i % 2 == 0))
    pdf.ln(1)

    pdf.sub_title("5.3", "Chunking Ablation")
    cols_c = ["Chunking", "Chunks", "Faithfulness", "Relevancy", "Total (s)"]
    ws_c = [pw * 0.28, pw * 0.14, pw * 0.19, pw * 0.19, pw * 0.20]
    pdf.table_header(cols_c, ws_c)
    if exp:
        chunk_labels = {
            "fixed": "Fixed (700c, 120 overlap)",
            "recursive": "Recursive (700c)",
        }
        chunk_rows = []
        for method in ["fixed", "recursive"]:
            cm = exp["ablation"]["chunking"]["summary_by_chunking"][method]
            chunk_rows.append((
                chunk_labels[method],
                str(cm.get("chunk_count", "N/A")),
                f"{cm['avg_faithfulness']:.1%}",
                f"{cm['avg_relevancy']:.1%}",
                f"{cm['avg_total_s']:.3f}",
            ))
        for i, row in enumerate(chunk_rows):
            pdf.table_row(row, ws_c, fill=(i % 2 == 0))
    pdf.ln(1)

    if exp:
        ra = exp["ablation"]["retrieval"]["summary_by_strategy"]
        best_faith_key = max(ra, key=lambda k: ra[k]["avg_faithfulness"])
        best_rel_key = max(ra, key=lambda k: ra[k]["avg_relevancy"])
        best_faith_label = {"semantic_only": "Semantic Only", "hybrid_rrf": "Hybrid+RRF", "hybrid_rrf_ce": "Hybrid+RRF+CE"}
        pdf.body_text(
            f"Selected pipeline: Hybrid + RRF + CE with full-document embeddings. "
            f"{best_faith_label[best_faith_key]} gave best Faithfulness ({ra[best_faith_key]['avg_faithfulness']:.1%}), "
            f"{best_faith_label[best_rel_key]} added highest Relevancy ({ra[best_rel_key]['avg_relevancy']:.1%}). "
            f"We use full-document embedding as resumes fit within the model's token limit."
        )
    else:
        pdf.body_text(
            "Selected pipeline: Hybrid + RRF + CE with full-document embeddings."
        )

    # ── Section 6: Claim Verification Examples ──
    pdf.section_title("6", "Claim Verification Examples")
    if exp:
        example_rows = exp["baseline_fixed_set"].get("example_verifications", [])[:3]
        for ex in example_rows:
            query = ex.get("query", "")
            faith = f"{ex.get('faithfulness', 0) * 100:.1f}%"
            rel = f"{ex.get('relevancy', 0) * 100:.1f}%"
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 5, f'Query: "{query}"  |  Faith: {faith}  |  Rel: {rel}',
                     new_x="LMARGIN", new_y="NEXT")
            for vc in ex.get("verified_claims", []):
                status = "SUPPORTED" if vc.get("supported") else "NOT SUPPORTED"
                claim_text = vc.get("claim", "")
                reason = vc.get("reason", "")
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(30, 30, 30)
                pdf.set_x(pdf.l_margin + 4)
                pdf.multi_cell(0, 4.5, f"- [{status}] {claim_text}")
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(80, 80, 80)
                pdf.set_x(pdf.l_margin + 8)
                pdf.multi_cell(0, 4.5, f"Reason: {reason}")
                pdf.set_text_color(30, 30, 30)
            pdf.ln(1)
    else:
        pdf.body_text("Run: python -m pipeline.experiments to generate claim verification examples.")

    # ── Section 7: Reproducibility & Deployment ──
    pdf.section_title("7", "Reproducibility and Deployment")
    pdf.body_text(
        "The application is deployed on Amazon EC2 and accessible at http://3.144.109.78. "
        "To reproduce locally:"
    )
    pdf.bullet("Configure .env with Pinecone, HuggingFace, and Groq API keys.")
    pdf.bullet("docker compose build && docker compose up -d")
    pdf.bullet("python -m pipeline.ingest (embed 3,500 resumes + 2,277 jobs to Pinecone)")
    pdf.bullet("python -m pipeline.experiments (evaluation + ablation)")
    pdf.bullet("Access at http://localhost:8501 (local) or http://3.144.109.78 (production)")
    pdf.ln(1)

    # ── Section 8: Urdu Bonus Track ──
    pdf.section_title("8", "Bonus: Urdu Low-Resource Language Track")
    pdf.body_text(
        "We built a parallel Urdu RAG pipeline with multilingual embeddings and a UI that accepts "
        "native Nastaliq script queries. Key implementation details:"
    )

    pdf.sub_title("8.1", "Data & Embeddings")
    pdf.bullet("Translated the entire English corpus to Urdu: 3,501 resumes + 63,763 job postings.")
    pdf.bullet("Used paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages) for Urdu embeddings.")
    pdf.bullet("Separate Pinecone indexes: urdu-candidates and urdu-jobs.")
    pdf.bullet("BM25 tokenization uses whitespace splitting (no lowercasing since Urdu has no case).")

    pdf.sub_title("8.2", "Challenges & Solutions")
    pdf.bullet("Script handling: Urdu Nastaliq is RTL with complex ligatures. Added RTL CSS + Google Fonts "
               "(Noto Nastaliq Urdu) in the Streamlit UI for correct rendering.")
    pdf.bullet("Translation artifacts: Technical terms like 'Python' become phonetic Urdu. Mitigated by "
               "retaining both English and Urdu fields, enabling hybrid cross-lingual matching.")
    pdf.bullet("Embedding model: English-only MiniLM-L6-v2 produces meaningless Urdu embeddings. Switched "
               "to multilingual-MiniLM-L12-v2 which supports Urdu natively.")
    pdf.bullet("LLM Judge bias: English-centric LLMs give poor Urdu evaluations. Used Qwen2.5-72B-Instruct "
               "as the multilingual judge with Urdu-language evaluation prompts.")

    pdf.sub_title("8.3", "Deliverables")
    pdf.bullet("Working live demo accepting native Urdu Nastaliq queries with Urdu answers.")
    pdf.bullet("Full hybrid pipeline: BM25 + Semantic + RRF + CrossEncoder for Urdu.")
    pdf.bullet("Multilingual evaluation with Qwen2.5-72B and heuristic Unicode fallback.")

    # ── Save ──
    out_path = Path("reports/Assignment_Report.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    print(f"Report saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB, {pdf.page_no()} pages)")


if __name__ == "__main__":
    build_report()
