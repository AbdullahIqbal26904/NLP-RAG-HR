# Assignment 3 Report: RAG-based Question-Answering System

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
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Embeddings are precomputed and upserted to Pinecone indexes for candidates and jobs.

### 3.2 Retrieval Pipeline
- Implemented retrieval variants:
  - Semantic only
  - Hybrid (BM25 + Semantic) with RRF fusion
  - Hybrid (BM25 + Semantic) + RRF + CrossEncoder re-ranking
- CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2

### 3.3 Generation and Evaluation
- LLM model configured: meta-llama/Llama-3.1-8B-Instruct
- Generation uses structured recruiter/candidate prompts with retrieved context.
- LLM-as-a-Judge evaluation implemented:
  - Faithfulness: claim extraction + claim verification against retrieved context
  - Relevancy: generate 3 alternate questions + cosine similarity with original query

### 3.4 Chunking Strategy Comparison
- Fixed chunking: constant character windows with overlap.
- Recursive chunking: paragraph/sentence-aware splitting with max chunk size.

## 4. Performance Metrics
### 4.1 Fixed Test Set Results
- Number of evaluated queries: 15
- Average Faithfulness: 36.7%
- Average Relevancy: 27.1%
- Average Retrieval Time: 1.168 s
- Average Generation Time: 1.393 s
- Average Evaluation Time: 5.113 s
- Average End-to-End Time: 7.673 s

### 4.2 Retrieval Ablation (Semantic vs Hybrid)
| Retrieval Strategy | Faithfulness | Relevancy | Avg Retrieval (s) | Avg Total (s) |
|---|---:|---:|---:|---:|
| Semantic only | 33.3% | 4.5% | 0.316 | 5.096 |
| Hybrid + RRF | 50.0% | 4.5% | 0.260 | 4.854 |
| Hybrid + RRF + CrossEncoder | 16.7% | 35.9% | 0.977 | 7.613 |

### 4.3 Chunking Ablation
| Chunking Strategy | Chunk Count | Faithfulness | Relevancy | Avg Retrieval (s) | Avg Total (s) |
|---|---:|---:|---:|---:|---:|
| Fixed + Hybrid/RRF/CE | 1045 | 75.0% | 2.9% | 0.675 | 8.162 |
| Recursive + Hybrid/RRF/CE | 984 | 25.0% | 22.0% | 0.691 | 8.903 |

## 5. Best Model and Pipeline Selection
- Selected retrieval strategy: hybrid_rrf
- Selection criterion: maximum combined score (Faithfulness + Relevancy) from ablation runs.
- Selected strategy metrics:
  - Faithfulness: 50.0%
  - Relevancy: 4.5%
  - Avg total latency: 4.854 s

## 6. Claim Verification Examples (at least 3)
### Example 1
- Query: Looking for a Python backend developer with FastAPI and PostgreSQL, 3+ years
- Mode: candidate
- Faithfulness: 0.0%
- Relevancy: 36.8%
- Claims Verification (sample):
  - Not supported: He has 4 years of experience as a Full Stack Python Developer.
  - Reason: The context does not mention Syed Muhammad Ali as a Full Stack Python Developer, but rather Umair Qasim is mentioned with that role.
  - Not supported: He has listed Python as his primary skill.
  - Reason: The claim is not supported because Syed Muhammad Ali has listed multiple skills as primary, including Python, but it does not explicitly state that Python is his primary skill.
- Answer excerpt: Based on the provided candidate profiles, the top 3 matching candidates for the Python backend developer position with FastAPI and PostgreSQL are:  1. **Candidate 2: Umair Qasim** Umair Qasim matches the requirement because: - He has 4 years of experience as a Full Stack Python Developer. - He has listed Python as his primary skill. - He has experi

### Example 2
- Query: Need a React frontend developer with TypeScript experience
- Mode: candidate
- Faithfulness: 100.0%
- Relevancy: 35.6%
- Claims Verification (sample):
  - Supported: Shakeel has 4 years of experience as a React Native developer
  - Reason: The claim is supported by the 'Experience' section of Shakeel's profile.
  - Supported: Shakeel has experience with React.js and TypeScript
  - Reason: Shakeel's skills include React.js and TypeScript, and his work experience mentions fixing TypeScript type annotations.
- Answer excerpt: Based on the provided candidate profiles, the top 3 matching candidates for the React frontend developer with TypeScript experience are:  1. **Candidate 1: Shakeel Mazhar** Shakeel has 4 years of experience as a React Native developer, but his profile also mentions that he has experience with React.js and TypeScript. He has worked on optimizing cod

### Example 3
- Query: Seeking a DevOps engineer with Docker and Kubernetes skills
- Mode: candidate
- Faithfulness: 0.0%
- Relevancy: 63.4%
- Claims Verification (sample):
  - Not supported: M. Zakaria Nazir matches the requirement due to his extensive experience with Docker, Kubernetes, and AWS Cloud.
  - Reason: The context does not mention any experience with Kubernetes, AWS Cloud, or Docker.
  - Not supported: His primary skills include Docker, Kubernetes, and AWS Cloud, which aligns with the job requirement.
  - Reason: The claim is not supported because the provided context does not mention Docker, Kubernetes, and AWS Cloud as the primary skills of M. Zakaria Nazir.
- Answer excerpt: Based on the provided candidate profiles, the top 3 matching candidates for the DevOps engineer position with Docker and Kubernetes skills are:  1. **Candidate 1: M. Zakaria Nazir** M. Zakaria Nazir matches the requirement due to his extensive experience with Docker, Kubernetes, and AWS Cloud. His primary skills include Docker, Kubernetes, and AWS 


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
Generated on: 2026-03-30T13:53:54.903436Z
