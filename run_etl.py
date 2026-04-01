"""
ETL Pipeline Runner — Extract structured data from resume PDFs to JSON.

Scans data/ for PDF resumes, runs the full ETL pipeline (extract text,
clean, LLM structured extraction, validate), and saves results to
data/candidates.json.

Run: python run_etl.py
"""

import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from etl import (
    extract_text,
    clean_text,
    ExtractionStatus,
)
from etl.resume_extractor import extract_resume_structured

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "candidates.json"
TARGET_COUNT = 60  # aim for ~60 to comfortably exceed the 50 minimum
PER_CATEGORY = 3   # resumes per category for diversity
RATE_LIMIT_DELAY = 2.5  # seconds between LLM calls (Groq free = 30 RPM)


def collect_pdfs() -> list[Path]:
    """Collect PDF paths, sampling PER_CATEGORY from each subdirectory."""
    pdfs = []
    categories = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    for category_dir in categories:
        category_pdfs = sorted(category_dir.glob("*.pdf"))[:PER_CATEGORY]
        pdfs.extend(category_pdfs)
    logger.info("Collected %d PDFs from %d categories", len(pdfs), len(categories))
    return pdfs


def process_resume(pdf_path: Path, candidate_id: str) -> dict | None:
    """Run full ETL pipeline on a single PDF. Returns dict or None on failure."""
    # Step 1: Extract text
    extraction = extract_text(str(pdf_path))
    if extraction.status != ExtractionStatus.SUCCESS:
        logger.warning("Text extraction failed for %s: %s", pdf_path.name, extraction.status)
        return None

    # Step 2: Clean text
    cleaned = clean_text(extraction.text)

    # Step 3: LLM structured extraction + validation
    result = extract_resume_structured(cleaned.text)
    if not result.is_success or result.resume is None:
        logger.warning(
            "LLM extraction failed for %s: %s — %s",
            pdf_path.name, result.status, result.error_message,
        )
        return None

    # Step 4: Serialize to dict
    resume = result.resume
    data = resume.model_dump(mode="json")
    data["candidate_id"] = candidate_id
    data["resume_category"] = pdf_path.parent.name
    data["source_file"] = pdf_path.name

    # Rename nested keys to match export_db.py / ingest.py expectations
    data["work_experiences"] = data.pop("experience", [])
    data["educations"] = data.pop("education", [])

    if result.warnings:
        logger.debug("Warnings for %s: %s", pdf_path.name, result.warnings)

    return data


def main():
    logger.info("=== ETL Pipeline Start ===")

    pdfs = collect_pdfs()
    if not pdfs:
        logger.error("No PDFs found in %s", DATA_DIR)
        return

    # Load existing results to support resuming after interruption
    candidates = []
    processed_files: set[str] = set()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        processed_files = {c["source_file"] for c in candidates}
        logger.info("Loaded %d existing results from %s", len(candidates), OUTPUT_FILE)

    failures = 0

    for i, pdf_path in enumerate(pdfs, 1):
        candidate_id = pdf_path.stem  # filename without extension
        category = pdf_path.parent.name

        # Skip already-processed resumes
        if pdf_path.name in processed_files:
            logger.info("[%d/%d] SKIP (already done) %s/%s", i, len(pdfs), category, pdf_path.name)
            continue

        logger.info("[%d/%d] Processing %s/%s ...", i, len(pdfs), category, pdf_path.name)

        try:
            data = process_resume(pdf_path, candidate_id)
            if data:
                candidates.append(data)
                logger.info(
                    "  -> OK: %s %s | %d skills, %d exp, %d edu",
                    data.get("first_name", "?"),
                    data.get("last_name", "?"),
                    len(data.get("skills", [])),
                    len(data.get("work_experiences", [])),
                    len(data.get("educations", [])),
                )
                # Save incrementally after each success
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(candidates, f, indent=2, ensure_ascii=False)
            else:
                failures += 1
        except Exception as e:
            logger.error("  -> EXCEPTION: %s", e)
            failures += 1

        # Rate limit between LLM calls
        if i < len(pdfs):
            time.sleep(RATE_LIMIT_DELAY)

    logger.info("=== ETL Pipeline Complete ===")
    logger.info("Total in output: %d | New failures: %d", len(candidates), failures)
    logger.info("Output: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
