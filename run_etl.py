"""
Data Verification — Check that the Kaggle datasets are present and valid.

Run: python run_etl.py
"""
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESUMES_PATH = Path("data/resumes_dataset.jsonl")
JOBS_PATH = Path("data/job_title_des.csv")


def verify_resumes():
    if not RESUMES_PATH.exists():
        logger.error("Missing %s", RESUMES_PATH)
        return False

    count = 0
    categories = set()
    with open(RESUMES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            count += 1
            categories.add(record.get("Category", ""))

    logger.info("Resumes: %d records, %d categories", count, len(categories))
    logger.info("Categories: %s", sorted(categories))
    return count > 0


def verify_jobs():
    if not JOBS_PATH.exists():
        logger.error("Missing %s", JOBS_PATH)
        return False

    count = 0
    titles = set()
    with open(JOBS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            count += 1
            titles.add(row.get("Job Title", ""))

    logger.info("Jobs: %d records, %d unique titles", count, len(titles))
    logger.info("Titles: %s", sorted(titles))
    return count > 0


def main():
    logger.info("=== Data Verification Start ===")
    ok_resumes = verify_resumes()
    ok_jobs = verify_jobs()

    if ok_resumes and ok_jobs:
        logger.info("=== All datasets verified. Ready for ingestion. ===")
        logger.info("Next step: python -m pipeline.ingest")
    else:
        logger.error("=== Some datasets are missing or empty. ===")


if __name__ == "__main__":
    main()
