"""
Export candidate profiles and job postings from PostgreSQL to JSON.
Run: python -m pipeline.export_db
"""
import json
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def rows_to_dicts(cursor) -> list[dict]:
    """Convert cursor results to list of plain dicts."""
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def export_candidates(conn) -> list[dict]:
    candidates = []
    with conn.cursor() as cur:
        # 1. Fetch all candidate profiles
        cur.execute("SELECT * FROM candidate_profiles ORDER BY candidate_id")
        profiles = rows_to_dicts(cur)

        for profile in profiles:
            cid = profile["candidate_id"]

            # 2. Skills
            cur.execute("SELECT * FROM candidate_skills WHERE candidate_id = %s", (cid,))
            profile["skills"] = rows_to_dicts(cur)

            # 3. Work experience
            cur.execute(
                "SELECT * FROM candidate_work_experience WHERE candidate_id = %s ORDER BY from_date DESC NULLS LAST",
                (cid,),
            )
            profile["work_experiences"] = rows_to_dicts(cur)

            # 4. Education
            cur.execute(
                "SELECT * FROM candidate_education WHERE candidate_id = %s ORDER BY from_date DESC NULLS LAST",
                (cid,),
            )
            profile["educations"] = rows_to_dicts(cur)

            # 5. Certifications
            cur.execute(
                "SELECT * FROM candidate_certifications_examinations WHERE candidate_id = %s",
                (cid,),
            )
            profile["certifications"] = rows_to_dicts(cur)

            # 6. Projects
            cur.execute("SELECT * FROM candidate_projects WHERE candidate_id = %s", (cid,))
            profile["projects"] = rows_to_dicts(cur)

            # 7. Languages
            cur.execute("SELECT * FROM candidate_languages WHERE candidate_id = %s", (cid,))
            profile["languages"] = rows_to_dicts(cur)

            # 8. Achievements
            cur.execute("SELECT * FROM candidate_achievements WHERE candidate_id = %s", (cid,))
            profile["achievements"] = rows_to_dicts(cur)

            # 9. Job preferences (one-to-one)
            cur.execute("SELECT * FROM candidate_job_preferences WHERE candidate_id = %s", (cid,))
            prefs = rows_to_dicts(cur)
            profile["job_preferences"] = prefs[0] if prefs else None

            candidates.append(profile)

    return candidates


def export_jobs(conn) -> list[dict]:
    jobs = []
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM job_postings ORDER BY job_id")
        postings = rows_to_dicts(cur)

        for posting in postings:
            jid = posting["job_id"]

            # Skills
            cur.execute("SELECT * FROM job_skill_requirements WHERE job_id = %s", (jid,))
            posting["skills"] = rows_to_dicts(cur)

            # Sections (sorted by order)
            cur.execute(
                "SELECT * FROM job_posting_sections WHERE job_id = %s ORDER BY section_order ASC",
                (jid,),
            )
            posting["sections"] = rows_to_dicts(cur)

            jobs.append(posting)

    return jobs


def serialize_dates(obj):
    """JSON serializer for dates and decimals."""
    import datetime
    import decimal

    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set in .env")

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(database_url)

    try:
        print("Exporting candidates...")
        candidates = export_candidates(conn)

        print("Exporting jobs...")
        jobs = export_jobs(conn)

        os.makedirs("data", exist_ok=True)

        with open("data/candidates.json", "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2, default=serialize_dates)

        with open("data/jobs.json", "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2, default=serialize_dates)

        print(f"[OK] Exported {len(candidates)} candidates -> data/candidates.json")
        print(f"[OK] Exported {len(jobs)} jobs -> data/jobs.json")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
