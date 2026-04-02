"""Convert resume/job dicts from Kaggle datasets into rich text for embedding."""


def serialize_candidate(resume: dict) -> str:
    """Serialize a resume record from resumes_dataset.jsonl into text."""
    parts = []

    name = resume.get("Name") or "Unknown"
    category = resume.get("Category") or "Not specified"
    location = resume.get("Location") or "Not specified"

    parts.append(f"Name: {name} | Category: {category} | Location: {location}")

    skills = resume.get("Skills") or ""
    if skills:
        parts.append(f"\nSKILLS:\n{skills}")

    summary = resume.get("Summary") or ""
    if summary:
        parts.append(f"\nSUMMARY:\n{summary}")

    experience = resume.get("Experience") or ""
    if experience:
        parts.append(f"\nEXPERIENCE:\n{experience[:2000]}")

    education = resume.get("Education") or ""
    if education:
        parts.append(f"\nEDUCATION:\n{education}")

    return "\n".join(parts)


def serialize_job(job: dict) -> str:
    """Serialize a job record from job_title_des.csv into text."""
    parts = []

    title = job.get("Job Title") or job.get("job_title") or "Unknown"
    parts.append(f"Job Title: {title}")

    description = job.get("Job Description") or job.get("job_description") or ""
    if description:
        parts.append(f"\nJOB DESCRIPTION:\n{description[:3000]}")

    return "\n".join(parts)
