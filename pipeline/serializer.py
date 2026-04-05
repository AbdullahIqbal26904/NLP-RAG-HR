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


def serialize_candidate_urdu(record: dict) -> str:
    """Serialize an Urdu resume record into text for embedding."""
    parts = []

    name = record.get("name") or "نامعلوم"
    category = record.get("category_urdu") or record.get("category") or ""
    parts.append(f"نام: {name} | زمرہ: {category}")

    skills = record.get("skills_urdu") or record.get("skills") or ""
    if skills:
        parts.append(f"\nمہارتیں:\n{skills}")

    summary = record.get("summary_urdu") or ""
    if summary:
        parts.append(f"\nخلاصہ:\n{summary}")

    experience = record.get("experience_urdu") or ""
    if experience:
        parts.append(f"\nتجربہ:\n{experience[:2000]}")

    education = record.get("education_urdu") or ""
    if education:
        parts.append(f"\nتعلیم:\n{education}")

    return "\n".join(parts)


def serialize_job_urdu(job: dict) -> str:
    """Serialize an Urdu job record into text for embedding."""
    parts = []

    title = job.get("title_urdu") or job.get("Job Title") or "نامعلوم"
    parts.append(f"عنوان ملازمت: {title}")

    description = job.get("desc_urdu") or job.get("Job Description") or ""
    if description:
        parts.append(f"\nتفصیل ملازمت:\n{description[:3000]}")

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
