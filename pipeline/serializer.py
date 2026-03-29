"""Convert structured candidate/job dicts into rich text for embedding."""


def serialize_candidate(candidate: dict) -> str:
    parts = []

    name = f"{candidate.get('first_name') or ''} {candidate.get('last_name') or ''}".strip() or "Unknown"
    role = candidate.get('current_role') or 'Not specified'
    exp = candidate.get('years_experience') or 0
    industry = candidate.get('associated_industry') or 'Not specified'
    city = candidate.get('city') or ''
    country = candidate.get('country') or ''
    location = ', '.join(filter(None, [city, country])) or 'Not specified'

    parts.append(f"Name: {name} | Role: {role} | Experience: {exp} years")
    parts.append(f"Industry: {industry} | Location: {location}")
    if candidate.get('qualification'):
        parts.append(f"Qualification: {candidate['qualification']}")

    # Skills
    skills = candidate.get('skills') or []
    if skills:
        skill_strs = []
        for s in skills:
            skill_str = s.get('skill_name') or ''
            if s.get('proficiency_level'):
                skill_str += f" ({s['proficiency_level']}"
                if s.get('years_of_experience'):
                    skill_str += f", {s['years_of_experience']} yrs"
                skill_str += ")"
            if s.get('is_primary'):
                skill_str += " [primary]"
            skill_strs.append(skill_str)
        parts.append(f"\nSKILLS:\n{', '.join(skill_strs)}")

    # Work Experience
    experiences = candidate.get('work_experiences') or []
    if experiences:
        exp_parts = ["\nWORK EXPERIENCE:"]
        for w in experiences:
            from_date = str(w.get('from_date') or '')[:7]
            end_date = 'Present' if w.get('is_current_employer') else str(w.get('end_date') or '')[:7]
            duration = f", {w['duration_months']} months" if w.get('duration_months') else ''
            exp_parts.append(
                f"{w.get('job_title') or ''} at {w.get('company_name') or ''} "
                f"({from_date} - {end_date}{duration})"
            )
            if w.get('industry'):
                exp_parts.append(f"Industry: {w['industry']}")
            if w.get('job_description'):
                exp_parts.append(w['job_description'])
        parts.append('\n'.join(exp_parts))

    # Education
    educations = candidate.get('educations') or []
    if educations:
        edu_parts = ["\nEDUCATION:"]
        for e in educations:
            from_y = str(e.get('from_date') or '')[:4]
            end_y = str(e.get('end_date') or '')[:4]
            line = f"{e.get('degree') or ''} - {e.get('institute_name') or ''} ({from_y}-{end_y})"
            if e.get('field_of_study'):
                line += f" | Field: {e['field_of_study']}"
            if e.get('cgpa'):
                line += f" | CGPA: {e['cgpa']}"
            edu_parts.append(line)
        parts.append('\n'.join(edu_parts))

    # Certifications
    certs = candidate.get('certifications') or []
    if certs:
        cert_strs = [
            f"{c.get('certification_name') or ''} - {c.get('issuing_organization') or ''}"
            for c in certs
        ]
        parts.append(f"\nCERTIFICATIONS:\n" + '\n'.join(cert_strs))

    # Projects
    projects = candidate.get('projects') or []
    if projects:
        proj_parts = ["\nPROJECTS:"]
        for p in projects:
            proj_parts.append(
                f"{p.get('project_name') or ''} "
                f"(Role: {p.get('role_in_project') or 'N/A'}, Domain: {p.get('domain') or 'N/A'})"
            )
            if p.get('project_description'):
                proj_parts.append(p['project_description'])
        parts.append('\n'.join(proj_parts))

    # Languages
    languages = candidate.get('languages') or []
    if languages:
        lang_strs = [
            f"{l.get('language_name') or ''} ({l.get('proficiency_level') or ''})"
            for l in languages
        ]
        parts.append(f"\nLANGUAGES:\n{', '.join(lang_strs)}")

    # Job Preferences
    prefs = candidate.get('job_preferences')
    if prefs:
        pref_parts = ["\nJOB PREFERENCES:"]
        if prefs.get('job_search_status'):
            pref_parts.append(f"Status: {str(prefs['job_search_status']).replace('_', ' ').title()}")
        emp_types = prefs.get('preferred_employment_types')
        if isinstance(emp_types, list):
            pref_parts.append(f"Types: {', '.join(emp_types)}")
        target_roles = prefs.get('target_roles')
        if isinstance(target_roles, list):
            pref_parts.append(f"Target Roles: {', '.join(target_roles)}")
        target_inds = prefs.get('target_industries')
        if isinstance(target_inds, list):
            pref_parts.append(f"Target Industries: {', '.join(target_inds)}")
        if prefs.get('notice_period_days'):
            pref_parts.append(f"Notice Period: {prefs['notice_period_days']} days")
        if prefs.get('expected_salary_max'):
            currency = prefs.get('salary_currency') or 'PKR'
            pref_parts.append(f"Expected Salary: {prefs['expected_salary_max']} {currency}")
        parts.append('\n'.join(pref_parts))

    return '\n'.join(parts)


def serialize_job(job: dict) -> str:
    parts = []

    parts.append(f"Job Title: {job.get('job_title') or 'Unknown'}")

    dept = job.get('department') or ''
    industry = job.get('industry') or ''
    if dept or industry:
        parts.append(f"Department: {dept} | Industry: {industry}")

    emp_type = job.get('employment_type') or ''
    workspace = job.get('work_space_type') or ''
    if emp_type or workspace:
        parts.append(f"Employment Type: {emp_type} | Workspace: {workspace}")

    city = job.get('city') or ''
    country = job.get('country') or ''
    location = ', '.join(filter(None, [city, country]))
    if location:
        parts.append(f"Location: {location}")

    min_exp = job.get('min_years_experience')
    max_exp = job.get('max_years_experience')
    if min_exp is not None or max_exp is not None:
        parts.append(f"Experience Required: {min_exp or 0}-{max_exp or '+'} years")

    if job.get('education_requirement'):
        parts.append(f"Education: {job['education_requirement']}")

    if job.get('job_brief'):
        parts.append(f"\nJOB BRIEF:\n{job['job_brief']}")

    skills = job.get('skills') or []
    required = [s for s in skills if s.get('is_required')]
    preferred = [s for s in skills if not s.get('is_required')]

    if required:
        req_strs = []
        for s in required:
            skill_str = s.get('skill_name') or ''
            if s.get('min_years_required'):
                skill_str += f" (min {s['min_years_required']} yrs)"
            req_strs.append(skill_str)
        parts.append(f"\nREQUIRED SKILLS:\n{', '.join(req_strs)}")

    if preferred:
        parts.append(f"\nPREFERRED SKILLS:\n{', '.join(s.get('skill_name') or '' for s in preferred)}")

    sections = sorted(job.get('sections') or [], key=lambda x: x.get('section_order') or 0)
    for section in sections:
        name = (section.get('section_name') or '').upper()
        content = section.get('section_content') or ''
        if name and content:
            parts.append(f"\n{name}:\n{content}")

    return '\n'.join(parts)
