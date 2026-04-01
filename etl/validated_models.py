"""
Pydantic Validation Models for Resume Data.

Provides validated versions of the BAML-generated resume types with:
  - Date normalization (YYYY-MM format correction)
  - Cross-field consistency checks (current employer ↔ end_date)
  - CGPA range validation
  - Skill deduplication (keeps most informative entry)
  - Semantic completeness warnings (missing name, empty sections)

All corrections and warnings are collected via Pydantic's validation
context, so callers receive a clean (validated_resume, warnings) tuple.

Usage:
    >>> from etl.validated_models import validate_resume
    >>> validated_resume, warnings = validate_resume(baml_resume)
    >>> print(validated_resume.first_name)
    >>> for w in warnings:
    ...     print(f"  ⚠ {w}")
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from etl.baml_client.types import (
    Resume as BAMLResume,
    # Reuse BAML-generated enums — no duplication
    AchievementType,
    AwardLevel,
    CertificationLevel,
    CertificationType,
    EmploymentType,
    InstitutionType,
    LanguageProficiency,
    ProficiencyLevel,
    ProjectType,
    SkillCategory,
    WorkspaceType,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_PRESENT_WORDS = frozenset({"present", "current", "ongoing", "now", "till date"})


def _add_warning(context: Any, message: str) -> None:
    """Safely append a warning to the validation context."""
    if isinstance(context, dict):
        context.setdefault("warnings", []).append(message)


def _normalize_date(
    value: Optional[str],
    *,
    field_label: str = "date",
    context: Any = None,
) -> Optional[str]:
    """
    Normalize a date string to YYYY-MM format.

    Corrects common LLM output issues:
      "2023"    → "2023-01"  (year-only → default January)
      "2023-1"  → "2023-01"  (single-digit month → zero-pad)
      "Present" → None       (textual "current" markers)
      ""        → None       (empty string)
    """
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    # Already valid
    if _DATE_RE.match(value):
        return value

    # Year only → default to January
    if re.match(r"^\d{4}$", value):
        corrected = f"{value}-01"
        _add_warning(context, f"{field_label}: '{value}' had no month; defaulted to '{corrected}'")
        return corrected

    # Single-digit month → zero-pad
    m = re.match(r"^(\d{4})-(\d)$", value)
    if m:
        corrected = f"{m.group(1)}-0{m.group(2)}"
        _add_warning(context, f"{field_label}: '{value}' single-digit month; corrected to '{corrected}'")
        return corrected

    # "Present", "Current", etc. → null
    if value.lower() in _PRESENT_WORDS:
        _add_warning(context, f"{field_label}: contained '{value}' instead of null; corrected")
        return None

    # Unrecoverable format
    _add_warning(context, f"{field_label}: invalid format '{value}'; set to null")
    return None


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def _extract_city_from_address(address: str) -> Optional[str]:
    """
    Heuristically extract a city name from a free-text address string.

    Strategy: split on commas, scan from the end for the first segment
    that is purely alphabetic (city names don't contain digits or #).
    Returns None if no candidate segment is found.

    Examples:
        "House no # 1794 Police Line, Abbottabad" → "Abbottabad"
        "Flat 5, F-8/3, Islamabad"                → "Islamabad"
    """
    parts = [p.strip() for p in address.split(",")]
    for part in reversed(parts):
        # Strip trailing punctuation (periods, colons, etc.)
        cleaned = part.strip().rstrip(".:;")
        # Match segments that are only letters, spaces, hyphens, or dots
        if cleaned and re.match(r'^[A-Za-z][A-Za-z\s\-\.]+$', cleaned):
            return cleaned
    return None


# ---------------------------------------------------------------------------
# Experience duration helper
# ---------------------------------------------------------------------------

def _months_between(from_date: str, end_date: str) -> int:
    """Return the number of months between two YYYY-MM date strings."""
    try:
        fy, fm = map(int, from_date.split("-"))
        ey, em = map(int, end_date.split("-"))
        return max(0, (ey - fy) * 12 + (em - fm))
    except (ValueError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# Reusable date field validator
# ---------------------------------------------------------------------------

def _validate_date_field(v: Optional[str], info: ValidationInfo) -> Optional[str]:
    """Field validator callback shared by all models with date fields."""
    return _normalize_date(v, field_label=info.field_name, context=info.context)


# ---------------------------------------------------------------------------
# Validated sub-models
# ---------------------------------------------------------------------------

class ValidatedWorkExperience(BaseModel):
    """Work experience with date normalization and current-employer consistency."""

    company_name: str
    job_title: str
    from_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current_employer: bool = False
    country: Optional[str] = None
    city: Optional[str] = None
    employment_type: Optional[EmploymentType] = None
    industry: Optional[str] = None
    job_description: Optional[str] = None
    workspace_type: Optional[WorkspaceType] = None

    # --- Validators ---

    @field_validator("from_date", "end_date", mode="before")
    @classmethod
    def normalize_dates(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_date_field(v, info)

    def model_post_init(self, __context: Any) -> None:
        """If current employer, end_date must be null."""
        if self.is_current_employer and self.end_date is not None:
            _add_warning(
                __context,
                f"experience['{self.job_title}']: is_current_employer=True "
                f"but end_date='{self.end_date}'; clearing end_date",
            )
            self.end_date = None


class ValidatedEducation(BaseModel):
    """Education entry with date normalization, CGPA check, and attendance consistency."""

    institute_name: str
    degree: str
    cgpa: Optional[float] = None
    is_currently_attending: bool = False
    from_date: Optional[str] = None
    end_date: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    institution_type: Optional[InstitutionType] = None
    field_of_study: Optional[str] = None

    # --- Validators ---

    @field_validator("from_date", "end_date", mode="before")
    @classmethod
    def normalize_dates(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_date_field(v, info)

    @field_validator("cgpa")
    @classmethod
    def validate_cgpa(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        """Warn if CGPA exceeds common scale ranges."""
        if v is not None and v > 10.0:
            _add_warning(info.context, f"education: CGPA {v} seems unreasonably high")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Consistency checks and city fallback for education entries."""
        # Enforce currently-attending ↔ end_date consistency
        if self.is_currently_attending and self.end_date is not None:
            _add_warning(
                __context,
                f"education['{self.degree}']: is_currently_attending=True "
                f"but end_date='{self.end_date}'; clearing end_date",
            )
            self.end_date = None

        # Fallback: extract city from institute_name if LLM left city null.
        # Very common pattern: "University Name, City" or "University Name, City."
        if not self.city and self.institute_name:
            extracted = _extract_city_from_address(self.institute_name)
            if extracted:
                self.city = extracted
                _add_warning(
                    __context,
                    f"education city inferred from institute name: '{extracted}'",
                )


class ValidatedProject(BaseModel):
    """Project entry with date normalization."""

    project_name: str
    associated_with: Optional[str] = None
    project_type: Optional[ProjectType] = None
    project_description: Optional[str] = None
    project_url: Optional[str] = None
    repository_url: Optional[str] = None
    role_in_project: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    domain: Optional[str] = None

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def normalize_dates(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_date_field(v, info)


class ValidatedCertification(BaseModel):
    """Certification entry with date normalization."""

    certification_name: str
    issuing_organization: Optional[str] = None
    certification_type: Optional[CertificationType] = None
    credential_url: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    score: Optional[str] = None
    skill_domain: Optional[str] = None
    level: Optional[CertificationLevel] = None
    description: Optional[str] = None

    @field_validator("issue_date", "expiry_date", mode="before")
    @classmethod
    def normalize_dates(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_date_field(v, info)


class ValidatedSkill(BaseModel):
    """Individual skill entry (deduplication handled at Resume level)."""

    skill_name: str
    skill_category: Optional[SkillCategory] = None
    proficiency_level: Optional[ProficiencyLevel] = None
    years_of_experience: Optional[float] = None
    is_primary: bool = False


class ValidatedLanguage(BaseModel):
    """Language proficiency entry."""

    language_name: str
    proficiency_level: Optional[LanguageProficiency] = None
    certification_name: Optional[str] = None
    certification_score: Optional[str] = None


class ValidatedAchievement(BaseModel):
    """Achievement entry with date normalization."""

    achievement_type: Optional[AchievementType] = None
    achievement_title: str
    issuing_organization: Optional[str] = None
    achievement_date: Optional[str] = None
    description: Optional[str] = None
    publication_url: Optional[str] = None
    competition_name: Optional[str] = None
    position_rank: Optional[str] = None
    award_level: Optional[AwardLevel] = None

    @field_validator("achievement_date", mode="before")
    @classmethod
    def normalize_dates(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_date_field(v, info)


# ---------------------------------------------------------------------------
# Top-level validated resume
# ---------------------------------------------------------------------------

class ValidatedResume(BaseModel):
    """
    Fully validated resume with all corrections applied.

    Validation rules applied automatically:
      1. All date fields normalized to YYYY-MM format
      2. Current employer / currently attending ↔ end_date consistency
      3. CGPA range validated (warn if > 10.0)
      4. Duplicate skills merged (keeps most informative entry)
      5. Missing candidate name warning
      6. Empty structured sections warning
      7. Experience chronological order check
    """

    # --- Personal information ---
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    salutation: Optional[str] = None
    gender: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    alternate_phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    nationality: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    current_address: Optional[str] = None
    current_role: Optional[str] = None
    qualification: Optional[str] = None
    associated_industry: Optional[str] = None
    # Computed post-validation from experience dates (not extracted by LLM)
    years_experience: Optional[int] = None

    # --- Nested sections (each applies its own validators) ---
    education: list[ValidatedEducation] = Field(default_factory=list)
    experience: list[ValidatedWorkExperience] = Field(default_factory=list)
    projects: list[ValidatedProject] = Field(default_factory=list)
    skills: list[ValidatedSkill] = Field(default_factory=list)
    certifications: list[ValidatedCertification] = Field(default_factory=list)
    languages: list[ValidatedLanguage] = Field(default_factory=list)
    achievements: list[ValidatedAchievement] = Field(default_factory=list)

    # --- Field-level validators ---

    @field_validator("skills", mode="after")
    @classmethod
    def deduplicate_skills(
        cls,
        skills: list[ValidatedSkill],
        info: ValidationInfo,
    ) -> list[ValidatedSkill]:
        """Remove duplicate skills, keeping the entry with the most detail."""
        if not skills:
            return skills

        seen: dict[str, int] = {}  # normalized_name → index in deduped
        deduped: list[ValidatedSkill] = []

        for skill in skills:
            key = skill.skill_name.lower().strip()

            if key in seen:
                # Score each entry by how many optional fields are filled
                existing = deduped[seen[key]]
                existing_score = sum([
                    existing.skill_category is not None,
                    existing.proficiency_level is not None,
                    existing.years_of_experience is not None,
                    existing.is_primary,
                ])
                new_score = sum([
                    skill.skill_category is not None,
                    skill.proficiency_level is not None,
                    skill.years_of_experience is not None,
                    skill.is_primary,
                ])
                if new_score > existing_score:
                    deduped[seen[key]] = skill
                _add_warning(info.context, f"Deduplicated skill: '{skill.skill_name}'")
            else:
                seen[key] = len(deduped)
                deduped.append(skill)

        return deduped

    # --- Model-level semantic checks ---

    def model_post_init(self, __context: Any) -> None:
        """Run semantic completeness checks after all fields are validated."""

        # 1. Check if candidate name exists
        if not self.first_name and not self.last_name:
            _add_warning(
                __context,
                "No candidate name could be extracted from the resume",
            )

        # 2. Check for suspiciously empty results
        total_sections = (
            len(self.education)
            + len(self.experience)
            + len(self.projects)
            + len(self.skills)
            + len(self.certifications)
        )
        if total_sections == 0:
            _add_warning(
                __context,
                "No structured sections (education, experience, projects, skills, "
                "certifications) were extracted — the resume may have unusual formatting",
            )

        # 3. Verify experience is in reverse chronological order
        for i in range(len(self.experience) - 1):
            curr = self.experience[i]
            next_ = self.experience[i + 1]
            if (
                curr.from_date
                and next_.from_date
                and curr.from_date < next_.from_date
            ):
                _add_warning(
                    __context,
                    f"Experience may not be in reverse chronological order: "
                    f"'{curr.job_title}' ({curr.from_date}) before "
                    f"'{next_.job_title}' ({next_.from_date})",
                )

        # 4. Compute years_experience from experience entries
        if self.experience:
            today_str = date.today().strftime("%Y-%m")
            total_months = 0
            for exp in self.experience:
                if not exp.from_date:
                    continue
                end = exp.end_date if (not exp.is_current_employer and exp.end_date) else today_str
                total_months += _months_between(exp.from_date, end)
            if total_months > 0:
                self.years_experience = max(1, total_months // 12)

        # 5. Fallback: extract city from current_address if LLM left city null
        if not self.city and self.current_address:
            extracted = _extract_city_from_address(self.current_address)
            if extracted:
                self.city = extracted
                _add_warning(
                    __context,
                    f"city not extracted by LLM; inferred from address: '{extracted}'",
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_resume(baml_resume: BAMLResume) -> tuple[ValidatedResume, list[str]]:
    """
    Convert a BAML-extracted Resume into a fully validated resume.

    Applies all Pydantic validation rules (date normalization, consistency
    checks, skill deduplication, semantic warnings) in a single pass.

    Args:
        baml_resume: Raw Resume object returned by BAML extraction.

    Returns:
        Tuple of (validated_resume, warnings_list).

    Example:
        >>> validated, warnings = validate_resume(baml_resume)
        >>> print(f"{validated.first_name} {validated.last_name}")
        >>> for w in warnings:
        ...     print(f"  ⚠ {w}")
    """
    warnings: list[str] = []
    validated = ValidatedResume.model_validate(
        baml_resume.model_dump(),
        context={"warnings": warnings},
    )
    return validated, warnings
