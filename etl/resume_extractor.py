"""
LLM Structured Extraction Layer for Resume Parser ETL System.

This module orchestrates BAML-based structured extraction of resume data.
It takes cleaned resume text, sends it through the BAML-defined extraction
function (which calls the LLM with strict schema enforcement), and returns
a validated, structured result ready for database insertion.

Architecture:
    Cleaned text → BAML ExtractResume() → ValidatedResume (Pydantic) → ExtractionResult

Design Decisions:
    - BAML handles JSON schema enforcement at the LLM level (the model
      is constrained to produce valid JSON matching our type definitions).
    - Temperature=0 ensures deterministic, reproducible outputs.
    - The round-robin client tries the primary model first, falls back.
    - All retry/error handling is in this Python layer, not in BAML.
    - Post-extraction validation (dates, dedup, consistency) is handled
      by Pydantic models in validated_models.py — not manual functions.

Usage:
    >>> from etl.resume_extractor import extract_resume_structured
    >>> result = extract_resume_structured(cleaned_text)
    >>> if result.is_success:
    ...     print(result.resume.first_name)
    ...     for exp in result.resume.experience:
    ...         print(f"{exp.job_title} at {exp.company_name}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from etl.baml_client import b
from etl.baml_client.types import Resume as BAMLResume

from etl.validated_models import ValidatedResume, validate_resume

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum input length (characters) to send to the LLM.
# Groq Llama 3.3 70B supports 128K context. 15K chars covers even dense
# multi-page resumes (projects and certifications sections must not be cut).
MAX_INPUT_LENGTH = 15_000

# Number of retry attempts for transient LLM failures.
MAX_RETRIES = 3

# Base delay between retries (seconds). Uses exponential backoff.
# Keep this low for faster batch throughput.
RETRY_BASE_DELAY = 5.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class LLMExtractionStatus(Enum):
    """Status of the LLM extraction operation."""
    SUCCESS = "success"
    FAILED = "failed"
    INPUT_TOO_LONG = "input_too_long"
    EMPTY_INPUT = "empty_input"
    PARSE_ERROR = "parse_error"      # BAML couldn't parse LLM output
    LLM_ERROR = "llm_error"          # LLM API error (rate limit, timeout, etc.)
    VALIDATION_ERROR = "validation_error"  # Post-extraction validation failed


@dataclass
class LLMExtractionResult:
    """
    Result of LLM-based structured resume extraction.

    Attributes:
        resume: The validated Resume object (None if extraction failed).
        status: Status of the extraction operation.
        error_message: Error details if extraction failed.
        warnings: Non-fatal issues found during post-processing.
        retries_used: Number of retry attempts that were needed.
        latency_ms: Total extraction time in milliseconds.
        input_length: Character count of the input text.
    """
    resume: Optional[ValidatedResume]
    status: LLMExtractionStatus
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    retries_used: int = 0
    latency_ms: float = 0.0
    input_length: int = 0

    @property
    def is_success(self) -> bool:
        """Check if extraction was successful."""
        return self.status == LLMExtractionStatus.SUCCESS

    @property
    def has_resume(self) -> bool:
        """Check if a Resume object is available."""
        return self.resume is not None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_resume_structured(
    cleaned_text: str,
    *,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_BASE_DELAY,
) -> LLMExtractionResult:
    """
    Extract structured resume data from cleaned text using BAML + LLM.

    This is the main entry point for the LLM Structured Extraction Layer.

    Pipeline:
        1. Input validation (length, emptiness)
        2. BAML ExtractResume() call with retry logic
        3. Pydantic validation (dates, dedup, consistency, semantics)

    Args:
        cleaned_text: Cleaned resume text from the Text Cleaning Layer.
        max_retries: Maximum number of retry attempts for transient failures.
        retry_delay: Base delay between retries (exponential backoff).

    Returns:
        LLMExtractionResult containing the structured Resume and metadata.

    Example:
        >>> result = extract_resume_structured(cleaned_text)
        >>> if result.is_success:
        ...     resume = result.resume
        ...     print(f"{resume.first_name} {resume.last_name}")
        ...     for exp in resume.experience:
        ...         print(f"  {exp.job_title} at {exp.company_name}")
    """
    start_time = time.monotonic()
    pre_warnings: list[str] = []

    # --- Input validation ---
    if not cleaned_text or not cleaned_text.strip():
        return LLMExtractionResult(
            resume=None,
            status=LLMExtractionStatus.EMPTY_INPUT,
            error_message="Input text is empty or whitespace-only",
            input_length=0,
            latency_ms=_elapsed_ms(start_time),
        )

    input_length = len(cleaned_text)

    if input_length > MAX_INPUT_LENGTH:
        pre_warnings.append(
            f"Input text was truncated from {input_length} to {MAX_INPUT_LENGTH} chars to fit model limits"
        )
        cleaned_text = cleaned_text[:MAX_INPUT_LENGTH]
        input_length = len(cleaned_text)

    # --- BAML extraction with retry logic ---
    last_error: Optional[Exception] = None
    retries_used = 0

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "BAML extraction attempt %d/%d (input: %d chars)",
                attempt, max_retries, input_length,
            )

            raw_resume: BAMLResume = b.ExtractResume(resume_text=cleaned_text)

            logger.info(
                "BAML extraction succeeded on attempt %d "
                "(name: %s %s, %d experience, %d education, %d skills)",
                attempt,
                raw_resume.first_name,
                raw_resume.last_name,
                len(raw_resume.experience),
                len(raw_resume.education),
                len(raw_resume.skills),
            )
            break  # Success — exit retry loop

        except Exception as e:
            last_error = e
            retries_used = attempt
            error_type = type(e).__name__

            logger.warning(
                "BAML extraction attempt %d/%d failed (%s): %s",
                attempt, max_retries, error_type, str(e),
            )

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.info("Retrying in %.1f seconds...", delay)
                time.sleep(delay)
    else:
        # All retries exhausted
        error_msg = f"All {max_retries} extraction attempts failed. Last error: {last_error}"
        logger.error(error_msg)

        # Classify the error
        error_str = str(last_error).lower()
        if "parse" in error_str or "json" in error_str or "deseriali" in error_str:
            status = LLMExtractionStatus.PARSE_ERROR
        else:
            status = LLMExtractionStatus.LLM_ERROR

        return LLMExtractionResult(
            resume=None,
            status=status,
            error_message=error_msg,
            retries_used=retries_used,
            input_length=input_length,
            latency_ms=_elapsed_ms(start_time),
        )

    # --- Post-extraction: Pydantic validation ---
    validated_resume, all_warnings = validate_resume(raw_resume)

    if all_warnings:
        logger.info(
            "Post-extraction validation generated %d warnings",
            len(all_warnings),
        )
        for w in all_warnings:
            logger.debug("  Warning: %s", w)

    return LLMExtractionResult(
        resume=validated_resume,
        status=LLMExtractionStatus.SUCCESS,
        warnings=pre_warnings + all_warnings,
        retries_used=retries_used - 1 if retries_used > 0 else 0,
        input_length=input_length,
        latency_ms=_elapsed_ms(start_time),
    )


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed time in milliseconds since start."""
    return (time.monotonic() - start) * 1000
