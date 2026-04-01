"""
Text Cleaning Layer for Resume Parser ETL System.

This module provides conservative text cleaning that prepares raw
extracted resume text for downstream processing (section segmentation,
LLM structured extraction).

Design Principles:
    - Conservative: preserve structural information (headings, line breaks, dates)
    - Modular: each cleaning step is an isolated, testable function
    - No external NLP dependencies: only stdlib + regex
    - Idempotent: running cleaning twice produces the same output

Pipeline order:
    1. Normalize Unicode characters
    2. Normalize whitespace (tabs, multiple spaces)
    3. Fix broken words across line boundaries
    4. Trim individual lines
    5. Remove page headers/footers
    6. Collapse excessive blank lines

Usage:
    >>> from etl.text_clean import clean_text
    >>> cleaned = clean_text(raw_resume_text)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unicode replacement mappings
# ---------------------------------------------------------------------------

# Bullet-like characters → normalized dash for list items
_BULLET_CHARS: dict[str, str] = {
    "\u2022": "-",   # • BULLET
    "\u2023": "-",   # ‣ TRIANGULAR BULLET
    "\u25E6": "-",   # ◦ WHITE BULLET
    "\u2043": "-",   # ⁃ HYPHEN BULLET
    "\u25AA": "-",   # ▪ BLACK SMALL SQUARE
    "\u25CF": "-",   # ● BLACK CIRCLE
    "\u25CB": "-",   # ○ WHITE CIRCLE
    "\uf0b7": "-",   # Private-use bullet (common in Word exports)
    "\uf0a7": "-",   # Private-use bullet variant
    "\uf076": "-",   # Private-use bullet variant
}

# Dash-like characters → standard ASCII hyphen-minus
_DASH_CHARS: dict[str, str] = {
    "\u2013": "-",   # – EN DASH
    "\u2014": "-",   # — EM DASH
    "\u2015": "-",   # ― HORIZONTAL BAR
    "\u2212": "-",   # − MINUS SIGN
}

# Whitespace-like characters → standard space
_WHITESPACE_CHARS: dict[str, str] = {
    "\u00A0": " ",   # NO-BREAK SPACE
    "\u200B": "",    # ZERO WIDTH SPACE (remove entirely)
    "\u200C": "",    # ZERO WIDTH NON-JOINER
    "\u200D": "",    # ZERO WIDTH JOINER
    "\uFEFF": "",    # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u2003": " ",   # EM SPACE
    "\u2002": " ",   # EN SPACE
    "\u2009": " ",   # THIN SPACE
    "\u202F": " ",   # NARROW NO-BREAK SPACE
}

# Quotation marks → ASCII equivalents (preserve meaning, normalize encoding)
_QUOTE_CHARS: dict[str, str] = {
    "\u2018": "'",   # ' LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # ' RIGHT SINGLE QUOTATION MARK
    "\u201C": '"',   # " LEFT DOUBLE QUOTATION MARK
    "\u201D": '"',   # " RIGHT DOUBLE QUOTATION MARK
}

# Build a combined translation table for single-pass replacement
_UNICODE_TABLE: dict[int, str] = {}
for _mapping in (_BULLET_CHARS, _DASH_CHARS, _WHITESPACE_CHARS, _QUOTE_CHARS):
    for _char, _replacement in _mapping.items():
        _UNICODE_TABLE[ord(_char)] = _replacement


# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at module level for performance)
# ---------------------------------------------------------------------------

# Tabs → single space
_RE_TAB = re.compile(r"\t")

# Multiple consecutive spaces (but NOT newlines) → single space
_RE_MULTI_SPACE = re.compile(r"[^\S\n]{2,}")

# Three or more consecutive newlines (with optional whitespace between)
# → collapse to exactly two newlines (one blank line)
_RE_EXCESSIVE_BLANKS = re.compile(r"(\n\s*){3,}")

# Page header/footer patterns (case-insensitive)
# Matches standalone lines like "Page 1 of 3", "Page 2", "- 1 -", "1 | Page"
_RE_PAGE_MARKERS = re.compile(
    r"^\s*(?:"
    r"[Pp]age\s+\d+\s+of\s+\d+"       # "Page 1 of 2"
    r"|[Pp]age\s+\d+"                   # "Page 3"
    r"|-\s*\d+\s*-"                     # "- 1 -"
    r"|\d+\s*\|\s*[Pp]age"             # "1 | Page"
    r"|\d+\s*/\s*\d+"                  # "1 / 3" (standalone)
    r")\s*$",
    re.MULTILINE,
)

# Broken word across a line boundary WITH a hyphen:
# A letter followed by a hyphen + newline + lowercase letter.
# This is high-confidence: explicit hyphens at line ends almost always
# indicate a word split by the PDF renderer (e.g., "experi-\nence").
_RE_BROKEN_WORD = re.compile(
    r"([a-zA-Z])-\n([a-z])"
)

# Page break markers inserted by the extraction layer
_RE_PAGE_BREAK = re.compile(
    r"\n*---\s*PAGE\s+BREAK\s*---\n*",
    re.IGNORECASE,
)

# PDF hyperlink text artifacts — standalone lines that are navigation/link labels
# extracted from web-template PDFs (e.g., "More Details", "View Certificate").
_RE_LINK_ARTIFACTS = re.compile(
    r"^\s*(?:More\s*Details?|View\s*(?:Certificate|Credential|Project|All)?|"
    r"Show\s*Credential|Click\s*Here|See\s*More|Read\s*More|Download\s*CV|"
    r"View\s*Profile|Open\s*Link)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# CamelCase word splitting: insert a space between a lowercase letter and an
# uppercase letter that starts a new word (e.g., "DataEngineering" → "Data Engineering").
# Also handles ACRONYM+TitleCase: "GCPSpecialization" → "GCP Specialization".
# Applied ONLY to lines that look like section content (not URLs or code snippets).
_RE_CAMEL_LOWER_UPPER = re.compile(r"([a-z])([A-Z])")
# Only split ACRONYM+TitleWord when the title word has 4+ lowercase chars
# to avoid false splits like "SQLfor" → "SQ Lfor".
_RE_CAMEL_ACRONYM = re.compile(r"([A-Z]{2,})([A-Z][a-z]{4,})")

# Residual spaced-out uppercase characters that weren't caught at
# extraction time (e.g. "S K I L L S" → "SKILLS").
# Matches 3+ single uppercase letters separated by single spaces,
# not preceded or followed by a letter.
_RE_SPACED_CAPS = re.compile(
    r"(?<![A-Za-z])([A-Z](?: [A-Z]){2,})(?![A-Za-z])"
)


# ---------------------------------------------------------------------------
# Dataclass for cleaning results
# ---------------------------------------------------------------------------

@dataclass
class CleaningResult:
    """
    Result of text cleaning operation.

    Attributes:
        text: The cleaned text (empty string if input was empty/None)
        original_length: Character count of the original input
        cleaned_length: Character count after cleaning
        steps_applied: Names of cleaning steps that were executed
        warnings: Any non-fatal issues encountered during cleaning
    """
    text: str
    original_length: int
    cleaned_length: int
    steps_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Ratio of cleaned length to original length (0.0–1.0)."""
        if self.original_length == 0:
            return 1.0
        return self.cleaned_length / self.original_length

    @property
    def is_empty(self) -> bool:
        """Check if the cleaned text is empty or whitespace-only."""
        return len(self.text.strip()) == 0


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Replace problematic Unicode characters with ASCII equivalents.

    Handles:
        - Non-breaking spaces → regular space
        - Bullet symbols → "-"
        - Long dashes (en/em) → "-"
        - Smart quotes → ASCII quotes
        - Zero-width characters → removed

    This uses str.translate() for a single-pass O(n) replacement,
    which is more efficient than chaining multiple str.replace() calls.

    Args:
        text: Raw text potentially containing Unicode artifacts.

    Returns:
        Text with normalized Unicode characters.
    """
    return text.translate(_UNICODE_TABLE)


def normalize_whitespace(text: str) -> str:
    """
    Normalize tabs and multiple spaces without destroying line structure.

    Steps:
        1. Replace tabs with a single space
        2. Collapse runs of multiple spaces (on the same line) into one

    Line breaks (\\n) are explicitly preserved — they carry structural
    meaning in resumes (section boundaries, list items, etc.).

    Args:
        text: Text with potential whitespace irregularities.

    Returns:
        Text with normalized horizontal whitespace.
    """
    # Step 1: tabs → space
    text = _RE_TAB.sub(" ", text)

    # Step 2: collapse multiple horizontal spaces into one
    text = _RE_MULTI_SPACE.sub(" ", text)

    return text


def fix_broken_words(text: str) -> str:
    """
    Rejoin words that were split across line boundaries with a hyphen.

    Handles the case where PDF renderers break a word with an explicit
    hyphen at line end: "experi-\\nence" → "experience".

    This is conservative by design — only hyphenated breaks are fixed.
    Non-hyphenated breaks (e.g., "exper\\nience") are NOT merged because
    distinguishing genuine mid-word breaks from normal line wraps
    (e.g., "Python\\nto ensure") is unreliable without a dictionary.
    With layout-aware extraction, non-hyphenated word breaks are rare.

    Guards:
        - Does NOT merge lines that start with uppercase (likely headings
          or new sentences).
        - Does NOT merge across blank lines (paragraph boundaries).

    Args:
        text: Text with potential broken words across lines.

    Returns:
        Text with hyphenated broken words rejoined.
    """
    # Explicit hyphen at end of line: "experi-\nence" → "experience"
    text = _RE_BROKEN_WORD.sub(r"\1\2", text)

    return text


def trim_lines(text: str) -> str:
    """
    Strip leading and trailing whitespace from each individual line.

    Preserves blank lines (they may be intentional paragraph separators)
    but removes invisible trailing/leading spaces that add no meaning.

    Args:
        text: Text with potential per-line whitespace issues.

    Returns:
        Text with each line individually trimmed.
    """
    lines = text.split("\n")
    trimmed = [line.strip() for line in lines]
    return "\n".join(trimmed)


def remove_page_markers(text: str) -> str:
    """
    Remove standalone page numbering lines.

    Targets patterns like:
        - "Page 1 of 2"
        - "Page 3"
        - "- 1 -"
        - "1 | Page"
        - "1 / 3" (standalone on a line)

    These are artifacts of PDF extraction and carry no resume content.
    Only removes lines that consist ENTIRELY of a page marker pattern
    to avoid accidentally removing content that happens to contain numbers.

    Args:
        text: Text potentially containing page markers.

    Returns:
        Text with page marker lines removed.
    """
    return _RE_PAGE_MARKERS.sub("", text)


def normalize_page_breaks(text: str) -> str:
    """
    Replace page break markers with clean double newlines.

    The extraction layer inserts "--- PAGE BREAK ---" between pages.
    We replace these with a consistent double newline to maintain
    page separation without the artificial marker text.

    Args:
        text: Text potentially containing page break markers.

    Returns:
        Text with page break markers replaced by double newlines.
    """
    return _RE_PAGE_BREAK.sub("\n\n", text)


def fix_spaced_characters(text: str) -> str:
    """
    Join residual spaced-out uppercase characters into normal words.

    Some resume templates use CSS letter-spacing to style section
    headings. If the extraction layer didn't fully resolve these,
    this step catches remaining patterns like "S K I L L S" → "SKILLS".

    Only targets sequences of 3+ single uppercase letters separated
    by single spaces. This is very conservative to avoid accidentally
    merging content like "I A M" that are separate words.

    Args:
        text: Text potentially containing spaced-out characters.

    Returns:
        Text with spaced characters joined.
    """
    def _join_spaced(match: re.Match) -> str:
        return match.group(0).replace(" ", "")

    return _RE_SPACED_CAPS.sub(_join_spaced, text)


def remove_link_artifacts(text: str) -> str:
    """
    Remove standalone hyperlink-label lines extracted from web-template PDFs.

    Multi-column PDF templates (common for modern resumes) often have
    "More Details", "View Certificate", etc. as clickable labels next to
    each list item. The PDF extractor strips the hyperlink but keeps the
    label text, producing noisy standalone lines that confuse the LLM.

    Only removes lines that consist ENTIRELY of such a label to avoid
    accidentally deleting content that happens to contain these words.
    """
    return _RE_LINK_ARTIFACTS.sub("", text)


def fix_concatenated_words(text: str) -> str:
    """
    Insert spaces into CamelCase-concatenated words caused by column PDF extraction.

    Multi-column PDFs sometimes produce runs of words with no spaces:
        "DataEngineering,BigData,andMachineLearningonGCPSpecialization"
        → "Data Engineering, Big Data, and Machine Learning on GCP Specialization"

    Two patterns are applied:
      1. lowercase→Uppercase boundary: "DataEngineering" → "Data Engineering"
      2. ACRONYM→TitleCase boundary:  "GCPSpecialization" → "GCP Specialization"

    This is conservative: it only fires where a letter-case transition implies
    a word boundary, which is almost always correct in resume text.
    URLs and email addresses are left unchanged because they don't contain
    these CamelCase transitions.
    """
    # Pattern 1: camelCase split (e.g., "andMachine" → "and Machine")
    text = _RE_CAMEL_LOWER_UPPER.sub(r"\1 \2", text)
    # Pattern 2: ACRONYM+Title split (e.g., "GCPSpecialization" → "GCP Specialization")
    text = _RE_CAMEL_ACRONYM.sub(r"\1 \2", text)
    return text


def collapse_blank_lines(text: str) -> str:
    """
    Reduce runs of 3+ consecutive blank lines to at most 2 newlines.

    A single blank line (two consecutive newlines) is a meaningful
    paragraph/section separator. More than that is visual noise from
    extraction artifacts.

    Args:
        text: Text with potentially excessive blank lines.

    Returns:
        Text with at most one blank line between content lines.
    """
    return _RE_EXCESSIVE_BLANKS.sub("\n\n", text)


def strip_outer_whitespace(text: str) -> str:
    """
    Remove leading and trailing whitespace from the entire text.

    This is the final cleanup step — ensures no stray whitespace
    at the very beginning or end of the document.

    Args:
        text: Text to trim.

    Returns:
        Text with outer whitespace removed.
    """
    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

# Ordered list of cleaning steps.
# Each entry is (step_name, function).
# Order matters: unicode normalization must happen before whitespace
# normalization, broken-word fixing must happen before blank-line collapsing, etc.
_CLEANING_PIPELINE: list[tuple[str, callable]] = [
    ("normalize_unicode", normalize_unicode),
    ("normalize_page_breaks", normalize_page_breaks),
    ("fix_spaced_characters", fix_spaced_characters),
    ("fix_concatenated_words", fix_concatenated_words),   # camelCase → spaced words
    ("normalize_whitespace", normalize_whitespace),
    ("fix_broken_words", fix_broken_words),
    ("trim_lines", trim_lines),
    ("remove_page_markers", remove_page_markers),
    ("remove_link_artifacts", remove_link_artifacts),     # strip "More Details" etc.
    ("collapse_blank_lines", collapse_blank_lines),
    ("strip_outer_whitespace", strip_outer_whitespace),
]


def clean_text(
    raw_text: Optional[str],
    *,
    skip_steps: Optional[set[str]] = None,
) -> CleaningResult:
    """
    Run the full text cleaning pipeline on extracted resume text.

    This is the main entry point for the Text Cleaning Layer.
    Each step is applied sequentially in a defined order. Individual
    steps can be skipped via the `skip_steps` parameter for testing
    or special-case handling.

    Args:
        raw_text: The raw extracted text from a resume. May be None or empty.
        skip_steps: Optional set of step names to skip. Valid names are
            the keys from :data:`_CLEANING_PIPELINE`.

    Returns:
        CleaningResult containing the cleaned text and metadata.

    Example:
        >>> from etl.text_clean import clean_text
        >>> result = clean_text("  Hello \\t World  \\n\\n\\n\\nTest  ")
        >>> result.text
        'Hello World\\n\\nTest'
        >>> result.steps_applied
        ['normalize_unicode', 'normalize_page_breaks', ...]
    """
    # Handle None / empty input gracefully
    if raw_text is None:
        logger.info("Received None input; returning empty CleaningResult")
        return CleaningResult(
            text="",
            original_length=0,
            cleaned_length=0,
            steps_applied=[],
            warnings=["Input text was None"],
        )

    original_length = len(raw_text)

    if original_length == 0:
        logger.info("Received empty string input")
        return CleaningResult(
            text="",
            original_length=0,
            cleaned_length=0,
            steps_applied=[],
            warnings=["Input text was empty"],
        )

    skip = skip_steps or set()
    text = raw_text
    steps_applied: list[str] = []
    warnings: list[str] = []

    for step_name, step_fn in _CLEANING_PIPELINE:
        if step_name in skip:
            logger.debug("Skipping cleaning step: %s", step_name)
            continue

        try:
            text = step_fn(text)
            steps_applied.append(step_name)
        except Exception as e:
            # A failing step should not crash the entire pipeline.
            # Log the error, record a warning, and continue with
            # the text as-is from the last successful step.
            logger.error(
                "Cleaning step '%s' failed: %s", step_name, e, exc_info=True
            )
            warnings.append(f"Step '{step_name}' failed: {e}")

    cleaned_length = len(text)

    # Sanity check: if cleaning removed all text, something may be wrong
    if cleaned_length == 0 and original_length > 0:
        warnings.append(
            "Cleaning reduced text to empty — original had "
            f"{original_length} chars"
        )
        logger.warning(
            "Text cleaning produced empty output from %d-char input",
            original_length,
        )

    return CleaningResult(
        text=text,
        original_length=original_length,
        cleaned_length=cleaned_length,
        steps_applied=steps_applied,
        warnings=warnings,
    )
