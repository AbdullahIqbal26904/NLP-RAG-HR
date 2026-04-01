"""
ETL Module for Resume Parser.

This module contains all ETL (Extract, Transform, Load) components
for the resume parsing pipeline.
"""

from .text_extractor import (
    # Main entry point
    extract_text,
    # Result types
    ExtractionResult,
    ExtractionStatus,
    FileType,
    # Extractors (for advanced usage)
    PdfTextExtractor,
    DocxTextExtractor,
    TextExtractorFactory,
    # Exceptions
    TextExtractorError,
    UnsupportedFileTypeError,
    CorruptedFileError,
)

from .text_clean import (
    # Main entry point
    clean_text,
    # Result type
    CleaningResult,
    # Individual step functions (for advanced / selective usage)
    normalize_unicode,
    normalize_whitespace,
    fix_broken_words,
    fix_spaced_characters,
    trim_lines,
    remove_page_markers,
    normalize_page_breaks,
    collapse_blank_lines,
    strip_outer_whitespace,
)

from .layout_analyzer import (
    # Main entry point
    extract_page_with_layout,
    # Analyzer class
    LayoutAnalyzer,
    # Result types
    LayoutInfo,
    ColumnRegion,
)

from .resume_extractor import (
    # Main entry point
    extract_resume_structured,
    # Result types
    LLMExtractionResult,
    LLMExtractionStatus,
)

from .validated_models import (
    # Conversion function
    validate_resume,
    # Validated model types
    ValidatedResume,
    ValidatedWorkExperience,
    ValidatedEducation,
    ValidatedProject,
    ValidatedCertification,
    ValidatedSkill,
    ValidatedLanguage,
    ValidatedAchievement,
)

__all__ = [
    # --- Extraction Layer ---
    "extract_text",
    "ExtractionResult",
    "ExtractionStatus",
    "FileType",
    "PdfTextExtractor",
    "DocxTextExtractor",
    "TextExtractorFactory",
    "TextExtractorError",
    "UnsupportedFileTypeError",
    "CorruptedFileError",
    # --- Layout Analysis Layer ---
    "extract_page_with_layout",
    "LayoutAnalyzer",
    "LayoutInfo",
    "ColumnRegion",
    # --- Cleaning Layer ---
    "clean_text",
    "CleaningResult",
    "normalize_unicode",
    "normalize_whitespace",
    "fix_broken_words",
    "trim_lines",
    "remove_page_markers",
    "normalize_page_breaks",
    "collapse_blank_lines",
    "strip_outer_whitespace",
    # --- LLM Structured Extraction Layer ---
    "extract_resume_structured",
    "LLMExtractionResult",
    "LLMExtractionStatus",
    # --- Pydantic Validation Models ---
    "validate_resume",
    "ValidatedResume",
    "ValidatedWorkExperience",
    "ValidatedEducation",
    "ValidatedProject",
    "ValidatedCertification",
    "ValidatedSkill",
    "ValidatedLanguage",
    "ValidatedAchievement",
]
