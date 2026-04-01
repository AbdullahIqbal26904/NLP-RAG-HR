"""
Layout Analysis Module for Resume Parser ETL System.

This module detects multi-column layouts in PDF pages by analyzing
the spatial distribution of characters/words. It enables the extraction
layer to handle modern resume templates that use sidebars, two-column
layouts, or text boxes.

Architecture:
    PDF Page → extract words with bounding boxes
            → detect columns via x-position gap analysis
            → identify full-width header/footer regions
            → return column regions for independent extraction

Design Principles:
    - No external dependencies beyond pdfplumber (already required)
    - Conservative detection: only split when columns are clearly present
    - Graceful fallback: single-column detection = use default extraction
    - Production-ready: handles edge cases, logs decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum gap (in PDF points) between x-position clusters to consider
# them separate columns. PDF points ≈ 1/72 inch, so 20pt ≈ 0.28 inches.
MIN_COLUMN_GAP = 20.0

# Minimum fraction of total words that must appear in each detected column
# to consider it a real column (avoids splitting on stray marginal text).
MIN_COLUMN_WORD_FRACTION = 0.08

# Minimum number of words required on a page to attempt layout detection.
# Pages with very few words are not worth analyzing for columns.
MIN_WORDS_FOR_DETECTION = 15

# Y-tolerance for grouping words into same-line rows (in PDF points).
# Words within this vertical distance are considered on the same line.
LINE_Y_TOLERANCE = 4.0

# A line is considered "full-width" if its text spans from the left column
# region into the right column region. Full-width lines are part of
# headers/footers that span the entire page.
FULL_WIDTH_SPAN_RATIO = 0.6

# --- Line-based gap voting constants ---
# Minimum horizontal gap within a single text line to signal a potential
# column break. Normal word spacing is 2-10pts; column breaks are 25+pts.
MIN_LINE_GAP = 25.0

# Minimum number of text lines that must show a gap at a consistent
# x-position to confirm a column boundary.
MIN_GAP_VOTES = 3

# Tolerance (in PDF points) for clustering gap midpoints across lines.
# Gaps whose midpoints are within this distance are considered the same
# column boundary.
GAP_CLUSTER_TOLERANCE = 40.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnRegion:
    """
    A detected column region on a PDF page.

    Coordinates are in PDF points, origin at top-left.

    Attributes:
        x0: Left edge of the column
        x1: Right edge of the column
        y0: Top edge (start of column content)
        y1: Bottom edge (end of column content)
        word_count: Number of words detected in this column
        label: Human-readable label for logging ("left", "right", etc.)
    """
    x0: float
    x1: float
    y0: float
    y1: float
    word_count: int = 0
    label: str = ""

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class LayoutInfo:
    """
    Complete layout analysis result for a single PDF page.

    Attributes:
        is_multi_column: Whether the page has a multi-column layout
        columns: List of detected column regions (left-to-right order)
        header_region: Full-width region at top of page (if detected)
        footer_region: Full-width region at bottom of page (if detected)
        page_width: Total page width in PDF points
        page_height: Total page height in PDF points
        confidence: Detection confidence ("high", "medium", "low")
        warnings: Non-fatal issues during detection
    """
    is_multi_column: bool
    columns: list[ColumnRegion] = field(default_factory=list)
    header_region: Optional[ColumnRegion] = None
    footer_region: Optional[ColumnRegion] = None
    page_width: float = 0.0
    page_height: float = 0.0
    confidence: str = "high"
    warnings: list[str] = field(default_factory=list)

    @property
    def column_count(self) -> int:
        return len(self.columns)


# ---------------------------------------------------------------------------
# Core layout detection
# ---------------------------------------------------------------------------

class LayoutAnalyzer:
    """
    Analyzes a pdfplumber page to detect multi-column layouts.

    The algorithm:
        1. Extract all words with bounding boxes
        2. Build a histogram of word x-positions (left edges)
        3. Find significant gaps in the histogram → column boundaries
        4. Identify full-width header/footer regions where text spans
           across multiple column zones
        5. Return LayoutInfo with column regions for independent extraction

    Usage:
        >>> analyzer = LayoutAnalyzer()
        >>> layout = analyzer.analyze(page)
        >>> if layout.is_multi_column:
        ...     for col in layout.columns:
        ...         cropped = page.crop((col.x0, col.y0, col.x1, col.y1))
        ...         text = cropped.extract_text()
    """

    def __init__(
        self,
        min_column_gap: float = MIN_COLUMN_GAP,
        min_word_fraction: float = MIN_COLUMN_WORD_FRACTION,
        line_y_tolerance: float = LINE_Y_TOLERANCE,
    ):
        self._min_column_gap = min_column_gap
        self._min_word_fraction = min_word_fraction
        self._line_y_tolerance = line_y_tolerance

    def analyze(self, page) -> LayoutInfo:
        """
        Analyze a pdfplumber page for column layout.

        Uses a two-phase approach:
            1. Detect column boundaries via per-line gap voting on ALL
               words (header lines are centered and don't contribute
               gap votes, so they don't interfere).
            2. Detect header region using the established column split.

        Args:
            page: A pdfplumber Page object.

        Returns:
            LayoutInfo describing the page layout.
        """
        page_width = float(page.width)
        page_height = float(page.height)

        # Extract words with bounding boxes
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
        )

        if len(words) < MIN_WORDS_FOR_DETECTION:
            logger.debug(
                "Page has only %d words, skipping layout detection", len(words)
            )
            return LayoutInfo(
                is_multi_column=False,
                page_width=page_width,
                page_height=page_height,
                confidence="high",
                warnings=["Too few words for layout detection"],
            )

        # Step 1: Group all words into lines
        lines = self._group_into_lines(words)

        # Step 2: Detect column boundaries using per-line gap voting
        #         on ALL words. Centered header lines won't produce
        #         column-gap votes, so they don't corrupt detection.
        column_boundaries = self._detect_column_boundaries(
            words, lines, page_width
        )

        if len(column_boundaries) < 2:
            return LayoutInfo(
                is_multi_column=False,
                page_width=page_width,
                page_height=page_height,
                confidence="high",
            )

        # Step 3: Detect header using established column split position
        split_x = column_boundaries[0][1]  # right edge of first column
        header_region, body_y_start = self._detect_header_region(
            words, lines, split_x, page_width, page_height
        )

        # Step 4: Filter body words (below header)
        body_words = [
            w for w in words if float(w["top"]) >= body_y_start
        ]

        if len(body_words) < MIN_WORDS_FOR_DETECTION:
            return LayoutInfo(
                is_multi_column=False,
                page_width=page_width,
                page_height=page_height,
                confidence="high",
                header_region=header_region,
            )

        # Step 5: Assign body words to columns and validate
        columns_words = self._assign_words_to_columns(
            body_words, column_boundaries
        )

        valid_columns = self._validate_columns(
            columns_words, len(body_words), column_boundaries
        )

        if len(valid_columns) < 2:
            return LayoutInfo(
                is_multi_column=False,
                page_width=page_width,
                page_height=page_height,
                confidence="medium",
                header_region=header_region,
                warnings=["Column candidates found but insufficient word count"],
            )

        # Step 6: Build final column regions
        column_regions = []
        for i, (col_x0, col_x1) in enumerate(valid_columns):
            col_words = [
                w for w in body_words
                if float(w["x0"]) >= col_x0
                and float(w["x0"]) < col_x1
            ]
            if not col_words:
                continue

            y0 = min(float(w["top"]) for w in col_words)
            y1 = max(float(w["bottom"]) for w in col_words)

            label = ["left", "right", "third"][i] if i < 3 else f"col_{i}"
            column_regions.append(ColumnRegion(
                x0=col_x0,
                x1=col_x1,
                y0=y0,
                y1=y1,
                word_count=len(col_words),
                label=label,
            ))

        if len(column_regions) < 2:
            return LayoutInfo(
                is_multi_column=False,
                page_width=page_width,
                page_height=page_height,
                confidence="medium",
                header_region=header_region,
                warnings=["Column regions collapsed after filtering"],
            )

        logger.info(
            "Detected %d-column layout (boundaries: %s)",
            len(column_regions),
            [(round(c.x0), round(c.x1)) for c in column_regions],
        )

        return LayoutInfo(
            is_multi_column=True,
            columns=column_regions,
            header_region=header_region,
            footer_region=None,
            page_width=page_width,
            page_height=page_height,
            confidence="high",
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _detect_column_boundaries(
        self, words: list[dict], lines: list[list[dict]], page_width: float
    ) -> list[tuple[float, float]]:
        """
        Detect column boundaries using per-line gap voting.

        For each text line, finds significant horizontal gaps between
        consecutive words. Gaps that appear consistently at the same
        x-position across multiple lines indicate column boundaries.

        This is more robust than the x0-histogram approach because:
        - Individually-spaced characters (e.g. "S K I L L S") don't
          create within-line column gaps (chars are uniformly spaced)
        - Centered header text spanning both columns is ignored
        - Only lines with actual two-column content vote

        Falls back to x0-histogram if line-based voting fails.

        Args:
            words: All words on the page.
            lines: Words grouped into lines (from _group_into_lines).
            page_width: Page width in PDF points.

        Returns:
            List of (col_x0, col_x1) tuples defining non-overlapping
            column boundaries. Returns [(0, page_width)] for single-column.
        """
        # --- Primary: per-line gap voting ---
        gap_edges: list[tuple[float, float]] = []

        for line_words in lines:
            if len(line_words) < 2:
                continue
            sorted_lw = sorted(line_words, key=lambda w: float(w["x0"]))

            for i in range(len(sorted_lw) - 1):
                x1_prev = float(sorted_lw[i]["x1"])
                x0_next = float(sorted_lw[i + 1]["x0"])
                gap_size = x0_next - x1_prev
                if gap_size >= MIN_LINE_GAP:
                    gap_edges.append((x1_prev, x0_next))

        if len(gap_edges) >= MIN_GAP_VOTES:
            boundaries = self._boundaries_from_gap_votes(
                gap_edges, words, page_width
            )
            if len(boundaries) >= 2:
                return boundaries

        # --- Fallback: x0-position histogram ---
        return self._detect_column_boundaries_histogram(words, page_width)

    def _boundaries_from_gap_votes(
        self,
        gap_edges: list[tuple[float, float]],
        words: list[dict],
        page_width: float,
    ) -> list[tuple[float, float]]:
        """
        Convert per-line gap votes into column boundaries.

        Clusters gap positions, then for each cluster uses the tightest
        gap (max of left edges, min of right edges) to find the safe
        split zone where no content exists.
        """
        # Sort by midpoint
        sorted_edges = sorted(
            gap_edges, key=lambda e: (e[0] + e[1]) / 2
        )

        # Cluster gaps whose midpoints are within CLUSTER_TOLERANCE
        clusters: list[list[tuple[float, float]]] = []
        current_cluster: list[tuple[float, float]] = [sorted_edges[0]]

        for i in range(1, len(sorted_edges)):
            prev_mid = (current_cluster[-1][0] + current_cluster[-1][1]) / 2
            curr_mid = (sorted_edges[i][0] + sorted_edges[i][1]) / 2
            if curr_mid - prev_mid <= GAP_CLUSTER_TOLERANCE:
                current_cluster.append(sorted_edges[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_edges[i]]
        clusters.append(current_cluster)

        # Keep clusters with enough supporting lines
        valid_clusters = [c for c in clusters if len(c) >= MIN_GAP_VOTES]

        if not valid_clusters:
            return [(0, page_width)]

        # For each cluster, determine the split position using the
        # tightest observed gap: max(left_edges) to min(right_edges).
        # This finds the narrowest zone guaranteed to have no content
        # on any observed line.
        split_positions: list[float] = []
        for cluster in valid_clusters:
            left_edges = [e[0] for e in cluster]
            right_edges = [e[1] for e in cluster]
            gap_left = max(left_edges)
            gap_right = min(right_edges)

            if gap_right > gap_left:
                split = (gap_left + gap_right) / 2
            else:
                # Overlapping edges (outlier) — use median midpoint
                mids = sorted([(e[0] + e[1]) / 2 for e in cluster])
                split = mids[len(mids) // 2]

            split_positions.append(split)

        split_positions.sort()

        # Build column boundaries
        left_edge = min(float(w["x0"]) for w in words)
        right_edge = max(float(w["x1"]) for w in words)

        boundaries: list[tuple[float, float]] = []
        boundaries.append((left_edge, split_positions[0]))
        for i in range(len(split_positions) - 1):
            boundaries.append((split_positions[i], split_positions[i + 1]))
        boundaries.append((split_positions[-1], right_edge))

        logger.debug(
            "Line-based gap voting found %d split(s) at x=%s (%d votes total)",
            len(split_positions),
            [round(s, 1) for s in split_positions],
            sum(len(c) for c in valid_clusters),
        )

        return boundaries

    def _detect_column_boundaries_histogram(
        self, words: list[dict], page_width: float
    ) -> list[tuple[float, float]]:
        """
        Detect column boundaries by finding significant gaps in word
        starting x-positions.

        Uses the gap between clusters of x0 positions to identify
        where one column ends and the next begins. Column x1 boundaries
        use the gap midpoint rather than the rightmost word extent,
        preventing overlap when a column's text wraps close to the gap.

        Args:
            words: Body-region words only (header already excluded).
            page_width: Page width in PDF points.

        Returns:
            List of (col_x0, col_x1) tuples defining non-overlapping
            column boundaries. Returns [(0, page_width)] for single-column.
        """
        # Collect unique x0 positions
        x_positions = sorted(set(round(float(w["x0"]), 1) for w in words))

        if not x_positions:
            return [(0, page_width)]

        # Find significant gaps between consecutive x0 values
        gaps: list[tuple[float, float, float]] = []
        for i in range(len(x_positions) - 1):
            gap = x_positions[i + 1] - x_positions[i]
            if gap >= self._min_column_gap:
                gaps.append((x_positions[i], x_positions[i + 1], gap))

        if not gaps:
            return [(0, page_width)]

        # Build non-overlapping column boundaries using gap midpoints
        # This prevents the left column's crop from including right-column
        # content, and vice versa.
        boundaries: list[tuple[float, float]] = []

        left_margin = min(float(w["x0"]) for w in words)

        # First column: from left margin to midpoint of first gap
        first_gap_mid = (gaps[0][0] + gaps[0][1]) / 2
        boundaries.append((left_margin, first_gap_mid))

        # Middle columns (for 3+ columns)
        for i in range(len(gaps) - 1):
            col_start = (gaps[i][0] + gaps[i][1]) / 2
            col_end = (gaps[i + 1][0] + gaps[i + 1][1]) / 2
            boundaries.append((col_start, col_end))

        # Last column: from midpoint of last gap to right edge
        last_gap_mid = (gaps[-1][0] + gaps[-1][1]) / 2
        right_margin = max(float(w["x1"]) for w in words)
        boundaries.append((last_gap_mid, right_margin))

        return boundaries

    def _assign_words_to_columns(
        self,
        words: list[dict],
        boundaries: list[tuple[float, float]],
    ) -> list[list[dict]]:
        """Assign each word to its closest column based on x0 position."""
        columns: list[list[dict]] = [[] for _ in boundaries]
        for word in words:
            x0 = float(word["x0"])
            # Find which column this word belongs to
            assigned = False
            for i, (col_x0, col_x1) in enumerate(boundaries):
                # Use midpoint of gap for boundary decision
                if i == 0 and x0 < col_x1:
                    columns[i].append(word)
                    assigned = True
                    break
                elif i > 0:
                    prev_x1 = boundaries[i - 1][1]
                    gap_mid = (prev_x1 + col_x0) / 2
                    if x0 >= gap_mid:
                        if i == len(boundaries) - 1 or x0 < (col_x1 + boundaries[i + 1][0]) / 2:
                            columns[i].append(word)
                            assigned = True
                            break
            if not assigned:
                # Assign to nearest column
                min_dist = float("inf")
                best_col = 0
                for i, (col_x0, col_x1) in enumerate(boundaries):
                    col_mid = (col_x0 + col_x1) / 2
                    dist = abs(x0 - col_mid)
                    if dist < min_dist:
                        min_dist = dist
                        best_col = i
                columns[best_col].append(word)
        return columns

    def _validate_columns(
        self,
        columns_words: list[list[dict]],
        total_words: int,
        boundaries: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Filter out columns that have too few words.

        A column must contain at least MIN_COLUMN_WORD_FRACTION of total
        words to be considered valid. This prevents false column detection
        from stray text in margins.
        """
        min_count = max(3, int(total_words * self._min_word_fraction))
        valid = []
        for i, col_words in enumerate(columns_words):
            if len(col_words) >= min_count:
                valid.append(boundaries[i])
            else:
                logger.debug(
                    "Column %d rejected: %d words < minimum %d",
                    i, len(col_words), min_count,
                )
        return valid

    def _detect_header_region(
        self,
        words: list[dict],
        lines: list[list[dict]],
        split_x: float,
        page_width: float,
        page_height: float,
    ) -> tuple[Optional[ColumnRegion], float]:
        """
        Detect the top-of-page header region before columns begin.

        Uses the established column split position to classify each line:
        - A line that crosses the split WITHOUT a significant gap is
          "centered/spanning" → part of the header (name, title).
        - A line with content only on one side of the split, or that
          crosses the split WITH a significant gap, is column content
          → body starts here.

        The header is the contiguous block of centered/spanning lines
        at the top of the page.

        Returns:
            (header_region or None, body_y_start)
        """
        if len(lines) < 2:
            y_start = min(float(w["top"]) for w in words)
            return None, y_start

        header_y_end = 0.0
        body_start_found = False

        for line_words in lines:
            line_min_x = min(float(w["x0"]) for w in line_words)
            line_max_x = max(float(w["x1"]) for w in line_words)

            # Does this line's content span across the column split?
            crosses_split = line_min_x < split_x and line_max_x > split_x

            if crosses_split:
                # Check whether there's a significant gap at the split
                # (two-column body) or content flows through (header).
                # Use x0 for classification to match column assignment.
                left_words = [
                    w for w in line_words if float(w["x0"]) < split_x
                ]
                right_words = [
                    w for w in line_words if float(w["x0"]) >= split_x
                ]

                if left_words and right_words:
                    left_max_x1 = max(float(w["x1"]) for w in left_words)
                    right_min_x0 = min(float(w["x0"]) for w in right_words)
                    gap = right_min_x0 - left_max_x1

                    if gap >= MIN_LINE_GAP:
                        # Large gap at split → two-column body line
                        body_start_found = True
                        break
                    else:
                        # Content flows through split → header line
                        header_y_end = max(
                            float(w["bottom"]) for w in line_words
                        )
                        continue
                else:
                    # Content crosses split but no clear left/right split
                    header_y_end = max(
                        float(w["bottom"]) for w in line_words
                    )
                    continue
            else:
                # Content is only on one side → body column content
                if header_y_end > 0:
                    # We've already seen header lines; body starts here
                    body_start_found = True
                    break
                else:
                    # No header lines yet — check if we're still in
                    # a plausible header zone (top 25% of page)
                    line_top = min(float(w["top"]) for w in line_words)
                    if line_top > page_height * 0.25:
                        break
                    # Could be a centered element on one side; skip
                    # and keep looking for spanning lines above
                    continue

        if not body_start_found or header_y_end <= 0:
            y_start = min(float(w["top"]) for w in words)
            return None, y_start

        # Build header region from all words above the body start
        header_words = [
            w for w in words if float(w["bottom"]) <= header_y_end + 1
        ]

        if not header_words:
            y_start = min(float(w["top"]) for w in words)
            return None, y_start

        header_region = ColumnRegion(
            x0=min(float(w["x0"]) for w in header_words),
            x1=max(float(w["x1"]) for w in header_words),
            y0=min(float(w["top"]) for w in header_words),
            y1=header_y_end,
            word_count=len(header_words),
            label="header",
        )

        logger.debug(
            "Header region: y=%.0f–%.0f (%d words)",
            header_region.y0, header_y_end, len(header_words),
        )

        return header_region, header_y_end

    def _group_into_lines(self, words: list[dict]) -> list[list[dict]]:
        """
        Group words into lines based on y-position proximity.

        Words within LINE_Y_TOLERANCE vertical distance are considered
        to be on the same line. Lines are returned sorted top-to-bottom.
        """
        if not words:
            return []

        # Sort by top position
        sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))

        lines: list[list[dict]] = []
        current_line: list[dict] = [sorted_words[0]]
        current_top = float(sorted_words[0]["top"])

        for word in sorted_words[1:]:
            word_top = float(word["top"])
            if abs(word_top - current_top) <= self._line_y_tolerance:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                current_top = word_top

        if current_line:
            lines.append(current_line)

        return lines


# ---------------------------------------------------------------------------
# Page-level extraction with layout awareness
# ---------------------------------------------------------------------------

def extract_page_with_layout(
    page,
    analyzer: Optional[LayoutAnalyzer] = None,
) -> tuple[str, LayoutInfo]:
    """
    Extract text from a pdfplumber page with layout-aware column handling.

    If the page has a multi-column layout, each column is extracted
    independently and the results are concatenated with clear separation.
    Single-column pages use standard extraction.

    Column ordering strategy for resumes:
        - Header region first (name, title, contact)
        - Wider column first (typically the main experience column)
        - Narrower column second (typically skills/sidebar)

    This ordering produces the most natural reading flow for resume content.

    Args:
        page: A pdfplumber Page object.
        analyzer: Optional LayoutAnalyzer instance (created if not provided).

    Returns:
        Tuple of (extracted_text, layout_info).
    """
    if analyzer is None:
        analyzer = LayoutAnalyzer()

    layout = analyzer.analyze(page)

    if not layout.is_multi_column:
        # Single column — use the default extraction
        text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
        return text, layout

    # Pre-extract all words — we'll reconstruct text from these rather
    # than re-extracting from crops. This avoids the exact problem we're
    # solving: extract_text() merging words across columns on the same y-line.
    all_words = page.extract_words(
        x_tolerance=3, y_tolerance=3, keep_blank_chars=False,
    )

    # Multi-column: reassemble text from each region independently
    parts: list[str] = []

    # 1. Header region (full-width, top of page)
    if layout.header_region:
        hr = layout.header_region
        header_words = [
            w for w in all_words
            if float(w["top"]) >= hr.y0 - 1
            and float(w["bottom"]) <= hr.y1 + 1
        ]
        header_text = _words_to_text(header_words, analyzer._line_y_tolerance)
        if header_text.strip():
            parts.append(header_text.strip())

    # 2. Body columns — sort by width descending (wider column first)
    #    For resumes, the wider column is typically "Experience" (main content)
    #    and the narrower column is "Skills" / sidebar
    sorted_columns = sorted(layout.columns, key=lambda c: c.width, reverse=True)

    for col in sorted_columns:
        # Select words within this column's bounding box
        col_words = [
            w for w in all_words
            if float(w["x0"]) >= col.x0 - 1
            and float(w["x0"]) < col.x1 + 1
            and float(w["top"]) >= col.y0 - 1
            and float(w["bottom"]) <= col.y1 + 1
        ]
        col_text = _words_to_text(col_words, analyzer._line_y_tolerance)
        if col_text.strip():
            parts.append(col_text.strip())

    combined = "\n\n".join(parts) if parts else ""
    return combined, layout


def _words_to_text(
    words: list[dict],
    y_tolerance: float = LINE_Y_TOLERANCE,
) -> str:
    """
    Reconstruct readable text from a list of pdfplumber word dicts.

    Words are grouped into lines by y-position, then sorted left-to-right
    within each line. Adjacent words are joined with a space. Lines are
    joined with newlines.

    This avoids the column-merging issues of extract_text() by only
    operating on words already filtered to a specific region.

    Args:
        words: List of pdfplumber word dicts (must have x0, top, text keys).
        y_tolerance: Vertical distance for grouping words into lines.

    Returns:
        Reconstructed text string.
    """
    if not words:
        return ""

    # Sort words top-to-bottom, then left-to-right
    sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))

    lines: list[str] = []
    current_line_words: list[dict] = [sorted_words[0]]
    current_top = float(sorted_words[0]["top"])

    for word in sorted_words[1:]:
        word_top = float(word["top"])
        if abs(word_top - current_top) <= y_tolerance:
            current_line_words.append(word)
        else:
            # Finish current line
            lines.append(_assemble_line(current_line_words))
            current_line_words = [word]
            current_top = word_top

    if current_line_words:
        lines.append(_assemble_line(current_line_words))

    return "\n".join(lines)


def _assemble_line(words: list[dict]) -> str:
    """
    Assemble a single line of text from words, inserting spaces
    based on the actual gap between word bounding boxes.

    Detects lines composed primarily of individually-spaced characters
    (common in resume templates with letter-spacing CSS) and uses
    adaptive gap analysis to determine actual word breaks within them.

    For normal word lines, inserts spaces between all words.

    Args:
        words: Words on the same line, not necessarily sorted.

    Returns:
        Single assembled line string.
    """
    if not words:
        return ""

    # Sort left-to-right
    sorted_words = sorted(words, key=lambda w: float(w["x0"]))

    # Detect spaced-character lines: mostly single-char "words"
    single_char_count = sum(1 for w in sorted_words if len(w["text"]) == 1)
    if single_char_count >= 3 and single_char_count >= len(sorted_words) * 0.6:
        return _assemble_spaced_chars(sorted_words)

    # Normal word assembly
    parts: list[str] = [sorted_words[0]["text"]]
    for i in range(1, len(sorted_words)):
        prev_x1 = float(sorted_words[i - 1]["x1"])
        curr_x0 = float(sorted_words[i]["x0"])
        gap = curr_x0 - prev_x1

        # Insert space between words. Even for small gaps, a space
        # is safer than gluing — the cleaning layer can normalize later.
        if gap >= 0:
            parts.append(" ")
        # Negative gap means overlapping bounding boxes (rare),
        # still insert space to avoid gluing
        else:
            parts.append(" ")

        parts.append(sorted_words[i]["text"])

    return "".join(parts)


def _assemble_spaced_chars(words: list[dict]) -> str:
    """
    Assemble a line of individually-spaced characters into readable text.

    Many resume templates use letter-spacing to style section headings
    (e.g. "S K I L L S"). pdfplumber extracts each character as a
    separate "word". This function uses gap analysis to detect actual
    word breaks: gaps significantly larger than the median inter-character
    gap indicate a word boundary.

    Example:
        Individual chars: U S A M A [gap] S H A F I Q U E
        Output: "USAMA SHAFIQUE"

    Args:
        words: Sorted left-to-right words (mostly single characters).

    Returns:
        Assembled text with characters joined and word breaks preserved.
    """
    if not words:
        return ""
    if len(words) == 1:
        return words[0]["text"]

    # Compute gaps between consecutive words
    gaps: list[float] = []
    for i in range(1, len(words)):
        prev_x1 = float(words[i - 1]["x1"])
        curr_x0 = float(words[i]["x0"])
        gaps.append(curr_x0 - prev_x1)

    if not gaps:
        return words[0]["text"]

    # Find the median gap (typical letter spacing)
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]

    # Gaps significantly larger than median indicate word breaks.
    # Use 1.8x median as threshold, with a minimum absolute increase
    # of 5pts to avoid false word breaks on tightly-spaced text.
    word_break_threshold = max(median_gap * 1.8, median_gap + 5.0)

    parts: list[str] = [words[0]["text"]]
    for i in range(1, len(words)):
        if gaps[i - 1] > word_break_threshold:
            parts.append(" ")  # Word break
        # else: join characters directly (no separator)
        parts.append(words[i]["text"])

    return "".join(parts)
