// ============================================================================
// BAML Functions for Structured Resume Extraction
// ============================================================================

function ExtractResume(resume_text: string) -> Resume {
  client ResumeExtractor

  prompt #"
You are an expert resume parser for a production HR technology platform.

Your task is to extract structured data from resume text.

There are TWO types of fields in this task:

────────────────────────────────────────
TYPE A — VERBATIM EXTRACTION (NO MODIFICATION)
────────────────────────────────────────

For all descriptive text fields, you MUST copy the content exactly as written.
You are NOT allowed to rewrite, summarize, improve grammar, or paraphrase.

This applies to:
- job_description
- project_description
- profile / summary text
- objective statements
- certification descriptions
- achievement descriptions

Rules for verbatim extraction:
• Preserve original wording.
• Preserve original capitalization.
• Preserve original phrasing.
• Combine ALL bullet points into a single text block by joining them with spaces.
• Do NOT rephrase.
• Do NOT correct grammar.
• Do NOT remove repetition.
• Do NOT add clarifications.
• Do NOT truncate or summarize descriptions — copy EVERY bullet point.
• Do NOT stop after the first bullet point — include all of them.

If you truncate, skip bullets, or modify the wording in any way, it is a critical error.

────────────────────────────────────────
TYPE B — INTELLIGENT STRUCTURED INFERENCE
────────────────────────────────────────

For structured fields, you ARE allowed to infer intelligently when strongly supported by the resume.
Prioritize high-coverage extraction over excessive nulls, but never fabricate facts.

You MAY infer:
- current_role (from most recent job if not explicitly labeled)
- associated_industry (from company type or responsibilities)
- city and country (from address block)
- institution_type (e.g., university, college)
- skill_category
- project_type (academic, professional, personal, research)
- domain (AI, Data Science, Web, Mobile, etc.)
- is_primary skill based on prominence
- role_in_project (from project bullets/headings)
- employment_type when clearly implied by wording (e.g., "Intern", "Contractor")
- workspace_type only when explicitly mentioned (remote/hybrid/onsite)

However:

• Do NOT invent companies, degrees, or skills.
• Do NOT fabricate dates.
• Do NOT fabricate contact information.
• If something is unclear or weakly implied, return null.

Inference confidence policy:
• High confidence (explicit or obvious context): fill the field.
• Medium confidence (strong contextual evidence): fill the field.
• Low confidence (ambiguous): keep null.

Do not leave inferable structured fields null if there is medium/high confidence evidence.

────────────────────────────────────────
STRICT RULES
────────────────────────────────────────

1. DATE FORMAT:
   All dates must be in "YYYY-MM".
   If only year is given → use "YYYY-01".
   If "Present" → end_date = null and set is_current_employer = true.

2. NULL POLICY:
   Use null only when evidence is absent or ambiguous.
   If there is medium/high confidence contextual evidence, fill the field.
   Do NOT guess randomly.

3. ARRAYS:
   If a section does not exist, return [].

4. URLS:
   Extract exactly as written. Do NOT construct URLs.

5. SKILLS:
   Extract from skills section AND experience AND projects.
   Deduplicate.
   Only set proficiency_level or years_of_experience if explicitly written.
   Set skill_category when it is obvious from the skill itself.

6. EXPERIENCE ORDER:
   Reverse chronological order (newest first).

7. QUALIFICATION:
   Set highest completed or in-progress degree.
   You may normalize degree name for clarity (e.g., BS(CS) → BS Computer Science).

8. CURRENT ROLE FALLBACK:
   If current_role is missing, derive it from the latest experience job_title.

9. PERSONAL LOCATION EXTRACTION:
   CITY: Extract the city name from the address block. The city is the last named
   location in the address before any postal code or country.
   Examples:
     "House no # 1794 Police Line, Abbottabad" → city = "Abbottabad"
     "Flat 5, Block B, F-8, Islamabad" → city = "Islamabad"
     "123 Main St, New York, NY 10001" → city = "New York"
   CRITICAL: NEVER leave city null if an address block is present in the resume.

   COUNTRY: Infer from ANY of these sources — do not require all three:
     (a) Explicit country mention in the address or contact block.
     (b) Known city in the address (Abbottabad/Lahore/Karachi/Islamabad/Rawalpindi → Pakistan;
         Dubai/Abu Dhabi/Sharjah → UAE; Mumbai/Delhi/Bengaluru → India).
     (c) Phone number country code (+92 → Pakistan; +971 → UAE; +966 → Saudi Arabia;
         +44 → United Kingdom; +1 → USA or Canada; +91 → India; +61 → Australia).
   CRITICAL: NEVER leave country null if any of the above evidence exists.

10. GENDER INFERENCE:
    Infer gender from the first name when it is clearly gender-specific (high confidence).
    Common male names: Muhammad, Usama, Usman, Omar, Ahmed, Ali, Hassan, Hamza,
      Bilal, Fahad, Zubair, Tariq, Faisal, Kashif, Adeel, Sohail, Imran, Asad.
    Common female names: Fatima, Ayesha, Zara, Sara, Maria, Amna, Hira, Nadia,
      Sana, Maryam, Rabia, Maham, Iqra, Saima, Noor.
    Set gender = "Male" or gender = "Female" for high-confidence names.
    Leave null only for genuinely ambiguous or gender-neutral names.

11. INDUSTRY INFERENCE:
    Set associated_industry based on job titles and experience with high confidence:
    - Data Analyst / Data Engineer / BI Analyst / Business Intelligence → "Information Technology"
    - Software Engineer / Web Developer / Full Stack / Backend / Frontend → "Software Development"
    - Machine Learning / AI Engineer / Data Scientist → "Artificial Intelligence"
    - Network Engineer / Systems Administrator → "Information Technology"
    - Finance / Accountant / Banking / Investment → "Finance"
    - Healthcare / Medical / Pharma → "Healthcare"
    - Marketing / Digital Marketing / SEO → "Marketing"
    CRITICAL: Do NOT leave associated_industry null when the job title clearly implies an industry.

12. PRIMARY SKILLS:
    Set is_primary = true for the 3-6 skills most central to the candidate's profile:
    - Skills that appear in or directly support the job title
    - Skills mentioned in the profile/summary/objective
    - Skills used across multiple work experience entries
    - Skills listed first in the skills section
    Example: A "Data Analyst/Engineer" with SQL, Python, Power BI, ETL → all four are is_primary=true.
    Do NOT set is_primary=true for every skill — only the most relevant core skills.

13. EMPLOYMENT TYPE DEFAULTS:
    Set employment_type for every work experience entry:
    - FULL_TIME: default for any standard professional/corporate role (senior/manager/engineer/analyst)
    - INTERNSHIP: if title or description contains "intern", "internship", or "trainee"
    - CONTRACT: if "contractor", "freelance", "consultant", "contract" appears
    - PART_TIME: if "part-time" or "part time" appears
    CRITICAL: Do NOT leave employment_type null for standard professional positions.

14. COMPLETE DESCRIPTIONS — NEVER TRUNCATE:
    For EVERY work experience entry: job_description must contain ALL bullet points/lines
    listed under that role in the resume text. Before finalizing your output, mentally
    count the bullets in the source text and verify your output has the same count.
    The description ends only at the next company name, job title, or major section header.
    SAME RULE applies to project_description — include every bullet of every project.
    CRITICAL: If you omit even one bullet point from any role or project, it is a critical error.

────────────────────────────────────────

Resume text to parse:
---
{{ resume_text }}
---

{{ ctx.output_format }}
"#
}
