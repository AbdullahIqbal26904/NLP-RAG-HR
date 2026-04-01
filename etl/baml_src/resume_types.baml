// ============================================================================
// BAML Type Definitions for Structured Resume Extraction
// ============================================================================
//
// These types map directly to the normalized PostgreSQL database schema.
// Each type corresponds to a database table (minus auto-generated fields
// like IDs, timestamps, and foreign keys which are added at the DB layer).
//
// Design decisions:
//   - All fields that cannot be reliably extracted are optional (nullable)
//   - Dates use "YYYY-MM" string format for partial dates (resumes rarely
//     include exact days). The Python layer converts to DATE as needed.
//   - Enums are used where the DB has CHECK constraints to ensure the LLM
//     only produces valid values.
//   - Skills are extracted as structured objects, not flat strings, to
//     support the candidate_skills table with category and proficiency.
//   - The LLM is NOT asked to fabricate data — missing fields stay null.
// ============================================================================


// ---------------------------------------------------------------------------
// Enums (matching DB CHECK constraints)
// ---------------------------------------------------------------------------

enum EmploymentType {
  FULL_TIME     @alias("full-time")  @description("Full-time employment")
  PART_TIME     @alias("part-time")  @description("Part-time employment")
  CONTRACT      @alias("contract")   @description("Contract or freelance work")
  INTERNSHIP    @alias("internship") @description("Internship position")
}

enum WorkspaceType {
  HYBRID  @alias("Hybrid")  @description("Mix of remote and on-site work")
  REMOTE  @alias("Remote")  @description("Fully remote position")
  ONSITE  @alias("OnSite")  @description("Fully on-site position")
}

enum SkillCategory {
  PROGRAMMING       @alias("Programming")
  DATABASE          @alias("Database")
  CLOUD             @alias("Cloud")
  ML                @alias("ML")
  BI                @alias("Business Intelligence") @description("Business Intelligence and analytics tools: Power BI, Tableau, Looker, QlikView, SSRS, BIRT")
  DEVOPS            @alias("DevOps")
  FRAMEWORK         @alias("Framework")
  TESTING           @alias("Testing")
  DESIGN            @alias("Design")
  MANAGEMENT        @alias("Management")
  DATA_ENGINEERING  @alias("Data Engineering")
  NETWORKING        @alias("Networking")
  SECURITY          @alias("Security")
  MOBILE            @alias("Mobile")
  OTHER             @alias("Other")
}

enum ProficiencyLevel {
  BEGINNER      @alias("beginner")
  INTERMEDIATE  @alias("intermediate")
  ADVANCED      @alias("advanced")
  EXPERT        @alias("expert")
}

enum InstitutionType {
  UNIVERSITY  @alias("university")
  COLLEGE     @alias("college")
  SCHOOL      @alias("school")
  ONLINE      @alias("online")   @description("Online learning platform like Coursera, edX")
}

enum ProjectType {
  ACADEMIC      @alias("academic")
  PROFESSIONAL  @alias("professional")
  PERSONAL      @alias("personal")
  RESEARCH      @alias("research")
}

enum CertificationType {
  CERTIFICATION   @alias("certification")
  COURSE          @alias("course")
  EXAM            @alias("exam")
  ACCREDITATION   @alias("accreditation")
}

enum CertificationLevel {
  BEGINNER      @alias("beginner")
  INTERMEDIATE  @alias("intermediate")
  ADVANCED      @alias("advanced")
  PROFESSIONAL  @alias("professional")
}

enum LanguageProficiency {
  NATIVE        @alias("native")
  FLUENT        @alias("fluent")
  PROFESSIONAL  @alias("professional")
  INTERMEDIATE  @alias("intermediate")
  BASIC         @alias("basic")
}

enum AchievementType {
  AWARD         @alias("award")
  HONOR         @alias("honor")
  SCHOLARSHIP   @alias("scholarship")
  COMPETITION   @alias("competition")
  PUBLICATION   @alias("publication")
}

enum AwardLevel {
  INTERNATIONAL  @alias("international")
  NATIONAL       @alias("national")
  UNIVERSITY     @alias("university")
  DEPARTMENT     @alias("department")
}


// ---------------------------------------------------------------------------
// Nested Types (matching DB tables)
// ---------------------------------------------------------------------------

// Maps to: candidate_work_experience
class WorkExperience {
  company_name     string        @description("Exact company name as written on the resume. Do NOT guess or fabricate.")
  job_title        string        @description("Exact job title/role as written on the resume.")

  from_date        string?       @description("Start date in YYYY-MM format. Use best effort from resume text. Null if not mentioned.")
  end_date         string?       @description("End date in YYYY-MM format. Null if this is the current position or not mentioned.")
  is_current_employer bool       @description("True if this is the candidate's current/present position. Look for keywords like 'Present', 'Current', 'Ongoing'.")

  country          string?       @description("Country where the job is located. Null if not mentioned.")
  city             string?       @description("City where the job is located. Null if not mentioned.")

  employment_type  EmploymentType? @description("Type of employment. Default to full-time for all standard professional/corporate roles. Use internship if title contains 'intern'/'trainee'; contract if 'contractor'/'freelance'/'consultant'; part-time if 'part-time'. Do NOT leave null for regular professional positions.")
  industry         string?       @description("Industry sector of the company. Null if not determinable from resume text.")

  job_description  string?       @description("ALL responsibilities and achievements listed under this position. Include EVERY bullet point and line from the first to the last, joining them with spaces. The description ends only when the next company name, job title, or major section header appears. Do NOT stop after the first few bullets — a 6-bullet role must have all 6 bullets in the output. Null only if there is genuinely no description text.")
  workspace_type   WorkspaceType?  @description("Remote, OnSite, or Hybrid. Only set if explicitly mentioned in resume.")
}


// Maps to: candidate_education
class Education {
  institute_name   string        @description("Full name of the educational institution as written on the resume.")
  degree           string        @description("Full degree name, e.g. 'Bachelor of Science in Computer Science', 'MS Computer Science'.")

  cgpa             float?        @description("GPA or CGPA as a decimal number (e.g. 3.50). Null if not mentioned.")

  is_currently_attending bool    @description("True if the candidate is still enrolled. Look for 'Expected', 'Present', 'Ongoing'.")

  from_date        string?       @description("Enrollment start date in YYYY-MM format. Null if not mentioned.")
  end_date         string?       @description("Graduation/completion date in YYYY-MM format. Null if still attending and no expected date.")

  city             string?       @description("City of the institution. ONLY extract if the city is LITERALLY written in the resume text — either after a comma in the institute name (e.g., 'University of Punjab, Lahore' → 'Lahore') or explicitly stated nearby. Do NOT use your knowledge of where universities are located. If the city is not written in the text, set null.")
  country          string?       @description("Country of the institution. ONLY extract if explicitly written in the resume text. Do NOT infer from the university name or your knowledge of institution locations. Null if not stated.")

  institution_type InstitutionType? @description("Type of institution. Infer from name if obvious (e.g. 'University of X' → university).")
  field_of_study   string?       @description("Major or field of study. E.g. 'Computer Science', 'Data Science'. Null if only degree name is given without a field.")
}


// Maps to: candidate_projects
class Project {
  project_name     string        @description("Name/title of the project as written on the resume.")
  associated_with  string?       @description("What this project is associated with: company name, university name, 'Personal', 'Internship', 'FYP'. Null if unclear.")

  project_type     ProjectType?  @description("Type of the project. Infer from context (university → academic, company → professional, self → personal).")

  project_description string?    @description("ALL description lines and bullet points listed under this project. Include EVERY bullet point from the first to the last, joining them with spaces. Do NOT truncate or summarize — copy every line. Null only if there is genuinely no description text.")

  project_url      string?       @description("Live project URL if mentioned. Null if not present.")
  repository_url   string?       @description("GitHub/GitLab repository URL if mentioned. Null if not present.")

  role_in_project  string?       @description("Candidate's role in the project (e.g. 'Team Lead', 'Developer'). Null if not mentioned.")

  start_date       string?       @description("Project start date in YYYY-MM format. Null if not mentioned.")
  end_date         string?       @description("Project end date in YYYY-MM format. Null if not mentioned.")

  domain           string?       @description("Technical domain: AI, Web, Data Science, Mobile, Systems, etc. Infer from description if clear.")
}


// Maps to: candidate_certifications_examinations
class Certification {
  certification_name      string  @description("Full name of the certification or course as written on the resume.")
  issuing_organization    string? @description("Organization that issued the certification. E.g. 'Microsoft', 'AWS', 'Coursera', 'Google', 'IBM', 'Udemy', 'Coursera'. Infer from the cert name if clearly implied (e.g., 'AWS Auto ML' → 'AWS', 'GCP Specialization' → 'Google', '(IBM)' in name → 'IBM'). Null only if truly unidentifiable.")

  certification_type CertificationType? @description("Type: certification, course, exam, or accreditation. Infer from context.")

  credential_url     string?     @description("URL for credential verification if mentioned. Null if not present.")

  issue_date         string?     @description("Date the certification was issued in YYYY-MM format. Null if not mentioned.")
  expiry_date        string?     @description("Expiry date in YYYY-MM format. Null if not mentioned or if it doesn't expire.")

  score              string?     @description("Score, grade, or pass/fail result. Null if not mentioned.")

  skill_domain       string?     @description("Technical domain the certification covers: Data Engineering, Cloud, SQL, BI, AI, etc.")
  level              CertificationLevel? @description("Difficulty level of the certification. Infer from name/context if obvious, otherwise null.")

  description        string?     @description("Additional details about the certification. Null if not provided.")
}


// Maps to: candidate_skills
class Skill {
  skill_name         string      @description("Name of the skill exactly as recognized. E.g. 'Python', 'SQL', 'Machine Learning', 'Docker'.")
  skill_category     SkillCategory? @description("Category of the skill. Classify based on the skill name.")
  proficiency_level  ProficiencyLevel? @description("Proficiency level. Only set if explicitly stated on the resume, otherwise null. Do NOT guess proficiency.")
  years_of_experience float?     @description("Years of experience with this skill. Only set if explicitly mentioned, otherwise null.")
  is_primary         bool        @description("True if this is a core skill. Set is_primary=true for: skills in the candidate's job title, skills used across multiple roles, skills listed first/prominently in the skills section, or skills mentioned in the profile/summary. Aim to mark 3-6 skills as primary per resume. Set false only for peripheral skills.")
}


// Maps to: candidate_languages
class Language {
  language_name      string      @description("Name of the language, e.g. 'English', 'Urdu', 'Arabic'.")
  proficiency_level  LanguageProficiency? @description("Proficiency level. Use the candidate's own description if provided, otherwise null.")
  certification_name string?     @description("Related certification name like 'IELTS', 'TOEFL'. Null if not mentioned.")
  certification_score string?    @description("Certification score like '7.5', 'C1'. Null if no certification mentioned.")
}


// Maps to: candidate_achievements
class Achievement {
  achievement_type    AchievementType? @description("Type of achievement. Infer from context.")
  achievement_title   string     @description("Title of the achievement as written on the resume.")
  issuing_organization string?   @description("Organization that granted the achievement. Null if not mentioned.")
  achievement_date    string?    @description("Date of the achievement in YYYY-MM format. Null if not mentioned.")
  description         string?    @description("Description or details of the achievement. Null if not provided.")

  // Publication-specific fields
  publication_url     string?    @description("URL of the publication. Null if not a publication or URL not provided.")

  // Competition-specific fields
  competition_name    string?    @description("Name of the competition if this is a competition achievement. Null otherwise.")
  position_rank       string?    @description("Rank or position: '1st', '2nd', 'finalist', 'winner'. Null if not mentioned.")

  // Award-specific fields
  award_level         AwardLevel? @description("Level of the award: international, national, university, department. Null if not determinable.")
}


// ---------------------------------------------------------------------------
// Top-Level Resume Type
// ---------------------------------------------------------------------------

// Maps to: candidate_profiles + all child tables
class Resume {
  // --- Personal Information (candidate_profiles) ---
  first_name         string?     @description("Candidate's first name. Null only if truly unidentifiable.")
  last_name          string?     @description("Candidate's last name. Null only if truly unidentifiable.")
  salutation         string?     @description("Title/salutation: Mr., Ms., Dr., etc. Null if not present on the resume.")
  gender             string?     @description("Gender if explicitly mentioned, inferable from salutation, or strongly implied by a clearly gender-specific first name (e.g., Muhammad/Usama/Ali/Omar/Hassan/Ahmed → 'Male'; Fatima/Ayesha/Hira/Sara/Maryam → 'Female'). Use 'Male' or 'Female'. Null only if ambiguous.")

  email              string?     @description("Email address. Extract exactly as written. Null if not present.")
  phone              string?     @description("Phone number including country code if present. Extract exactly as written. Null if not present.")
  alternate_phone    string?     @description("Secondary phone number if present. Null if not present.")

  linkedin           string?     @description("LinkedIn profile URL. Null if not present.")
  github             string?     @description("GitHub profile URL. Null if not present.")

  nationality        string?     @description("Nationality if mentioned. Null if not present.")
  city               string?     @description("Current city of residence. MUST extract from the address block — city is typically the last named location in the address (e.g., 'House no 1794, Police Line, Abbottabad' → 'Abbottabad'). Never leave null if an address block is present.")
  country            string?     @description("Current country of residence. Infer from: (a) explicit mention, (b) known city (Abbottabad/Lahore/Karachi/Islamabad → Pakistan; Dubai/Abu Dhabi → UAE), (c) phone country code (+92 → Pakistan, +971 → UAE, +966 → Saudi Arabia, +44 → UK, +1 → USA/Canada, +91 → India). Null only if no evidence exists.")
  current_address    string?     @description("Full current address if mentioned. Null if not present.")

  current_role       string?     @description("Current job title or professional headline from the resume summary/header. Null if not present.")
  qualification      string?     @description("Highest qualification, e.g., 'BS Computer Science', 'MBA'. Infer from education section if present.")

  associated_industry string?    @description("Primary industry the candidate works in. MUST infer from job titles with high confidence. E.g., Data Analyst/Data Engineer/BI Analyst → 'Information Technology'; Software Engineer/Developer → 'Software Development'; ML/AI Engineer → 'Artificial Intelligence'; Finance/Accountant → 'Finance'; Healthcare → 'Healthcare'. Do NOT leave null when job titles clearly indicate the industry.")

  // --- Nested Sections ---
  education          Education[]      @description("All education entries found on the resume. Empty array if no education section.")
  experience         WorkExperience[] @description("All work experience entries. Empty array if no experience section. List in chronological order (newest first).")
  projects           Project[]        @description("All projects mentioned. Empty array if no projects section.")
  skills             Skill[]          @description("All skills identified. Extract from skills section AND from experience/project descriptions. Empty array if none found.")
  certifications     Certification[]  @description("All certifications, courses, and exams. Empty array if none found.")
  languages          Language[]       @description("Languages the candidate speaks. Empty array if no language section.")
  achievements       Achievement[]    @description("Awards, honors, publications, scholarships, competition results. Empty array if none found.")
}
