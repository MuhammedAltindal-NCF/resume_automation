"""
Resume Parser Module

Handles extraction of text and structured information from resume files (PDF, DOCX).
Uses regex patterns and spaCy NLP for contact info, skills, experience, and education extraction.
"""
import io
import re
import logging
from typing import Optional, List, Dict, Any, Tuple

import pdfplumber
from docx import Document

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    logging.warning("spaCy model 'en_core_web_sm' not found. Some NLP features will be limited.")

from models import Resume

logger = logging.getLogger(__name__)

# --- Regex Patterns for Contact Information ---
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

PHONE_PATTERNS = [
    re.compile(r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
    re.compile(r'\([0-9]{3}\)\s*[0-9]{3}[-.\s]?[0-9]{4}'),
    re.compile(r'[0-9]{3}[-.\s][0-9]{3}[-.\s][0-9]{4}'),
    # Shorter format: (123) 456-789 (9 digits total)
    re.compile(r'\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}'),
]

LINKEDIN_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?',
    re.IGNORECASE
)

# --- Section Header Patterns ---
# Match section headers anywhere in text (not just line starts)
# For experience, require WORK or PROFESSIONAL prefix to avoid matching "user experience" etc.
EXPERIENCE_HEADERS = re.compile(
    r'\b(WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT(?:\s+HISTORY)?|CAREER\s+HISTORY)\b',
    re.IGNORECASE
)

EDUCATION_HEADERS = re.compile(
    r'\b(EDUCATION(?:AL\s+BACKGROUND)?|ACADEMIC\s+BACKGROUND|QUALIFICATIONS)\b',
    re.IGNORECASE
)

SKILLS_HEADERS = re.compile(
    r'\b(TECHNICAL\s+SKILLS(?:\s+&\s+TOOLS)?|SKILLS|COMPETENCIES|EXPERTISE|TECHNOLOGIES|PROFICIENCIES)\b',
    re.IGNORECASE
)

PROJECTS_HEADERS = re.compile(
    r'\b(PERSONAL\s+PROJECTS|PROJECTS|PORTFOLIO)\b',
    re.IGNORECASE
)

# --- Date Patterns ---
DATE_PATTERN = re.compile(
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'[\s,]*\d{4}|'
    r'\d{1,2}/\d{4}|'
    r'\d{4}\s*[-–]\s*(?:\d{4}|present|current)',
    re.IGNORECASE
)

# --- Skills Taxonomy ---
TECH_SKILLS = {
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'golang',
    'rust', 'scala', 'kotlin', 'swift', 'php', 'r', 'matlab', 'perl', 'bash', 'shell',
    # Web Technologies
    'html', 'css', 'react', 'reactjs', 'react.js', 'angular', 'vue', 'vuejs', 'vue.js',
    'node', 'nodejs', 'node.js', 'express', 'django', 'flask', 'fastapi', 'spring',
    'asp.net', '.net', 'rails', 'ruby on rails', 'next.js', 'nextjs', 'nuxt',
    # Databases
    'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch',
    'cassandra', 'dynamodb', 'oracle', 'sqlite', 'mariadb', 'neo4j', 'graphql',
    # Cloud & DevOps
    'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
    'k8s', 'terraform', 'ansible', 'jenkins', 'ci/cd', 'github actions', 'gitlab ci',
    'circleci', 'travis ci', 'helm', 'prometheus', 'grafana',
    # Data & ML
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
    'sklearn', 'pandas', 'numpy', 'scipy', 'spark', 'pyspark', 'hadoop', 'airflow',
    'tableau', 'power bi', 'looker', 'data analysis', 'data science', 'nlp',
    'natural language processing', 'computer vision', 'opencv',
    # Tools & Misc
    'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'agile', 'scrum',
    'linux', 'unix', 'windows', 'macos', 'rest', 'restful', 'api', 'microservices',
    'kafka', 'rabbitmq', 'celery', 'nginx', 'apache', 'graphql', 'grpc',
}

SOFT_SKILLS = {
    'leadership', 'communication', 'teamwork', 'problem solving', 'problem-solving',
    'critical thinking', 'project management', 'time management', 'analytical',
    'collaboration', 'adaptability', 'creativity', 'attention to detail',
    'interpersonal', 'presentation', 'negotiation', 'decision making', 'strategic',
    'mentoring', 'coaching', 'cross-functional', 'stakeholder management',
}

# Skill normalization mapping
SKILL_ALIASES = {
    'js': 'javascript',
    'ts': 'typescript',
    'py': 'python',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'ai': 'artificial intelligence',
    'react.js': 'react',
    'reactjs': 'react',
    'vue.js': 'vue',
    'vuejs': 'vue',
    'node.js': 'node',
    'nodejs': 'node',
    'postgres': 'postgresql',
    'k8s': 'kubernetes',
    'sklearn': 'scikit-learn',
    'gcp': 'google cloud',
    'aws': 'amazon web services',
}


# --- PDF Text Extraction ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_bytes: Raw bytes of the PDF file

    Returns:
        Extracted text as a string, or empty string if extraction fails
    """
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return '\n'.join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            raise ValueError("PDF is encrypted or password-protected")
        raise ValueError(f"Failed to extract text from PDF: {e}")


# --- DOCX Text Extraction ---
def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text content from a DOCX file.

    Args:
        file_bytes: Raw bytes of the DOCX file

    Returns:
        Extracted text as a string
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
        text_parts = []

        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_parts.append(' | '.join(row_text))

        return '\n'.join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise ValueError(f"Failed to extract text from DOCX: {e}")


# --- Unified Parser Interface ---
def parse_resume(file_bytes: bytes, file_type: str) -> str:
    """
    Parse a resume file and extract text content.

    Args:
        file_bytes: Raw bytes of the file
        file_type: File extension/type ('pdf' or 'docx')

    Returns:
        Cleaned, normalized text ready for further processing
    """
    file_type = file_type.lower().strip('.')

    if file_type == 'pdf':
        raw_text = extract_text_from_pdf(file_bytes)
    elif file_type in ('docx', 'doc'):
        raw_text = extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Supported types: pdf, docx")

    # Clean and normalize text
    cleaned = clean_resume_text(raw_text)
    logger.info(f"Extracted {len(cleaned)} characters from {file_type} file")

    return cleaned


def clean_resume_text(text: str) -> str:
    """Clean and normalize resume text."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Preserve line breaks for section detection
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# --- Contact Information Extraction ---
def extract_email(text: str) -> Optional[str]:
    """Extract email address from text."""
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text."""
    for pattern in PHONE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def extract_linkedin(text: str) -> Optional[str]:
    """Extract LinkedIn URL from text."""
    match = LINKEDIN_PATTERN.search(text)
    return match.group(0) if match else None


def extract_name_with_spacy(text: str) -> Optional[str]:
    """Extract person name using spaCy NER."""
    if nlp is None:
        return None

    # Check first 500 chars - name is usually at the top
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    return None


def extract_location_with_spacy(text: str) -> Optional[str]:
    """Extract location using spaCy NER."""
    # First try regex for "City, ST" pattern in header (first 500 chars)
    location_pattern = re.compile(
        r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s*([A-Z]{2})\b'
    )
    header_text = text[:500]
    loc_match = location_pattern.search(header_text)
    if loc_match:
        city = loc_match.group(1)
        state = loc_match.group(2)
        # Filter out false positives (degree abbreviations)
        if state not in ('BS', 'BA', 'MS', 'MA', 'MD', 'DO', 'JD', 'PhD'):
            return f"{city}, {state}"

    if nlp is None:
        return None

    # Check first 1000 chars for location
    doc = nlp(text[:1000])
    locations = []
    # Filter out common false positives
    exclude_locations = {'B.S.', 'BS', 'B.A.', 'BA', 'M.S.', 'MS', 'M.A.', 'MA', 'PhD', 'Ph.D.'}
    for ent in doc.ents:
        if ent.label_ in ('GPE', 'LOC') and ent.text not in exclude_locations:
            locations.append(ent.text)

    # Return first unique location found
    return locations[0] if locations else None


# --- Skills Extraction ---
def normalize_skill(skill: str) -> str:
    """Normalize a skill name to its canonical form."""
    skill_lower = skill.lower().strip()
    return SKILL_ALIASES.get(skill_lower, skill_lower)


def extract_skills(text: str) -> List[str]:
    """
    Extract skills from resume text using taxonomy matching.

    Args:
        text: Resume text

    Returns:
        List of unique, normalized skill names
    """
    text_lower = text.lower()
    found_skills = set()

    # Match against tech skills taxonomy
    for skill in TECH_SKILLS:
        # Word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_skills.add(normalize_skill(skill))

    # Match against soft skills
    for skill in SOFT_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_skills.add(skill)

    return sorted(list(found_skills))


# --- Section Detection ---
def find_section_boundaries(text: str) -> Dict[str, Tuple[int, int]]:
    """
    Find the start and end positions of major resume sections.

    Returns:
        Dictionary mapping section names to (start, end) positions
    """
    sections = {}
    section_patterns = [
        ('experience', EXPERIENCE_HEADERS),
        ('education', EDUCATION_HEADERS),
        ('skills', SKILLS_HEADERS),
        ('projects', PROJECTS_HEADERS),
    ]

    all_matches = []
    for section_name, pattern in section_patterns:
        for match in pattern.finditer(text):
            all_matches.append((match.start(), section_name, match.end()))

    # Sort by position
    all_matches.sort(key=lambda x: x[0])

    # Determine section boundaries
    for i, (start, name, header_end) in enumerate(all_matches):
        if i + 1 < len(all_matches):
            end = all_matches[i + 1][0]
        else:
            end = len(text)
        # Only keep first occurrence of each section type
        if name not in sections:
            sections[name] = (header_end, end)

    return sections


# --- Experience Extraction ---
def extract_experience(text: str) -> List[Dict[str, Any]]:
    """
    Extract work experience entries from resume text.

    Returns:
        List of experience dictionaries with keys:
        - company: Company name
        - title: Job title
        - dates: Date range string
        - description: List of bullet points/responsibilities
    """
    sections = find_section_boundaries(text)

    if 'experience' not in sections:
        return []

    start, end = sections['experience']
    experience_text = text[start:end]

    experiences = []

    # Date range pattern: "Month Year – Month Year" or "Month Year - Present"
    date_range_pattern = re.compile(
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})'
        r'\s*[-–]\s*'
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|Present|Current)',
        re.IGNORECASE
    )

    # Job title patterns - common titles
    title_pattern = re.compile(
        r'\b((?:Senior|Junior|Lead|Staff|Principal|Associate|Assistant)?\s*'
        r'(?:Software|Frontend|Backend|Full[\s-]?Stack|Data|DevOps|Cloud|Machine Learning|ML|AI)?\s*'
        r'(?:Engineer|Developer|Architect|Manager|Analyst|Consultant|Scientist|Director|'
        r'Specialist|Administrator|Designer|Intern|Coordinator|Lead|Superintendent|Supervisor|'
        r'Vice President|VP|CEO|CTO|CFO|COO))\b',
        re.IGNORECASE
    )

    # Find all date ranges in experience section
    date_matches = list(date_range_pattern.finditer(experience_text))

    if date_matches:
        for i, date_match in enumerate(date_matches):
            date_str = date_match.group(0)
            date_start = date_match.start()

            # Get text before this date (company and title info)
            if i == 0:
                before_text = experience_text[:date_start]
            else:
                prev_end = date_matches[i - 1].end()
                before_text = experience_text[prev_end:date_start]

            # Get text after this date (bullet points)
            if i + 1 < len(date_matches):
                after_text = experience_text[date_match.end():date_matches[i + 1].start()]
            else:
                after_text = experience_text[date_match.end():]

            # Extract job title from before text
            title_matches = title_pattern.findall(before_text)
            title = title_matches[-1].strip() if title_matches else None

            # Extract company - try pattern first, then spaCy
            company = None

            # Try pattern: "Company, City, STATE" or "Company, STATE" before job title
            # Look for text that appears before the city/state pattern
            company_loc_pattern = re.compile(
                r'([A-Za-z][A-Za-z\s&\.\-]+?),\s*(?:[A-Za-z\s]+,\s*)?([A-Z]{2})\s',
                re.IGNORECASE
            )
            comp_match = company_loc_pattern.search(before_text)
            if comp_match:
                company = comp_match.group(1).strip()

            # If no company found, try spaCy
            if not company and nlp:
                doc = nlp(before_text[-200:] if len(before_text) > 200 else before_text)
                orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
                if orgs:
                    company = orgs[0]

            # Final fallback: look for capitalized words before the title
            if not company and title:
                fallback_pattern = re.compile(r'([A-Z][A-Za-z\s&]+?)\s+' + re.escape(title), re.IGNORECASE)
                fallback_match = fallback_pattern.search(before_text)
                if fallback_match:
                    company = fallback_match.group(1).strip()

            # Extract bullet points from after_text
            bullets = []
            bullet_pattern = re.compile(r'[•\-\*]\s*(.+?)(?=[•\-\*]|$)')
            bullet_matches = bullet_pattern.findall(after_text)
            bullets = [b.strip() for b in bullet_matches if len(b.strip()) > 10][:5]

            entry = {
                'company': company,
                'title': title,
                'dates': date_str,
                'description': bullets
            }
            experiences.append(entry)

    # Fallback to old behavior if no date ranges found
    elif nlp:
        date_matches = list(DATE_PATTERN.finditer(experience_text))
        doc = nlp(experience_text)
        orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']

        for i, date_match in enumerate(date_matches[:3]):  # Limit to 3 entries
            entry = {
                'company': orgs[i] if i < len(orgs) else None,
                'title': None,
                'dates': date_match.group(0),
                'description': []
            }
            experiences.append(entry)

    return experiences


# --- Education Extraction ---
def extract_education(text: str) -> List[Dict[str, Any]]:
    """
    Extract education entries from resume text.

    Returns:
        List of education dictionaries with keys:
        - institution: School/University name
        - degree: Degree type (BS, MS, PhD, etc.)
        - field: Field of study
        - dates: Graduation date or date range
        - gpa: GPA if present
    """
    sections = find_section_boundaries(text)

    if 'education' not in sections:
        return []

    start, end = sections['education']
    education_text = text[start:end]

    educations = []

    # University/College pattern - include well-known universities and common patterns
    # This pattern handles "X University", "University of X", "X State University", and known names
    university_pattern = re.compile(
        r'\b((?:Oregon State|Northeastern|Stanford|MIT|Harvard|Yale|Berkeley|UCLA|USC|NYU|'
        r'Columbia|Cornell|Princeton|Duke|Northwestern|Carnegie Mellon|Georgia Tech|'
        r'Boston|Michigan|Texas|Penn State|Ohio State|Florida State|Arizona State|'
        r'Washington|Virginia|Illinois|Wisconsin|Minnesota|Purdue|Maryland|'
        r'Rutgers|Indiana|Iowa State|Colorado|Tennessee|Kentucky|Alabama|LSU|'
        r'Notre Dame|Georgetown|Brown|Dartmouth|Vanderbilt|Rice|Emory)\s+'
        r'(?:University|College|Institute)|'
        r'University\s+of\s+[A-Z][a-z]+(?:\s+[A-Za-z]+)*|'
        r'[A-Z][a-z]+\s+State\s+University|'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:College|Institute|Technical Institute))\b',
        re.IGNORECASE
    )

    # Simpler fallback pattern
    simple_uni_pattern = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+University)\b'
    )

    # Degree patterns - expanded to catch more formats
    # Be careful not to match state abbreviations like MA, OR, etc.
    degree_pattern = re.compile(
        r'(Bachelor(?:\s+of\s+(?:Science|Arts))?|Master(?:\s+of\s+(?:Science|Arts))?|'
        r'B\.S\.?|B\.A\.?|M\.S\.?|M\.A\.|Ph\.?D\.?|MBA|Associate|'
        r'Postbaccalaureate|Doctorate)',
        re.IGNORECASE
    )

    # Field of study - follows "in" after degree
    field_pattern = re.compile(
        r'(?:Bachelor|Master|B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?)\s+(?:of\s+\w+\s+)?in\s+([A-Za-z\s]+?)(?:\s*[\|,\(]|$)',
        re.IGNORECASE
    )

    # GPA pattern
    gpa_pattern = re.compile(r'GPA[:\s]*([0-4]\.[0-9]{1,2})', re.IGNORECASE)

    # Date pattern for education (month year or just year)
    edu_date_pattern = re.compile(
        r'(?:Expected\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|'
        r'\b\d{4}\b',
        re.IGNORECASE
    )

    # Find all universities using both patterns
    universities = university_pattern.findall(education_text)
    if not universities:
        universities = simple_uni_pattern.findall(education_text)

    # Clean up university names - remove false positives
    exclude_words = {'bachelor', 'master', 'education', 'science', 'arts', 'engineering',
                    'computer', 'data', 'business', 'management', 'associate', 'doctor'}
    cleaned_universities = []
    for u in universities:
        u = u.strip()
        if not u:
            continue
        first_word = u.split()[0].lower() if u.split() else ''
        if first_word not in exclude_words:
            cleaned_universities.append(u)
    universities = cleaned_universities

    if universities:
        # Split education text by university names to get each entry
        for i, uni in enumerate(universities):
            uni_clean = uni.strip()

            # Find the text chunk for this university
            uni_start = education_text.find(uni)
            if i + 1 < len(universities):
                next_uni_start = education_text.find(universities[i + 1])
                chunk = education_text[uni_start:next_uni_start]
            else:
                chunk = education_text[uni_start:]

            # Extract details from this chunk
            degree_matches = degree_pattern.findall(chunk)
            field_matches = field_pattern.findall(chunk)
            date_matches = edu_date_pattern.findall(chunk)
            gpa_match = gpa_pattern.search(chunk)

            entry = {
                'institution': uni_clean,
                'degree': degree_matches[0] if degree_matches else None,
                'field': field_matches[0].strip() if field_matches else None,
                'dates': date_matches[0] if date_matches else None,
                'gpa': gpa_match.group(1) if gpa_match else None
            }
            educations.append(entry)
    else:
        # Fallback: try spaCy if no universities found with regex
        if nlp:
            doc = nlp(education_text)
            institutions = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            degree_matches = degree_pattern.findall(education_text)
            gpa_match = gpa_pattern.search(education_text)

            if institutions or degree_matches:
                entry = {
                    'institution': institutions[0] if institutions else None,
                    'degree': degree_matches[0] if degree_matches else None,
                    'field': None,
                    'dates': None,
                    'gpa': gpa_match.group(1) if gpa_match else None
                }
                educations.append(entry)

    return educations


# --- Projects Extraction ---
def extract_projects(text: str) -> List[Dict[str, Any]]:
    """
    Extract project entries from resume text.

    Returns:
        List of project dictionaries with keys:
        - name: Project name
        - technologies: List of technologies used
        - description: List of bullet points describing the project
    """
    sections = find_section_boundaries(text)

    if 'projects' not in sections:
        return []

    start, end = sections['projects']
    projects_text = text[start:end]

    projects = []

    # Pattern for project name with technologies in parentheses
    # e.g., "Festival Playlist Creator for Spotify (Java | JavaScript | React):"
    project_pattern = re.compile(
        r'([A-Z][A-Za-z0-9\s\-]+?)[\s]*\(([^)]+)\)[:\s]*',
        re.IGNORECASE
    )

    # Find all project headers
    project_matches = list(project_pattern.finditer(projects_text))

    if project_matches:
        for i, match in enumerate(project_matches):
            name = match.group(1).strip()
            tech_str = match.group(2)

            # Parse technologies (split by | or ,)
            technologies = [t.strip() for t in re.split(r'[|,]', tech_str) if t.strip()]

            # Get description (bullet points) - text between this project and next
            desc_start = match.end()
            if i + 1 < len(project_matches):
                desc_end = project_matches[i + 1].start()
            else:
                desc_end = len(projects_text)

            desc_text = projects_text[desc_start:desc_end]

            # Extract bullet points
            bullets = []
            bullet_pattern = re.compile(r'[•\-\*]\s*(.+?)(?=[•\-\*]|$)', re.DOTALL)
            for bullet_match in bullet_pattern.finditer(desc_text):
                bullet = bullet_match.group(1).strip()
                # Clean up and limit length
                bullet = re.sub(r'\s+', ' ', bullet)
                if len(bullet) > 20:  # Skip very short bullets
                    bullets.append(bullet[:300])  # Limit bullet length

            project = {
                'name': name,
                'technologies': technologies,
                'description': bullets[:5]  # Limit to 5 bullets per project
            }
            projects.append(project)

    return projects


# --- Summary Extraction ---
def extract_summary(text: str) -> Optional[str]:
    """Extract professional summary/objective from resume."""
    summary_pattern = re.compile(
        r'(?:summary|objective|profile|about\s*me)[:\s]*(.+?)(?=\n\n|experience|education|skills)',
        re.IGNORECASE | re.DOTALL
    )

    match = summary_pattern.search(text[:2000])  # Summary is usually near the top
    if match:
        summary = match.group(1).strip()
        # Limit to reasonable length
        if len(summary) > 50:
            return summary[:500] + '...' if len(summary) > 500 else summary

    return None


# --- Main Orchestrator ---
def extract_resume_details(raw_text: str) -> Resume:
    """
    Extract all structured information from resume text.

    Args:
        raw_text: Raw text extracted from resume file

    Returns:
        Populated Resume object
    """
    extraction_methods = {'regex': [], 'spacy': []}

    # Extract contact information
    email = extract_email(raw_text)
    if email:
        extraction_methods['regex'].append('email')

    phone = extract_phone(raw_text)
    if phone:
        extraction_methods['regex'].append('phone')

    linkedin = extract_linkedin(raw_text)
    if linkedin:
        extraction_methods['regex'].append('linkedin')

    name = extract_name_with_spacy(raw_text)
    if name:
        extraction_methods['spacy'].append('name')

    location = extract_location_with_spacy(raw_text)
    if location:
        extraction_methods['spacy'].append('location')

    # Extract sections
    skills = extract_skills(raw_text)
    if skills:
        extraction_methods['regex'].append('skills')

    experience = extract_experience(raw_text)
    if experience:
        extraction_methods['spacy'].append('experience')

    education = extract_education(raw_text)
    if education:
        extraction_methods['spacy'].append('education')

    projects = extract_projects(raw_text)
    if projects:
        extraction_methods['regex'].append('projects')

    # Build metadata
    metadata = {
        'extraction_methods': extraction_methods,
        'linkedin_url': linkedin,
    }

    return Resume(
        full_name=name,
        email=email,
        phone=phone,
        location=location,
        summary=None,  # Not used for American resumes
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        raw_text=raw_text,
        metadata=metadata
    )
