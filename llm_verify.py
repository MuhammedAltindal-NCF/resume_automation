"""
LLM Verification Module

Uses OpenAI's gpt-4o-mini to verify and auto-correct extracted resume data.
This runs automatically and silently after initial extraction.
Two-pass approach: first extraction, then verification/refinement.
"""
import os
import json
import logging
import re
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from models import Resume

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_openai_client():
    """Get OpenAI client, returns None if not configured."""
    # Check multiple possible env var names
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('open_ai_api_key')
    if not api_key:
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def _call_llm(client, prompt: str, system_msg: str, model: str = "gpt-4o-mini", max_tokens: int = 2500) -> Optional[Dict]:
    """Helper to make LLM call and parse JSON response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            # Remove ```json or ``` at start and ``` at end
            result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)

        # Parse JSON response
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Attempting extraction...")
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse extracted JSON")
                    return None
            logger.error(f"No JSON found in response: {result_text[:200]}...")
            return None

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def _normalize_descriptions(items: list) -> list:
    """Normalize 'bullets' key to 'description' for consistency."""
    for item in items:
        if isinstance(item, dict):
            # LLM sometimes uses 'bullets' instead of 'description'
            if 'bullets' in item and 'description' not in item:
                item['description'] = item.pop('bullets')
    return items


def _apply_corrections(resume: Resume, corrections: Dict) -> Resume:
    """Apply corrections dict to resume object."""
    if not corrections:
        return resume

    # Contact fields
    if 'full_name' in corrections:
        resume.full_name = corrections['full_name']
    if 'email' in corrections:
        resume.email = corrections['email']
    if 'phone' in corrections:
        resume.phone = corrections['phone']
    if 'location' in corrections:
        resume.location = corrections['location']

    # Experience - replace if LLM provided extraction
    if 'experience' in corrections and isinstance(corrections['experience'], list):
        resume.experience = _normalize_descriptions(corrections['experience'])

    # Projects - replace if LLM provided extraction
    if 'projects' in corrections and isinstance(corrections['projects'], list):
        resume.projects = _normalize_descriptions(corrections['projects'])

    # Education - replace if LLM provided extraction
    if 'education' in corrections and isinstance(corrections['education'], list):
        resume.education = corrections['education']

    # Skills - replace if LLM provided extraction
    if 'skills' in corrections and isinstance(corrections['skills'], list):
        resume.skills = corrections['skills']

    return resume


def auto_correct_resume(resume: Resume, model: str = "gpt-4o-mini") -> Resume:
    """
    Automatically verify and correct resume extraction using LLM.

    Uses a two-pass approach:
    1. First pass: Extract all structured data from raw text
    2. Second pass: Verify and refine the extraction for completeness

    If the API is unavailable, returns the original resume unchanged.

    Args:
        resume: The Resume object with initial extraction
        model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)

    Returns:
        Resume object with corrected fields
    """
    client = get_openai_client()
    if not client:
        return resume

    raw_text = resume.raw_text[:8000]  # Increased for better context

    # =========================================================================
    # PASS 1: Full extraction from raw text
    # =========================================================================
    pass1_prompt = f"""Extract ALL structured data from this resume text. Parse carefully - do not miss any information.

RESUME TEXT:
{raw_text}

Return a JSON object with ALL of these fields populated:

{{
  "full_name": "string or null if placeholder",
  "email": "string or null if placeholder",
  "phone": "string or null if not found",
  "location": "City, State format or null",
  "skills": ["skill1", "skill2", ...],
  "experience": [
    {{
      "title": "Job Title",
      "company": "Company Name",
      "dates": "Start - End",
      "description": ["bullet 1", "bullet 2", ...]
    }}
  ],
  "education": [
    {{
      "institution": "School Name",
      "degree": "Degree Type",
      "field": "Field of Study",
      "dates": "Date range",
      "gpa": "GPA or null"
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "technologies": ["tech1", "tech2"],
      "description": ["bullet 1", "bullet 2", ...],
      "url": "URL or null"
    }}
  ]
}}

CRITICAL RULES:
1. SKILLS: Extract ALL technical skills from everywhere - job bullets, projects, skills sections. Include: programming languages, frameworks, tools, platforms, concepts.
2. EXPERIENCE: Include ONLY paid work (jobs, internships). Extract ALL bullet points (•) for each job. THE DESCRIPTION ARRAY IS REQUIRED - do not leave it empty!
3. PROJECTS: Look for sections like "Projects", "Projects & Coursework", "Personal Projects". Format may be "Project Name | Tech1, Tech2" or similar. Extract ALL bullet points. THE DESCRIPTION ARRAY IS REQUIRED!
4. VOLUNTEER: Do NOT include volunteer work in experience - skip sections labeled "Volunteer".
5. DESCRIPTIONS ARE MANDATORY: Each bullet point (•) must be extracted as a string in the description array. If a job has 7 bullets, the description array must have 7 strings.
6. URLs: Look for GitHub links, portfolio URLs, demo links in projects section or contact header.
7. Placeholders like [Your Name], [email], [phone] should be null.

EXAMPLE OUTPUT WITH DESCRIPTIONS:
{{
  "experience": [{{
    "title": "Software Engineer Intern",
    "company": "Applied Materials",
    "dates": "May 2025 - Present",
    "description": [
      "Developed a C++ hardware simulator that emulates...",
      "Implemented a SEMI SECS-II/HSMS protocol handler...",
      "Engineered a low-latency messaging pipeline..."
    ]
  }}]
}}

Respond with ONLY the JSON object, no other text."""

    pass1_result = _call_llm(
        client,
        pass1_prompt,
        "You are a precise resume parser. You MUST extract all data including every bullet point. The description arrays MUST NOT be empty. Respond with only valid JSON - no markdown, no explanation.",
        model,
        max_tokens=4500
    )

    if pass1_result:
        resume = _apply_corrections(resume, pass1_result)
        resume.metadata['llm_pass1'] = pass1_result

    # =========================================================================
    # PASS 2: Verification and refinement
    # =========================================================================
    # Build summary of what pass 1 extracted
    exp_details = []
    for exp in resume.experience[:5]:
        desc_count = len(exp.get('description', []))
        exp_details.append(f"  - {exp.get('title')} at {exp.get('company')} ({exp.get('dates')}) [{desc_count} bullets]")

    proj_details = []
    for proj in resume.projects[:5]:
        desc_count = len(proj.get('description', []))
        has_url = "✓ URL" if proj.get('url') else "no URL"
        proj_details.append(f"  - {proj.get('name')} [{', '.join(proj.get('technologies', [])[:3])}] [{desc_count} bullets, {has_url}]")

    edu_details = []
    for edu in resume.education[:3]:
        edu_details.append(f"  - {edu.get('degree')} from {edu.get('institution')}")

    current_extraction = f"""CURRENT EXTRACTION:
Contact: {resume.full_name or 'None'} | {resume.email or 'None'} | {resume.phone or 'None'} | {resume.location or 'None'}
Skills ({len(resume.skills)}): {', '.join(resume.skills[:15])}
Experience ({len(resume.experience)}):
{chr(10).join(exp_details) if exp_details else '  None'}
Education ({len(resume.education)}):
{chr(10).join(edu_details) if edu_details else '  None'}
Projects ({len(resume.projects)}):
{chr(10).join(proj_details) if proj_details else '  None'}"""

    pass2_prompt = f"""Review this resume extraction for completeness. Compare against the original text.

ORIGINAL RESUME TEXT:
{raw_text}

{current_extraction}

VERIFY THESE ITEMS:
1. SKILLS: Are ALL programming languages, frameworks, tools mentioned in the resume captured? Look in experience bullets AND project descriptions.
2. EXPERIENCE: Does each job have ALL its bullet points? Count the bullets in the original text vs extraction.
3. PROJECTS: Does each project have its full name (not just "Project"), technologies, and ALL bullet points?
4. Are project names correct? They should NOT be generic like "Project" - extract actual names like "Operating Systems 1" or "Portfolio App".

If ANY data is missing or wrong, return the COMPLETE corrected version of that field.

Return JSON with corrections (use EXACTLY these field names):
{{
  "skills": ["skill1", "skill2"],
  "experience": [{{"title": "", "company": "", "dates": "", "description": ["bullet1", "bullet2"]}}],
  "projects": [{{"name": "", "technologies": [], "description": ["bullet1"], "url": null}}],
  "education": [{{"institution": "", "degree": "", "field": "", "dates": "", "gpa": null}}]
}}

IMPORTANT: Use "description" NOT "bullets" for the array of bullet points.

If extraction is complete, return {{}}.
Respond with ONLY valid JSON."""

    pass2_result = _call_llm(
        client,
        pass2_prompt,
        "You are a resume extraction verifier. If ANY descriptions are missing or empty, you MUST return the complete corrected data with all bullet points. Respond with only valid JSON.",
        model,
        max_tokens=4500
    )

    if pass2_result:
        resume = _apply_corrections(resume, pass2_result)
        resume.metadata['llm_pass2'] = pass2_result

    # Track that LLM processing completed
    resume.metadata['llm_processed'] = True

    return resume
