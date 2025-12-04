"""
Resume Analysis Module

Compares resumes against job listings, calculates match scores,
and generates optimization suggestions.
"""
import re
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

from models import Resume, JobListing, AnalysisResult
from resume_parser import TECH_SKILLS, SOFT_SKILLS, normalize_skill, SKILL_ALIASES


# Skill similarity mappings for fuzzy matching
SIMILAR_SKILLS = {
    'react': {'reactjs', 'react.js', 'react native'},
    'node': {'nodejs', 'node.js'},
    'vue': {'vuejs', 'vue.js'},
    'angular': {'angularjs', 'angular.js'},
    'python': {'python3', 'python2'},
    'javascript': {'js', 'ecmascript', 'es6'},
    'typescript': {'ts'},
    'machine learning': {'ml', 'machine-learning'},
    'deep learning': {'dl', 'deep-learning'},
    'postgresql': {'postgres', 'psql'},
    'kubernetes': {'k8s'},
    'amazon web services': {'aws'},
    'google cloud': {'gcp', 'google cloud platform'},
    'ci/cd': {'cicd', 'continuous integration', 'continuous deployment'},
}


def extract_skills_from_job(job: JobListing) -> List[str]:
    """
    Extract skills mentioned in a job listing.

    Args:
        job: The JobListing object to analyze.

    Returns:
        List of normalized skill names found in the job description.
    """
    text = job.description.lower()
    found_skills = set()

    # Match against tech skills
    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(normalize_skill(skill))

    # Match against soft skills
    for skill in SOFT_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(skill)

    return sorted(list(found_skills))


def skills_match(skill1: str, skill2: str) -> bool:
    """
    Check if two skills are equivalent (exact or fuzzy match).

    Args:
        skill1: First skill name
        skill2: Second skill name

    Returns:
        True if skills match, False otherwise
    """
    s1 = normalize_skill(skill1.lower())
    s2 = normalize_skill(skill2.lower())

    # Exact match
    if s1 == s2:
        return True

    # Check similarity mappings
    for canonical, variants in SIMILAR_SKILLS.items():
        all_variants = {canonical} | variants
        if s1 in all_variants and s2 in all_variants:
            return True

    return False


def compare_skills(
    resume_skills: List[str],
    job_skills: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Compare resume skills against job requirements.

    Args:
        resume_skills: List of skills from the resume
        job_skills: List of skills required by the job

    Returns:
        Tuple of (matching_skills, missing_skills)
    """
    matching = []
    missing = []

    resume_skills_normalized = {normalize_skill(s) for s in resume_skills}

    for job_skill in job_skills:
        job_skill_normalized = normalize_skill(job_skill)
        found = False

        # Check for exact or fuzzy match
        for resume_skill in resume_skills_normalized:
            if skills_match(resume_skill, job_skill_normalized):
                matching.append(job_skill)
                found = True
                break

        if not found:
            missing.append(job_skill)

    return matching, missing


def calculate_match_score(
    matching_skills: List[str],
    missing_skills: List[str],
    resume: Resume,
    job: JobListing
) -> float:
    """
    Calculate an overall match score between resume and job.

    Score components:
    - Skills match (60% weight)
    - Experience relevance (20% weight)
    - Education match (10% weight)
    - Keyword overlap (10% weight)

    Args:
        matching_skills: Skills that match between resume and job
        missing_skills: Skills required by job but missing from resume
        resume: The Resume object
        job: The JobListing object

    Returns:
        Match score as a percentage (0-100)
    """
    total_job_skills = len(matching_skills) + len(missing_skills)

    # Skills score (60% weight)
    if total_job_skills > 0:
        skills_score = (len(matching_skills) / total_job_skills) * 60
    else:
        skills_score = 30  # Neutral if no skills identified

    # Experience score (20% weight) - based on keyword overlap
    experience_score = 0
    if resume.experience:
        job_text_lower = job.description.lower()
        experience_keywords = set()
        for exp in resume.experience:
            if exp.get('company'):
                experience_keywords.add(exp['company'].lower())
            if exp.get('title'):
                experience_keywords.add(exp['title'].lower())

        matches = sum(1 for kw in experience_keywords if kw in job_text_lower)
        if experience_keywords:
            experience_score = min((matches / len(experience_keywords)) * 20, 20)

    # Education score (10% weight)
    education_score = 5  # Default base score
    if resume.education:
        # Check if job mentions degree requirements
        job_lower = job.description.lower()
        degree_keywords = ['bachelor', 'master', 'phd', 'degree', 'bs', 'ms', 'ba', 'ma']
        if any(kw in job_lower for kw in degree_keywords):
            education_score = 10  # Full score if education present and job requires it

    # Keyword overlap score (10% weight)
    keyword_score = _calculate_keyword_overlap(resume.raw_text, job.description) * 10

    total_score = skills_score + experience_score + education_score + keyword_score
    return round(min(total_score, 100), 1)


def _calculate_keyword_overlap(resume_text: str, job_text: str) -> float:
    """Calculate keyword overlap between resume and job description."""
    # Extract words (3+ characters, alphanumeric)
    resume_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume_text.lower()))
    job_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', job_text.lower()))

    # Remove common stopwords
    stopwords = {
        'the', 'and', 'for', 'with', 'you', 'your', 'that', 'this', 'are', 'will',
        'have', 'has', 'from', 'our', 'can', 'all', 'what', 'about', 'more', 'work',
        'experience', 'team', 'ability', 'skills', 'including', 'working', 'strong',
    }
    resume_words -= stopwords
    job_words -= stopwords

    if not job_words:
        return 0.5

    overlap = len(resume_words & job_words)
    return min(overlap / len(job_words), 1.0)


def extract_important_keywords(job: JobListing, top_n: int = 20) -> List[str]:
    """
    Extract the most important keywords from a job description.

    Uses term frequency to identify key terms.

    Args:
        job: The JobListing object
        top_n: Number of top keywords to return

    Returns:
        List of important keywords
    """
    text = job.description.lower()

    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)

    # Remove stopwords
    stopwords = {
        'the', 'and', 'for', 'with', 'you', 'your', 'that', 'this', 'are', 'will',
        'have', 'has', 'from', 'our', 'can', 'all', 'what', 'about', 'more', 'work',
        'experience', 'team', 'ability', 'skills', 'including', 'working', 'strong',
        'looking', 'join', 'company', 'role', 'position', 'job', 'opportunity',
        'responsibilities', 'requirements', 'qualifications', 'benefits',
    }
    words = [w for w in words if w not in stopwords]

    # Count frequencies
    word_counts = Counter(words)

    # Return top keywords
    return [word for word, _ in word_counts.most_common(top_n)]


def identify_missing_keywords(
    resume: Resume,
    job: JobListing,
    top_n: int = 10
) -> List[str]:
    """
    Identify important job keywords missing from the resume.

    Args:
        resume: The Resume object
        job: The JobListing object
        top_n: Maximum number of keywords to return

    Returns:
        List of missing keywords that should be added to the resume
    """
    job_keywords = set(extract_important_keywords(job, top_n=30))
    resume_text_lower = resume.raw_text.lower()

    missing = []
    for keyword in job_keywords:
        if keyword not in resume_text_lower:
            missing.append(keyword)
            if len(missing) >= top_n:
                break

    return missing


def generate_improvement_suggestions(
    resume: Resume,
    job: JobListing,
    matching_skills: List[str],
    missing_skills: List[str],
    match_score: float
) -> List[str]:
    """
    Generate actionable improvement suggestions based on the analysis.

    Args:
        resume: The Resume object
        job: The JobListing object
        matching_skills: Skills that match
        missing_skills: Skills that are missing
        match_score: The calculated match score

    Returns:
        List of improvement suggestions
    """
    suggestions = []

    # Missing skills suggestions
    if missing_skills:
        critical_skills = missing_skills[:5]  # Top 5 missing skills
        if len(critical_skills) == 1:
            suggestions.append(
                f"Add '{critical_skills[0]}' to your skills section if you have experience with it."
            )
        else:
            skills_list = ', '.join(f"'{s}'" for s in critical_skills)
            suggestions.append(
                f"Consider adding these key skills to your resume: {skills_list}"
            )

    # Experience suggestions
    if not resume.experience:
        suggestions.append(
            "Add a work experience section with bullet points describing your achievements."
        )
    else:
        # Check for quantifiable achievements
        has_numbers = bool(re.search(r'\d+%|\$\d+|\d+ (years?|months?|projects?)', resume.raw_text))
        if not has_numbers:
            suggestions.append(
                "Add quantifiable achievements (e.g., 'Increased sales by 20%', 'Managed team of 5')."
            )

    # Education suggestions
    if not resume.education:
        job_lower = job.description.lower()
        if any(w in job_lower for w in ['degree', 'bachelor', 'master', 'phd', 'education']):
            suggestions.append(
                "Add an education section as the job listing mentions educational requirements."
            )

    # Summary suggestions
    if not resume.summary:
        suggestions.append(
            "Add a professional summary at the top highlighting your key qualifications."
        )

    # Length check
    word_count = len(resume.raw_text.split())
    if word_count < 200:
        suggestions.append(
            "Your resume may be too brief. Consider adding more detail about your experience."
        )
    elif word_count > 1000:
        suggestions.append(
            "Your resume may be too long. Consider condensing to highlight the most relevant experience."
        )

    # Contact info check
    if not resume.email:
        suggestions.append("Ensure your email address is clearly visible on the resume.")

    # Score-based suggestions
    if match_score < 50:
        suggestions.append(
            "Consider customizing your resume to better align with this specific job's requirements."
        )

    return suggestions


def analyze_resume_against_job(
    resume: Resume,
    job: JobListing,
    resume_id: int = 0,
    job_listing_id: int = 0
) -> AnalysisResult:
    """
    Perform a complete analysis of a resume against a job listing.

    Args:
        resume: The Resume object to analyze
        job: The JobListing to compare against
        resume_id: Database ID of the resume (for persistence)
        job_listing_id: Database ID of the job (for persistence)

    Returns:
        An AnalysisResult object containing all analysis data
    """
    # Extract skills from job
    job_skills = extract_skills_from_job(job)

    # Compare skills
    matching_skills, missing_skills = compare_skills(resume.skills, job_skills)

    # Calculate match score
    match_score = calculate_match_score(
        matching_skills, missing_skills, resume, job
    )

    # Identify missing keywords
    keyword_suggestions = identify_missing_keywords(resume, job)

    # Generate improvement suggestions
    improvement_suggestions = generate_improvement_suggestions(
        resume, job, matching_skills, missing_skills, match_score
    )

    # Build metadata
    metadata = {
        'job_skills_found': len(job_skills),
        'resume_skills_count': len(resume.skills),
        'keyword_analysis': {
            'total_keywords_analyzed': 30,
            'missing_count': len(keyword_suggestions)
        }
    }

    return AnalysisResult(
        resume_id=resume_id,
        job_listing_id=job_listing_id,
        match_score=match_score,
        matching_skills=matching_skills,
        missing_skills=missing_skills,
        keyword_suggestions=keyword_suggestions,
        improvement_suggestions=improvement_suggestions,
        metadata=metadata
    )
