"""
Core data structures (models) used across the application.

This file serves as the single source of truth for shared data structures,
preventing circular dependencies between modules.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class JobListing:
    """A structured representation of a job listing."""
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    apply_url: Optional[str] = None
    description: str = ""
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the JobListing to a dictionary."""
        return {
            'job_title': self.job_title,
            'company': self.company,
            'location': self.location,
            'apply_url': self.apply_url,
            'description': self.description,
            'source_url': self.source_url,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobListing':
        """Create a JobListing from a dictionary."""
        return cls(
            job_title=data.get('job_title'),
            company=data.get('company'),
            location=data.get('location'),
            apply_url=data.get('apply_url'),
            description=data.get('description', ''),
            source_url=data.get('source_url'),
            metadata=data.get('metadata', {})
        )


@dataclass
class Resume:
    """A structured representation of a parsed resume."""
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    experience: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    projects: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Resume to a dictionary."""
        return {
            'full_name': self.full_name,
            'email': self.email,
            'phone': self.phone,
            'location': self.location,
            'summary': self.summary,
            'skills': self.skills,
            'experience': self.experience,
            'education': self.education,
            'projects': self.projects,
            'raw_text': self.raw_text,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resume':
        """Create a Resume from a dictionary."""
        return cls(
            full_name=data.get('full_name'),
            email=data.get('email'),
            phone=data.get('phone'),
            location=data.get('location'),
            summary=data.get('summary'),
            skills=data.get('skills', []),
            experience=data.get('experience', []),
            education=data.get('education', []),
            projects=data.get('projects', []),
            raw_text=data.get('raw_text', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class AnalysisResult:
    """Result of comparing a resume against a job listing."""
    resume_id: int
    job_listing_id: int
    match_score: float
    matching_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    keyword_suggestions: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the AnalysisResult to a dictionary."""
        return {
            'resume_id': self.resume_id,
            'job_listing_id': self.job_listing_id,
            'match_score': self.match_score,
            'matching_skills': self.matching_skills,
            'missing_skills': self.missing_skills,
            'keyword_suggestions': self.keyword_suggestions,
            'improvement_suggestions': self.improvement_suggestions,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create an AnalysisResult from a dictionary."""
        return cls(
            resume_id=data.get('resume_id', 0),
            job_listing_id=data.get('job_listing_id', 0),
            match_score=data.get('match_score', 0.0),
            matching_skills=data.get('matching_skills', []),
            missing_skills=data.get('missing_skills', []),
            keyword_suggestions=data.get('keyword_suggestions', []),
            improvement_suggestions=data.get('improvement_suggestions', []),
            metadata=data.get('metadata', {})
        )
