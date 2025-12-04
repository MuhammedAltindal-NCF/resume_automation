-- =============================================================================
-- Resume Optimizer - Complete Database Setup
-- Run this script once to create all tables for a fresh installation.
-- =============================================================================

-- Drop existing tables (in correct order due to foreign keys)
DROP TABLE IF EXISTS analysis_results CASCADE;
DROP TABLE IF EXISTS resumes CASCADE;
DROP TABLE IF EXISTS job_listings CASCADE;

-- =============================================================================
-- Table: job_listings (Phase 1)
-- Stores processed job listings from various sources
-- =============================================================================
CREATE TABLE job_listings (
    id SERIAL PRIMARY KEY,
    job_title VARCHAR(255),
    company VARCHAR(255),
    location VARCHAR(255),
    description TEXT NOT NULL,
    apply_url TEXT,
    description_hash VARCHAR(64) UNIQUE NOT NULL,
    source_url TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_job_description_hash ON job_listings(description_hash);

-- =============================================================================
-- Table: resumes (Phase 2)
-- Stores parsed resume data
-- =============================================================================
CREATE TABLE resumes (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    location VARCHAR(255),
    summary TEXT,
    skills JSONB,
    experience JSONB,
    education JSONB,
    projects JSONB,
    raw_text TEXT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_resume_file_hash ON resumes(file_hash);
CREATE INDEX idx_resume_skills ON resumes USING GIN(skills);
CREATE INDEX idx_resume_email ON resumes(email);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_resume_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_resume_updated_at
    BEFORE UPDATE ON resumes
    FOR EACH ROW
    EXECUTE FUNCTION update_resume_timestamp();

-- =============================================================================
-- Table: analysis_results (Phase 2)
-- Stores resume-job comparison results
-- =============================================================================
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    resume_id INTEGER REFERENCES resumes(id) ON DELETE CASCADE,
    job_listing_id INTEGER REFERENCES job_listings(id) ON DELETE CASCADE,
    match_score DECIMAL(5,2),
    matching_skills JSONB,
    missing_skills JSONB,
    keyword_suggestions JSONB,
    improvement_suggestions JSONB,
    analysis_metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(resume_id, job_listing_id)
);

CREATE INDEX idx_analysis_resume ON analysis_results(resume_id);
CREATE INDEX idx_analysis_job ON analysis_results(job_listing_id);
CREATE INDEX idx_analysis_score ON analysis_results(match_score DESC);

-- =============================================================================
-- Verification
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Database setup complete!';
    RAISE NOTICE 'Tables created: job_listings, resumes, analysis_results';
END $$;
