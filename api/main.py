"""
FastAPI application for the resume parser.
"""

import os
import logging
import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers import get_parser
from extractors.ner import NERExtractor
from extractors.classifier import ResumeClassifier
from extractors.section_segmenter import SectionSegmenter
from normalizers.date_normalizer import DateNormalizer
from normalizers.skill_normalizer import SkillNormalizer
from normalizers.education_normalizer import EducationNormalizer
from utils.validation import validate_output
from utils.helpers import ensure_directory, save_json

import config


# Initialize FastAPI app
app = FastAPI(
    title="Resume Parser API",
    description="API for parsing resumes and extracting structured information",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory storage for parsing tasks
tasks = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ensure upload and result directories exist
ensure_directory(config.API_UPLOAD_DIR)
ensure_directory(config.API_RESULT_DIR)


class ParseRequest(BaseModel):
    """Request model for job description comparison."""
    job_description: Optional[str] = None


class ParseResponse(BaseModel):
    """Response model for parse requests."""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Resume Parser API", "version": "1.0.0"}


@app.post("/parse", response_model=ParseResponse)
async def parse_resume(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    job_description: Optional[str] = None
):
    """
    Parse resume file(s) and extract structured information.
    
    Args:
        background_tasks: FastAPI background tasks
        files: Resume file(s) to parse
        job_description: Optional job description for comparison
        
    Returns:
        Task ID for tracking the parsing task
    """
    # Validate files
    for file in files:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in config.SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(config.SUPPORTED_FILE_TYPES)}"
            )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task directory
    task_dir = os.path.join(config.API_UPLOAD_DIR, task_id)
    ensure_directory(task_dir)
    
    # Save files
    file_paths = []
    for file in files:
        file_path = os.path.join(task_dir, file.filename)
        
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_paths.append(file_path)
    
    # Save job description if provided
    job_desc_path = None
    if job_description:
        job_desc_path = os.path.join(task_dir, "job_description.txt")
        with open(job_desc_path, "w", encoding="utf-8") as f:
            f.write(job_description)
    
    # Initialize task status
    tasks[task_id] = {
        "status": "queued",
        "files": file_paths,
        "job_description": job_desc_path,
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    
    # Start processing in background
    background_tasks.add_task(
        process_resume_task, 
        task_id, 
        file_paths, 
        job_desc_path
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": f"Processing {len(files)} resume(s)"
    }


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a parsing task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status and result if completed
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task["result"],
        "error": task["error"]
    }


@app.get("/tasks")
async def list_tasks(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    List parsing tasks.
    
    Args:
        limit: Maximum number of tasks to return
        offset: Offset for pagination
        
    Returns:
        List of tasks
    """
    task_ids = list(tasks.keys())
    task_ids.sort(key=lambda tid: tasks[tid]["created_at"], reverse=True)
    
    paginated_ids = task_ids[offset:offset + limit]
    
    return {
        "tasks": [
            {
                "task_id": tid,
                "status": tasks[tid]["status"],
                "created_at": tasks[tid]["created_at"],
                "file_count": len(tasks[tid]["files"])
            }
            for tid in paginated_ids
        ],
        "total": len(task_ids),
        "limit": limit,
        "offset": offset
    }


async def process_resume_task(task_id, file_paths, job_desc_path=None):
    """
    Process resume parsing task in the background.
    
    Args:
        task_id: Task ID
        file_paths: List of resume file paths
        job_desc_path: Path to job description file
    """
    try:
        # Update task status
        tasks[task_id]["status"] = "processing"
        
        # Load job description if provided
        job_desc_text = None
        if job_desc_path:
            with open(job_desc_path, "r", encoding="utf-8") as f:
                job_desc_text = f.read()
        
        # Process each resume
        results = {}
        
        # Initialize components (reuse for all files)
        ner_extractor = NERExtractor()
        resume_classifier = ResumeClassifier()
        section_segmenter = SectionSegmenter()
        date_normalizer = DateNormalizer()
        skill_normalizer = SkillNormalizer()
        education_normalizer = EducationNormalizer()
        
        for file_path in file_paths:
            try:
                # Get file extension
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # Get parser for file type
                parser = get_parser(file_ext)
                
                # Parse resume
                resume_text = parser.parse(file_path)
                
                # Segment the resume into sections
                sections = section_segmenter.segment(resume_text)
                
                # Extract entities using NER
                entities = ner_extractor.extract(resume_text, sections)
                
                # Classify the resume
                job_role = resume_classifier.classify(resume_text)
                
                # Normalize extracted data
                normalized_entities = {
                    "name": entities.get("NAME", ""),
                    "contact": {
                        "email": entities.get("EMAIL", ""),
                        "phone": entities.get("PHONE", ""),
                        "location": entities.get("LOCATION", "")
                    },
                    "education": [
                        {
                            "degree": education_normalizer.normalize_degree(edu.get("EDUCATION_DEGREE", "")),
                            "institution": edu.get("EDUCATION_INSTITUTION", ""),
                            "year": date_normalizer.normalize_year(edu.get("EDUCATION_YEAR", ""))
                        }
                        for edu in entities.get("EDUCATION", [])
                    ],
                    "skills": [
                        skill_normalizer.normalize(skill)
                        for skill in entities.get("SKILL", [])
                    ],
                    "experience": [
                        {
                            "title": exp.get("EXPERIENCE_TITLE", ""),
                            "company": exp.get("EXPERIENCE_COMPANY", ""),
                            "duration": date_normalizer.normalize_duration(exp.get("EXPERIENCE_DURATION", ""))
                        }
                        for exp in entities.get("EXPERIENCE", [])
                    ],
                    "certifications": entities.get("CERTIFICATION", []),
                    "projects": entities.get("PROJECT", []),
                    "job_role": job_role
                }
                
                # If job description is provided, calculate similarity score
                if job_desc_text:
                    similarity = resume_classifier.calculate_similarity(resume_text, job_desc_text)
                    normalized_entities["job_match_score"] = similarity
                
                # Validate output
                validation_result = validate_output(normalized_entities)
                if validation_result["warnings"]:
                    normalized_entities["warnings"] = validation_result["warnings"]
                
                # Add to results
                file_name = os.path.basename(file_path)
                results[file_name] = normalized_entities
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                file_name = os.path.basename(file_path)
                results[file_name] = {"error": str(e)}
        
        # Save results
        result_path = os.path.join(config.API_RESULT_DIR, f"{task_id}.json")
        save_json(results, result_path)
        
        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = results
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=config.API_HOST, 
        port=config.API_PORT, 
        reload=True
    )
