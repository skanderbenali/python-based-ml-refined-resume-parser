"""
Configuration settings for the resume parser.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
NER_MODEL_PATH = os.path.join(MODEL_DIR, "ner_model")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "classifier_model")

# Supported file types
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt", ".html"]

# NER configuration
NER_LABELS = [
    "NAME", "EMAIL", "PHONE", "LOCATION", 
    "EDUCATION_DEGREE", "EDUCATION_INSTITUTION", "EDUCATION_YEAR",
    "EXPERIENCE_TITLE", "EXPERIENCE_COMPANY", "EXPERIENCE_DURATION",
    "SKILL", "CERTIFICATION", "PROJECT"
]

# Section segmentation configuration
RESUME_SECTIONS = [
    "PERSONAL_INFO", "EDUCATION", "EXPERIENCE", 
    "SKILLS", "CERTIFICATIONS", "PROJECTS"
]

# Mapping file paths
SKILL_MAPPING_PATH = os.path.join(MODEL_DIR, "skill_mappings.json")
EDUCATION_MAPPING_PATH = os.path.join(MODEL_DIR, "education_mappings.json")

# The skill and education mappings are now stored in JSON files in the model directory

# Processing configuration
MAX_WORKERS = os.cpu_count() or 4  # Default to 4 if cpu_count returns None
BATCH_SIZE = 10  # Number of resumes to process in parallel

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
API_RESULT_DIR = os.path.join(BASE_DIR, "results")

# Create required directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(API_UPLOAD_DIR, exist_ok=True)
os.makedirs(API_RESULT_DIR, exist_ok=True)
