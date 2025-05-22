#!/usr/bin/env python3
"""
Resume Parser - Main Script
"""

import os
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from parsers import get_parser
from extractors.ner import NERExtractor
from extractors.classifier import ResumeClassifier
from extractors.section_segmenter import SectionSegmenter
from normalizers.date_normalizer import DateNormalizer
from normalizers.skill_normalizer import SkillNormalizer
from normalizers.education_normalizer import EducationNormalizer
from utils.validation import validate_output
from utils.helpers import setup_logging

import config

def setup_argparse():
    parser = argparse.ArgumentParser(description="Parse resumes and extract structured information")
    parser.add_argument("--input", required=True, help="Path to a resume file or directory containing resume files")
    parser.add_argument("--output", required=True, help="Path to output JSON file or directory for multiple resumes")
    parser.add_argument("--job-description", help="Optional job description file to compare resume against")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    return parser.parse_args()

def process_single_resume(file_path, job_description=None):
    try:
        logging.info(f"Processing resume: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in config.SUPPORTED_FILE_TYPES:
            logging.error(f"Unsupported file type: {file_ext}")
            return {"error": f"Unsupported file type: {file_ext}"}
        
        parser = get_parser(file_ext)
        resume_text = parser.parse(file_path)
        
        ner_extractor = NERExtractor()
        resume_classifier = ResumeClassifier()
        section_segmenter = SectionSegmenter()
        date_normalizer = DateNormalizer()
        skill_normalizer = SkillNormalizer()
        education_normalizer = EducationNormalizer()
        
        sections = section_segmenter.segment(resume_text)
        entities = ner_extractor.extract(resume_text, sections)
        job_role = resume_classifier.classify(resume_text)
        
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
        
        if job_description:
            similarity = resume_classifier.calculate_similarity(resume_text, job_description)
            normalized_entities["job_match_score"] = similarity
        
        validation_result = validate_output(normalized_entities)
        if validation_result["warnings"]:
            normalized_entities["warnings"] = validation_result["warnings"]
        
        return normalized_entities
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return {"error": str(e), "file": str(file_path)}

def process_resumes(input_path, output_path, job_description=None, workers=config.MAX_WORKERS):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    job_desc_text = None
    if job_description:
        with open(job_description, 'r', encoding='utf-8') as f:
            job_desc_text = f.read()
    
    if input_path.is_file():
        result = process_single_resume(str(input_path), job_desc_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        logging.info(f"Results saved to {output_path}")
    
    elif input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)
        
        resume_files = []
        for ext in config.SUPPORTED_FILE_TYPES:
            resume_files.extend(list(input_path.glob(f"*{ext}")))
        
        if not resume_files:
            logging.warning(f"No supported resume files found in {input_path}")
            return
        
        logging.info(f"Found {len(resume_files)} resume files to process")
        
        results = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_resume, str(file), job_desc_text): file 
                       for file in resume_files}
            
            for future in tqdm(futures, desc="Processing resumes"):
                file = futures[future]
                try:
                    result = future.result()
                    results[file.name] = result
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                    results[file.name] = {"error": str(e)}
        
        for file_name, result in results.items():
            output_file = output_path / f"{file_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        
        summary_file = output_path / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            summary = {
                "total": len(results),
                "successful": sum(1 for r in results.values() if "error" not in r),
                "failed": sum(1 for r in results.values() if "error" in r),
                "job_roles": {}
            }
            
            for result in results.values():
                if "error" not in result and "job_role" in result:
                    role = result["job_role"]
                    summary["job_roles"][role] = summary["job_roles"].get(role, 0) + 1
            
            json.dump(summary, f, indent=2)
        
        logging.info(f"Results saved to {output_path}")
        logging.info(f"Processed {summary['successful']} resumes successfully, {summary['failed']} failed")
    
    else:
        logging.error(f"Input path does not exist: {input_path}")

def main():
    args = setup_argparse()
    setup_logging(args.log_level)
    
    process_resumes(
        args.input, 
        args.output, 
        args.job_description, 
        args.workers
    )

if __name__ == "__main__":
    main()
