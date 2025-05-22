"""
Validation utility functions for the resume parser.
"""

import re
import logging
from datetime import datetime


def validate_output_structure(parsed_data):
    """
    Validate the extracted resume data and identify missing or incomplete sections.
    
    Args:
        parsed_data: Dictionary containing extracted resume data
        
    Returns:
        Dictionary with validation results and warnings
    """
    warnings = []
    
    try:
        # Check for missing name
        if not parsed_data.get("name"):
            warnings.append("Missing name information")
        
        # Check contact information
        contact = parsed_data.get("contact", {})
        if not contact.get("email"):
            warnings.append("Missing email address")
        if not contact.get("phone"):
            warnings.append("Missing phone number")
        
        # Validate email format
        if contact.get("email") and not validate_email(contact["email"]):
            warnings.append("Invalid email format")
        
        # Check education
        education = parsed_data.get("education", [])
        if not education:
            warnings.append("Missing education information")
        else:
            for i, edu in enumerate(education):
                if not edu.get("degree"):
                    warnings.append(f"Missing degree in education entry {i+1}")
                if not edu.get("institution"):
                    warnings.append(f"Missing institution in education entry {i+1}")
        
        # Check experience
        experience = parsed_data.get("experience", [])
        if not experience:
            warnings.append("Missing work experience information")
        else:
            for i, exp in enumerate(experience):
                if not exp.get("title"):
                    warnings.append(f"Missing job title in experience entry {i+1}")
                if not exp.get("company"):
                    warnings.append(f"Missing company in experience entry {i+1}")
        
        # Check skills
        skills = parsed_data.get("skills", [])
        if not skills:
            warnings.append("Missing skills information")
        elif len(skills) < 3:
            warnings.append("Limited skills information (fewer than 3 skills)")
        
        # Check for job role classification
        if not parsed_data.get("job_role"):
            warnings.append("Unable to classify resume by job role")
        
        return {"valid": len(warnings) == 0, "warnings": warnings}
        
    except Exception as e:
        logging.error(f"Error validating resume data: {str(e)}")
        return {"valid": False, "warnings": ["Error during validation", str(e)]}


def validate_email(email):
    import re
    email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    return bool(email_pattern.match(email))


def validate_file(file_path, supported_extensions):
    """
    Validate if a file exists and has a supported extension.
    
    Args:
        file_path: Path to the file
        supported_extensions: List of supported file extensions
        
    Returns:
        Dictionary with validation results and errors
    """
    import os
    
    errors = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        errors.append(f"File does not exist: {file_path}")
        return {"valid": False, "errors": errors}
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        errors.append(f"Not a file: {file_path}")
        return {"valid": False, "errors": errors}
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in supported_extensions:
        errors.append(f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_extensions)}")
        return {"valid": False, "errors": errors}
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as f:
            # Just try to read a small part to check access
            f.read(10)
    except Exception as e:
        errors.append(f"File is not readable: {str(e)}")
        return {"valid": False, "errors": errors}
    
    return {"valid": True, "errors": errors}


def check_model_compatibility(model_path, required_version):
    """
    Check if a model is compatible with the current code.
    
    Args:
        model_path: Path to the model
        required_version: Required model version
        
    Returns:
        True if compatible, False otherwise
    """
    import os
    import json
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        logging.warning(f"Model directory does not exist: {model_path}")
        return False
    
    # Check for version file
    version_file = os.path.join(model_path, "version.json")
    if not os.path.exists(version_file):
        logging.warning(f"Model version file not found: {version_file}")
        return False
    
    try:
        with open(version_file, 'r') as f:
            version_info = json.load(f)
        
        model_version = version_info.get("version")
        if model_version != required_version:
            logging.warning(f"Model version mismatch. Required: {required_version}, Found: {model_version}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error checking model compatibility: {str(e)}")
        return False
