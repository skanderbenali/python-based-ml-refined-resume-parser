"""
Education normalizer module for standardizing education information in resume data.
"""

import re
import json
import logging
from pathlib import Path

import config


class EducationNormalizer:
    
    def __init__(self, mapping_path=None):
        self.mapping_path = mapping_path or config.EDUCATION_MAPPING_PATH
        self.degree_map = {}
        self.institution_map = {}
        self.level_map = {}
        self._load_mappings()

    def _load_mappings(self):
        try:
            mapping_file = Path(self.mapping_path)
            
            if mapping_file.exists():
                logging.info(f"Loading education mappings from {mapping_file}")
                with open(mapping_file, 'r') as f:
                    mappings = json.load(f)
                
                self.degree_map = mappings.get('degrees', {})
                self.institution_map = mappings.get('institutions', {})
                self.level_map = mappings.get('levels', {})
            else:
                logging.warning(f"Education mapping file not found at {mapping_file}")
                self._create_default_mappings()
                
        except Exception as e:
            logging.error(f"Error loading education mappings: {str(e)}")
            self._create_default_mappings()

    def _create_default_mappings(self):
        logging.info("Creating default education mappings")
        
        # Default degree normalizations
        self.degree_map = {
            "bs": "Bachelor of Science",
            "b.s.": "Bachelor of Science",
            "b.s": "Bachelor of Science",
            "bachelor of science": "Bachelor of Science",
            "ba": "Bachelor of Arts",
            "b.a.": "Bachelor of Arts",
            "b.a": "Bachelor of Arts",
            "bachelor of arts": "Bachelor of Arts",
            "bba": "Bachelor of Business Administration",
            "b.b.a.": "Bachelor of Business Administration",
            "bachelor of business administration": "Bachelor of Business Administration",
            "bfa": "Bachelor of Fine Arts",
            "b.f.a.": "Bachelor of Fine Arts",
            "bachelor of fine arts": "Bachelor of Fine Arts",
            
            "ms": "Master of Science",
            "m.s.": "Master of Science",
            "m.s": "Master of Science",
            "master of science": "Master of Science",
            "ma": "Master of Arts",
            "m.a.": "Master of Arts",
            "m.a": "Master of Arts",
            "master of arts": "Master of Arts",
            "mba": "Master of Business Administration",
            "m.b.a.": "Master of Business Administration",
            "master of business administration": "Master of Business Administration",
            "mfa": "Master of Fine Arts",
            "m.f.a.": "Master of Fine Arts",
            "master of fine arts": "Master of Fine Arts",
            
            "phd": "Doctor of Philosophy",
            "ph.d.": "Doctor of Philosophy",
            "ph.d": "Doctor of Philosophy",
            "doctor of philosophy": "Doctor of Philosophy",
            "doctorate": "Doctor of Philosophy",
            "doctoral": "Doctor of Philosophy",
            
            "aa": "Associate of Arts",
            "a.a.": "Associate of Arts",
            "associate of arts": "Associate of Arts",
            "as": "Associate of Science",
            "a.s.": "Associate of Science",
            "associate of science": "Associate of Science",
            
            "high school diploma": "High School Diploma",
            "high school": "High School Diploma",
            "hs": "High School Diploma",
            "h.s.": "High School Diploma",
            "secondary education": "High School Diploma",
            
            "certificate": "Certificate",
            "certification": "Certificate",
            "professional certificate": "Certificate"
        }
        
        # Simple institution mapping for common universities
        self.institution_map = {
            "mit": "Massachusetts Institute of Technology",
            "harvard": "Harvard University",
            "stanford": "Stanford University",
            "berkeley": "University of California, Berkeley",
            "uc berkeley": "University of California, Berkeley",
            "ucla": "University of California, Los Angeles",
            "nyu": "New York University",
            "columbia": "Columbia University",
            "princeton": "Princeton University",
            "yale": "Yale University",
            "cmu": "Carnegie Mellon University",
            "carnegie mellon": "Carnegie Mellon University"
        }
        
        # Map degrees to academic levels
        self.level_map = {
            "High School Diploma": "High School",
            "Associate of Arts": "Associate",
            "Associate of Science": "Associate",
            "Bachelor of Science": "Bachelor",
            "Bachelor of Arts": "Bachelor",
            "Bachelor of Business Administration": "Bachelor",
            "Bachelor of Fine Arts": "Bachelor",
            "Master of Science": "Master",
            "Master of Arts": "Master",
            "Master of Business Administration": "Master",
            "Master of Fine Arts": "Master",
            "Doctor of Philosophy": "Doctorate",
            "Certificate": "Certificate"
        }

    def get_degree_level(self, degree):
        normalized_degree = self.normalize_degree(degree)
        return self.level_map.get(normalized_degree, "Unknown")

    def normalize_degree(self, degree):
        if not degree or not isinstance(degree, str):
            return ""
        
        degree_lower = degree.strip().lower()
        
        try:
            # Direct match in aliases
            if degree_lower in self.degree_map:
                return self.degree_map[degree_lower]
            
            # Clean degree name (remove punctuation, extra spaces)
            cleaned_degree = re.sub(r'[^\w\s]', '', degree_lower)
            cleaned_degree = re.sub(r'\s+', ' ', cleaned_degree).strip()
            
            if cleaned_degree in self.degree_map:
                return self.degree_map[cleaned_degree]
            
            # Try fuzzy matching
            # close_matches = get_close_matches(cleaned_degree, self.all_degrees, n=1, cutoff=0.85)
            # if close_matches:
            #     close_match = close_matches[0]
                
            #     # If match is an alias, return its normalized form
            #     if close_match in self.aliases:
            #         return self.aliases[close_match]
                
            #     # Otherwise return the match itself (it's a normalized degree)
            #     return close_match
            
            # If no match is found, try to infer the degree type and field
            return self._infer_degree(degree)
                
        except Exception as e:
            logging.error(f"Error normalizing degree '{degree}': {str(e)}")
            return degree  # Return original if error occurs

    def _infer_degree(self, degree):
        """
        Infer the degree type and field from a degree string.
        
        Args:
            degree: Degree string
            
        Returns:
            Normalized degree string
        """
        degree_lower = degree.lower()
        
        # Check for degree type
        degree_type = None
        if "bachelor" in degree_lower:
            degree_type = "Bachelor"
        elif "master" in degree_lower:
            degree_type = "Master"
        elif "phd" in degree_lower or "doctor" in degree_lower:
            degree_type = "Doctor of Philosophy"
        elif "associate" in degree_lower:
            degree_type = "Associate"
        elif "mba" in degree_lower:
            return "Master of Business Administration"
        
        # Check for field of study
        field_match = re.search(r'\bin\s+(\w+(?:\s+\w+)*)', degree_lower)
        field = field_match.group(1).title() if field_match else ""
        
        # Determine field based on keywords if not found with regex
        if not field:
            # Common fields and their keywords
            field_keywords = {
                "Computer Science": ["computer", "computing", "cs", "information technology", "it"],
                "Engineering": ["engineering", "engineer"],
                "Business": ["business", "management", "administration", "marketing", "finance"],
                "Arts": ["arts", "liberal arts", "humanities"],
                "Science": ["science", "sciences"],
                "Mathematics": ["mathematics", "math", "maths"],
                "Economics": ["economics", "econ"],
                "Law": ["law", "legal", "llb"]
            }
            
            for field_name, keywords in field_keywords.items():
                if any(keyword in degree_lower for keyword in keywords):
                    field = field_name
                    break
        
        # Construct normalized degree
        if degree_type and field:
            return f"{degree_type} of {field}"
        elif degree_type:
            return f"{degree_type}'s Degree"
        else:
            # Return original with proper capitalization
            return degree.title()

    def normalize_institution(self, institution):
        if not institution or not isinstance(institution, str):
            return ""
        
        institution = institution.strip()
        
        # Common abbreviations for institutions
        institution_abbrevs = {
            "mit": "Massachusetts Institute of Technology",
            "caltech": "California Institute of Technology",
            "cmu": "Carnegie Mellon University",
            "nyu": "New York University",
            "ucla": "University of California, Los Angeles",
            "usc": "University of Southern California",
            "ucb": "University of California, Berkeley"
            # Add more as needed
        }
        
        # Check for direct matches
        inst_lower = institution.lower()
        if inst_lower in institution_abbrevs:
            return institution_abbrevs[inst_lower]
        
        # Basic cleaning: ensure proper capitalization
        words_to_lowercase = {'of', 'the', 'and', 'in', 'at', 'by', 'for', 'with'}
        
        words = institution.split()
        for i, word in enumerate(words):
            if i > 0 and word.lower() in words_to_lowercase:
                words[i] = word.lower()
            else:
                words[i] = word.capitalize()
        
        return ' '.join(words)

    def add_alias(self, alias, normalized_form):
        """
        Add a custom degree alias.
        
        Args:
            alias: Alias or variation of a degree name
            normalized_form: Standardized form of the degree
        """
        alias_lower = alias.strip().lower()
        self.aliases[alias_lower] = normalized_form
        
        # Update the reverse lookup
        if normalized_form not in self.all_degrees:
            self.all_degrees.append(normalized_form)
    
    def batch_add_aliases(self, alias_dict):
        """
        Add multiple custom degree aliases.
        
        Args:
            alias_dict: Dictionary mapping aliases to normalized forms
        """
        for alias, norm in alias_dict.items():
            self.add_alias(alias, norm)
