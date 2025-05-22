"""
Skill normalizer module for standardizing skill names in resume data.
"""

import re
import json
import logging
from pathlib import Path

import config


class SkillNormalizer:
    
    def __init__(self, skill_mapping_path=None):
        self.mapping_path = skill_mapping_path or config.SKILL_MAPPING_PATH
        self.skill_map = {}
        self.category_map = {}
        self.acronym_map = {}
        self._load_mappings()

    def _load_mappings(self):
        try:
            mapping_file = Path(self.mapping_path)
            
            if mapping_file.exists():
                logging.info(f"Loading skill mappings from {mapping_file}")
                with open(mapping_file, 'r') as f:
                    mappings = json.load(f)
                
                self.skill_map = mappings.get('normalizations', {})
                self.category_map = mappings.get('categories', {})
                self.acronym_map = mappings.get('acronyms', {})
            else:
                logging.warning(f"Skill mapping file not found at {mapping_file}")
                self._create_default_mappings()
                
        except Exception as e:
            logging.error(f"Error loading skill mappings: {str(e)}")
            self._create_default_mappings()

    def normalize(self, skill):
        if not skill or not isinstance(skill, str):
            return ""
        
        # Clean and lowercase the skill
        cleaned_skill = re.sub(r'[^\w\s.#+]', '', skill).strip().lower()
        
        # Check if it's a known variation
        normalized = self.skill_map.get(cleaned_skill, None)
        if normalized:
            return normalized
        
        # Check if it's an acronym
        acronym_expansion = self.acronym_map.get(cleaned_skill.upper(), None)
        if acronym_expansion:
            return acronym_expansion
        
        # If no mapping found, use the original with proper capitalization
        words = cleaned_skill.split()
        capitalized = []
        
        for word in words:
            # Don't capitalize certain words like 'and', 'or', etc.
            if word in ['and', 'or', 'of', 'the', 'in', 'on', 'at', 'for', 'with']:
                capitalized.append(word)
            else:
                capitalized.append(word[0].upper() + word[1:] if word else '')
        
        return ' '.join(capitalized)

    def normalize_skills(self, skills):
        if not skills or not isinstance(skills, list):
            return {}
        
        categorized = {}
        
        for skill in skills:
            normalized = self.normalize(skill)
            category = self.get_category(normalized)
            
            if category not in categorized:
                categorized[category] = []
            
            # Avoid duplicates
            if normalized not in categorized[category]:
                categorized[category].append(normalized)
        
        return categorized

    def get_category(self, skill):
        return self.category_map.get(skill, 'Unknown')

    def _create_default_mappings(self):
        logging.info("Creating default skill mappings")
        
        # Default skill normalizations
        self.skill_map = {
            "javascript": "JavaScript",
            "js": "JavaScript",
            "typescript": "TypeScript",
            "ts": "TypeScript",
            "react": "React.js",
            "reactjs": "React.js",
            "vue": "Vue.js",
            "vuejs": "Vue.js",
            "node": "Node.js",
            "nodejs": "Node.js",
            "python": "Python",
            "py": "Python",
            "java": "Java",
            "c#": "C#",
            "csharp": "C#",
            "c++": "C++",
            "cpp": "C++",
            "html": "HTML",
            "css": "CSS",
            "sql": "SQL"
        }
        
        # Default skill categories
        self.category_map = {
            "JavaScript": "Programming Languages",
            "TypeScript": "Programming Languages",
            "Python": "Programming Languages",
            "Java": "Programming Languages",
            "C#": "Programming Languages",
            "C++": "Programming Languages",
            "HTML": "Web Technologies",
            "CSS": "Web Technologies",
            "React.js": "Frontend Frameworks",
            "Vue.js": "Frontend Frameworks",
            "Node.js": "Backend Technologies",
            "SQL": "Database Technologies"
        }
        
        # Default acronym expansions
        self.acronym_map = {
            "AI": "Artificial Intelligence",
            "ML": "Machine Learning",
            "NLP": "Natural Language Processing",
            "API": "Application Programming Interface",
            "UI": "User Interface",
            "UX": "User Experience",
            "CSS": "Cascading Style Sheets",
            "HTML": "Hypertext Markup Language"
        }
    
    def batch_add_aliases(self, alias_dict):
        if not alias_dict or not isinstance(alias_dict, dict):
            return
            
        for alias, normalized in alias_dict.items():
            self.skill_map[alias.lower()] = normalized
