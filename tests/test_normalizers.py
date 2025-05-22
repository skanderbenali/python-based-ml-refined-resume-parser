"""
Tests for the normalizers.
"""

import pytest
from unittest.mock import patch

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalizers.date_normalizer import DateNormalizer
from normalizers.skill_normalizer import SkillNormalizer
from normalizers.education_normalizer import EducationNormalizer
import config


class TestDateNormalizer:
    """Test suite for date normalizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = DateNormalizer()
    
    def test_normalize_month_year(self):
        """Test normalizing month-year format."""
        assert self.normalizer.normalize("Jan 2020") == "2020-01"
        assert self.normalizer.normalize("January 2020") == "2020-01"
        assert self.normalizer.normalize("Dec 2019") == "2019-12"
        assert self.normalizer.normalize("December 2019") == "2019-12"
    
    def test_normalize_slash_date(self):
        """Test normalizing dates with slashes."""
        assert self.normalizer.normalize("01/15/2020") == "2020-01-15"
        assert self.normalizer.normalize("1/15/2020") == "2020-01-15"
        assert self.normalizer.normalize("01/15/20") == "2020-01-15"
    
    def test_normalize_hyphen_date(self):
        """Test normalizing dates with hyphens."""
        assert self.normalizer.normalize("01-15-2020") == "2020-01-15"
        assert self.normalizer.normalize("1-15-2020") == "2020-01-15"
        assert self.normalizer.normalize("01-15-20") == "2020-01-15"
    
    def test_normalize_year_only(self):
        """Test normalizing year only."""
        assert self.normalizer.normalize("2020") == "2020"
        assert self.normalizer.normalize("  2020  ") == "2020"
    
    def test_normalize_duration(self):
        """Test normalizing duration."""
        assert self.normalizer.normalize_duration("2018 - 2020") == "2018 to 2020"
        assert self.normalizer.normalize_duration("Jan 2018 - Dec 2020") == "2018-01 to 2020-12"
        assert self.normalizer.normalize_duration("2018 - Present") == "2018 to Present"
    
    def test_normalize_year(self):
        """Test extracting and normalizing year."""
        assert self.normalizer.normalize_year("Jan 2020") == "2020"
        assert self.normalizer.normalize_year("01/15/2020") == "2020"
        assert self.normalizer.normalize_year("2020") == "2020"
        assert self.normalizer.normalize_year("Graduated in 2020") == "2020"


class TestSkillNormalizer:
    """Test suite for skill normalizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = SkillNormalizer()
    
    def test_normalize_aliases(self):
        """Test normalizing skill aliases."""
        # Test with configured aliases
        assert self.normalizer.normalize("js") == "JavaScript"
        assert self.normalizer.normalize("JS") == "JavaScript"
        assert self.normalizer.normalize("py") == "Python"
        assert self.normalizer.normalize("ml") == "Machine Learning"
        assert self.normalizer.normalize("react.js") == "React"
    
    def test_normalize_formatting(self):
        """Test normalizing skill formatting."""
        # Multi-word skills should be title case
        assert self.normalizer.normalize("deep learning") == "Deep Learning"
        assert self.normalizer.normalize("computer vision") == "Computer Vision"
        
        # Acronyms should be uppercase
        assert self.normalizer.normalize("css") == "CSS"
        assert self.normalizer.normalize("aws") == "AWS"
    
    def test_normalize_skills_list(self):
        """Test normalizing a list of skills."""
        skills = ["js", "python", "machine learning", "CSS"]
        normalized = self.normalizer.normalize_skills(skills)
        
        assert "JavaScript" in normalized
        assert "Python" in normalized
        assert "Machine Learning" in normalized
        assert "CSS" in normalized
        assert len(normalized) == 4  # No duplicates
    
    def test_add_alias(self):
        """Test adding custom skill aliases."""
        # Add custom alias
        self.normalizer.add_alias("vue", "Vue.js")
        assert self.normalizer.normalize("vue") == "Vue.js"
        
        # Test batch add
        self.normalizer.batch_add_aliases({
            "ts": "TypeScript",
            "next": "Next.js"
        })
        assert self.normalizer.normalize("ts") == "TypeScript"
        assert self.normalizer.normalize("next") == "Next.js"


class TestEducationNormalizer:
    """Test suite for education normalizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = EducationNormalizer()
    
    def test_normalize_degree_aliases(self):
        """Test normalizing degree aliases."""
        # Test with configured aliases
        assert self.normalizer.normalize_degree("bsc") == "Bachelor of Science"
        assert self.normalizer.normalize_degree("BSc") == "Bachelor of Science"
        assert self.normalizer.normalize_degree("B.S.") == "Bachelor of Science"
        assert self.normalizer.normalize_degree("phd") == "Doctor of Philosophy"
    
    def test_infer_degree(self):
        """Test inferring degree type and field."""
        assert self.normalizer.normalize_degree("Bachelor in Computer Science") == "Bachelor of Computer Science"
        assert self.normalizer.normalize_degree("Master's in Business") == "Master of Business"
        assert self.normalizer.normalize_degree("PhD in Physics") == "Doctor of Philosophy in Physics"
    
    def test_normalize_institution(self):
        """Test normalizing institution names."""
        assert self.normalizer.normalize_institution("mit") == "Massachusetts Institute of Technology"
        assert self.normalizer.normalize_institution("new york university") == "New York University"
        assert self.normalizer.normalize_institution("university of california") == "University of California"
    
    def test_add_alias(self):
        """Test adding custom degree aliases."""
        # Add custom alias
        self.normalizer.add_alias("llb", "Bachelor of Laws")
        assert self.normalizer.normalize_degree("llb") == "Bachelor of Laws"
        
        # Test batch add
        self.normalizer.batch_add_aliases({
            "md": "Doctor of Medicine",
            "jd": "Juris Doctor"
        })
        assert self.normalizer.normalize_degree("md") == "Doctor of Medicine"
        assert self.normalizer.normalize_degree("jd") == "Juris Doctor"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
