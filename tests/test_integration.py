"""
Integration tests for the resume parser.
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_parser import process_single_resume
from parsers import get_parser
from extractors.ner import NERExtractor
from extractors.classifier import ResumeClassifier
from extractors.section_segmenter import SectionSegmenter
import config


class TestResumeParser:
    """Integration test suite for the resume parser."""
    
    @pytest.fixture
    def sample_resume_file(self):
        """Create a sample resume file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            resume_content = """
            John Doe
            john.doe@example.com
            (123) 456-7890
            New York, NY
            
            EDUCATION
            Bachelor of Science in Computer Science
            MIT
            2015-2019
            
            EXPERIENCE
            Software Engineer
            Acme Corporation
            Jan 2019 - Present
            • Developed web applications using React and Node.js
            • Implemented machine learning algorithms for data analysis
            
            Junior Developer
            Tech Startups Inc.
            Jun 2017 - Dec 2018
            • Assisted in developing mobile applications
            • Worked on database optimization
            
            SKILLS
            Python, JavaScript, React, Node.js, Machine Learning, SQL, Git
            
            CERTIFICATIONS
            AWS Certified Developer
            Google Cloud Professional
            
            PROJECTS
            Personal Website - A portfolio site built with React
            ML Model - A machine learning model for predicting stock prices
            """
            temp.write(resume_content.encode('utf-8'))
            temp_name = temp.name
        
        yield temp_name
        
        # Cleanup
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    
    @pytest.fixture
    def sample_job_description(self):
        """Create a sample job description for testing."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            job_content = """
            Software Engineer
            
            We are seeking an experienced Software Engineer to join our team.
            
            Requirements:
            - 3+ years of experience in software development
            - Proficiency in Python and JavaScript
            - Experience with React and Node.js
            - Knowledge of machine learning is a plus
            - Bachelor's degree in Computer Science or related field
            
            Responsibilities:
            - Develop and maintain web applications
            - Implement machine learning algorithms
            - Collaborate with cross-functional teams
            """
            temp.write(job_content.encode('utf-8'))
            temp_name = temp.name
        
        yield temp_name
        
        # Cleanup
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    
    @patch.object(NERExtractor, "_load_model")
    @patch.object(ResumeClassifier, "_load_model")
    @patch.object(ResumeClassifier, "_load_spacy")
    def test_process_single_resume(self, mock_load_spacy, mock_load_classifier, mock_load_ner, sample_resume_file):
        """Test processing a single resume."""
        # Process the sample resume
        result = process_single_resume(sample_resume_file)
        
        # Verify basic structure
        assert isinstance(result, dict)
        assert "name" in result
        assert "contact" in result
        assert "education" in result
        assert "skills" in result
        assert "experience" in result
        assert "job_role" in result
        
        # Verify content extraction
        assert "John Doe" in result["name"]
        assert "john.doe@example.com" in result["contact"]["email"]
        assert any("Computer Science" in edu["degree"] for edu in result["education"])
        assert any("Software Engineer" in exp["title"] for exp in result["experience"])
        assert any("Acme" in exp["company"] for exp in result["experience"])
        assert any("Python" in result["skills"])
        assert any("JavaScript" in result["skills"])
    
    @patch.object(NERExtractor, "_load_model")
    @patch.object(ResumeClassifier, "_load_model")
    @patch.object(ResumeClassifier, "_load_spacy")
    def test_resume_with_job_description(self, mock_load_spacy, mock_load_classifier, mock_load_ner, 
                                       sample_resume_file, sample_job_description):
        """Test processing a resume with job description comparison."""
        # Mock the similarity calculation
        with patch.object(ResumeClassifier, "calculate_similarity", return_value=0.85):
            # Load job description
            with open(sample_job_description, 'r', encoding='utf-8') as f:
                job_desc = f.read()
            
            # Process the resume with job description
            result = process_single_resume(sample_resume_file, job_desc)
            
            # Verify job match score
            assert "job_match_score" in result
            assert result["job_match_score"] == 0.85
    
    @patch.object(NERExtractor, "_load_model")
    @patch.object(ResumeClassifier, "_load_model")
    @patch.object(ResumeClassifier, "_load_spacy")
    def test_unsupported_file_format(self, mock_load_spacy, mock_load_classifier, mock_load_ner):
        """Test processing an unsupported file format."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp:
            temp.write(b"Test content")
            temp_name = temp.name
        
        try:
            # Process the file
            result = process_single_resume(temp_name)
            
            # Verify error message
            assert "error" in result
            assert "Unsupported file type" in result["error"]
            
        finally:
            # Cleanup
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    
    @patch("parsers.txt_parser.TXTParser.parse")
    @patch.object(NERExtractor, "_load_model")
    @patch.object(ResumeClassifier, "_load_model")
    @patch.object(ResumeClassifier, "_load_spacy")
    def test_parser_exception_handling(self, mock_load_spacy, mock_load_classifier, 
                                     mock_load_ner, mock_parse, sample_resume_file):
        """Test handling of parser exceptions."""
        # Mock parser to raise exception
        mock_parse.side_effect = Exception("Parser error")
        
        # Process the resume
        result = process_single_resume(sample_resume_file)
        
        # Verify error handling
        assert "error" in result
        assert "Parser error" in result["error"]


class TestAPIIntegration:
    """Integration test suite for the API."""
    
    @pytest.mark.asyncio
    @patch("api.main.NERExtractor")
    @patch("api.main.ResumeClassifier")
    @patch("api.main.SectionSegmenter")
    async def test_parse_endpoint(self, mock_segmenter, mock_classifier, mock_ner):
        """Test the parse endpoint of the API."""
        from api.main import app
        from fastapi.testclient import TestClient
        import aiofiles
        
        # Setup test client
        client = TestClient(app)
        
        # Create a temporary resume file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write(b"Test resume content")
            temp_name = temp.name
        
        try:
            # Mock file upload
            with open(temp_name, "rb") as f:
                files = {"files": (os.path.basename(temp_name), f, "text/plain")}
                response = client.post("/parse", files=files)
            
            # Verify response
            assert response.status_code == 200
            assert "task_id" in response.json()
            assert response.json()["status"] == "queued"
            
        finally:
            # Cleanup
            if os.path.exists(temp_name):
                os.unlink(temp_name)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
