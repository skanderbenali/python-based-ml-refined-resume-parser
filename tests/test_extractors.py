"""
Tests for the extractors.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractors.ner import NERExtractor
from extractors.classifier import ResumeClassifier
from extractors.section_segmenter import SectionSegmenter


class TestNERExtractor:
    """Test suite for NER extractor."""
    
    @patch("spacy.load")
    def test_initialization(self, mock_load):
        """Test NER extractor initialization."""
        # Mock spacy.load
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp
        
        # Create extractor
        extractor = NERExtractor()
        
        # Verify results
        assert extractor.nlp == mock_nlp
        mock_load.assert_called_once_with("en_core_web_lg")
    
    @patch("spacy.load")
    def test_extract_entities(self, mock_load):
        """Test entity extraction."""
        # Create mock entities
        mock_ent1 = MagicMock()
        mock_ent1.text = "John Doe"
        mock_ent1.label_ = "PERSON"
        
        mock_ent2 = MagicMock()
        mock_ent2.text = "Acme Corp"
        mock_ent2.label_ = "ORG"
        
        # Mock spaCy doc and its entities
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2]
        
        # Mock nlp function to return the mock doc
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        # Create extractor
        extractor = NERExtractor()
        
        # Mock regex methods
        extractor.email_pattern.findall = MagicMock(return_value=["john@example.com"])
        extractor.phone_pattern.findall = MagicMock(return_value=["123-456-7890"])
        
        # Extract entities
        result = extractor.extract("Sample resume text")
        
        # Verify results
        assert "John Doe" in result.get("NAME", [])
        assert "Acme Corp" in result.get("EXPERIENCE_COMPANY", [])
        assert result.get("EMAIL") == "john@example.com"
        assert result.get("PHONE") == "123-456-7890"
    
    @patch("spacy.load")
    def test_extract_with_sections(self, mock_load):
        """Test entity extraction with resume sections."""
        # Mock spaCy doc and its entities
        mock_doc = MagicMock()
        mock_doc.ents = []
        
        # Mock NLP function to return the mock doc
        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp
        
        # Create extractor
        extractor = NERExtractor()
        
        # Mock regex pattern methods
        extractor.degree_pattern.findall = MagicMock(return_value=["BSc"])
        extractor.job_title_pattern.findall = MagicMock(return_value=["Software Engineer"])
        extractor.date_pattern.findall = MagicMock(return_value=["2020"])
        
        # Extract entities with sections
        sections = {
            "EDUCATION": "BSc Computer Science, MIT, 2020",
            "EXPERIENCE": "Software Engineer at Acme Corp, 2018-2020"
        }
        
        result = extractor.extract("Sample resume text", sections)
        
        # Verify results
        assert "BSc" in result.get("EDUCATION")[0].get("EDUCATION_DEGREE", "")
        assert "Software Engineer" in result.get("EXPERIENCE")[0].get("EXPERIENCE_TITLE", "")


class TestResumeClassifier:
    """Test suite for resume classifier."""
    
    @patch("spacy.load")
    @patch("pickle.load")
    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    def test_initialization(self, mock_exists, mock_open, mock_pickle, mock_spacy_load):
        """Test classifier initialization."""
        # Mock path.exists to return False (no saved model)
        mock_exists.return_value = False
        
        # Mock spacy load
        mock_nlp = MagicMock()
        mock_spacy_load.return_value = mock_nlp
        
        # Create classifier
        classifier = ResumeClassifier()
        
        # Verify results
        assert not classifier.is_trained
        mock_spacy_load.assert_called_once()
    
    @patch("spacy.load")
    def test_heuristic_classification(self, mock_load):
        """Test classification without trained model."""
        # Mock spacy load
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp
        
        # Create classifier with no trained model
        classifier = ResumeClassifier()
        classifier.is_trained = False
        
        # Test with software engineer keywords
        result1 = classifier.classify("java python software developer code programming")
        assert result1 == "Software Engineer"
        
        # Test with data scientist keywords
        result2 = classifier.classify("data scientist machine learning statistics python tensorflow")
        assert result2 == "Data Scientist"
    
    @patch("spacy.load")
    def test_similarity_calculation(self, mock_load):
        """Test similarity calculation between resume and job description."""
        # Create mock docs with similarity method
        mock_resume_doc = MagicMock()
        mock_job_doc = MagicMock()
        mock_resume_doc.similarity.return_value = 0.75
        
        # Mock NLP function to return different docs based on input
        mock_nlp = MagicMock()
        mock_nlp.side_effect = lambda text: mock_resume_doc if "resume" in text else mock_job_doc
        mock_load.return_value = mock_nlp
        
        # Create classifier
        classifier = ResumeClassifier()
        
        # Calculate similarity
        result = classifier.calculate_similarity("resume text", "job description")
        
        # Verify results
        assert result == 0.75
        mock_resume_doc.similarity.assert_called_once_with(mock_job_doc)


class TestSectionSegmenter:
    """Test suite for section segmenter."""
    
    def test_initialization(self):
        """Test segmenter initialization."""
        segmenter = SectionSegmenter()
        assert segmenter.section_headers is not None
        assert not segmenter.use_ml
    
    @patch("pickle.load")
    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    def test_model_loading(self, mock_exists, mock_open, mock_pickle):
        """Test model loading."""
        # Mock path.exists to return True (model exists)
        mock_exists.return_value = True
        
        # Mock pickle.load to return a model
        mock_model = MagicMock()
        mock_pickle.return_value = mock_model
        
        # Create segmenter with model path
        segmenter = SectionSegmenter(model_path="/path/to/model")
        
        # Verify results
        assert segmenter.use_ml
        assert segmenter.model == mock_model
    
    def test_rule_based_segmentation(self):
        """Test rule-based segmentation."""
        segmenter = SectionSegmenter()
        segmenter.use_ml = False
        
        # Test with resume text containing section headers
        resume_text = """
        John Doe
        john@example.com
        123-456-7890
        
        EDUCATION
        BSc Computer Science, MIT, 2020
        
        EXPERIENCE
        Software Engineer at Acme Corp, 2018-2020
        
        SKILLS
        Python, Java, Machine Learning
        """
        
        result = segmenter.segment(resume_text)
        
        # Verify results
        assert "John Doe" in result.get("PERSONAL_INFO", "")
        assert "BSc Computer Science" in result.get("EDUCATION", "")
        assert "Software Engineer" in result.get("EXPERIENCE", "")
        assert "Python" in result.get("SKILLS", "")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
