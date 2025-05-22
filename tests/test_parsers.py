"""
Tests for the file parsers.
"""

import os
import pytest
from unittest.mock import patch, mock_open

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.txt_parser import TXTParser
from parsers.html_parser import HTMLParser
from parsers import get_parser


class TestParsers:
    """Test suite for file parsers."""
    
    def test_get_parser(self):
        """Test the parser factory function."""
        assert isinstance(get_parser(".pdf"), PDFParser)
        assert isinstance(get_parser(".docx"), DOCXParser)
        assert isinstance(get_parser(".txt"), TXTParser)
        assert isinstance(get_parser(".html"), HTMLParser)
        assert get_parser(".unknown") is None
    
    @patch("parsers.pdf_parser.extract_text")
    def test_pdf_parser(self, mock_extract_text):
        """Test PDF parser."""
        # Mock the extract_text function
        mock_extract_text.return_value = "Sample PDF Content"
        
        # Create parser and parse
        parser = PDFParser()
        result = parser.parse("sample.pdf")
        
        # Verify results
        assert result == "Sample PDF Content"
        mock_extract_text.assert_called_once_with("sample.pdf")
    
    @patch("docx.Document")
    def test_docx_parser(self, mock_document):
        """Test DOCX parser."""
        # Mock Document class
        mock_doc = mock_document.return_value
        mock_doc.paragraphs = [
            type('obj', (object,), {'text': 'Paragraph 1'}),
            type('obj', (object,), {'text': 'Paragraph 2'})
        ]
        mock_doc.tables = [
            type('obj', (object,), {
                'rows': [
                    type('obj', (object,), {
                        'cells': [
                            type('obj', (object,), {'text': 'Cell 1'}),
                            type('obj', (object,), {'text': 'Cell 2'})
                        ]
                    })
                ]
            })
        ]
        
        # Create parser and parse
        parser = DOCXParser()
        result = parser.parse("sample.docx")
        
        # Verify results
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result
        assert "Cell 1 | Cell 2" in result
        mock_document.assert_called_once_with("sample.docx")
    
    @patch("builtins.open", new_callable=mock_open, read_data="Sample TXT Content")
    def test_txt_parser(self, mock_file):
        """Test TXT parser."""
        # Create parser and parse
        parser = TXTParser()
        result = parser.parse("sample.txt")
        
        # Verify results
        assert result == "Sample TXT Content"
        mock_file.assert_called_once_with("sample.txt", 'r', encoding='utf-8', errors='replace')
    
    @patch("builtins.open", new_callable=mock_open, read_data="<html><body>Sample HTML Content</body></html>")
    def test_html_parser(self, mock_file):
        """Test HTML parser."""
        # Create parser and parse
        parser = HTMLParser()
        result = parser.parse("sample.html")
        
        # Verify results
        assert "Sample HTML Content" in result
        mock_file.assert_called_once_with("sample.html", 'r', encoding='utf-8', errors='replace')
    
    @patch("builtins.open")
    def test_txt_parser_encoding_fallback(self, mock_file):
        """Test TXT parser with encoding fallback."""
        # First open raises UnicodeDecodeError, second succeeds
        mock_file.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'Test error'),
            mock_open(read_data="Sample TXT Content").return_value
        ]
        
        # Create parser and parse
        parser = TXTParser()
        result = parser.parse("sample.txt")
        
        # Verify results
        assert result == "Sample TXT Content"
        assert mock_file.call_count == 2


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
