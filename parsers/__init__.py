"""
File parsers module for extracting text from different file formats.
"""

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.txt_parser import TXTParser
from parsers.html_parser import HTMLParser


def get_parser(file_extension):
    """
    Factory function to get the appropriate parser based on file extension.
    
    Args:
        file_extension: The file extension (including the dot)
        
    Returns:
        A parser instance for the given file type
    """
    parsers = {
        ".pdf": PDFParser(),
        ".docx": DOCXParser(),
        ".txt": TXTParser(),
        ".html": HTMLParser(),
    }
    
    return parsers.get(file_extension.lower())
