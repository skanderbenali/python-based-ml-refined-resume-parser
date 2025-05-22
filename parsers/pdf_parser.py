"""
PDF parser module for extracting text from PDF files.
"""

import os
import io
import logging
from pathlib import Path

import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage


class PDFParser:
    """Parser for extracting text from PDF files."""
    
    def __init__(self, fallback_to_pdfminer=True):
        self.fallback_to_pdfminer = fallback_to_pdfminer

    def parse(self, file_path):
        """
        Extract text content from a PDF file.
        
        This method first tries using PyPDF2 which is faster. If PyPDF2 fails or returns empty text, 
        it falls back to pdfminer.six which gives better results with text layout preservation.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content as a string
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""
        
        logging.info(f"Extracting text from PDF: {file_path}")
        
        try:
            # First try with PyPDF2
            text = self._extract_with_pypdf2(file_path)
            
            # If PyPDF2 fails or returns empty text, try pdfminer
            if not text.strip() and self.fallback_to_pdfminer:
                logging.info(f"PyPDF2 returned empty text, trying pdfminer for: {file_path}")
                text = self._extract_with_pdfminer(file_path)
            
            return text
            
        except Exception as e:
            logging.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""

    def _extract_with_pypdf2(self, file_path):
        """
        Extract text using PyPDF2.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content as a string
        """
        text = ""
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        return text

    def _extract_with_pdfminer(self, file_path):
        """
        Extract text using pdfminer with custom parameters.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content as a string
        """
        text_output = io.StringIO()
        with open(file_path, 'rb') as pdf_file:
            resource_manager = PDFResourceManager()
            device = TextConverter(resource_manager, text_output, laparams=LAParams())
            interpreter = PDFPageInterpreter(resource_manager, device)
            
            for page in PDFPage.get_pages(pdf_file):
                interpreter.process_page(page)
        
        text = text_output.getvalue()
        text_output.close()
        return text
