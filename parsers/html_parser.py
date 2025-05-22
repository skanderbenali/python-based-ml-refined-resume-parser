"""
HTML Parser module for extracting text from HTML documents.
"""

import logging
from bs4 import BeautifulSoup


class HTMLParser:
    """Parser for extracting text from HTML files."""
    
    def parse(self, file_path):
        """
        Extract text content from an HTML file.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Extracted text content as a string
        """
        try:
            logging.debug(f"Parsing HTML file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                html_content = file.read()
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text: normalize whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
                
        except UnicodeDecodeError:
            # Try with different encodings if utf-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    html_content = file.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
            except Exception as e2:
                logging.error(f"Failed to decode HTML file with alternate encoding: {str(e2)}")
                raise
                
        except Exception as e:
            logging.error(f"Error parsing HTML file {file_path}: {str(e)}")
            raise ValueError(f"Failed to extract text from HTML: {str(e)}")
