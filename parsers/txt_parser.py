"""
TXT Parser module for extracting text from plain text documents.
"""

import logging


class TXTParser:
    """Parser for extracting text from TXT files."""
    
    def parse(self, file_path):
        """
        Extract text content from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Extracted text content as a string
        """
        try:
            logging.debug(f"Parsing TXT file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                return file.read()
                
        except UnicodeDecodeError:
            # Try with different encodings if utf-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e2:
                logging.error(f"Failed to decode TXT file with alternate encoding: {str(e2)}")
                raise
                
        except Exception as e:
            logging.error(f"Error parsing TXT file {file_path}: {str(e)}")
            raise ValueError(f"Failed to extract text from TXT: {str(e)}")
