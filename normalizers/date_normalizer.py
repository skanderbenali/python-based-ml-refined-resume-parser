"""
Date normalizer module for standardizing date formats in resume data.
"""

import re
import logging
from datetime import datetime


class DateNormalizer:
    
    def __init__(self):
        # Month abbreviation to number mapping
        self.month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Regex patterns for different date formats
        self.patterns = {
            # Month Year: Jan 2020, January 2020
            'month_year': re.compile(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})', re.IGNORECASE),
            
            # MM/DD/YYYY or DD/MM/YYYY
            'slash_date': re.compile(r'(\d{1,2})/(\d{1,2})/(\d{2,4})'),
            
            # MM-DD-YYYY or DD-MM-YYYY
            'hyphen_date': re.compile(r'(\d{1,2})-(\d{1,2})-(\d{2,4})'),
            
            # Year only: 2020
            'year': re.compile(r'\b(19|20)\d{2}\b'),
            
            # Date range: 2018 - 2020, Jan 2018 - Mar 2020
            'date_range': re.compile(r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(\d{4})\s*[-–—to]\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(\d{4}|present|current|now)', re.IGNORECASE),
        }
    
    def normalize(self, date_string):
        if not date_string or not isinstance(date_string, str):
            return ""
        
        date_string = date_string.strip().lower()
        
        try:
            # Try Month Year format (Jan 2020)
            match = self.patterns['month_year'].search(date_string)
            if match:
                month_abbr = match.group(1).lower()[:3]
                year = match.group(2)
                month = self.month_map.get(month_abbr, '01')
                return f"{year}-{month}"
            
            # Try slash date format (MM/DD/YYYY or DD/MM/YYYY)
            match = self.patterns['slash_date'].search(date_string)
            if match:
                # Assume MM/DD/YYYY for simplicity
                month, day, year = match.groups()
                # Handle 2-digit years
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                month = month.zfill(2)
                day = day.zfill(2)
                return f"{year}-{month}-{day}"
            
            # Try hyphen date format (MM-DD-YYYY or DD-MM-YYYY)
            match = self.patterns['hyphen_date'].search(date_string)
            if match:
                # Assume MM-DD-YYYY for simplicity
                month, day, year = match.groups()
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                month = month.zfill(2)
                day = day.zfill(2)
                return f"{year}-{month}-{day}"
            
            # Try year only
            match = self.patterns['year'].search(date_string)
            if match:
                return match.group(0)
            
            return date_string
            
        except Exception as e:
            logging.error(f"Error normalizing date '{date_string}': {str(e)}")
            return date_string
    
    def normalize_duration(self, duration_string):
        if not duration_string or not isinstance(duration_string, str):
            return ""
        
        duration_string = duration_string.strip().lower()
        
        try:
            # Handle date ranges like "2018 - 2020" or "Jan 2018 - Mar 2020"
            match = self.patterns['date_range'].search(duration_string)
            if match:
                start_month_str, start_year, end_month_str, end_year = match.groups()
                
                # Format start date
                start_date = ""
                if start_month_str:
                    start_month = self.month_map.get(start_month_str.lower()[:3], '01')
                    start_date = f"{start_year}-{start_month}"
                else:
                    start_date = start_year
                
                # Format end date
                end_date = ""
                if end_year.lower() in ('present', 'current', 'now'):
                    end_date = "Present"
                elif end_month_str:
                    end_month = self.month_map.get(end_month_str.lower()[:3], '01')
                    end_date = f"{end_year}-{end_month}"
                else:
                    end_date = end_year
                
                return f"{start_date} to {end_date}"
            
            # Try as a single date if not a range
            normalized_date = self.normalize(duration_string)
            if normalized_date != duration_string:
                return normalized_date
            
            return duration_string
            
        except Exception as e:
            logging.error(f"Error normalizing duration '{duration_string}': {str(e)}")
            return duration_string
    
    def normalize_year(self, year_string):
        if not year_string or not isinstance(year_string, str):
            return ""
        
        year_string = year_string.strip().lower()
        
        try:
            # Extract a year value from various formats
            match = self.patterns['year'].search(year_string)
            if match:
                return match.group(0)
            
            match = self.patterns['month_year'].search(year_string)
            if match:
                return match.group(2)  # Return the year part
            
            for pattern_name in ['slash_date', 'hyphen_date']:
                match = self.patterns[pattern_name].search(year_string)
                if match:
                    year = match.group(3)
                    # Handle 2-digit years
                    if len(year) == 2:
                        year = '20' + year if int(year) < 50 else '19' + year
                    return year
            
            return year_string
            
        except Exception as e:
            logging.error(f"Error extracting year from '{year_string}': {str(e)}")
            return year_string
