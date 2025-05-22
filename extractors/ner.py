"""
Named Entity Recognition (NER) module for extracting entities from resume text.
"""

import re
import logging
import spacy
from pathlib import Path
import os

import config


class NERExtractor:
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self._load_model()
        self._initialize_patterns()
    
    def _load_model(self):
        try:
            if self.model_path and Path(self.model_path).exists():
                logging.info(f"Loading custom NER model from {self.model_path}")
                self.nlp = spacy.load(self.model_path)
            else:
                logging.info("Loading default spaCy model (en_core_web_lg)")
                self.nlp = spacy.load("en_core_web_lg")
                
            # Optimize pipeline by disabling components we don't need
            disabled_pipes = ['tagger', 'parser', 'attribute_ruler', 'lemmatizer']
            self.nlp.select_pipes(disable=disabled_pipes)
            
        except Exception as e:
            logging.error(f"Error loading spaCy model: {str(e)}")
            raise
    
    def _initialize_patterns(self):
        # Common regex patterns for entity extraction
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'(?:\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}')
        self.degree_pattern = re.compile(r'\b(?:B\.?S\.?|M\.?S\.?|Ph\.?D\.?|B\.?A\.?|M\.?A\.?|MBA|Bachelor|Master|Doctor|Doctorate|Associate)\b')
        self.job_title_pattern = re.compile(r'\b(?:Engineer|Developer|Manager|Director|Analyst|Specialist|Designer|Consultant|Coordinator|Assistant|Lead|Senior|Junior)\b')
        self.date_pattern = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b')
    
    def extract(self, text, sections=None):
        try:
            logging.info("Extracting entities from resume text")
            
            doc = self.nlp(text)
            entities = {label: [] for label in config.NER_LABELS}
            
            self._extract_ner_entities(doc, entities)
            self._extract_regex_entities(text, entities)
            
            if sections:
                self._extract_section_entities(sections, entities)
            
            # Group related entities together (education, experience)
            self._consolidate_education_entities(entities)
            self._consolidate_experience_entities(entities)
            
            return entities
            
        except Exception as e:
            logging.error(f"Error extracting entities: {str(e)}")
            raise
    
    def _extract_ner_entities(self, doc, entities):
        # Map spaCy entity labels to our custom labels
        label_mapping = {
            "PERSON": "NAME",
            "ORG": "EXPERIENCE_COMPANY",
            "GPE": "LOCATION",
            "DATE": "EDUCATION_YEAR",  # Might also be experience duration
            "CARDINAL": None,  # Ignore
            "ORDINAL": None,  # Ignore
            "MONEY": None,  # Ignore
            "PERCENT": None  # Ignore
        }
        
        for ent in doc.ents:
            mapped_label = label_mapping.get(ent.label_)
            if mapped_label:
                entities[mapped_label].append(ent.text)
    
    def _extract_regex_entities(self, text, entities):
        # Extract contact information and dates
        emails = self.email_pattern.findall(text)
        if emails:
            entities["EMAIL"] = emails[0]  # Assume first email is the primary one
        
        phones = self.phone_pattern.findall(text)
        if phones:
            entities["PHONE"] = phones[0]  # Assume first phone is the primary one
        
        dates = self.date_pattern.findall(text)
        if dates:
            entities["EDUCATION_YEAR"].extend(dates)
    
    def _extract_section_entities(self, sections, entities):
        # Process education section
        if "EDUCATION" in sections:
            education_text = sections["EDUCATION"]
            education_doc = self.nlp(education_text)
            
            # Get institutions from education section
            for ent in education_doc.ents:
                if ent.label_ == "ORG":
                    entities["EDUCATION_INSTITUTION"].append(ent.text)
            
            degrees = self.degree_pattern.findall(education_text)
            entities["EDUCATION_DEGREE"].extend(degrees)
        
        # Process experience section
        if "EXPERIENCE" in sections:
            experience_text = sections["EXPERIENCE"]
            experience_doc = self.nlp(experience_text)
            
            for ent in experience_doc.ents:
                if ent.label_ == "ORG":
                    entities["EXPERIENCE_COMPANY"].append(ent.text)
            
            job_titles = self.job_title_pattern.findall(experience_text)
            entities["EXPERIENCE_TITLE"].extend(job_titles)
            
            dates = self.date_pattern.findall(experience_text)
            entities["EXPERIENCE_DURATION"].extend(dates)
        
        # Process skills section
        if "SKILLS" in sections:
            skills_text = sections["SKILLS"]
            skill_doc = self.nlp(skills_text)
            
            # Extract noun phrases as potential skills
            for chunk in skill_doc.noun_chunks:
                entities["SKILL"].append(chunk.text)
            
            # Also get individual proper nouns
            for token in skill_doc:
                if token.pos_ == "PROPN" and token.text not in entities["SKILL"]:
                    entities["SKILL"].append(token.text)
    
    def _consolidate_education_entities(self, entities):
        # Group education details (degree, institution, year) together
        degrees = entities.pop("EDUCATION_DEGREE", [])
        institutions = entities.pop("EDUCATION_INSTITUTION", [])
        years = entities.pop("EDUCATION_YEAR", [])
        
        education_entries = []
        
        for i in range(max(len(degrees), len(institutions))):
            entry = {}
            if i < len(degrees):
                entry["EDUCATION_DEGREE"] = degrees[i]
            if i < len(institutions):
                entry["EDUCATION_INSTITUTION"] = institutions[i]
            if i < len(years):
                entry["EDUCATION_YEAR"] = years[i]
            
            if entry:
                education_entries.append(entry)
        
        entities["EDUCATION"] = education_entries
    
    def _consolidate_experience_entities(self, entities):
        # Group experience details (job title, company, duration) together
        titles = entities.pop("EXPERIENCE_TITLE", [])
        companies = entities.pop("EXPERIENCE_COMPANY", [])
        durations = entities.pop("EXPERIENCE_DURATION", [])
        
        experience_entries = []
        
        for i in range(max(len(titles), len(companies))):
            entry = {}
            if i < len(titles):
                entry["EXPERIENCE_TITLE"] = titles[i]
            if i < len(companies):
                entry["EXPERIENCE_COMPANY"] = companies[i]
            if i < len(durations):
                entry["EXPERIENCE_DURATION"] = durations[i]
            
            if entry:
                experience_entries.append(entry)
        
        entities["EXPERIENCE"] = experience_entries
