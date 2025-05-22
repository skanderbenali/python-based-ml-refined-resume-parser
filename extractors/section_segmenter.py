"""
Section segmenter module for identifying different sections in a resume.
"""

import re
import logging
from pathlib import Path
import pickle
import os
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

import config


class SectionSegmenter:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.section_headers = self._get_section_headers()
        self._load_model()

    def _get_section_headers(self):
        return {
            "PERSONAL_INFO": [
                r"personal\s+information",
                r"contact\s+information",
                r"contact",
                r"personal\s+details",
                r"personal\s+profile"
            ],
            "EDUCATION": [
                r"education",
                r"academic\s+background",
                r"educational\s+qualifications",
                r"academic\s+qualifications",
                r"qualifications"
            ],
            "EXPERIENCE": [
                r"experience",
                r"work\s+experience",
                r"employment\s+history",
                r"professional\s+experience",
                r"career\s+history",
                r"work\s+history"
            ],
            "SKILLS": [
                r"skills",
                r"technical\s+skills",
                r"core\s+competencies",
                r"competencies",
                r"key\s+skills",
                r"professional\s+skills"
            ],
            "CERTIFICATIONS": [
                r"certifications",
                r"professional\s+certifications",
                r"certificates",
                r"accreditations",
                r"licenses"
            ],
            "PROJECTS": [
                r"projects",
                r"key\s+projects",
                r"project\s+experience",
                r"academic\s+projects",
                r"personal\s+projects"
            ]
        }

    def _load_model(self):
        try:
            if self.model_path and Path(self.model_path).exists():
                logging.info(f"Loading section segmenter model from {self.model_path}")
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.use_ml = True
            else:
                logging.info("No section segmenter model found. Using rule-based approach.")
                self.use_ml = False
        except Exception as e:
            logging.error(f"Error loading section segmenter model: {str(e)}")
            self.use_ml = False

    def segment(self, text):
        try:
            logging.info("Segmenting resume into sections")
            if self.use_ml:
                return self._segment_with_ml(text)
            else:
                return self._segment_with_rules(text)
        except Exception as e:
            logging.error(f"Error segmenting resume: {str(e)}")
            return {section: "" for section in config.RESUME_SECTIONS}

    def _segment_with_rules(self, text):
        sections = {}
        lines = text.split('\n')
        section_indices = {}
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for section, patterns in self.section_headers.items():
                for pattern in patterns:
                    if re.search(pattern, line_lower, re.IGNORECASE):
                        section_indices[i] = section
                        break
        sorted_indices = sorted(section_indices.keys())
        for i in range(len(sorted_indices)):
            start_idx = sorted_indices[i]
            section = section_indices[start_idx]
            if i < len(sorted_indices) - 1:
                end_idx = sorted_indices[i + 1]
            else:
                end_idx = len(lines)
            section_content = '\n'.join(lines[start_idx + 1:end_idx]).strip()
            sections[section] = section_content
        if "PERSONAL_INFO" not in sections and len(lines) > 0:
            first_section_idx = sorted_indices[0] if sorted_indices else len(lines)
            personal_info = '\n'.join(lines[:first_section_idx]).strip()
            sections["PERSONAL_INFO"] = personal_info
        for section in config.RESUME_SECTIONS:
            if section not in sections:
                sections[section] = ""
        return sections

    def _segment_with_ml(self, text):
        sections = {section: "" for section in config.RESUME_SECTIONS}
        paragraphs = self._split_into_paragraphs(text)
        if not paragraphs:
            return sections
        for paragraph in paragraphs:
            if paragraph.strip():
                prediction = self.model.predict([paragraph])[0]
                section = prediction if prediction in config.RESUME_SECTIONS else "OTHER"
                if section != "OTHER":
                    if sections[section]:
                        sections[section] += "\n\n" + paragraph
                    else:
                        sections[section] = paragraph
        return sections

    def _split_into_paragraphs(self, text):
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def train(self, labeled_data):
        try:
            logging.info(f"Training section segmenter with {len(labeled_data)} samples")
            texts, labels = zip(*labeled_data)
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', OneVsRestClassifier(LinearSVC()))
            ])
            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test)
            logging.info(f"Section segmenter accuracy: {accuracy:.4f}")
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.use_ml = True
            logging.info("Section segmenter training completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error training section segmenter: {str(e)}")
            return False
