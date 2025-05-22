"""
Resume classifier module for classifying resumes and calculating similarity scores.
"""

import os
import logging
import pickle
import numpy as np
from pathlib import Path
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import config


class ResumeClassifier:
    """
    Classifier for categorizing resumes by job role and calculating similarity.
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path or config.CLASSIFIER_MODEL_PATH
        self._load_model()
        self._load_spacy()
    
    def _load_model(self):
        try:
            model_file = Path(self.model_path) / "classifier.pkl"
            label_encoder_file = Path(self.model_path) / "label_encoder.pkl"
            
            if model_file.exists() and label_encoder_file.exists():
                logging.info(f"Loading classifier model from {model_file}")
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(label_encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                self.is_trained = True
            else:
                logging.warning("No trained classifier model found. Using fallback.")
                self._create_fallback_model()
                self.is_trained = False
        
        except Exception as e:
            logging.error(f"Error loading classifier model: {str(e)}")
            self._create_fallback_model()
            self.is_trained = False
    
    def _create_fallback_model(self):
        logging.info("Creating fallback classifier model")
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LinearSVC(C=1.0))
        ])
        
        # Create a basic label encoder with common job roles
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array([
            'Software Engineer', 
            'Data Scientist', 
            'Product Manager',
            'UX Designer',
            'Marketing Specialist',
            'Sales Representative',
            'Project Manager',
            'Business Analyst',
            'Frontend Developer',
            'Backend Developer'
        ])
    
    def _load_spacy(self):
        try:
            logging.info("Loading spaCy model for similarity calculations")
            self.nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner', 'tagger'])
        except Exception as e:
            logging.error(f"Error loading spaCy model: {str(e)}")
            raise
    
    def classify(self, text):
        try:
            if not self.is_trained:
                # Without a trained model, use a heuristic approach
                return self._heuristic_classification(text)
            
            processed_text = self._preprocess_text(text)
            predicted_label = self.model.predict([processed_text])[0]
            job_role = self.label_encoder.inverse_transform([predicted_label])[0]
            
            logging.info(f"Classified resume as: {job_role}")
            return job_role
            
        except Exception as e:
            logging.error(f"Error classifying resume: {str(e)}")
            return "Unknown"
    
    def _heuristic_classification(self, text):
        # Keyword-based classification when no ML model is available
        text = text.lower()
        
        # Define keywords for different roles
        role_keywords = {
            'Software Engineer': ['software engineer', 'software developer', 'programming', 'code', 'develop', 
                               'java', 'python', 'c++', 'javascript', 'full stack', 'backend', 'frontend'],
            'Data Scientist': ['data scientist', 'machine learning', 'ml', 'ai', 'artificial intelligence', 
                            'analytics', 'statistics', 'statistical', 'python', 'r', 'tensorflow', 'pytorch'],
            'Product Manager': ['product manager', 'product management', 'product owner', 'roadmap', 
                             'requirements', 'user stories', 'agile', 'scrum', 'backlog'],
            'UX Designer': ['ux', 'ui', 'user experience', 'user interface', 'design', 'wireframe', 
                         'prototype', 'usability', 'figma', 'sketch', 'adobe xd'],
            'Marketing Specialist': ['marketing', 'digital marketing', 'seo', 'social media', 'content', 
                                  'campaign', 'brand', 'advertising'],
            'Sales Representative': ['sales', 'account', 'client', 'customer', 'revenue', 'pipeline', 
                                  'quota', 'crm', 'salesforce'],
            'Project Manager': ['project manager', 'project management', 'pmp', 'scrum master', 
                             'agile', 'timeline', 'milestone', 'deliverable', 'gantt'],
            'Business Analyst': ['business analyst', 'requirements', 'business process', 'data analysis', 
                              'sql', 'reporting', 'stakeholder', 'documentation'],
            'Frontend Developer': ['frontend', 'front-end', 'ui', 'react', 'angular', 'vue', 'html', 
                                'css', 'javascript', 'web development'],
            'Backend Developer': ['backend', 'back-end', 'server', 'api', 'database', 'sql', 'nosql', 
                               'node', 'django', 'express', 'java', 'python']
        }
        
        # Count matches for each role
        role_scores = {role: 0 for role in role_keywords}
        
        for role, keywords in role_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    role_scores[role] += 1
        
        # Find role with highest score
        if max(role_scores.values()) > 0:
            best_role = max(role_scores.items(), key=lambda x: x[1])[0]
            return best_role
        else:
            return "Unknown"
    
    def calculate_similarity(self, resume_text, job_description):
        try:
            # Use spaCy's vector-based similarity
            resume_doc = self.nlp(resume_text)
            job_doc = self.nlp(job_description)
            similarity = resume_doc.similarity(job_doc)
            
            logging.info(f"Resume-job similarity score: {similarity:.2f}")
            return round(similarity, 2)
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _preprocess_text(self, text):
        # Process with spaCy and filter to meaningful tokens
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and token.is_alpha]
        return ' '.join(tokens)
    
    def train(self, texts, labels):
        try:
            logging.info(f"Training classifier with {len(texts)} samples")
            
            processed_texts = [self._preprocess_text(text) for text in texts]
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', LinearSVC(C=1.0))
            ])
            
            self.model.fit(processed_texts, encoded_labels)
            
            # Save the trained model
            os.makedirs(self.model_path, exist_ok=True)
            
            with open(Path(self.model_path) / "classifier.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(Path(self.model_path) / "label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            self.is_trained = True
            logging.info("Classifier training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error training classifier: {str(e)}")
            return False
