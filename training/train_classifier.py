#!/usr/bin/env python3
"""
Train a resume classifier model for job role classification.

This script trains a classifier to categorize resumes into different job roles
using machine learning techniques (TF-IDF + SVM by default).
"""

import os
import json
import logging
import pickle
import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datetime import datetime
from tqdm import tqdm

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import setup_logging, ensure_directory
import config


def setup_argparse():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="Train a resume classifier model for job role classification")
    parser.add_argument(
        "--training-data", 
        required=True, 
        help="Path to JSON training data"
    )
    parser.add_argument(
        "--output", 
        default=config.CLASSIFIER_MODEL_PATH, 
        help="Output directory for the trained model"
    )
    parser.add_argument(
        "--model-type", 
        default="svm", 
        choices=["svm", "random_forest", "logistic_regression"],
        help="Type of classifier model to train"
    )
    parser.add_argument(
        "--max-features", 
        type=int, 
        default=5000, 
        help="Maximum number of features for TF-IDF vectorizer"
    )
    parser.add_argument(
        "--ngram-range", 
        default="1,2", 
        help="N-gram range for TF-IDF vectorizer (e.g., '1,2' for unigrams and bigrams)"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true", 
        help="Perform hyperparameter optimization using grid search"
    )
    parser.add_argument(
        "--eval-split", 
        type=float, 
        default=0.2, 
        help="Fraction of data to use for evaluation"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    return parser.parse_args()


def load_training_data(file_path):
    """
    Load training data from JSON file.
    
    Expected format:
    [
        {
            "text": "Resume text content...",
            "label": "Software Engineer"
        },
        ...
    ]
    
    Args:
        file_path: Path to JSON training data
        
    Returns:
        Tuple of (texts, labels)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            text = item.get("text", "")
            label = item.get("label", "")
            
            # Skip invalid items
            if not text or not label:
                continue
            
            texts.append(text)
            labels.append(label)
        
        logging.info(f"Loaded {len(texts)} training examples with {len(set(labels))} unique labels")
        return texts, labels
        
    except Exception as e:
        logging.error(f"Error loading training data: {str(e)}")
        raise


def preprocess_text(text):
    """
    Preprocess text for classification.
    
    Args:
        text: Text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Basic preprocessing
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def create_model_pipeline(model_type, max_features, ngram_range):
    """
    Create a scikit-learn pipeline for text classification.
    
    Args:
        model_type: Type of classifier model
        max_features: Maximum number of features for TF-IDF vectorizer
        ngram_range: N-gram range for TF-IDF vectorizer
        
    Returns:
        Pipeline object
    """
    # Parse ngram_range
    ngram_min, ngram_max = map(int, ngram_range.split(','))
    
    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        stop_words='english'
    )
    
    # Classifier
    if model_type == "svm":
        clf = LinearSVC(C=1.0, class_weight='balanced')
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    elif model_type == "logistic_regression":
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])
    
    return pipeline


def optimize_hyperparameters(pipeline, X_train, y_train):
    """
    Optimize hyperparameters using grid search.
    
    Args:
        pipeline: Pipeline object
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Optimized pipeline
    """
    # Define parameter grid based on classifier type
    clf_type = pipeline.named_steps['clf'].__class__.__name__
    
    if clf_type == 'LinearSVC':
        param_grid = {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__C': [0.1, 1.0, 10.0]
        }
    elif clf_type == 'RandomForestClassifier':
        param_grid = {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 10, 20]
        }
    elif clf_type == 'LogisticRegression':
        param_grid = {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1.0, 10.0],
            'clf__solver': ['liblinear', 'lbfgs']
        }
    else:
        logging.warning(f"No parameter grid defined for {clf_type}, skipping optimization")
        return pipeline
    
    # Run grid search
    logging.info("Starting hyperparameter optimization with grid search")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Log best parameters
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, label_encoder):
    """
    Train and evaluate the classifier model.
    
    Args:
        pipeline: Pipeline object
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder object
        
    Returns:
        Tuple of (trained_pipeline, metrics)
    """
    # Train the model
    logging.info("Training classifier model")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    logging.info("Evaluating classifier model")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Generate classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Log metrics
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score (weighted): {f1:.4f}")
    logging.info(f"Classification Report:\n{report}")
    
    # Create metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "classification_report": report
    }
    
    return pipeline, metrics


def save_model(pipeline, label_encoder, output_dir, metrics):
    """
    Save the trained model and metadata.
    
    Args:
        pipeline: Trained pipeline
        label_encoder: LabelEncoder object
        output_dir: Output directory
        metrics: Evaluation metrics
        
    Returns:
        Path to the saved model
    """
    # Create output directory
    ensure_directory(output_dir)
    
    # Save pipeline
    pipeline_path = os.path.join(output_dir, "classifier.pkl")
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    # Save label encoder
    label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Create version info
    version_info = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "model_type": pipeline.named_steps['clf'].__class__.__name__,
        "feature_type": pipeline.named_steps['tfidf'].__class__.__name__,
        "classes": label_encoder.classes_.tolist()
    }
    
    version_path = os.path.join(output_dir, "version.json")
    with open(version_path, 'w', encoding='utf-8') as f:
        json.dump(version_info, f, indent=2)
    
    logging.info(f"Model saved to {output_dir}")
    return output_dir


def main():
    """Main entry point for the classifier model trainer."""
    # Parse command line arguments
    args = setup_argparse()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Load training data
        texts, labels = load_training_data(args.training_data)
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, encoded_labels, 
            test_size=args.eval_split, 
            random_state=42, 
            stratify=encoded_labels
        )
        
        # Create model pipeline
        pipeline = create_model_pipeline(
            args.model_type, 
            args.max_features, 
            args.ngram_range
        )
        
        # Optimize hyperparameters if requested
        if args.optimize:
            pipeline = optimize_hyperparameters(pipeline, X_train, y_train)
        
        # Train and evaluate
        trained_pipeline, metrics = train_and_evaluate(
            pipeline, X_train, y_train, X_test, y_test, label_encoder
        )
        
        # Save model
        save_model(trained_pipeline, label_encoder, args.output, metrics)
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
