#!/usr/bin/env python3
"""
Train a custom spaCy NER model for resume parsing.

This script trains a Named Entity Recognition model to identify 
resume-specific entities like skills, job titles, and education details.
"""

import os
import json
import logging
import random
import spacy
import argparse
from pathlib import Path
from spacy.tokens import DocBin
from spacy.training import Example
from tqdm import tqdm

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import setup_logging, ensure_directory
import config


def setup_argparse():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="Train a custom spaCy NER model for resume parsing")
    parser.add_argument(
        "--training-data", 
        required=True, 
        help="Path to JSON training data"
    )
    parser.add_argument(
        "--base-model", 
        default="en_core_web_lg", 
        help="Base spaCy model to use"
    )
    parser.add_argument(
        "--output", 
        default=config.NER_MODEL_PATH, 
        help="Output directory for the trained model"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.2, 
        help="Dropout rate"
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
            "text": "John Doe has 5 years of experience in Python and JavaScript.",
            "entities": [
                [0, 8, "NAME"],
                [25, 31, "EXPERIENCE_DURATION"],
                [45, 51, "SKILL"],
                [56, 66, "SKILL"]
            ]
        },
        ...
    ]
    
    Args:
        file_path: Path to JSON training data
        
    Returns:
        List of (text, entities) tuples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        training_data = []
        for item in data:
            text = item.get("text", "")
            entities = item.get("entities", [])
            
            # Skip invalid items
            if not text or not entities:
                continue
            
            # Convert to (start, end, label) format
            spans = []
            for start, end, label in entities:
                if start < end and start >= 0 and end <= len(text):
                    spans.append((start, end, label))
            
            training_data.append((text, {"entities": spans}))
        
        logging.info(f"Loaded {len(training_data)} training examples")
        return training_data
        
    except Exception as e:
        logging.error(f"Error loading training data: {str(e)}")
        raise


def split_training_data(training_data, eval_split=0.2):
    """
    Split training data into training and evaluation sets.
    
    Args:
        training_data: List of (text, entities) tuples
        eval_split: Fraction of data to use for evaluation
        
    Returns:
        Tuple of (train_data, eval_data)
    """
    # Shuffle data
    random.shuffle(training_data)
    
    # Split data
    split_point = int(len(training_data) * (1 - eval_split))
    train_data = training_data[:split_point]
    eval_data = training_data[split_point:]
    
    logging.info(f"Split data into {len(train_data)} training and {len(eval_data)} evaluation examples")
    return train_data, eval_data


def create_docbin(nlp, data, output_path):
    """
    Create a DocBin from training data and save to disk.
    
    Args:
        nlp: spaCy model
        data: List of (text, entities) tuples
        output_path: Path to save the DocBin
    """
    doc_bin = DocBin()
    
    for text, annotations in tqdm(data, desc="Creating DocBin"):
        doc = nlp.make_doc(text)
        ents = []
        
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        
        doc.ents = ents
        doc_bin.add(doc)
    
    doc_bin.to_disk(output_path)
    logging.info(f"Saved DocBin to {output_path}")


def train_model(train_path, eval_path, base_model, output_dir, epochs, batch_size, dropout):
    """
    Train a spaCy NER model using the provided training data.
    
    Args:
        train_path: Path to training DocBin
        eval_path: Path to evaluation DocBin
        base_model: Base spaCy model to use
        output_dir: Output directory for the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        dropout: Dropout rate
        
    Returns:
        Path to the trained model
    """
    # Create config file
    config_path = os.path.join(output_dir, "config.cfg")
    
    # Define training configuration
    config = f"""
    [paths]
    train = {train_path}
    dev = {eval_path}
    
    [system]
    gpu_allocator = null
    seed = 42
    
    [nlp]
    lang = "en"
    pipeline = ["tok2vec", "ner"]
    batch_size = {batch_size}
    
    [components]
    
    [components.tok2vec]
    factory = "tok2vec"
    
    [components.tok2vec.model]
    @architectures = "spacy.Tok2Vec.v2"
    
    [components.tok2vec.model.embed]
    @architectures = "spacy.MultiHashEmbed.v2"
    width = 96
    attrs = ["LOWER", "PREFIX", "SUFFIX", "SHAPE"]
    rows = [5000, 1000, 2500, 2500]
    include_static_vectors = true
    
    [components.tok2vec.model.encode]
    @architectures = "spacy.MaxoutWindowEncoder.v2"
    width = 96
    depth = 4
    window_size = 1
    maxout_pieces = 3
    
    [components.ner]
    factory = "ner"
    
    [components.ner.model]
    @architectures = "spacy.TransitionBasedParser.v2"
    state_type = "ner"
    extra_state_tokens = false
    hidden_width = 64
    maxout_pieces = 2
    use_upper = true
    nO = null
    
    [corpora]
    
    [corpora.train]
    @readers = "spacy.Corpus.v1"
    path = ${train_path}
    max_length = 0
    
    [corpora.dev]
    @readers = "spacy.Corpus.v1"
    path = ${eval_path}
    max_length = 0
    
    [training]
    dev_corpus = "corpora.dev"
    train_corpus = "corpora.train"
    
    [training.optimizer]
    @optimizers = "Adam.v1"
    beta1 = 0.9
    beta2 = 0.999
    L2_is_weight_decay = true
    L2 = 0.01
    grad_clip = 1.0
    use_averages = false
    eps = 0.00000001
    
    [training.optimizer.learn_rate]
    @schedules = "warmup_linear.v1"
    warmup_steps = 250
    total_steps = {epochs * 10000}
    initial_rate = 0.00005
    
    [training.batcher]
    @batchers = "spacy.batch_by_words.v1"
    discard_oversize = false
    tolerance = 0.2
    
    [training.batcher.size]
    @schedules = "compounding.v1"
    start = 100
    stop = 1000
    compound = 1.001
    
    [training.logger]
    @loggers = "spacy.ConsoleLogger.v1"
    progress_bar = true
    
    [training.freezing]
    
    [initialize]
    vectors = ${base_model}
    init_tok2vec = ${base_model}
    vocab_data = ${base_model}
    lookups = ${base_model}
    
    [initialize.components]
    
    [initialize.tokenizer]
    """
    
    # Write config to file
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config)
    
    # Run training using spaCy CLI
    train_cmd = f"python -m spacy train {config_path} --output {output_dir} --epochs {epochs}"
    logging.info(f"Starting training with command: {train_cmd}")
    os.system(train_cmd)
    
    return output_dir


def create_version_info(output_dir):
    """
    Create version info file for the model.
    
    Args:
        output_dir: Path to the model directory
    """
    version_info = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "spacy_version": spacy.__version__,
        "labels": config.NER_LABELS
    }
    
    version_path = os.path.join(output_dir, "version.json")
    with open(version_path, 'w', encoding='utf-8') as f:
        json.dump(version_info, f, indent=2)
    
    logging.info(f"Created version info file at {version_path}")


def evaluate_model(model_path, eval_data):
    """
    Evaluate the trained model on the evaluation data.
    
    Args:
        model_path: Path to the trained model
        eval_data: List of (text, entities) tuples
        
    Returns:
        Evaluation metrics
    """
    try:
        # Load the trained model
        nlp = spacy.load(model_path)
        
        # Initialize metrics
        correct = 0
        incorrect = 0
        partial = 0
        missed = 0
        total_gold = 0
        total_pred = 0
        
        # Evaluate on each example
        for text, annotations in tqdm(eval_data, desc="Evaluating"):
            # Get gold entities
            gold_entities = set()
            for start, end, label in annotations["entities"]:
                gold_entities.add((start, end, label))
            total_gold += len(gold_entities)
            
            # Get predicted entities
            doc = nlp(text)
            pred_entities = set()
            for ent in doc.ents:
                pred_entities.add((ent.start_char, ent.end_char, ent.label_))
            total_pred += len(pred_entities)
            
            # Count matches
            for pred in pred_entities:
                pred_start, pred_end, pred_label = pred
                
                # Check for exact matches
                if pred in gold_entities:
                    correct += 1
                    continue
                
                # Check for label matches with overlap
                found_partial = False
                for gold_start, gold_end, gold_label in gold_entities:
                    # Check if entities overlap and have same label
                    overlap = min(pred_end, gold_end) - max(pred_start, gold_start)
                    if overlap > 0 and pred_label == gold_label:
                        partial += 1
                        found_partial = True
                        break
                
                if not found_partial:
                    incorrect += 1
            
            # Count missed entities
            for gold in gold_entities:
                gold_start, gold_end, gold_label = gold
                
                # Check if entity was missed
                found = False
                for pred_start, pred_end, pred_label in pred_entities:
                    # Check if entities overlap and have same label
                    overlap = min(pred_end, gold_end) - max(pred_start, gold_start)
                    if overlap > 0 and pred_label == gold_label:
                        found = True
                        break
                
                if not found:
                    missed += 1
        
        # Calculate metrics
        precision = correct / total_pred if total_pred > 0 else 0
        recall = correct / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
            "incorrect": incorrect,
            "partial": partial,
            "missed": missed,
            "total_gold": total_gold,
            "total_pred": total_pred
        }
        
        # Log metrics
        logging.info(f"Evaluation metrics: {metrics}")
        
        # Save metrics to file
        metrics_path = os.path.join(model_path, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        return None


def main():
    """Main entry point for the NER model trainer."""
    # Parse command line arguments
    args = setup_argparse()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Load training data
        training_data = load_training_data(args.training_data)
        
        # Split data into training and evaluation sets
        train_data, eval_data = split_training_data(training_data, args.eval_split)
        
        # Create output directory
        output_dir = ensure_directory(args.output)
        
        # Load base model
        logging.info(f"Loading base model: {args.base_model}")
        nlp = spacy.load(args.base_model)
        
        # Prepare temporary directories for DocBins
        temp_dir = ensure_directory(os.path.join(output_dir, "temp"))
        train_path = os.path.join(temp_dir, "train.spacy")
        eval_path = os.path.join(temp_dir, "eval.spacy")
        
        # Create DocBins
        create_docbin(nlp, train_data, train_path)
        create_docbin(nlp, eval_data, eval_path)
        
        # Train model
        model_path = train_model(
            train_path, 
            eval_path, 
            args.base_model, 
            output_dir, 
            args.epochs, 
            args.batch_size, 
            args.dropout
        )
        
        # Create version info
        create_version_info(model_path)
        
        # Evaluate model
        evaluate_model(model_path, eval_data)
        
        logging.info(f"Training completed. Model saved to {model_path}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
