"""
Helper utility functions for the resume parser.
"""

import os
import logging
import json
from pathlib import Path
import concurrent.futures
from functools import wraps
import time


def setup_logging(log_level="INFO"):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def ensure_directory(directory_path):
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, file_path):
    try:
        # Ensure directory exists
        ensure_directory(os.path.dirname(file_path))
        
        # Write JSON file with proper indentation
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False


def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {str(e)}")
        return None


def batch_process(items, process_func, max_workers=None, batch_size=None, **kwargs):
    # Processes a list of items in parallel batches using ProcessPoolExecutor
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size or len(items)):
        batch = items[i:i + (batch_size or len(items))]
        
        # Process batch in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_func, item, **kwargs): item for item in batch}
            
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error processing {item}: {str(e)}")
                    results.append({"error": str(e), "item": item})
    
    return results


def timer(func):
    # Decorator that logs execution time of the wrapped function
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1].lower()


def is_supported_file(file_path, supported_extensions):
    ext = get_file_extension(file_path)
    return ext in supported_extensions


def find_files(directory, extensions=None, recursive=True):
    file_paths = []
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        logging.error(f"Directory not found: {directory}")
        return file_paths
    
    # Get all files with or without recursion
    if recursive:
        all_files = list(directory.glob('**/*'))
    else:
        all_files = list(directory.glob('*'))
    
    # Filter by extension if specified
    if extensions:
        file_paths = [str(f) for f in all_files if f.is_file() and f.suffix.lower() in extensions]
    else:
        file_paths = [str(f) for f in all_files if f.is_file()]
    
    return file_paths
