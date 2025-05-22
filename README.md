# Resume Parser

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.2%2B-green)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Made with ML](https://img.shields.io/badge/Made%20with-ML-ff69b4)](https://madewithml.com/)
[![NLP](https://img.shields.io/badge/NLP-Resume%20Parsing-orange)](https://en.wikipedia.org/wiki/Natural_language_processing)

A professional Python-based resume parser using spaCy and machine learning to extract, classify, and structure resume data into a standardized JSON format. This tool efficiently processes resumes from multiple file formats and provides detailed, structured information for ATS systems and HR professionals.

## Features

- **Multi-format Support**: Process resumes in PDF, DOCX, TXT, and HTML formats
- **Advanced NER Extraction**:
  - Personal information (name, email, phone, location)
  - Education details (institution, degree, dates)
  - Work experience (company, title, responsibilities, duration)
  - Skills with categorization
  - Certifications and projects
- **Machine Learning Capabilities**:
  - Resume classification by job role (e.g., Software Engineer, Data Scientist)
  - Job description similarity scoring for matching
  - Intelligent section segmentation
- **Smart Normalization**:
  - Standardizes dates into consistent formats
  - Normalizes skills and groups by category
  - Standardizes education degree names and levels
- **Robust Architecture**:
  - Structured JSON output for easy integration
  - Comprehensive error handling and validation
  - Performance-optimized with parallel processing
  - Modular design for easy extension and customization

## Installation

```bash
# Clone the repository
git clone https://github.com/skanderbenali/python-based-ml-refined-resume-parser.git

# Navigate to the project directory
cd python-based-ml-refined-resume-parser

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (required for NER and similarity scoring)
python -m spacy download en_core_web_lg

# Create necessary directories
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['models/ner_model', 'models/classifier_model', 'uploads', 'results']]"
```

## Usage

### Command Line Interface

```bash
python resume_parser.py --input path/to/resume.pdf --output result.json
```

For processing multiple resumes:

```bash
python resume_parser.py --input path/to/resume/directory --output path/to/output/directory
```

### API

Start the API server:

```bash
uvicorn api.main:app --reload
```

Then, you can send requests to `http://localhost:8000/parse` with resume files.

## Training Custom Models

### Named Entity Recognition (NER)

Train a custom NER model with your own labeled data:

```bash
python training/train_ner.py --training-data path/to/training/data --output models/ner_model
```

### Resume Classifier

Train the resume classifier on your dataset:

```bash
python training/train_classifier.py --training-data path/to/labeled/resumes --output models/classifier_model
```

### Custom Mapping Files

Create custom mapping files for skills and education normalization:

```json
// models/skill_mappings.json example
{
  "normalizations": {
    "js": "JavaScript",
    "py": "Python"
  },
  "categories": {
    "JavaScript": "Programming Languages",
    "Python": "Programming Languages"
  },
  "acronyms": {
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence"
  }
}
```

## Testing

Run the test suite:

```bash
pytest
```

## Project Structure

```
refined-resume-parser-script/
├── resume_parser.py         # Main script with processing pipeline
├── requirements.txt         # Project dependencies
├── config.py                # Configuration settings
├── parsers/                 # File format parsing modules
│   ├── __init__.py
│   ├── pdf_parser.py        # PDF text extraction (PyPDF2 & pdfminer)
│   ├── docx_parser.py       # DOCX parsing
│   ├── txt_parser.py        # TXT file handling
│   └── html_parser.py       # HTML parsing with BeautifulSoup
├── extractors/              # Information extraction modules
│   ├── __init__.py
│   ├── ner.py               # Named Entity Recognition with spaCy
│   ├── section_segmenter.py # Resume section identification
│   └── classifier.py        # Resume classification (job roles)
├── normalizers/             # Data standardization modules
│   ├── __init__.py
│   ├── date_normalizer.py   # Date format standardization
│   ├── skill_normalizer.py  # Skill name normalization & categorization
│   └── education_normalizer.py # Education degree standardization
├── models/                  # Trained ML models and mappings
│   ├── __init__.py
│   ├── ner_model/           # Custom NER model
│   ├── classifier_model/    # Role classifier model
│   ├── skill_mappings.json  # Skill normalization mappings
│   └── education_mappings.json # Education normalization mappings
├── training/                # Model training scripts
│   ├── __init__.py
│   ├── train_ner.py         # NER model training
│   └── train_classifier.py  # Classifier training
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── validation.py        # Data validation utilities
│   └── helpers.py           # General helper functions
├── api/                     # FastAPI integration
│   ├── __init__.py
│   └── main.py              # API endpoints definition
└── tests/                   # Comprehensive test suite
    ├── __init__.py
    ├── test_parsers.py
    ├── test_extractors.py
    ├── test_normalizers.py
    └── test_integration.py
```

## Recent Updates

- **Code Cleanup**: Removed unnecessary comments while preserving complex logic explanations
- **Configuration Enhancement**: Moved from hardcoded mappings to JSON configuration files
- **Normalizer Improvements**: Enhanced skill and education normalizers with better categorization
- **Architecture Refinement**: Improved module consistency and error handling
- **Documentation**: Updated inline documentation and this README for better clarity

## License

MIT
