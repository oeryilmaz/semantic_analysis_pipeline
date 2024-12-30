"""
paths.py: Central configuration of project paths with environment-based setup
Implements platform-independent path handling with configurable root directory.
If no path is explicitly set it will default to the directory containing this file ('paths.py').

"""

from pathlib import Path
import os

# Environment-based root configuration
DEFAULT_PROJECT_ROOT = Path(__file__).parent  # Base dir is project root
PROJECT_ROOT = Path(os.getenv("NLP_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT)))

# Data directory structure
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "Project_Gutenberg" / "txt"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
WORD2VEC_MODEL_PATH = MODELS_DIR / "gutenberg_word2vec.model"

# Runtime artifacts
LOGS_DIR = PROJECT_ROOT / "logs"

# File to store the preprocessed corpus
PREPROCESSED_CORPUS_PATH = DATA_DIR / "preprocessed_corpus.json"

# Validate critical paths exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Export path constants
__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'CORPUS_DIR',
    'MODELS_DIR',
    'WORD2VEC_MODEL_PATH',
    'LOGS_DIR',
    'PREPROCESSED_CORPUS_PATH'
]
