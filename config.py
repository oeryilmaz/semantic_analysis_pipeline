"""
config.py: Configuration settings for semantic analysis pipeline

Defines semantic categories based on political science frameworks and
model training parameters optimized for institutional discourse analysis.
"""

# Semantic categories for political discourse analysis
# These categories define the thematic groups for semantic analysis.
# Adjusting these categories impacts the thematic distinctiveness analysis.
# Ensure changes reflect the domain-specific terminology of your dataset.

SEMANTIC_CATEGORIES = {
    'governance': ['union', 'constitution', 'government', 'law'],
    'rights': ['freedom', 'liberty', 'rights', 'justice'],
    'conflict': ['war', 'battle', 'military', 'army', 'enemy'],
    'democracy': ['people', 'citizen', 'vote', 'election'],
    'economy' : ['market', 'capital', 'labor', 'profit', 'prosperity', 'wealth']
}

# Parameters for Word2Vec embedding training
# These control the training process for Word2Vec embeddings.
# Adjust these parameters based on corpus size and domain-specific language.
MODEL_TRAINING_PARAMS = {
    'vector_size': 300,  # Higher dimensions can capture more nuanced relationships
                         # but are more ressource intensive
    'window': 8,         # How many words before/after to consider as context.
                         # Larger windows are more likely to capture semantic relationships.
    'epochs': 10,        # Number of passes through all texts. More passes usually imply better
                         # learning but may run the risk of overfitting.
    'min_count': 20,     # Ignore vocabulary with fewer than 20 occurrences in the corpus.
    'workers': 12,       # Number of parallel processes for faster training
    'sg': 1,             # Use Skip-gram architecture (vs CBOW). For this application,
                         # we want to predict the context, given one of the target words
                         # in SEMANTIC_CATEGORIES.
    'negative': 10,      # Think of this as a penalty term for nonsensical relationships #
                         # (e.g. justice:espresso) but also frequent fillers like "the" or "and".
                         # Higher values mean higher penalties.
    'seed': 1215
}

# Parameters for semantic analysis
SEMANTIC_ANALYSIS_PARAMS = {
    'min_samples': 3,            # Require at least 3 words per category 
                                 # (e.g., need 3+ democracy-related terms)
}

# Parameters for text preprocessing
# These define thresholds for token filtering and sampling from the corpus.
PREPROCESSING_PARAMS = {
    'min_tokens': 5,     # Skip texts shorter than 5 words. Discards very short 
                         # documents that may lack semantic value.
    'sample_size': 1000   # Process X documents, e.g., for testing (None = process all)
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': 'logs/nlp_pipeline.log',
            'mode': 'a'
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'INFO'
        }
    }
}
