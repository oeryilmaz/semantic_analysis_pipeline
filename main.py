"""
main.py: Main execution module

This module orchestrates the entire NLP pipeline, from text preprocessing
to embedding training and semantic analysis.
"""

import json
import logging
import logging.config

from text_processing import process_corpus
from embedding_training import train_word2vec
from semantic_analysis import analyze_semantic_relationships
from config import (
    SEMANTIC_CATEGORIES,
    PREPROCESSING_PARAMS,
    SEMANTIC_ANALYSIS_PARAMS,
    LOGGING_CONFIG,
)
from paths import CORPUS_DIR, WORD2VEC_MODEL_PATH, PREPROCESSED_CORPUS_PATH


def initialize_logging():
    """Configure logging based on settings from config."""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)


def main():
    """Execute the complete NLP semantic analysis pipeline."""
    logger = initialize_logging()
    logger.info("Starting NLP Semantic Analysis Pipeline")

    try:
        # Step 1: Load and preprocess the corpus
        logger.info("Step 1: Loading and preprocessing corpus")
        if not CORPUS_DIR.exists():
            raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

        process_corpus(
            corpus_path=str(CORPUS_DIR),
            output_path=str(PREPROCESSED_CORPUS_PATH),  # Save to standard path
            **PREPROCESSING_PARAMS
        )
        logger.info(f"Preprocessing complete. Corpus saved at {PREPROCESSED_CORPUS_PATH}.")

        # Step 2: Train word embeddings
        logger.info("Step 2: Training word embeddings")
        with open(PREPROCESSED_CORPUS_PATH, "r") as f:
            preprocessed_corpus = json.load(f)  # Load preprocessed corpus

        word2vec_model = train_word2vec(
            corpus=preprocessed_corpus,
            save_path=str(WORD2VEC_MODEL_PATH)
        )

        logger.info(f"Word2Vec model training complete. Model saved at {WORD2VEC_MODEL_PATH}.")

        # Step 3: Perform semantic analysis
        logger.info("Step 3: Performing semantic analysis")
        results = analyze_semantic_relationships(
            embeddings=word2vec_model,
            word_groups=SEMANTIC_CATEGORIES,
            min_samples=SEMANTIC_ANALYSIS_PARAMS['min_samples']
        )

        # Log key results for semantic analysis
        for category, metrics in results.items():
            logger.info(f"\nCategory: {category}")
            logger.info(f"Effect size: {metrics['comparative_metrics']['effect_size']:.3f}")
            logger.info(
                f"Semantic distinctiveness: "
                f"{metrics['comparative_metrics']['semantic_distinctiveness']:.3f}"
            )

            # Log exemplar pairs
            within_exemplars = metrics['within_category']['exemplar_pairs']
            logger.info("Within-category exemplar pairs:")
            for pair_type, (pair, sim) in within_exemplars.items():
                logger.info(f"  {pair_type}: {pair[0]}-{pair[1]} ({sim:.3f})")

            across_exemplars = metrics['across_category']['exemplar_pairs']
            logger.info("Across-category exemplar pairs:")
            for pair_type, (pair, sim) in across_exemplars.items():
                logger.info(f"  {pair_type}: {pair[0]}-{pair[1]} ({sim:.3f})")

        logger.info("Pipeline completed successfully.")

    except FileNotFoundError as e:
        logger.error(f"File system error: {str(e)}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
