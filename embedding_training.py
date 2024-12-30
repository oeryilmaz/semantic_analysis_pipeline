"""
embedding_training.py: Word Embedding Training Module

This module handles the training of word embeddings using Word2Vec. Word embeddings
capture semantic relationships between words based on their usage contexts.
"""


import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from gensim.models import Word2Vec

from paths import DATA_DIR, WORD2VEC_MODEL_PATH
from config import MODEL_TRAINING_PARAMS



def train_word2vec(
    corpus: List[List[str]],
    save_path: Optional[str] = None,
    **kwargs
) -> Word2Vec:
    """
    Train Word2Vec embeddings on the preprocessed corpus using skip-gram architecture.

    Args:
        corpus: List of tokenized documents (each document is a list of tokens).
        save_path: Optional path to save the trained model.
        **kwargs: Optional overrides for MODEL_TRAINING_PARAMS configuration.

    Returns:
        Word2Vec: Trained word embedding model.

    Raises:
        ValueError: If corpus is empty or contains no valid documents.
        RuntimeError: If training fails.
    """
    # Merge default parameters with any overrides
    params = MODEL_TRAINING_PARAMS.copy()
    params.update(kwargs)

    logging.info(
        f"Starting Word2Vec training with parameters: "
        f"vector_size={params['vector_size']}, window={params['window']}, "
        f"epochs={params['epochs']}"
    )

    # Validate corpus
    if not corpus or not any(doc for doc in corpus):
        raise ValueError("Corpus is empty or contains no valid documents.")

    # Log corpus statistics
    vocab_size = len(set(word for doc in corpus for word in doc))
    avg_doc_length = np.mean([len(doc) for doc in corpus])
    logging.info(
        f"Corpus statistics: {len(corpus)} documents, {vocab_size} unique tokens, "
        f"average document length: {avg_doc_length:.1f}"
    )

    try:
        # Initialize and train Word2Vec model
        model = Word2Vec(sentences=corpus, **params)

        # Log training results
        final_vocab_size = len(model.wv)
        coverage = final_vocab_size / vocab_size * 100
        logging.info(
            f"Training complete. Final vocabulary size: {final_vocab_size} "
            f"({coverage:.1f}% of original vocabulary)"
        )

        # Save model if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
            logging.info(f"Model saved to {save_path}")

        return model

    except Exception as e:
        logging.error(f"Word2Vec training failed: {str(e)}")
        raise RuntimeError(f"Failed to train word embeddings: {str(e)}")

# Inline Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Ensure the preprocessed corpus exists
        preprocessed_path = DATA_DIR / "preprocessed_corpus_test.json"  # "_test" file from text_processing.py

        # Check for preprocessed corpus
        preprocessed_path = DATA_DIR / "preprocessed_corpus_test.json"
        if not preprocessed_path.exists():
            raise FileNotFoundError(
                f"Preprocessed corpus not found at {preprocessed_path}. "
                "Please run the inline testing in text_processing.py first."
            )

        # Load the preprocessed corpus
        logging.info(f"Loading preprocessed corpus from {preprocessed_path}")
        with open(preprocessed_path, "r") as f:
            sample_corpus = json.load(f)

        # Train and save a test Word2Vec model
        test_model_path = WORD2VEC_MODEL_PATH.with_name(
            "gutenberg_word2vec_test.model"
        )
        logging.info(f"Training Word2Vec model on preprocessed corpus")
        model = train_word2vec(corpus=sample_corpus)
        logging.info(f"Saving test Word2Vec model to {test_model_path}")
        model.save(str(test_model_path))

        # Log success
        logging.info(f"Test Word2Vec model saved to {test_model_path}")
        print(f"Test Word2Vec model saved to {test_model_path}")

    except Exception as e:
        logging.error(f"Embedding training failed: {str(e)}", exc_info=True)
