"""
text_processing.py: Text Corpus Preprocessing Module

Implements robust text preprocessing for semantic analysis of institutional discourse,
focusing on:
- Efficient corpus handling with configurable sampling
- Multi-encoding support (UTF-8/Latin-1)
- Token preservation strategies for domain-specific analysis
- Comprehensive processing metrics and validation

Design Principles:
- Modular processing pipeline with configurable parameters
- Robust error handling for large-scale text processing
- Memory-efficient streaming for large corpora
- Detailed logging for process monitoring and debugging
"""
import logging
import os
import random
import json
from typing import List

from nltk.tokenize import word_tokenize

from config import PREPROCESSING_PARAMS
from paths import CORPUS_DIR, DATA_DIR

def log_processing_metrics(corpus: List[List[str]], total_files: int) -> None:
    """
    Log comprehensive corpus processing metrics.

    Calculates and logs:
    - Total processed files
    - Generated text segments
    - Average segment length
    - Vocabulary statistics

    Args:
        corpus: Processed text segments
        total_files: Number of processed files
    """
    total_segments = len(corpus)
    avg_length = sum(len(seg) for seg in corpus) / max(1, total_segments)
    unique_words = len(set(word for seg in corpus for word in seg))

    logging.info("\nProcessing Results:")
    logging.info(f"Files processed: {total_files}")
    logging.info(f"Segments generated: {total_segments}")
    logging.info(f"Average segment length: {avg_length:.1f}")
    logging.info(f"Unique words: {unique_words}")


def process_corpus(
    corpus_path: str,
    output_path: str,
    **kwargs
) -> None:
    """
    Load, process, and save texts from the corpus for semantic analysis.

    Args:
        corpus_path: Path to the directory containing text files.
        output_path: Path to save the processed corpus as a JSON file.
        **kwargs: Optional overrides for PREPROCESSING_PARAMS.

    Returns:
        None. The processed corpus is saved to the specified output path.
    """
    params = PREPROCESSING_PARAMS.copy()
    params.update(kwargs)

    logging.info(f"Starting corpus processing from {corpus_path}")
    processed_corpus = []

    try:
        # Get all available text files
        all_files = [f for f in os.listdir(corpus_path) if f.endswith('.txt')]
        logging.info(f"Found {len(all_files)} total text files")

        # Select files to process
        files_to_process = all_files
        if params['sample_size'] and params['sample_size'] < len(all_files):
            files_to_process = random.sample(all_files, params['sample_size'])
            logging.info(f"Selected {params['sample_size']} files for processing")

        # Process each selected file
        for filename in files_to_process:
            file_path = os.path.join(corpus_path, filename)
            logging.info(f"Processing {filename}")

            try:
                # Read file with error handling for different encodings
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        text = file.read()

                # Split into paragraphs
                paragraphs = text.split('\n\n')

                # Process each paragraph
                for para in paragraphs:
                    if not para.strip():  # Skip empty paragraphs
                        continue

                    # Tokenize the paragraph to split it into individual words
                    tokens = word_tokenize(para.lower())

                    # Filter tokens but preserve important elements
                    filtered_tokens = [
                        token for token in tokens
                        if (token.isalnum() or  # Regular words
                            token in ["'s", "n't", "'ll", "'ve", "'re", "'m"] or  # Contractions
                            (token.startswith("'") and token.endswith("'")))  # Quoted terms
                    ]

                    # Only add paragraphs with meaningful content
                    if len(filtered_tokens) >= params['min_tokens']:
                        processed_corpus.append(filtered_tokens)

            except Exception as e:
                logging.warning(f"Error processing {filename}: {str(e)}")
                continue

        # Log processing results
        log_processing_metrics(processed_corpus, len(files_to_process))

        if not processed_corpus:
            raise ValueError("No valid text segments were generated from the corpus")

        # Save the processed corpus to the specified output path
        with open(output_path, "w") as f:  # Use the passed output_path
            json.dump(processed_corpus, f)
        logging.info(f"Processed corpus saved to {output_path}")

    except Exception as e:
        logging.error(f"Corpus processing failed: {str(e)}")
        raise

# Inline testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Specify the actual corpus directory and the test output file path
        corpus_path = CORPUS_DIR
        output_path = DATA_DIR / "preprocessed_corpus_test.json"

        # Create a modified copy of PREPROCESSING_PARAMS for testing
        test_params = PREPROCESSING_PARAMS.copy()
        test_params['sample_size'] = 10  # Override the sample size for the test

        # Process and save a subsample of the actual corpus
        logging.info(f"Testing preprocessing on a subsample of the corpus at {corpus_path}")
        process_corpus(
            corpus_path=str(corpus_path),
            output_path=str(output_path),
            **test_params
        )

        # Log test results
        logging.info(f"Test output saved to {output_path}")
        print(f"\nTest Results: Processed and saved to {output_path}")

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        logging.error(f"Test execution failed: {str(e)}", exc_info=True)
