"""
semantic_analysis.py: Semantic Relationship Analysis Module

Implements comprehensive semantic analysis of word embeddings for institutional discourse,
with focus on:
- Statistical validation of semantic relationships
- Cross-category thematic analysis
- Effect size quantification
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
from gensim.models import Word2Vec
from config import SEMANTIC_CATEGORIES, SEMANTIC_ANALYSIS_PARAMS
from paths import MODELS_DIR


def standardized_diff(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate standardized differences.

    Args:
        group1: First sample distribution.
        group2: Second sample distribution.

    Returns:
        float: standardized differences.

    Why:
        Effect size quantifies the strength of thematic distinctions.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def calculate_similarity_stats(similarities: List[float]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for similarity distributions.

    Args:
        similarities: List of similarity scores.

    Returns:
        dict: Distribution statistics including mean, std, and skewness.
    """
    if not similarities:
        return {}

    sim_array = np.array(similarities)
    return {
        'mean': np.mean(sim_array),
        'std': np.std(sim_array),
        'median': np.median(sim_array),
        'quartiles': np.percentile(sim_array, [25, 50, 75]),
        'range': (float(np.min(sim_array)), float(np.max(sim_array))),
        'skewness': float(stats.skew(sim_array)),
        'kurtosis': float(stats.kurtosis(sim_array))
    }


def get_exemplar_pairs(
    similarities: List[float],
    word_pairs: List[Tuple[str, str]]
) -> Dict[str, Tuple[Tuple[str, str], float]]:
    """
    Identify representative word pairs at similarity extremes.

    Why:
        Helps users understand which word pairs exhibit the strongest and weakest relationships.

    Args:
        similarities: List of similarity scores.
        word_pairs: List of corresponding word pairs.

    Returns:
        Dictionary mapping categories to (word_pair, similarity_score) tuples.
    """
    if not similarities or not word_pairs:
        return {}

    max_idx = np.argmax(similarities)
    min_idx = np.argmin(similarities)

    return {
        'strongest': (word_pairs[max_idx], similarities[max_idx]),
        'weakest': (word_pairs[min_idx], similarities[min_idx])
    }


def analyze_semantic_relationships(
    embeddings: Word2Vec,
    word_groups: Dict[str, List[str]],
    min_samples: int = SEMANTIC_ANALYSIS_PARAMS['min_samples']
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze semantic relationships between word categories with statistical validation.

    Why:
        Captures thematic coherence and distinctiveness across predefined word groups.

    Args:
        embeddings: Trained Word2Vec model.
        word_groups: Mapping of categories to their respective word lists.
        min_samples: Minimum sample size for statistical validity.

    Returns:
        Dict containing within/across-category statistics, effect size, and exemplar pairs.
    """
    logging.info("Initiating semantic relationship analysis")
    results = {}
    categories = list(word_groups.keys())

    for category in categories:
        words = [w for w in word_groups[category] if w in embeddings.wv]
        if len(words) < min_samples:
            logging.warning(
                f"Insufficient valid words for category '{category}'. "
                f"Found {len(words)}, minimum required: {min_samples}"
            )
            continue

        # Track within and across-category similarities
        within_sims, across_sims = [], []
        within_pairs, across_pairs = [], []

        # Within-category similarities
        for i, word1 in enumerate(words):
            for word2 in words[i + 1:]:
                try:
                    sim = embeddings.wv.similarity(word1, word2)
                    within_sims.append(sim)
                    within_pairs.append((word1, word2))
                except KeyError as e:
                    logging.error(f"Within-category calculation failed: {str(e)}")

        # Across-category similarities
        for other_cat in categories:
            if other_cat != category:
                other_words = [w for w in word_groups[other_cat] if w in embeddings.wv]
                for word1 in words:
                    for word2 in other_words:
                        try:
                            sim = embeddings.wv.similarity(word1, word2)
                            across_sims.append(sim)
                            across_pairs.append((word1, word2))
                        except KeyError as e:
                            logging.error(f"Across-category calculation failed: {str(e)}")

        if within_sims and across_sims:
            within_stats = calculate_similarity_stats(within_sims)
            across_stats = calculate_similarity_stats(across_sims)

            # Get exemplar pairs
            within_exemplars = get_exemplar_pairs(within_sims, within_pairs)
            across_exemplars = get_exemplar_pairs(across_sims, across_pairs)

            # Calculate effect size and statistical significance
            effect_size = standardized_diff(np.array(within_sims), np.array(across_sims))
            t_stat, p_value = stats.ttest_ind(within_sims, across_sims)

            results[category] = {
                'within_category': {
                    **within_stats,
                    'exemplar_pairs': within_exemplars
                },
                'across_category': {
                    **across_stats,
                    'exemplar_pairs': across_exemplars
                },
                'comparative_metrics': {
                    'effect_size': effect_size,
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'semantic_distinctiveness': within_stats['mean'] - across_stats['mean']
                }
            }

            logging.info(f"\nResults for category '{category}':")
            logging.info(f"Effect size: {effect_size:.3f}")
            logging.info(f"Semantic distinctiveness: {results[category]['comparative_metrics']['semantic_distinctiveness']:.3f}")

    return results

# Inline Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Import required paths
        test_model_path = MODELS_DIR / "gutenberg_word2vec_test.model"

        # Check for test Word2Vec model
        if not test_model_path.exists():
            raise FileNotFoundError(
                f"Test Word2Vec model not found at {test_model_path}. "
                "Please run the inline testing in embedding_training.py first."
            )

        # Load the test Word2Vec model
        logging.info(f"Loading test Word2Vec model from {test_model_path}")
        model = Word2Vec.load(str(test_model_path))

        # Perform semantic analysis
        results = analyze_semantic_relationships(model, SEMANTIC_CATEGORIES)

        # Display results
        print("\nSemantic Analysis Results:")
        for category, metrics in results.items():
            print(f"\n{category.title()} Category Analysis:")

            # Display within-category exemplars
            within_exemplars = metrics['within_category']['exemplar_pairs']
            print("\nWithin-category exemplar pairs:")
            for pair_type, (pair, sim) in within_exemplars.items():
                print(f"  {pair_type}: {pair[0]}-{pair[1]} ({sim:.3f})")

            # Display cross-category exemplars
            across_exemplars = metrics['across_category']['exemplar_pairs']
            print("\nCross-category exemplar pairs:")
            for pair_type, (pair, sim) in across_exemplars.items():
                print(f"  {pair_type}: {pair[0]}-{pair[1]} ({sim:.3f})")

    except Exception as e:
        logging.error(f"Semantic analysis failed: {str(e)}", exc_info=True)
