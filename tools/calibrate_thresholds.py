#!/usr/bin/env python3
"""
Calibrate optimal feedback thresholds for a given model and vocabulary.

This script analyzes your actual game vocabulary to recommend thresholds
that will create a good distribution of feedback across the game.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import argparse


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between normalized vectors."""
    return float(np.dot(v1, v2))


def load_vocabulary(data_file: Path) -> List[str]:
    """Extract all unique words from category JSON file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = set()
    categories = data.get('categories', {})

    # Handle both dict and list structures
    if isinstance(categories, dict):
        for category_name, words in categories.items():
            vocab.update(words)
    else:
        for category in categories:
            vocab.update(category.get('words', []))

    return sorted(vocab)


def analyze_similarities(
    model: SentenceTransformer,
    vocab: List[str],
    sample_size: int = 50
) -> Tuple[np.ndarray, dict]:
    """
    Compute pairwise similarities and return statistics.

    Args:
        model: The sentence transformer model
        vocab: List of words to analyze
        sample_size: Number of words to sample (for large vocabs)

    Returns:
        Tuple of (all_similarities, statistics_dict)
    """
    # Sample vocabulary if too large
    if len(vocab) > sample_size:
        import random
        sampled = random.sample(vocab, sample_size)
    else:
        sampled = vocab

    print(f"Computing embeddings for {len(sampled)} words...")
    embeddings = model.encode(sampled, normalize_embeddings=True, show_progress_bar=True)

    print("Computing pairwise similarities...")
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            similarities.append(sim)

    sims = np.array(similarities)

    stats = {
        'min': float(sims.min()),
        'max': float(sims.max()),
        'mean': float(sims.mean()),
        'median': float(np.median(sims)),
        'std': float(sims.std()),
        'q25': float(np.percentile(sims, 25)),
        'q50': float(np.percentile(sims, 50)),
        'q75': float(np.percentile(sims, 75)),
        'q90': float(np.percentile(sims, 90)),
        'q95': float(np.percentile(sims, 95)),
        'q99': float(np.percentile(sims, 99)),
    }

    return sims, stats


def recommend_thresholds(stats: dict, target_distribution: str = "balanced") -> List[float]:
    """
    Recommend feedback thresholds based on similarity distribution.

    Args:
        stats: Dictionary of similarity statistics
        target_distribution: "balanced" or "challenging"

    Returns:
        List of [hot_threshold, warm_threshold, mild_threshold]
    """
    if target_distribution == "balanced":
        # Aim for roughly equal distribution across bins
        return [
            stats['q90'],  # Top 10% are "hot"
            stats['q75'],  # Top 25% are "warm"
            stats['q50'],  # Top 50% are "mild"
        ]
    elif target_distribution == "challenging":
        # Make it harder - fewer hot/warm ratings
        return [
            stats['q95'],  # Top 5% are "hot"
            stats['q90'],  # Top 10% are "warm"
            stats['q75'],  # Top 25% are "mild"
        ]
    else:
        raise ValueError(f"Unknown distribution: {target_distribution}")


def print_analysis(stats: dict, thresholds: List[float], model_name: str):
    """Print analysis results."""
    print("\n" + "=" * 80)
    print(f"ANALYSIS RESULTS FOR: {model_name}")
    print("=" * 80)

    print("\nSimilarity Score Distribution:")
    print("-" * 80)
    print(f"  Minimum:        {stats['min']:.4f}")
    print(f"  25th percentile: {stats['q25']:.4f}")
    print(f"  Median (50th):   {stats['median']:.4f}")
    print(f"  75th percentile: {stats['q75']:.4f}")
    print(f"  90th percentile: {stats['q90']:.4f}")
    print(f"  95th percentile: {stats['q95']:.4f}")
    print(f"  Maximum:        {stats['max']:.4f}")
    print(f"  Mean:           {stats['mean']:.4f}")
    print(f"  Std Dev:        {stats['std']:.4f}")
    print(f"  Range:          {stats['max'] - stats['min']:.4f}")

    print("\nRecommended Thresholds:")
    print("-" * 80)
    print(f"  🔥 Hot (Kuuma):      >= {thresholds[0]:.4f}  (top ~10% of pairs)")
    print(f"  🌡️  Warm (Lämmin):   >= {thresholds[1]:.4f}  (top ~25% of pairs)")
    print(f"  😊 Mild (Lämpöinen): >= {thresholds[2]:.4f}  (top ~50% of pairs)")
    print(f"  ❄️  Cold (Kylmä):     <  {thresholds[2]:.4f}  (bottom ~50% of pairs)")

    print("\nConfiguration Update:")
    print("-" * 80)
    print("Update src/core/scoring.py:")
    print(f"""
THRESHOLDS = [{thresholds[0]:.2f}, {thresholds[1]:.2f}, {thresholds[2]:.2f}]
""")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate feedback thresholds for Semantic Seek game"
    )
    parser.add_argument(
        '--model',
        default='intfloat/multilingual-e5-base',
        help='Model name to test'
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/fi/categories_fi.json'),
        help='Path to category data file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='Number of words to sample for analysis'
    )
    parser.add_argument(
        '--distribution',
        choices=['balanced', 'challenging'],
        default='balanced',
        help='Target difficulty distribution'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Also run analysis for paraphrase-multilingual model'
    )

    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return

    # Load vocabulary
    print(f"Loading vocabulary from {args.data}...")
    vocab = load_vocabulary(args.data)
    print(f"Found {len(vocab)} unique words")

    # Analyze primary model
    print(f"\nLoading model: {args.model}...")
    model = SentenceTransformer(args.model)

    sims, stats = analyze_similarities(model, vocab, args.sample_size)
    thresholds = recommend_thresholds(stats, args.distribution)
    print_analysis(stats, thresholds, args.model)

    # Compare with paraphrase model if requested
    if args.compare:
        compare_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"\nLoading comparison model: {compare_model_name}...")
        compare_model = SentenceTransformer(compare_model_name)

        sims2, stats2 = analyze_similarities(compare_model, vocab, args.sample_size)
        thresholds2 = recommend_thresholds(stats2, args.distribution)
        print_analysis(stats2, thresholds2, compare_model_name)

        # Print comparison
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"\n{'Metric':<20} {args.model[:30]:<15} {compare_model_name[:30]:<15}")
        print("-" * 80)
        print(f"{'Range (max-min)':<20} {stats['max']-stats['min']:>14.4f} {stats2['max']-stats2['min']:>14.4f}")
        print(f"{'Std Dev':<20} {stats['std']:>14.4f} {stats2['std']:>14.4f}")
        print(f"{'Hot threshold':<20} {thresholds[0]:>14.4f} {thresholds2[0]:>14.4f}")
        print(f"{'Warm threshold':<20} {thresholds[1]:>14.4f} {thresholds2[1]:>14.4f}")
        print(f"{'Mild threshold':<20} {thresholds[2]:>14.4f} {thresholds2[2]:>14.4f}")

        # Recommendation
        better_discrimination = stats2['std'] > stats['std']
        if better_discrimination:
            print(f"\n✓ {compare_model_name} shows better discrimination (higher std dev)")
            print(f"  Recommendation: Consider switching models")
        else:
            print(f"\n✓ Current model shows adequate discrimination")


if __name__ == '__main__':
    main()
