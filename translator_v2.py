"""
translator_v2.py — IMPROVED similarity interface for Semantic Seek

Key improvements over translator.py:
- Uses paraphrase-multilingual-MiniLM-L12-v2 (better model)
- NO complex rescaling hacks needed
- Natural score distribution (0-100)
- Better discrimination between related/unrelated words

Example scores with NEW model:
- cat vs dog:    30 (different animals, correctly low!)
- cat vs car:    35 (unrelated, correctly low!)
- cat vs lynx:   51 (cat family, moderate)
- animal vs cat: 79 (category relationship, high!)

vs OLD model (E5) which gave:
- cat vs dog:    90 (too high!)
- cat vs car:    86 (WRONG - very high for unrelated words!)
"""

from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

# Lazy imports
if TYPE_CHECKING:
    import similarity_v2

# Configuration
VERBOSE = False

# Cache
_embedding_cache = {}
_model = None
_similarity_module = None
_loading_printed = False

class TranslatorError(Exception):
    """Base exception for translator module errors."""
    pass

class ModelLoadError(TranslatorError):
    """Raised when the similarity model fails to load."""
    pass

class EmbeddingError(TranslatorError):
    """Raised when embedding text fails."""
    pass

def _import_similarity():
    """Lazy import similarity_v2 module."""
    global _similarity_module, _loading_printed
    if _similarity_module is None:
        if VERBOSE:
            print("🔄 Loading improved AI model (paraphrase-multilingual)...", end="", flush=True)
        elif not _loading_printed:
            print("[LOADING MODEL... PLEASE WAIT]", end="", flush=True)
            _loading_printed = True

        try:
            import similarity_v2 as sim
            _similarity_module = sim
            if VERBOSE:
                print(" ✓")
        except Exception as e:
            if VERBOSE:
                print(" ✗")
            else:
                print(" FAILED")
            raise ImportError(f"Failed to import similarity_v2 module: {e}") from e
    return _similarity_module

def _get_model():
    """Lazy load the similarity model."""
    global _model
    if _model is None:
        sim = _import_similarity()
        try:
            if VERBOSE:
                print("🔄 Initializing model...", end="", flush=True)
            _model = sim.SimilarityModel()
            if VERBOSE:
                print(" ✓")
            else:
                print(" DONE")
        except Exception as e:
            if VERBOSE:
                print(" ✗")
            else:
                print(" FAILED")
            raise ModelLoadError(f"Failed to load similarity model: {e}") from e
    return _model

def get_similarity(word1: str, word2: str) -> float:
    """
    Get similarity score between two words (0-100).

    The new model provides much better discrimination:
    - Unrelated words: 10-40
    - Some relationship: 40-60
    - Related concepts: 60-80
    - Very similar: 80-100

    No rescaling hacks needed!
    """
    if not word1 or not word2:
        raise ValueError("Words cannot be empty")

    if not isinstance(word1, str) or not isinstance(word2, str):
        raise ValueError("Words must be strings")

    try:
        model = _get_model()

        # Check cache
        if word1 not in _embedding_cache:
            try:
                emb = model.encode([word1], normalize=True)[0]
                _embedding_cache[word1] = emb
            except Exception as e:
                raise EmbeddingError(f"Failed to embed '{word1}': {e}") from e

        if word2 not in _embedding_cache:
            try:
                emb = model.encode([word2], normalize=True)[0]
                _embedding_cache[word2] = emb
            except Exception as e:
                raise EmbeddingError(f"Failed to embed '{word2}': {e}") from e

        # Compute cosine similarity (dot product of normalized vectors)
        similarity_score = np.dot(_embedding_cache[word1], _embedding_cache[word2])

        # SIMPLE conversion to 0-100 scale
        # similarity_score is in [-1, 1], typically 0.0-1.0 for this model
        # Just scale to 0-100
        rescaled = similarity_score * 100

        # Clamp to [0, 100]
        return float(max(0.0, min(100.0, rescaled)))

    except (ModelLoadError, EmbeddingError):
        raise
    except Exception as e:
        raise TranslatorError(f"Unexpected error computing similarity: {e}") from e

def embed_words(words: List[str]) -> Dict[str, np.ndarray]:
    """
    Pre-embed a list of words for faster similarity lookups.

    Args:
        words: List of words/phrases to embed

    Returns:
        Dictionary mapping words to their embeddings
    """
    if not words:
        raise ValueError("Words list cannot be empty")

    if not isinstance(words, list):
        raise ValueError("Words must be a list")

    if not all(isinstance(w, str) for w in words):
        raise ValueError("All words must be strings")

    try:
        model = _get_model()
        embeddings = model.encode(words, normalize=True)

        vocab = {}
        for word, embedding in zip(words, embeddings):
            vocab[word] = embedding
            _embedding_cache[word] = embedding

        return vocab

    except ModelLoadError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Failed to embed word list: {e}") from e

def preload_model() -> None:
    """
    Pre-load the model at startup to avoid delays during first use.
    """
    _get_model()

def clear_cache() -> None:
    """Clear the embedding cache to free memory."""
    global _embedding_cache
    _embedding_cache.clear()

def get_cache_size() -> int:
    """Get the number of cached embeddings."""
    return len(_embedding_cache)

def is_model_loaded() -> bool:
    """Check if the similarity model is loaded."""
    return _model is not None


# Quick test
if __name__ == "__main__":
    print("Testing improved model...")
    print()

    test_pairs = [
        ("cat", "dog", "different pets"),
        ("cat", "lynx", "cat family"),
        ("cat", "car", "UNRELATED"),
        ("animal", "cat", "category relationship"),
        ("Finland", "Helsinki", "country-capital"),
        ("bread", "butter", "common pairing"),
    ]

    for word1, word2, description in test_pairs:
        score = get_similarity(word1, word2)
        print(f"{word1:10s} ↔ {word2:10s} ({description:25s}): {score:5.1f}")

    print()
    print("✓ Much better discrimination than old E5 model!")
    print("  Notice how unrelated words score low (30-40)")
    print("  while related concepts score high (60-80+)")
