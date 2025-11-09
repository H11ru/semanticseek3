"""
translator.py — simplified interface for similarity.py

Provides easy-to-use functions for the Semantic Seek game:
- get_similarity(word1, word2) -> float: Returns similarity score (0-100)
- embed_words(words) -> dict: Pre-embeds a list of words for faster lookups
- load_vocab(filepath) -> dict: Loads pre-embedded vocabulary
- save_vocab(words, filepath): Saves embedded vocabulary to disk
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

# Lazy imports - only load similarity when actually needed
if TYPE_CHECKING:
    import similarity

# Configuration
VERBOSE = False  # Set to True for detailed loading messages

# Cache for embeddings to avoid recomputing
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

class VocabError(TranslatorError):
    """Raised when loading or saving vocabulary fails."""
    pass

def _import_similarity():
    """Lazy import similarity module with loading message."""
    global _similarity_module, _loading_printed
    if _similarity_module is None:
        if VERBOSE:
            print("🔄 Loading AI model (first time only)...", end="", flush=True)
        elif not _loading_printed:
            print("[LOADING MODEL... PLEASE WAIT]", end="", flush=True)
            _loading_printed = True
        
        try:
            import similarity as sim
            _similarity_module = sim
            if VERBOSE:
                print(" ✓")
        except Exception as e:
            if VERBOSE:
                print(" ✗")
            else:
                print(" FAILED")
            raise ImportError(f"Failed to import similarity module: {e}") from e
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
    """Get similarity score between two words."""
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
        
        # FIX: Rescale from compressed range [0.6, 1.0] to [0, 100]
        # Original formula: (similarity_score + 1) * 50
        # New formula: Stretch the 0.6-1.0 range to use full 0-100
        # similarity_score is in [-1, 1], typically 0.6-1.0 for this model
        # Map 0.6 -> 0, 1.0 -> 100
        rescaled = (similarity_score - 0.6) / 0.4 * 100
        
        # map 0.6-0.85 to 0.2-0.8
        rescaled /= 100
        rescaled = (rescaled - 0.6) * 240 + 50

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
        
    Raises:
        TranslatorError: If model loading or embedding fails
        ValueError: If words list is empty or invalid
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

def load_vocab(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load pre-embedded vocabulary from disk.
    
    Args:
        filepath: Path to saved vocabulary directory
        
    Returns:
        Dictionary mapping words to their embeddings
        
    Raises:
        VocabError: If loading vocabulary fails
        FileNotFoundError: If vocabulary files don't exist
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary directory not found: {filepath}")
    
    if not path.is_dir():
        raise VocabError(f"Path must be a directory: {filepath}")
    
    try:
        sim = _import_similarity()
        bundle = sim.EmbeddingBundle.load(path)
        
        # Update cache
        vocab_dict = {}
        for word, embedding in zip(bundle.vocab, bundle.embeddings):
            vocab_dict[word] = embedding
            _embedding_cache[word] = embedding
        
        return vocab_dict
        
    except Exception as e:
        raise VocabError(f"Failed to load vocabulary from {filepath}: {e}") from e

def save_vocab(words: List[str], filepath: str) -> None:
    """
    Embed and save vocabulary to disk for faster loading.
    
    Args:
        words: List of words/phrases to embed and save
        filepath: Path where to save the vocabulary (directory will be created)
        
    Raises:
        TranslatorError: If embedding or saving fails
        ValueError: If inputs are invalid
    """
    if not words:
        raise ValueError("Words list cannot be empty")
    
    if not filepath:
        raise ValueError("Filepath cannot be empty")
    
    try:
        sim = _import_similarity()
        vocab = embed_words(words)
        embeddings = np.stack([vocab[word] for word in words])
        
        model = _get_model()
        bundle = sim.EmbeddingBundle(
            vocab=words,
            embeddings=embeddings,
            model_name=model.model_name
        )
        
        bundle.save(filepath)
        
    except (EmbeddingError, ModelLoadError):
        raise
    except Exception as e:
        raise VocabError(f"Failed to save vocabulary to {filepath}: {e}") from e

def preload_model() -> None:
    """
    Pre-load the model at startup to avoid delays during first use.
    Call this during game initialization.
    """
    _get_model()

def clear_cache() -> None:
    """Clear the embedding cache to free memory."""
    global _embedding_cache
    _embedding_cache.clear()

def get_cache_size() -> int:
    """
    Get the number of cached embeddings.
    
    Returns:
        Number of words currently cached
    """
    return len(_embedding_cache)

def is_model_loaded() -> bool:
    """
    Check if the similarity model is loaded.
    
    Returns:
        True if model is loaded, False otherwise
    """
    return _model is not None