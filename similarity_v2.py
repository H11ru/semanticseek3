from __future__ import annotations
"""
similarity_v2.py — IMPROVED embedding utilities for Semantic Seek v3.0

Key improvements over similarity.py:
- Uses paraphrase-multilingual-MiniLM-L12-v2 (better for word-level similarity)
- No complex rescaling needed - scores are naturally well-distributed
- Simpler, cleaner code
- Better discrimination between related/unrelated words

Dependencies: sentence-transformers, numpy
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import json
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from e


# BETTER MODEL: Optimized for word-level similarity
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class EmbeddingBundle:
    """Container for a vocabulary and its corresponding embeddings."""
    vocab: List[str]
    embeddings: np.ndarray  # shape: (N, D), float32, L2-normalized
    model_name: str = DEFAULT_MODEL

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.save(out / "embeddings.npy", self.embeddings)
        (out / "model_id.txt").write_text(self.model_name + "\n", encoding="utf-8")

    @staticmethod
    def load(in_dir: str | Path) -> "EmbeddingBundle":
        p = Path(in_dir)
        vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        embeddings = np.load(p / "embeddings.npy")
        model_name = (p / "model_id.txt").read_text(encoding="utf-8").strip() if (p / "model_id.txt").exists() else DEFAULT_MODEL
        return EmbeddingBundle(vocab=vocab, embeddings=embeddings, model_name=model_name)


class SimilarityModel:
    """
    Wrapper around SentenceTransformer with paraphrase-multilingual model.

    This model works MUCH better for word-level similarity than E5:
    - No special prefixes needed
    - Natural score distribution (not compressed)
    - Better discrimination between related/unrelated words
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self,
               texts: Iterable[str],
               batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        """
        Embed an iterable of strings.

        Returns float32 ndarray of shape (N, D).
        Vectors are L2-normalized for cosine similarity.
        """
        texts_list = list(texts)

        # No special prefixes needed for paraphrase-multilingual!
        vecs = self.model.encode(
            texts_list,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return vecs


def embed_words(words: Iterable[str],
                model: Optional[SimilarityModel] = None,
                model_name: str = DEFAULT_MODEL,
                batch_size: int = 64,
                normalize: bool = True) -> Tuple[List[str], np.ndarray, str]:
    """
    Embed a list of words and return (vocab_list, embeddings, model_name).

    Args:
        words: Iterable of words/phrases to embed
        model: Optional pre-loaded SimilarityModel
        model_name: Model to use if model not provided
        batch_size: Batch size for encoding
        normalize: Whether to L2-normalize embeddings

    Returns:
        Tuple of (word list, embeddings array, model name)
    """
    if model is None:
        model = SimilarityModel(model_name=model_name)

    vocab_list = list(words)
    embeddings = model.encode(vocab_list, batch_size=batch_size, normalize=normalize)

    return vocab_list, embeddings, model.model_name


def save_vocab(words: Iterable[str],
               out_dir: str | Path,
               model: Optional[SimilarityModel] = None,
               model_name: str = DEFAULT_MODEL,
               **kwargs) -> None:
    """
    Embed words and save to disk.

    Args:
        words: Words to embed and save
        out_dir: Output directory path
        model: Optional pre-loaded model
        model_name: Model to use if model not provided
    """
    vocab_list, embeddings, used_model = embed_words(
        words, model=model, model_name=model_name, **kwargs
    )
    bundle = EmbeddingBundle(vocab=vocab_list, embeddings=embeddings, model_name=used_model)
    bundle.save(out_dir)


def load_vocab(in_dir: str | Path) -> EmbeddingBundle:
    """
    Load saved vocabulary and embeddings from disk.

    Args:
        in_dir: Directory containing vocab.json and embeddings.npy

    Returns:
        EmbeddingBundle with loaded data
    """
    return EmbeddingBundle.load(in_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed vocabulary with paraphrase-multilingual model")
    parser.add_argument("--words", nargs="+", help="Words to embed")
    parser.add_argument("--file", help="JSON file containing list of words")
    parser.add_argument("--output", "-o", help="Output directory for embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                words = data
            elif isinstance(data, dict) and "words" in data:
                words = data["words"]
            else:
                raise ValueError("JSON must be a list or dict with 'words' key")
    elif args.words:
        words = args.words
    else:
        parser.print_help()
        exit(1)

    print(f"Embedding {len(words)} words with {args.model}...")
    vocab_list, embeddings, model_name = embed_words(words, model_name=args.model)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Model used: {model_name}")

    if args.output:
        bundle = EmbeddingBundle(vocab=vocab_list, embeddings=embeddings, model_name=model_name)
        bundle.save(args.output)
        print(f"Saved to {args.output}/")
    else:
        # Show first few embeddings
        for word, emb in zip(vocab_list[:3], embeddings[:3]):
            print(f"{word}: {emb[:5]}... (L2 norm: {np.linalg.norm(emb):.4f})")
