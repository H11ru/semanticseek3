from __future__ import annotations
"""
similarity.py — lightweight embedding utilities for Semantic Seek v3.0

- Loads a multilingual sentence-transformers model (default: intfloat/multilingual-e5-base)
- Provides functions to embed a list of words/phrases with optional L2-normalization
- Includes simple save/load helpers for vocab + embeddings
- Can be used as a module or invoked as a CLI to embed a JSON vocab file

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


DEFAULT_MODEL = "intfloat/multilingual-e5-base"


@dataclass
class EmbeddingBundle:
    """Container for a vocabulary and its corresponding embeddings."""
    vocab: List[str]
    embeddings: np.ndarray  # shape: (N, D), float32, (optionally) L2-normalized
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
    Thin wrapper around SentenceTransformer with sane defaults.

    Notes for E5 models:
    - REQUIRES instruction prefixes (e.g., "query: ", "passage: ")
    - Without prefixes, all embeddings collapse to similar vectors
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # Check if this is an E5 model
        self.is_e5_model = "e5" in model_name.lower()

    def encode(self,
               texts: Iterable[str],
               batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        """
        Embed an iterable of strings.

        Returns float32 ndarray of shape (N, D). If normalize=True, vectors are L2-normalized.
        """
        texts_list = list(texts)
        
        # E5 models: use "passage: " for content embeddings
        if self.is_e5_model:
            texts_list = [f"passage: {text}" for text in texts_list]
        
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
    Convenience function to get embeddings for a list of words/phrases.

    Returns (vocab_list, embeddings, model_name).
    """
    vocab = list(words)
    mdl = model or SimilarityModel(model_name=model_name)
    vecs = mdl.encode(vocab, batch_size=batch_size, normalize=normalize)
    return vocab, vecs, mdl.model_name


# -----------------
# CLI entry point
# -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for a vocabulary file or inline words.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name")
    parser.add_argument("--input-json", help="Path to JSON with {'categories': {...}} or {'vocab': [...]}.")
    parser.add_argument("--out", required=True, help="Output directory for vocab.json, embeddings.npy, model_id.txt")
    parser.add_argument("--lang", default=None, help="Optional language code for heuristics (unused for now)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of embeddings")
    parser.add_argument("--words", nargs="*", help="Embed these words instead of reading a file")

    args = parser.parse_args()

    if args.words:
        vocab = args.words
    else:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        if "vocab" in payload:
            vocab = payload["vocab"]
        elif "categories" in payload:
            vocab = []
            for lst in payload["categories"].values():
                vocab.extend(lst)
            # dedup preserving order
            seen = set(); dedup = []
            for w in vocab:
                if w not in seen:
                    seen.add(w)
                    dedup.append(w)
            vocab = dedup
        else:
            raise ValueError("Input JSON must contain 'vocab' or 'categories'.")

    print(f"Embedding {len(vocab)} items with {args.model}…")
    sim = SimilarityModel(model_name=args.model)
    vecs = sim.encode(vocab, batch_size=args.batch_size, normalize=not args.no_normalize)

    bundle = EmbeddingBundle(vocab=vocab, embeddings=vecs, model_name=args.model)
    bundle.save(args.out)
    print(f"Saved to {args.out}")
