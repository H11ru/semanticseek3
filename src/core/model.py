from __future__ import annotations
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around sentence-transformers for encoding text to embeddings."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (None = auto, 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode texts to normalized embeddings.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding

        Returns:
            Normalized float32 embeddings with shape (len(texts), dim)
        """
        # E5 models work best with instruction prefixes, but keeping plain for v1 simplicity
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return vecs.astype(np.float32)
