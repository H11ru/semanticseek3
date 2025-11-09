from __future__ import annotations
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around sentence-transformers for encoding text to embeddings.

    Default model: paraphrase-multilingual-MiniLM-L12-v2
    - Optimized for word-level semantic similarity (better than E5 for single words)
    - Supports 50+ languages including Finnish and English
    - Produces 384-dim embeddings (smaller and faster than E5's 768-dim)
    - Wide similarity score distribution creates fun "almost correct" gameplay moments
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None
    ):
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

        Note:
            The paraphrase-multilingual model works well with plain text (no special prefixes needed).
            All embeddings are L2-normalized, so cosine similarity = dot product.
        """
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return vecs.astype(np.float32)
