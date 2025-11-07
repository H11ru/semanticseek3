from __future__ import annotations
import os
import numpy as np
import hnswlib
from typing import Tuple


class VectorIndex:
    """Wrapper around hnswlib for efficient approximate nearest neighbor search."""

    def __init__(self, dim: int, space: str = "cosine"):
        """
        Initialize the vector index.

        Args:
            dim: Dimensionality of vectors
            space: Distance metric ('cosine', 'l2', 'ip')
        """
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self._size = 0

    def build(self, vectors: np.ndarray, M: int = 32, ef_construction: int = 200, ef_search: int = 256):
        """
        Build the HNSW index from vectors.

        Args:
            vectors: Array of shape (N, dim) containing vectors to index
            M: HNSW M parameter (number of connections per layer)
            ef_construction: HNSW ef_construction parameter
            ef_search: HNSW ef parameter for search
        """
        n = vectors.shape[0]
        self.index.init_index(max_elements=n, M=M, ef_construction=ef_construction)
        self.index.add_items(vectors)
        self.index.set_ef(ef_search)
        self._size = n

    def add(self, vectors: np.ndarray):
        """
        Add vectors to existing index (requires capacity).

        Args:
            vectors: Array of shape (N, dim) containing vectors to add
        """
        self.index.add_items(vectors)
        self._size += vectors.shape[0]

    def search(self, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            qvec: Query vector of shape (1, dim)
            k: Number of neighbors to retrieve

        Returns:
            Tuple of (labels, similarities) where labels are indices and
            similarities are converted from distances (1 - distance for cosine)
        """
        labels, dists = self.index.knn_query(qvec, k=k)
        # For cosine space: similarity = 1 - distance
        sims = 1.0 - dists
        return labels[0], sims[0]

    def save(self, path: str):
        """Save the index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.index.save_index(path)

    def load(self, path: str):
        """Load the index from disk."""
        self.index.load_index(path)
        self._size = self.index.get_current_count()
