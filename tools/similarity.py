"""
similarity.py
----------------

This module provides classes and utilities for working with word or sentence
embeddings and computing cosine similarity between them.  It wraps a
pre‑trained embedding model from the `sentence_transformers` package to
generate dense vector representations for strings, manages a persistent cache
for previously computed embeddings, offers optional approximate nearest
neighbour indexing using `hnswlib`, and includes simple functions for
visualising high‑dimensional vectors with dimensionality reduction (PCA or
t‑SNE).

Overview
~~~~~~~~

Embeddings map words or sentences into high‑dimensional vectors which capture
semantic similarity.  Two common metrics for comparing embeddings are
Euclidean distance and cosine similarity.  Cosine similarity measures the
angle between vectors and yields values in the range –1 to 1, with 1 for
identical directions and values near 0 for unrelated vectors.  For unit
length vectors produced by many transformer models, the dot product of two
vectors is equal to their cosine similarity【108359925649917†L158-L189】.  This module normalises
all vectors before computing similarities to ensure consistent results.

The underlying embedding model is loaded lazily and can be customised via
the `model_name` parameter.  By default it uses the multilingual model
``intfloat/multilingual-e5-base`` which supports over 100 languages and
produces 768‑dimensional unit vectors.  You must have the
``sentence_transformers`` and ``torch`` packages available to use this
functionality.  An optional HNSW index can be constructed for fast
nearest‑neighbour queries; this requires the ``hnswlib`` package.  The
index as well as the embedding cache are persisted to disk so that
previously computed vectors need not be recomputed on subsequent runs.

Example usage from the command line::

    # Compute the cosine similarity between two words
    python similarity.py similarity koira susi

    # Precompute embeddings for a list of words and save the cache/index
    python similarity.py embed --words koira kissa susi

    # Visualise the relationship between several words via PCA and write
    # the figure to a file
    python similarity.py plot --words koira kissa susi --method pca --output plot.png

If the requisite packages are unavailable at runtime, informative
``ImportError`` exceptions will be raised when the model or index is
instantiated.

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional dependencies.  We attempt to import these at module import time
# but only raise when used if unavailable.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore[misc]

try:
    import hnswlib  # type: ignore
except ImportError:
    hnswlib = None  # type: ignore[misc]

try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.manifold import TSNE  # type: ignore
except ImportError:
    PCA = None  # type: ignore[misc]
    TSNE = None  # type: ignore[misc]

# Matplotlib is used for plotting.  It will only be imported when
# visualisation is requested.


def _normalise(vec: np.ndarray) -> np.ndarray:
    """Return a normalised copy of the input vector.

    The vector is scaled to unit length.  If the input has zero norm, the
    vector is returned unchanged to avoid division by zero.

    Parameters
    ----------
    vec: ndarray
        The input vector.

    Returns
    -------
    ndarray
        The normalised vector.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


@dataclass
class EmbeddingBundle:
    """Persistent storage of embeddings and lookup tables.

    Attributes
    ----------
    embeddings : Dict[str, np.ndarray]
        Mapping from tokens (lower‑cased) to their embedding vectors.  All
        embeddings are stored as 1D NumPy arrays.
    word_to_id : Dict[str, int]
        Mapping from tokens to the integer identifiers used in the HNSW index.
    """

    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    word_to_id: Dict[str, int] = field(default_factory=dict)

    @property
    def dimension(self) -> Optional[int]:
        """Return the dimensionality of the stored embeddings.

        Returns
        -------
        int or None
            The dimensionality of the first embedding or ``None`` if no
            embeddings have been stored.
        """
        for vec in self.embeddings.values():
            return len(vec)
        return None

    def to_json(self) -> Dict[str, list]:
        """Serialise embeddings to a JSON‑compatible dictionary.

        Embeddings are converted to lists of floats.  The ``word_to_id``
        mapping is also included in the output.
        """
        return {
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "word_to_id": self.word_to_id,
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "EmbeddingBundle":
        """Construct a bundle from a JSON dictionary previously emitted by
        :meth:`to_json`.

        Parameters
        ----------
        data : Dict[str, object]
            The JSON object loaded from disk.

        Returns
        -------
        EmbeddingBundle
        """
        embeddings = {k: np.array(v, dtype=np.float32) for k, v in data.get("embeddings", {}).items()}
        word_to_id = {k: int(v) for k, v in data.get("word_to_id", {}).items()}
        return cls(embeddings=embeddings, word_to_id=word_to_id)


class SimilarityModel:
    """Compute and store word embeddings and cosine similarity values.

    When first constructed, this object will attempt to load previously
    computed embeddings and, if available, an HNSW index from the specified
    cache directory.  New embeddings are lazily computed using a
    ``SentenceTransformer`` model.  All vectors are automatically
    normalised to unit length to simplify cosine similarity computation.

    Parameters
    ----------
    model_name : str
        Name of the pre‑trained model on the HuggingFace hub.  A good default
        for multilingual similarity tasks is ``intfloat/multilingual-e5-base``.
    cache_dir : str, optional
        Directory where the embedding cache and index will be stored.  If
        omitted, a ``.similarity_cache`` folder in the current working
        directory is used.
    max_elements : int, optional
        Maximum number of elements to reserve in the HNSW index.  This should
        exceed the expected number of unique words you intend to embed.  Only
        relevant when ``hnswlib`` is available.
    build_index : bool, optional
        Whether to construct an HNSW index.  If ``False``, the index is not
        created and all similarity computations fall back to direct dot
        products.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        cache_dir: Optional[str] = None,
        max_elements: int = 100_000,
        build_index: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".similarity_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        # Paths for persisting the embeddings and index
        self.bundle_path = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_bundle.json")
        self.index_path = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_index.bin")

        # Load or initialise the embedding bundle
        if os.path.exists(self.bundle_path):
            with open(self.bundle_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._bundle = EmbeddingBundle.from_json(data)
        else:
            self._bundle = EmbeddingBundle()

        # The embedding model is loaded lazily on first use to avoid the cost
        # of initialising it if no embeddings are requested.
        self._model: Optional[SentenceTransformer] = None

        # Optional HNSW index for fast nearest neighbour search
        self._index = None
        self._index_enabled = build_index and (hnswlib is not None)
        self._max_elements = max_elements
        if self._index_enabled:
            # Determine the dimensionality either from existing embeddings or by
            # deferring until the first embedding is computed.
            dim = self._bundle.dimension
            if dim is not None:
                self._init_index(dim)
            # If we have persisted an index on disk, attempt to load it.
            if os.path.exists(self.index_path) and dim is not None:
                # When loading, we must supply max_elements equal to or greater
                # than the number of items contained in the index
                self._index.load_index(self.index_path, max_elements=self._max_elements)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> SentenceTransformer:
        """Load and cache the embedding model.

        Raises
        ------
        ImportError
            If ``sentence_transformers`` is not installed.
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence_transformers is required for embedding computation. "
                "Install it with `pip install sentence-transformers`"
            )
        if self._model is None:
            # The SentenceTransformer constructor downloads the model from the
            # HuggingFace hub if necessary.
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _init_index(self, dim: int) -> None:
        """Initialise the HNSW index for a given embedding dimension.

        Parameters
        ----------
        dim : int
            Dimensionality of the vectors to be indexed.
        """
        if not self._index_enabled:
            return
        # Only construct a new index if one does not already exist
        if self._index is None:
            # Use cosine space; for unit vectors the inner product is the cosine
            self._index = hnswlib.Index(space="cosine", dim=dim)
            self._index.init_index(
                max_elements=self._max_elements, ef_construction=200, M=16
            )
            self._index.set_ef(50)
            # If there are already embeddings cached, build the index from them
            if self._bundle.embeddings:
                all_vecs = np.vstack(list(self._bundle.embeddings.values())).astype(np.float32)
                ids = np.array(list(range(len(all_vecs))), dtype=np.int32)
                self._index.add_items(all_vecs, ids)

    def _update_index(self, vec: np.ndarray, idx: int) -> None:
        """Add a vector with a specific identifier to the HNSW index.

        If the index has not yet been initialised because we did not know
        the dimensionality a priori, this function initialises it on first
        insertion.

        Parameters
        ----------
        vec : ndarray
            The vector to add.
        idx : int
            The identifier corresponding to the word in the bundle.
        """
        if not self._index_enabled:
            return
        # Initialise the index if necessary
        if self._index is None:
            self._init_index(len(vec))
        # It is possible that the index has already reached its capacity.
        if self._index.get_current_count() >= self._index.get_max_elements():
            # At this point we cannot add more elements without rebuilding.
            # For simplicity we skip adding to the index and rely on direct
            # similarity computations.  A production system might rebuild the
            # index with a larger capacity here.
            return
        self._index.add_items(vec.reshape(1, -1).astype(np.float32), np.array([idx], dtype=np.int32))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        """Generate or retrieve a normalised embedding for the given text.

        The input is lower‑cased and stripped before lookup.  If the embedding
        has been seen previously, it is returned from the cache.  Otherwise,
        a new embedding is computed via the underlying model, normalised, and
        added to the cache (and the HNSW index, if available).

        Parameters
        ----------
        text : str
            The word or sentence to embed.

        Returns
        -------
        ndarray
            The normalised embedding vector.
        """
        token = text.strip().lower()
        if token in self._bundle.embeddings:
            return self._bundle.embeddings[token]

        # Compute the embedding using the model
        model = self._load_model()
        # normalise embeddings directly from the model; ensures unit length
        vec = model.encode([token], normalize_embeddings=True)[0]
        # Ensure numpy float32 for compatibility with hnswlib
        vec = np.asarray(vec, dtype=np.float32)
        # Cache the embedding and assign an id
        new_id = len(self._bundle.embeddings)
        self._bundle.embeddings[token] = vec
        self._bundle.word_to_id[token] = new_id
        # Update the HNSW index if available
        self._update_index(vec, new_id)
        return vec

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute the cosine similarity between two strings.

        Both strings are embedded (using the cache if possible), normalised,
        and their cosine similarity is computed.  Values range from –1 to 1
        (for antiparallel vectors)【108359925649917†L158-L189】.  Because the embeddings produced by
        ``SentenceTransformer`` models are normalised, the dot product of
        the two vectors yields the cosine similarity directly【108359925649917†L158-L189】.

        Parameters
        ----------
        text1 : str
            The first word or sentence.
        text2 : str
            The second word or sentence.

        Returns
        -------
        float
            The cosine similarity between the two vectors.
        """
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        # Although the vectors are already normalised, normalise again to be
        # defensive in case the underlying model is changed.
        v1 = _normalise(v1)
        v2 = _normalise(v2)
        return float(np.dot(v1, v2))

    def nearest_neighbours(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return the ``k`` nearest neighbours of ``text`` from the cache.

        This uses the HNSW index if available.  If the index is not
        available (because ``hnswlib`` is missing or the index has not
        been built), it falls back to computing cosine similarity with
        every cached embedding.  The results include the input token itself.

        Parameters
        ----------
        text : str
            The query word or sentence.
        k : int, optional
            The number of neighbours to return.

        Returns
        -------
        List[Tuple[str, float]]
            A list of tuples ``(neighbour, similarity)`` sorted by decreasing
            similarity.  If no embeddings have been computed yet, returns an
            empty list.
        """
        if not self._bundle.embeddings:
            return []
        query_vec = self.embed(text)
        if self._index is not None:
            # Ensure k does not exceed the number of indexed items
            k = min(k, self._index.get_current_count())
            labels, distances = self._index.knn_query(query_vec.reshape(1, -1), k=k)
            results: List[Tuple[str, float]] = []
            for lbl, dist in zip(labels[0], distances[0]):
                # Convert distance to similarity; hnswlib returns cosine distances
                sim = 1.0 - dist
                # Map id back to word
                for w, idx in self._bundle.word_to_id.items():
                    if idx == lbl:
                        results.append((w, float(sim)))
                        break
            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        else:
            # Brute force fallback
            sims: List[Tuple[str, float]] = []
            for w, v in self._bundle.embeddings.items():
                sims.append((w, float(np.dot(query_vec, v))))
            sims.sort(key=lambda x: x[1], reverse=True)
            return sims[:k]

    def visualise(
        self,
        words: Iterable[str],
        method: str = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Visualise the embeddings of ``words`` in two or three dimensions.

        This function reduces the dimensionality of the embeddings using either
        principal component analysis (PCA) or t‑distributed stochastic neighbour
        embedding (t‑SNE) and returns the projected coordinates.  A scatter
        plot with word labels is also produced using ``matplotlib``.  The
        figure is saved to ``output_path`` if provided.

        Parameters
        ----------
        words : Iterable[str]
            Collection of words or sentences to visualise.
        method : {"pca", "tsne"}, optional
            Dimensionality reduction algorithm to use.  PCA is linear and
            preserves global structure; t‑SNE is non‑linear and better at
            capturing local neighbourhoods but is slower【448492695386158†L27-L77】.  Default is ``pca``.
        n_components : int, optional
            Number of dimensions of the output space (2 or 3).  Default is 2.
        perplexity : float, optional
            t‑SNE perplexity parameter; ignored when using PCA.  Typical
            values are between 5 and 50.
        output_path : str or None, optional
            If provided, path to which the generated plot will be saved.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple ``(points, vectors)``.  ``points`` has shape
            ``(len(words), n_components)`` and contains the projected
            coordinates.  ``vectors`` has shape ``(len(words), d)`` and
            contains the original d‑dimensional embeddings.

        Raises
        ------
        ImportError
            If scikit‑learn or matplotlib is not available.
        ValueError
            If ``method`` is not recognised or ``n_components`` is not 2 or 3.
        """
        if PCA is None or TSNE is None:
            raise ImportError(
                "scikit-learn must be installed to perform dimensionality reduction"
            )
        import matplotlib
        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt  # type: ignore

        if method not in {"pca", "tsne"}:
            raise ValueError("method must be 'pca' or 'tsne'")
        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3")

        # Compute embeddings for all words
        word_list = [w.strip().lower() for w in words]
        vectors = np.vstack([self.embed(w) for w in word_list])

        if method == "pca":
            reducer = PCA(n_components=n_components)
        else:
            reducer = TSNE(n_components=n_components, perplexity=perplexity, init="random")
        points = reducer.fit_transform(vectors)

        # Plot the points
        fig = plt.figure()
        if n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D  # type: ignore
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])
            for i, w in enumerate(word_list):
                ax.text(points[i, 0], points[i, 1], points[i, 2], w)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
        else:
            ax = fig.add_subplot(111)
            ax.scatter(points[:, 0], points[:, 1])
            for i, w in enumerate(word_list):
                ax.text(points[i, 0], points[i, 1], w)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
        title = f"{method.upper()} projection of {len(word_list)} embeddings"
        ax.set_title(title)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        plt.close(fig)
        return points, vectors

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Save the embedding bundle and HNSW index to disk.

        The embeddings and mapping are written as JSON to ``bundle_path``.
        The index is saved to ``index_path`` if ``hnswlib`` is available and
        indexing is enabled.  Existing files are overwritten.
        """
        # Serialise the embeddings to JSON
        with open(self.bundle_path, "w", encoding="utf-8") as fh:
            json.dump(self._bundle.to_json(), fh)
        # Save the index
        if self._index is not None:
            self._index.save_index(self.index_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarities, generate embeddings and visualise them."
        )
    )
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-base",
        help=(
            "HuggingFace model name to use for embeddings.  "
            "Default is intfloat/multilingual-e5-base."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory where the cache and index will be stored",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Similarity command
    sim_parser = subparsers.add_parser(
        "similarity", help="Compute the cosine similarity between two texts"
    )
    sim_parser.add_argument("text1", help="First word or sentence")
    sim_parser.add_argument("text2", help="Second word or sentence")

    # Embed command
    embed_parser = subparsers.add_parser(
        "embed", help="Precompute embeddings for a list of words or sentences"
    )
    embed_parser.add_argument(
        "--words",
        nargs="+",
        required=True,
        help="Words or sentences to embed",
    )

    # Plot command
    plot_parser = subparsers.add_parser(
        "plot", help="Visualise embeddings in 2D or 3D using PCA or t-SNE"
    )
    plot_parser.add_argument(
        "--words",
        nargs="+",
        required=True,
        help="Words or sentences to visualise",
    )
    plot_parser.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="Dimensionality reduction method (pca or tsne)",
    )
    plot_parser.add_argument(
        "--components",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of dimensions for the projection (2 or 3)",
    )
    plot_parser.add_argument(
        "--output",
        default=None,
        help="Filename where the plot will be saved",
    )
    plot_parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE (ignored for PCA)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sim = SimilarityModel(
        model_name=args.model, cache_dir=args.cache_dir, build_index=True
    )
    if args.command == "similarity":
        sim_value = sim.cosine_similarity(args.text1, args.text2)
        print(f"Cosine similarity between '{args.text1}' and '{args.text2}' is {sim_value:.4f}")
        sim.save()
    elif args.command == "embed":
        for word in args.words:
            vec = sim.embed(word)
            print(f"Embedded '{word}' (first 5 values): {vec[:5]}...")
        sim.save()
    elif args.command == "plot":
        points, _ = sim.visualise(
            args.words,
            method=args.method,
            n_components=args.components,
            perplexity=args.perplexity,
            output_path=args.output,
        )
        print(f"Projection coordinates:\n{points}")
        sim.save()
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()