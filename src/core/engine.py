from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import numpy as np

from .scoring import cosine_score, feedback


@dataclass
class GameState:
    """Represents the state of an ongoing game."""
    language: str
    category: str
    target_word: str
    target_vec: np.ndarray
    guesses: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    best_score: float = 0.0
    best_guess: str = ""
    seed: int = 0
    hints_used: int = 0
    # Cache top words for hint system (computed once at game start)
    _top_words: List[Tuple[str, float]] = field(default_factory=list, init=False, repr=False)


class GameEngine:
    """Core game logic for Semantic Seek."""

    def __init__(self, vocab: List[str], vectors: np.ndarray, word_to_ix: dict[str, int]):
        """
        Initialize the game engine.

        Args:
            vocab: List of words in vocabulary (must match vectors order)
            vectors: Precomputed L2-normalized embeddings (N, dim)
            word_to_ix: Mapping from word to index in vocab/vectors
        """
        self.vocab = vocab
        self.vectors = vectors
        self.word_to_ix = word_to_ix

    def start(
        self,
        language: str,
        category: str | None,
        words_in_category: List[str],
        seed: int | None = None
    ) -> GameState:
        """
        Start a new game by selecting a random target word.

        Args:
            language: Language code
            category: Category name (or None for random)
            words_in_category: List of candidate words for target selection
            seed: Random seed for reproducibility

        Returns:
            New GameState with selected target
        """
        rng = random.Random(seed)
        target_word = rng.choice(words_in_category)
        target_vec = self.vectors[self.word_to_ix[target_word]]

        return GameState(
            language=language,
            category=category or "random",
            target_word=target_word,
            target_vec=target_vec,
            seed=seed or 0
        )

    def guess(
        self,
        state: GameState,
        word: str,
        embed_fn: Callable[[List[str]], np.ndarray] | None = None
    ) -> Tuple[float, str]:
        """
        Process a guess and update game state.

        Args:
            state: Current game state (modified in place)
            word: The guessed word
            embed_fn: Function to embed OOV words (required if word not in vocab)

        Returns:
            Tuple of (similarity_score, feedback_message)
        """
        state.guesses.append(word)

        # Get vector for guessed word
        if word in self.word_to_ix:
            vec = self.vectors[self.word_to_ix[word]]
        else:
            # Out-of-vocabulary: embed on the fly
            if embed_fn is None:
                raise ValueError(f"Word '{word}' not in vocabulary and no embed_fn provided")
            vec = embed_fn([word])[0]

        # Calculate similarity
        score = cosine_score(state.target_vec, vec)
        state.scores.append(score)

        # Update best score
        if score > state.best_score:
            state.best_score = score
            state.best_guess = word

        # Generate feedback
        fb = feedback(score, language=state.language)

        return score, fb

    def top_suggestions(
        self,
        state: GameState,
        index_search_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most similar words to the target.

        Args:
            state: Current game state
            index_search_fn: Function that takes (query_vec, k) and returns (indices, similarities)
            k: Number of suggestions to return

        Returns:
            List of (word, similarity) tuples, excluding the target word
        """
        idxs, sims = index_search_fn(state.target_vec.reshape(1, -1), k + 1)

        pairs = []
        for idx, sim in zip(idxs, sims):
            word = self.vocab[int(idx)]
            if word != state.target_word:
                pairs.append((word, float(sim)))

        return pairs[:k]

    def check_win(self, state: GameState, threshold: float = 0.95) -> bool:
        """Check if the player has won (scored above threshold)."""
        return state.best_score >= threshold

    def compute_top_words(self, state: GameState, k: int = 20) -> List[Tuple[str, float]]:
        """
        Compute and cache the top-k most similar words to the target.
        These are the "best possible guesses" the player could make.

        Args:
            state: Current game state
            k: Number of top words to compute

        Returns:
            List of (word, similarity_score) tuples, sorted by score descending
        """
        # Return cached result if already computed
        if state._top_words:
            return state._top_words[:k]

        # Compute similarities for all vocab words
        all_scores = []
        for i, word in enumerate(self.vocab):
            if word == state.target_word:
                continue  # Skip the target itself
            score = cosine_score(state.target_vec, self.vectors[i])
            all_scores.append((word, float(score)))

        # Sort by score descending
        all_scores.sort(key=lambda x: x[1], reverse=True)

        # Cache result
        state._top_words = all_scores

        return all_scores[:k]

    def get_hint(self, state: GameState, hint_type: str = "category") -> str:
        """
        Provide a hint to the player based on the target word.

        Args:
            state: Current game state
            hint_type: Type of hint - "category", "best_word", or "top_3"

        Returns:
            Hint string in the appropriate language
        """
        state.hints_used += 1

        if hint_type == "category":
            # Simple category hint
            if state.language == "fi":
                return f"💡 Vihje: Sana kuuluu kategoriaan '{state.category}'"
            else:
                return f"💡 Hint: The word belongs to category '{state.category}'"

        elif hint_type == "best_word":
            # Show the best possible guess
            top_words = self.compute_top_words(state, k=1)
            if top_words:
                word, score = top_words[0]
                if state.language == "fi":
                    return f"💡 Vihje: Paras mahdollinen arvaus: '{word}' (pisteet: {score:.2f})"
                else:
                    return f"💡 Hint: Best possible guess: '{word}' (score: {score:.2f})"

        elif hint_type == "top_3":
            # Show top 3 best guesses
            top_words = self.compute_top_words(state, k=3)
            if state.language == "fi":
                hint = "💡 Vihje: Kolme parasta arvausta:\n"
            else:
                hint = "💡 Hint: Three best guesses:\n"

            for i, (word, score) in enumerate(top_words, 1):
                hint += f"   {i}. {word} (pisteet: {score:.2f})\n"

            return hint.strip()

        return "❓ Unknown hint type"

    def get_best_possible_score(self, state: GameState) -> float:
        """
        Get the highest possible score (best word in vocabulary).
        Useful for showing player how close they are to optimal.

        Args:
            state: Current game state

        Returns:
            The similarity score of the best possible guess
        """
        top_words = self.compute_top_words(state, k=1)
        if top_words:
            return top_words[0][1]
        return 0.0
