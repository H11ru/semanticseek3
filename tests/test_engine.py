"""Tests for game engine."""

import numpy as np
import pytest

from src.core.engine import GameEngine, GameState


@pytest.fixture
def simple_vocab():
    """Create a simple test vocabulary."""
    return ["apple", "banana", "orange", "cat", "dog", "bird"]


@pytest.fixture
def simple_vectors():
    """Create simple test vectors (not semantically meaningful)."""
    # Just random normalized vectors for testing
    np.random.seed(42)
    vecs = np.random.randn(6, 10).astype(np.float32)
    # Normalize
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def simple_engine(simple_vocab, simple_vectors):
    """Create a simple game engine for testing."""
    word_to_ix = {w: i for i, w in enumerate(simple_vocab)}
    return GameEngine(vocab=simple_vocab, vectors=simple_vectors, word_to_ix=word_to_ix)


class TestGameEngine:
    """Tests for GameEngine."""

    def test_start_game(self, simple_engine):
        """Starting a game should create valid state."""
        state = simple_engine.start(
            language="en",
            category="test",
            words_in_category=["apple", "banana"],
            seed=123
        )

        assert state.language == "en"
        assert state.category == "test"
        assert state.target_word in ["apple", "banana"]
        assert state.target_vec is not None
        assert len(state.guesses) == 0
        assert state.best_score == 0.0

    def test_start_game_reproducible(self, simple_engine):
        """Same seed should give same target."""
        state1 = simple_engine.start("en", "test", ["apple", "banana", "orange"], seed=42)
        state2 = simple_engine.start("en", "test", ["apple", "banana", "orange"], seed=42)

        assert state1.target_word == state2.target_word

    def test_guess_in_vocab(self, simple_engine):
        """Guessing a word in vocab should work."""
        state = simple_engine.start("en", "test", ["apple"], seed=1)

        score, feedback = simple_engine.guess(state, "banana")

        assert len(state.guesses) == 1
        assert state.guesses[0] == "banana"
        assert 0.0 <= score <= 1.0
        assert isinstance(feedback, str)

    def test_guess_updates_best(self, simple_engine):
        """Better guess should update best score."""
        state = simple_engine.start("en", "test", ["apple"], seed=1)

        score1, _ = simple_engine.guess(state, "cat")
        score2, _ = simple_engine.guess(state, "dog")

        if score2 > score1:
            assert state.best_score == score2
            assert state.best_guess == "dog"
        else:
            assert state.best_score == score1
            assert state.best_guess == "cat"

    def test_guess_oov_without_embed_fn(self, simple_engine):
        """Guessing OOV word without embed_fn should raise error."""
        state = simple_engine.start("en", "test", ["apple"], seed=1)

        with pytest.raises(ValueError, match="not in vocabulary"):
            simple_engine.guess(state, "unknown_word", embed_fn=None)

    def test_guess_oov_with_embed_fn(self, simple_engine):
        """Guessing OOV word with embed_fn should work."""
        state = simple_engine.start("en", "test", ["apple"], seed=1)

        def mock_embed(words):
            # Return a random normalized vector
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            return np.array([vec])

        score, feedback = simple_engine.guess(state, "unknown_word", embed_fn=mock_embed)

        assert len(state.guesses) == 1
        assert state.guesses[0] == "unknown_word"
        assert 0.0 <= score <= 1.0

    def test_check_win(self, simple_engine):
        """check_win should return True for high scores."""
        state = simple_engine.start("en", "test", ["apple"], seed=1)

        # Manually set high score
        state.best_score = 0.96

        assert simple_engine.check_win(state, threshold=0.95)

    def test_check_win_below_threshold(self, simple_engine):
        """check_win should return False for low scores."""
        state = simple_engine.start("en", "test", ["apple"], seed=1)

        state.best_score = 0.80

        assert not simple_engine.check_win(state, threshold=0.95)

    def test_top_suggestions(self, simple_engine, simple_vectors):
        """top_suggestions should return valid suggestions."""
        state = simple_engine.start("en", "test", ["apple"], seed=1)

        def mock_search(qvec, k):
            # Return indices and fake similarities
            indices = np.array([0, 1, 2, 3, 4])
            similarities = np.array([0.99, 0.85, 0.75, 0.65, 0.55])
            return indices, similarities

        suggestions = simple_engine.top_suggestions(state, mock_search, k=3)

        assert len(suggestions) <= 3
        assert all(isinstance(word, str) for word, _ in suggestions)
        assert all(0.0 <= sim <= 1.0 for _, sim in suggestions)
        # Target word should be excluded
        assert state.target_word not in [word for word, _ in suggestions]
