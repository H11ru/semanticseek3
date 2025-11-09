"""Tests for scoring module."""

import numpy as np
import pytest

from src.core.scoring import cosine_score, feedback, score_to_percentage


class TestCosineScore:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have score 1.0."""
        vec = np.array([0.5, 0.5, 0.5, 0.5])
        vec = vec / np.linalg.norm(vec)  # normalize
        score = cosine_score(vec, vec)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have score ~0.0."""
        vec1 = np.array([1.0, 0.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0, 0.0])
        score = cosine_score(vec1, vec2)
        assert abs(score - 0.0) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should be clipped to 0.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        score = cosine_score(vec1, vec2)
        assert score >= 0.0


class TestFeedback:
    """Tests for feedback generation."""

    def test_excellent_score(self):
        """Score >= 0.90 should give excellent feedback."""
        msg = feedback(0.95, language="fi")
        assert "MAHTAVA" in msg

    def test_very_close_score(self):
        """Score in [0.80, 0.90) should give very close feedback."""
        msg = feedback(0.85, language="fi")
        assert "Erittäin lähellä" in msg

    def test_close_score(self):
        """Score in [0.70, 0.80) should give close feedback."""
        msg = feedback(0.75, language="fi")
        assert "Lähellä" in msg

    def test_far_score(self):
        """Score < 0.70 should give far feedback."""
        msg = feedback(0.50, language="fi")
        assert "Kaukana" in msg

    def test_english_feedback(self):
        """English feedback should work."""
        msg = feedback(0.95, language="en")
        assert "AMAZING" in msg


class TestScoreToPercentage:
    """Tests for score to percentage conversion."""

    def test_perfect_score(self):
        """Score 1.0 should be 100%."""
        assert score_to_percentage(1.0) == 100

    def test_zero_score(self):
        """Score 0.0 should be 0%."""
        assert score_to_percentage(0.0) == 0

    def test_mid_score(self):
        """Score 0.5 should be 50%."""
        assert score_to_percentage(0.5) == 50

    def test_rounding(self):
        """Scores should be rounded to nearest integer."""
        assert score_to_percentage(0.856) == 86
