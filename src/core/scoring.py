from __future__ import annotations
import numpy as np

# Feedback thresholds and messages
FEEDBACK_BINS = [0.90, 0.80, 0.70]
FEEDBACK_TEXT = [
    "MAHTAVA: lähes identtinen merkitys",
    "Erittäin lähellä",
    "Lähellä",
    "Kaukana"
]

# English feedback (for future use)
FEEDBACK_TEXT_EN = [
    "AMAZING: nearly identical meaning",
    "Very close",
    "Close",
    "Far"
]


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector (should be L2-normalized)
        b: Second vector (should be L2-normalized)

    Returns:
        Cosine similarity in range [0.0, 1.0]

    Note:
        If vectors are pre-normalized, this is just the dot product.
    """
    score = float(np.dot(a, b))
    return float(np.clip(score, 0.0, 1.0))


def feedback(score: float, language: str = "fi") -> str:
    """
    Map a similarity score to qualitative feedback.

    Args:
        score: Similarity score in range [0.0, 1.0]
        language: Language code ('fi' or 'en')

    Returns:
        Feedback string appropriate for the score
    """
    text = FEEDBACK_TEXT if language == "fi" else FEEDBACK_TEXT_EN

    for i, threshold in enumerate(FEEDBACK_BINS):
        if score >= threshold:
            return text[i]

    return text[-1]


def score_to_percentage(score: float) -> int:
    """Convert similarity score to percentage (0-100)."""
    return int(round(score * 100))
