from __future__ import annotations
import numpy as np

# Feedback thresholds and messages
# Calibrated for paraphrase-multilingual-MiniLM-L12-v2 model based on Finnish vocabulary analysis
# These thresholds create a fun, educational experience with interesting "almost correct" moments:
# - Hot (0.75+): Very strong semantic relationship - you're on the right track!
# - Warm (0.55+): Clear connection - maybe same category or related concept
# - Mild (0.40+): Some relationship - could teach interesting connections
# - Cold (<0.40): Unrelated - try a different semantic direction
#
# The wider spread (vs old E5 model) makes "almost correct" answers more discoverable,
# helping players learn semantic relationships in Finnish.
FEEDBACK_BINS = [0.75, 0.55, 0.40]
FEEDBACK_TEXT = [
    "🔥 KUUMA! Olet hyvin lähellä!",           # Hot! You're very close!
    "🌡️  Lämmin - oikeaan suuntaan",          # Warm - right direction
    "😊 Lämpöinen - jotain yhteistä",         # Mild - something in common
    "❄️  Kylmä - kokeile muuta"               # Cold - try something else
]

# English feedback (for future use)
FEEDBACK_TEXT_EN = [
    "🔥 HOT! You're very close!",
    "🌡️  Warm - right direction",
    "😊 Mild - some connection",
    "❄️  Cold - try something else"
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
