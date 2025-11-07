from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List


class CategoryStore:
    """Loads and manages word categories from JSON files."""

    def __init__(self, json_path: str):
        """
        Load categories from a JSON file.

        Args:
            json_path: Path to categories JSON file

        Expected JSON format:
        {
            "language": "fi",
            "version": "2025-11-07",
            "categories": {
                "Category1": ["word1", "word2", ...],
                "Category2": ["word3", "word4", ...]
            }
        }
        """
        self.path = Path(json_path)
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.language: str = payload.get("language", "fi")
        self.version: str = payload.get("version", "unknown")
        self.categories: Dict[str, List[str]] = payload["categories"]

    @property
    def vocab(self) -> List[str]:
        """
        Extract all unique words from categories in stable order.

        Returns:
            Deduplicated list of words maintaining first occurrence order
        """
        words = []
        for lst in self.categories.values():
            words.extend(lst)

        # Deduplicate while preserving order
        seen = set()
        dedup = []
        for w in words:
            if w not in seen:
                seen.add(w)
                dedup.append(w)

        return dedup

    def get_category_words(self, category: str) -> List[str]:
        """Get all words in a specific category."""
        return self.categories.get(category, [])

    def list_categories(self) -> List[str]:
        """Get list of all category names."""
        return list(self.categories.keys())
