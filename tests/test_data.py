"""Tests for data loading."""

import json
import pytest
from pathlib import Path
from src.core.data import CategoryStore


@pytest.fixture
def temp_categories_file(tmp_path):
    """Create a temporary categories file."""
    data = {
        "language": "test",
        "version": "1.0",
        "categories": {
            "Fruits": ["apple", "banana", "orange"],
            "Animals": ["cat", "dog", "cat"]  # Intentional duplicate
        }
    }

    file_path = tmp_path / "categories.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return str(file_path)


class TestCategoryStore:
    """Tests for CategoryStore."""

    def test_load_categories(self, temp_categories_file):
        """Should load categories from JSON."""
        store = CategoryStore(temp_categories_file)

        assert store.language == "test"
        assert store.version == "1.0"
        assert len(store.categories) == 2
        assert "Fruits" in store.categories
        assert "Animals" in store.categories

    def test_vocab_deduplication(self, temp_categories_file):
        """Vocab should deduplicate words."""
        store = CategoryStore(temp_categories_file)
        vocab = store.vocab

        # "cat" appears twice in the data
        assert vocab.count("cat") == 1

    def test_vocab_order(self, temp_categories_file):
        """Vocab should maintain stable order."""
        store = CategoryStore(temp_categories_file)
        vocab = store.vocab

        # First category is Fruits, second is Animals
        assert vocab.index("apple") < vocab.index("cat")

    def test_get_category_words(self, temp_categories_file):
        """Should get words from specific category."""
        store = CategoryStore(temp_categories_file)

        fruits = store.get_category_words("Fruits")
        assert fruits == ["apple", "banana", "orange"]

    def test_get_nonexistent_category(self, temp_categories_file):
        """Should return empty list for nonexistent category."""
        store = CategoryStore(temp_categories_file)

        words = store.get_category_words("Vegetables")
        assert words == []

    def test_list_categories(self, temp_categories_file):
        """Should list all category names."""
        store = CategoryStore(temp_categories_file)

        categories = store.list_categories()
        assert len(categories) == 2
        assert "Fruits" in categories
        assert "Animals" in categories
