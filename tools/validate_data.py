#!/usr/bin/env python3
"""
Validate category data for quality issues.

Usage:
    python tools/validate_data.py data/fi/categories_fi.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def validate_categories(json_path: str):
    """Validate category JSON file."""
    print(f"Validating: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check required fields
    if "categories" not in data:
        print("❌ Error: Missing 'categories' field")
        return False

    categories = data["categories"]
    language = data.get("language", "unknown")
    version = data.get("version", "unknown")

    print(f"Language: {language}")
    print(f"Version: {version}")
    print(f"Categories: {len(categories)}")

    all_valid = True
    all_words = []

    # Check each category
    for cat_name, words in categories.items():
        if not words:
            print(f"⚠️  Warning: Empty category '{cat_name}'")
            all_valid = False

        if len(words) < 5:
            print(f"⚠️  Warning: Category '{cat_name}' has only {len(words)} words (recommended: 5+)")

        # Check for non-string words
        for word in words:
            if not isinstance(word, str):
                print(f"❌ Error: Non-string word in '{cat_name}': {word}")
                all_valid = False

        all_words.extend(words)

    # Check for duplicates
    word_counts = Counter(all_words)
    duplicates = [(word, count) for word, count in word_counts.items() if count > 1]

    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} duplicate words:")
        for word, count in sorted(duplicates, key=lambda x: -x[1])[:20]:
            print(f"  • '{word}' appears {count} times")
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more")
        all_valid = False
    else:
        print("\n✓ No duplicates found")

    # Summary
    print(f"\nTotal words: {len(all_words)}")
    print(f"Unique words: {len(word_counts)}")

    # Check for empty/whitespace words
    empty_words = [w for w in all_words if not w.strip()]
    if empty_words:
        print(f"❌ Error: Found {len(empty_words)} empty/whitespace words")
        all_valid = False

    if all_valid:
        print("\n✓ Validation passed!")
        return True
    else:
        print("\n⚠️  Validation completed with warnings/errors")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_data.py <path_to_categories.json>")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    success = validate_categories(json_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
