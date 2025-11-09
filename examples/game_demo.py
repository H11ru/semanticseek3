#!/usr/bin/env python3
"""
Demonstration of Semantic Seek gameplay with hint system.
Shows how the game engine knows the best possible guesses.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from src.core.model import EmbeddingModel
from src.core.data import CategoryStore
from src.core.engine import GameEngine
from src.core.scoring import feedback

def print_banner():
    print("=" * 80)
    print("🎮 SEMANTIC SEEK v3.0 - GAME DEMO WITH HINT SYSTEM")
    print("=" * 80)
    print()

def demo_game():
    """Demonstrate a game session with hints."""

    print_banner()

    # Load Finnish data
    lang = "fi"
    data_path = f"data/{lang}/categories_{lang}.json"
    store = CategoryStore(data_path)

    # Load artifacts
    vocab = json.loads(Path(f"artifacts/{lang}/vocab.json").read_text())
    embeddings = np.load(f"artifacts/{lang}/embeddings.npy")
    word_to_ix = {w: i for i, w in enumerate(vocab)}

    # Initialize engine and model
    engine = GameEngine(vocab=vocab, vectors=embeddings, word_to_ix=word_to_ix)
    model = EmbeddingModel()

    print(f"✓ Loaded {len(vocab)} words")
    print(f"✓ Model: {model.model_name.split('/')[-1]}")
    print()

    # Start game
    state = engine.start(
        language="fi",
        category="Eläimet",
        words_in_category=store.categories["Eläimet"],
        seed=42
    )

    print(f"🎯 GAME STARTED!")
    print(f"   Category: {state.category} (Animals)")
    print(f"   Target word: ??? (try to guess!)")
    print(f"   Secret target: {state.target_word} (shhh!)")
    print()

    # Show what the engine knows (best possible guesses)
    print("🧠 ENGINE KNOWLEDGE - Best possible guesses:")
    print("-" * 80)
    top_10 = engine.compute_top_words(state, k=10)
    for i, (word, score) in enumerate(top_10, 1):
        fb = feedback(score, language="fi")
        print(f"   {i:2d}. {word:15s} → {score:.4f}  {fb}")

    best_score = engine.get_best_possible_score(state)
    print(f"\n💡 Best possible score in vocabulary: {best_score:.4f}")
    print()

    # Simulate player guesses
    print("👤 PLAYER GUESSING:")
    print("=" * 80)
    print()

    player_guesses = [
        "koira",     # Dog - related animal
        "auto",      # Car - unrelated
        "eläin",     # Animal - category word!
    ]

    for guess in player_guesses:
        print(f"Guess: '{guess}'")
        score, fb = engine.guess(state, guess, model.encode)
        print(f"   → Score: {score:.4f}  {fb}")

        # Show how close to best possible
        gap = best_score - score
        if gap > 0.2:
            print(f"   💭 Could be better! Gap to best: {gap:.4f}")
        elif gap > 0.1:
            print(f"   👍 Pretty good! Gap to best: {gap:.4f}")
        else:
            print(f"   🌟 Excellent! Very close to optimal!")

        print()

    # Show hint system
    print("=" * 80)
    print("💡 HINT SYSTEM DEMONSTRATION:")
    print("=" * 80)
    print()

    # Category hint
    hint1 = engine.get_hint(state, hint_type="category")
    print(f"Hint Level 1 (Category):")
    print(f"   {hint1}")
    print()

    # Top 3 hint
    hint2 = engine.get_hint(state, hint_type="top_3")
    print(f"Hint Level 2 (Top 3 words):")
    for line in hint2.split('\n'):
        print(f"   {line}")
    print()

    # Best word hint
    hint3 = engine.get_hint(state, hint_type="best_word")
    print(f"Hint Level 3 (Best word):")
    print(f"   {hint3}")
    print()

    print(f"Total hints used: {state.hints_used}")
    print()

    # Final summary
    print("=" * 80)
    print("📊 GAME SUMMARY:")
    print("=" * 80)
    print(f"Target word:          {state.target_word}")
    print(f"Category:             {state.category}")
    print(f"Total guesses:        {len(state.guesses)}")
    print(f"Best guess:           {state.best_guess} ({state.best_score:.4f})")
    print(f"Best possible score:  {best_score:.4f}")
    print(f"Hints used:           {state.hints_used}")
    print()

    # Educational insight
    print("=" * 80)
    print("🎓 EDUCATIONAL INSIGHTS:")
    print("=" * 80)
    print("""
The game engine KNOWS the best possible guesses:
1. Category words like "eläin" (animal) score highest
2. Related concepts score well (koira/dog for hirvi/moose)
3. Unrelated words score low (auto/car)

This creates a fun learning experience where:
- Players discover semantic relationships
- Hints guide without spoiling the fun
- Players learn Finnish vocabulary naturally
- The game can show "you're getting warmer!" feedback

The hint system can be used for:
- Beginner mode: Show top 3 words
- Learning mode: Show best possible after 5 wrong guesses
- Challenge mode: No hints, just explore!
""")

if __name__ == '__main__':
    demo_game()
