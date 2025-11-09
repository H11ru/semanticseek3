#!/usr/bin/env python3
"""
Testaa embedausten laatua vertailemalla sanojen samankaltaisuuksia.
Vertailee sekä samaan kategoriaan kuuluvia sanoja että eri kategorioiden sanoja.
"""

import numpy as np
from pathlib import Path
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from similarity import SimilarityModel


def cosine_similarity(a, b):
    """Laske vektoreiden välinen kosini-similaarisuus."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_embeddings():
    """Testaa embedausten laatua muutamilla testitapauksilla."""
    # Test cases: [(word1, word2, expected_similar)]
    test_cases = [
        # Samankaltaiset sanat (pitäisi olla korkeampi samankaltaisuus)
        ("koira", "kissa", True),      # Eläimet
        ("auto", "bussi", True),       # Ajoneuvot
        ("suru", "ilo", True),         # Tunteet (vastakohdat, mutta samaa kategoriaa)
        ("vasara", "saha", True),      # Työkalut
        ("punainen", "sininen", True), # Värit
        
        # Erilaiset sanat (pitäisi olla matalampi samankaltaisuus)
        ("koira", "auto", False),      # Eläin vs. ajoneuvo
        ("suru", "vasara", False),     # Tunne vs. työkalu
        ("punainen", "kissa", False),  # Väri vs. eläin
        ("saha", "ilo", False),        # Työkalu vs. tunne
        ("bussi", "sininen", False),   # Ajoneuvo vs. väri
    ]
    
    # Luo embedaukset testidata
    print("Luodaan embedaukset testisanoille...")
    model = SimilarityModel()
    
    # Kerää kaikki uniikit sanat
    all_words = list(set(word for pair in test_cases for word in pair[:2]))
    
    # Embedaa sanat
    embeddings = model.encode(all_words)
    word_to_vec = dict(zip(all_words, embeddings))
    
    # Testaa parit
    print("\nTestitulokset:")
    print("-" * 60)
    print("Sanapari                      Samankaltaisuus  Odotettu  Tulos")
    print("-" * 60)
    
    successes = 0
    total = 0
    
    for word1, word2, expected_similar in test_cases:
        sim = cosine_similarity(word_to_vec[word1], word_to_vec[word2])
        
        # Arvioi tulos: samankaltaisten pitäisi olla > 0.4, erilaisten < 0.3
        threshold = 0.35
        is_similar = sim > threshold
        success = is_similar == expected_similar
        
        if success:
            successes += 1
        total += 1
        
        # Tulosta tulos
        result_mark = "✓" if success else "✗"
        word_pair = f"{word1:12} <-> {word2:12}"
        expected = "samanlaiset" if expected_similar else "erilaiset"
        print(f"{word_pair}  {sim:.3f}        {expected:10} {result_mark}")
    
    # Tulosta yhteenveto
    print("-" * 60)
    success_rate = (successes / total) * 100
    print(f"\nOnnistumisprosentti: {success_rate:.1f}% ({successes}/{total})")

    return success_rate > 80  # Testi läpi jos yli 80% onnistuu


if __name__ == "__main__":
    try:
        success = test_embeddings()
        print("\n" + ("✓ Testi läpäisty!" if success else "✗ Testi epäonnistui!"))
    except Exception as e:
        print(f"\n✗ Virhe testin ajossa: {str(e)}")
        raise