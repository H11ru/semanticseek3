import sys

# Clear any loaded modules
for mod in list(sys.modules.keys()):
    if 'translator' in mod or 'similarity' in mod:
        del sys.modules[mod]

import translator
import numpy as np

# Force fresh model load
translator._model = None
translator._similarity_module = None
translator._embedding_cache.clear()
translator._loading_printed = False

translator.VERBOSE = True

print("=== Testing FRESH Downloaded Model ===\n")

model = translator._get_model()
print(f"\nModel loaded: {model.model_name}")
print(f"Is E5 model: {model.is_e5_model}\n")

# Test critical pairs
test_pairs = [
    ("calm", "peaceful"),      # Should be HIGH (synonyms)
    ("calm", "explosion"),     # Should be LOW (opposites)
    ("DARK", "LIGHT"),         # Should be LOW (opposites)
    ("happy", "joyful"),       # Should be HIGH (synonyms)
    ("hot", "cold"),           # Should be LOW (opposites)
    ("calm", "qwerty123"),     # Should be MEDIUM-LOW (gibberish)
]

print("Testing word pairs:\n")
for word1, word2 in test_pairs:
    score = translator.get_similarity(word1, word2)
    
    # Determine if result is expected
    is_synonym = word2 in ["peaceful", "joyful"]
    is_opposite = word2 in ["explosion", "LIGHT", "cold"]
    is_gibberish = "qwerty" in word2
    
    if is_synonym:
        status = "✓" if score > 80 else "✗"
        expected = "HIGH (>80)"
    elif is_opposite:
        status = "✓" if score < 70 else "✗"
        expected = "LOW (<70)"
    elif is_gibberish:
        status = "✓" if 50 < score < 75 else "✗"
        expected = "MED (50-75)"
    else:
        status = "?"
        expected = "?"
    
    print(f"  {status} {word1:12} ↔ {word2:12}: {score:5.2f}%  (expected: {expected})")

print("\n=== DIAGNOSIS ===")
dark_light = translator.get_similarity("DARK", "LIGHT")
if dark_light < 70:
    print("✅ MODEL IS WORKING! Opposites score low.")
else:
    print(f"❌ MODEL STILL BROKEN! DARK↔LIGHT = {dark_light:.2f}% (should be <70)")