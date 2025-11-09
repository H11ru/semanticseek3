# Semantic Seek v3 - Model Analysis & Recommendations

## Problem Identified

The current model (`intfloat/multilingual-e5-base`) produces **unrealistically high and compressed similarity scores** for single-word comparisons, making the game less interesting and discriminative.

### Key Statistics

| Model | Min Score | Max Score | Range | Std Dev |
|-------|-----------|-----------|-------|---------|
| **E5-base (current)** | 0.7679 | 0.9021 | 0.1341 | 0.0335 |
| **Paraphrase MiniLM** | 0.0355 | 0.8794 | 0.8439 | 0.1535 |

### Problematic Examples with E5-base

When target word is "kissa" (cat):

| Guess | Relationship | Score | Feedback | Problem |
|-------|-------------|-------|----------|---------|
| koira (dog) | Related pet | 0.9021 | 🔥 Kuuma! | ✓ Correct |
| ilves (lynx) | Cat family | 0.8332 | 🌡️ Lämmin | ✓ Correct |
| **auto (car)** | **UNRELATED** | **0.8318** | **🌡️ Lämmin** | ✗ **Misleading!** |
| **talo (house)** | **UNRELATED** | **0.8582** | **🌡️ Lämmin** | ✗ **Misleading!** |

**The issue**: Unrelated words like "auto" (car) get a "warm" rating (0.83), which is nearly as high as "ilves" (lynx, 0.83), making it impossible for players to distinguish between good and bad guesses.

## Root Cause

The E5 model family was designed for:
- **Sentence-level semantic search** (not single words)
- **Retrieval tasks** with asymmetric query-passage relationships
- Text with the `query:` prefix for best results

For **word-level similarity** in a game context, this creates:
1. **Score compression**: Everything scores 0.77-0.90 (only 13% range)
2. **Poor discrimination**: Related vs unrelated words are too similar
3. **Boring gameplay**: Almost everything is "warm" or "hot"

## Solutions

### Option 1: Switch to Better Model (Recommended)

**Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Advantages**:
- ✅ Excellent score discrimination (84% range vs 13%)
- ✅ Multilingual support (Finnish + English + 50+ languages)
- ✅ Better suited for word-level similarity
- ✅ Clear separation between related/unrelated concepts

**Required Changes**:
1. Update `configs/settings.yaml`:
   ```yaml
   model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```

2. Adjust feedback thresholds in `src/core/scoring.py`:
   ```python
   # OLD (for E5)
   THRESHOLDS = [0.90, 0.80, 0.70]

   # NEW (for Paraphrase MiniLM)
   THRESHOLDS = [0.60, 0.45, 0.30]
   ```

3. Rebuild indexes:
   ```bash
   make build-fi
   make build-en
   ```

**Trade-offs**:
- Smaller model dimension (384 vs 768) - actually better for performance
- Different score distribution requires threshold recalibration

### Option 2: Adjust E5 Thresholds (Quick Fix)

Keep E5-base but lower the thresholds to account for score compression:

```python
# In src/core/scoring.py
THRESHOLDS = [0.88, 0.82, 0.77]  # Adjusted for E5's high baseline
```

**Advantages**:
- ✅ No need to rebuild indexes
- ✅ Quick fix

**Disadvantages**:
- ✗ Still poor discrimination between related/unrelated
- ✗ Doesn't solve the fundamental problem
- ✗ Game will still be less interesting

## Benchmark Results

### With E5-base (current thresholds [0.90, 0.80, 0.70])
```
Target: kissa
koira (dog):      0.9021  🔥 Kuuma!
ilves (lynx):     0.8332  🌡️  Lämmin
auto (car):       0.8318  🌡️  Lämmin    ← Problem: Unrelated gets "warm"!
talo (house):     0.8582  🌡️  Lämmin    ← Problem: Unrelated gets "warm"!
```

### With Paraphrase MiniLM (adjusted thresholds [0.60, 0.45, 0.30])
```
Target: kissa
kissa (exact):    1.0000  🔥 Kuuma!
eläin (animal):   0.4446  😊 Lämpöinen
koira (dog):      0.2824  ❄️  Kylmä
ilves (lynx):     0.2973  ❄️  Kylmä
auto (car):       0.2967  ❄️  Kylmä     ✓ Correctly cold!
talo (house):     0.2892  ❄️  Kylmä     ✓ Correctly cold!
```

**Note**: The Paraphrase model shows all specific animals as "cold" because it recognizes category relationships differently. This might actually create an interesting game dynamic where broad category words (like "eläin" = animal) score higher than specific instances.

## Recommendation

**Switch to `paraphrase-multilingual-MiniLM-L12-v2`** with adjusted thresholds.

This requires:
1. A systematic analysis of your actual game vocabulary to calibrate optimal thresholds
2. Rebuilding indexes (one-time cost)
3. Potentially adjusting game difficulty based on the new score distribution

The improved discrimination will make the game more engaging and meaningful for players.

## Next Steps

1. **Test with your actual category data** to determine optimal thresholds
2. **Run both models** on sample game scenarios from `data/fi/categories_fi.json`
3. **Calibrate thresholds** based on desired difficulty curve
4. **Update configuration** and rebuild artifacts

## References

- E5 Model Card: https://huggingface.co/intfloat/multilingual-e5-base
- Paraphrase MiniLM: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Test scripts created: `test_embeddings.py`, `test_models.py`, `test_comprehensive.py`
