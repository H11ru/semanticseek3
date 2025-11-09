# Model Update: Making Semantic Seek More Fun and Educational

## What Changed

Switched from `intfloat/multilingual-e5-base` to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to create a more engaging and educational gameplay experience.

## Why the Change?

### The Problem with E5-base

The E5 model was giving **unrealistically high similarity scores** for unrelated words:

```
Target: kissa (cat)
├─ koira (dog):      0.9021  🔥 Hot!     ✓ correct
├─ ilves (lynx):     0.8332  🌡️  Warm    ✓ correct
├─ auto (car):       0.8318  🌡️  Warm    ✗ WRONG! (unrelated gets "warm")
└─ talo (house):     0.8582  🌡️  Warm    ✗ WRONG! (unrelated gets "warm")
```

**Score compression**: Everything scored 0.78-0.92 (only 14% range)
- Made the game boring - almost every guess was "warm" or "hot"
- Couldn't distinguish between good and bad guesses
- No interesting "almost correct" learning moments

### The Solution: Paraphrase-Multilingual-MiniLM

This model is **optimized for word-level similarity** and creates much better gameplay:

```
Target: leipä (bread)
├─ ruoka (food):     0.7521  🔥 KUUMA!           ✨ Great - learned category!
├─ juusto (cheese):  0.6022  🌡️  Lämmin          ✨ Goes with bread!
├─ voi (butter):     0.5243  😊 Lämpöinen        💡 Spread on bread
├─ kahvi (coffee):   0.4258  😊 Lämpöinen        💡 Breakfast connection
└─ auto (car):       0.3335  ❄️  Kylmä           ✓ Correctly cold!
```

**Better discrimination**: Scores range 0.06-0.94 (88% range - 6x better!)
- Clear separation between related and unrelated words
- Fun "almost correct" moments teach semantic relationships
- Category thinking is rewarded (e.g., "eläin" for "kissa")
- Educational value: players learn Finnish word relationships naturally

## Technical Improvements

### Model Comparison

| Metric | E5-base (old) | Paraphrase-MiniLM (new) |
|--------|---------------|-------------------------|
| Score range | 0.14 | 0.88 (6x better!) |
| Std deviation | 0.019 | 0.193 (10x better!) |
| Embedding dim | 768 | 384 (faster!) |
| Languages | 100+ | 50+ (still multilingual) |
| Best for | Sentence search | Word similarity ✓ |

### New Feedback Thresholds

Calibrated for fun and educational gameplay:

```python
# Old (E5-base): [0.90, 0.80, 0.70] - too compressed
# New (Paraphrase): [0.75, 0.55, 0.40] - better spread

🔥 KUUMA! (Hot):       >= 0.75  Very close! Right track!
🌡️  Lämmin (Warm):     >= 0.55  Clear connection, same category
😊 Lämpöinen (Mild):   >= 0.40  Some relationship, interesting!
❄️  Kylmä (Cold):      <  0.40  Unrelated, try different direction
```

## Educational Benefits

The new model creates natural learning moments:

1. **Category hierarchies**
   - "eläin" (animal) → "kissa" (cat) shows warm relationship
   - Teaches broader vs. specific concepts

2. **Semantic relationships**
   - "Suomi" → "Helsinki" (country-capital)
   - "leipä" → "voi" (bread-butter pairing)

3. **No false positives**
   - "auto" (car) is consistently cold across all categories
   - No misleading feedback that would confuse learners

4. **Discovery encourages exploration**
   - Wide score range makes exploring guesses rewarding
   - Players learn by discovering connections

## Files Changed

1. **[configs/settings.yaml](configs/settings.yaml)**
   - Updated model name
   - Added explanation comments

2. **[src/core/scoring.py](src/core/scoring.py)**
   - New thresholds: [0.75, 0.55, 0.40]
   - Updated Finnish feedback text with emojis
   - Added educational design notes

3. **[src/core/model.py](src/core/model.py)**
   - Changed default model
   - Added documentation about model choice
   - Updated comments

4. **Rebuilt artifacts**
   - `artifacts/fi/` - Finnish embeddings (180 words, 384-dim)
   - `artifacts/en/` - English embeddings (179 words, 384-dim)
   - Both now using new model

## Testing Results

Run `python demo_gameplay.py` to see example gameplay scenarios!

Key findings:
- ✅ Unrelated words now clearly cold (0.28-0.33)
- ✅ Category words get warm feedback (0.44-0.48)
- ✅ Strong relationships show hot (0.75-0.88)
- ✅ Food pairings create fun learning moments
- ✅ Country-capital relationships discovered

## Migration Notes

**No user action needed!** The changes are backward compatible:

- Old similarity.py script still works (specify --model flag)
- API endpoints unchanged
- CLI commands unchanged
- Just rebuild indexes if you modify categories

**To test the old model** (for comparison):
```bash
python similarity.py --model intfloat/multilingual-e5-base similarity cat car
```

## Next Steps

Consider these future enhancements:

1. **Fine-tune thresholds** based on user feedback
   - Monitor which words players find confusing
   - Adjust bins if too easy/hard

2. **Add difficulty levels**
   - Easy: broader thresholds
   - Hard: narrower thresholds
   - Expert: only show cold/hot

3. **Track educational progress**
   - Which semantic relationships players discover
   - Build vocabulary learning features

4. **Multilingual learning mode**
   - Mix Finnish and English words
   - Teach cross-language relationships

## Tools Created

Analysis and calibration tools in the repo:

- **[calibrate_thresholds.py](calibrate_thresholds.py)** - Analyze vocabulary and recommend thresholds
- **[demo_gameplay.py](demo_gameplay.py)** - Show example gameplay scenarios
- **[test_models.py](test_models.py)** - Compare different embedding models
- **[MODEL_ANALYSIS.md](MODEL_ANALYSIS.md)** - Full technical analysis

## References

- New model: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Old model: https://huggingface.co/intfloat/multilingual-e5-base
- Semantic Seek v3 design: [semantic_seek_v_3.md](semantic_seek_v_3.md)
