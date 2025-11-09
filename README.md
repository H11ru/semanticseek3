# Semantic Seek v3.0

```

Welcome To
 _____                            _   _        _____           _
/  ___|                          | | (_)      /  ___|         | |
\ `--.  ___ _ __ ___   __ _ _ __ | |_ _  ___  \ `--.  ___  ___| | __
 `--. \/ _ \ '_ ` _ \ / _` | '_ \| __| |/ __|  `--. \/ _ \/ _ \ |/ /
/\__/ /  __/ | | | | | (_| | | | | |_| | (__  /\__/ /  __/  __/   <
\____/ \___|_| |_| |_|\__,_|_| |_|\__|_|\___| \____/ \___|\___|_|\_\
_____________________________________________________________________
By: ChatGPT, Caneli and H11rustan.

           ____                      _         _      ____            _
          / ___|  ___  ___ _ __ ___(_)_ __   / \    / ___|  ___  ___| |_
          \___ \ / _ \/ __| '__/ _ \ | '_ \ / _ \   \___ \ / _ \/ __| __|
           ___) |  __/ (__| | |  __/ | | | / ___ \   ___) |  __/ (__| |_
          |____/ \___|\___|_|  \___|_|_| |_/_/   \_\ |____/ \___|\___|\__|

                         — Semantic Seek v3.0 —
```

A multilingual semantic word similarity game that teaches language through fun discovery!

## 🎮 How to Play

The game shows you a **category** (like "Animals"), then picks a secret **target word**. You make guesses, and the game tells you how semantically close you are:

- 🔥 **KUUMA!** (Hot) - Very close! (score ≥ 0.75)
- 🌡️ **Lämmin** (Warm) - Right direction (score ≥ 0.55)
- 😊 **Lämpöinen** (Mild) - Some connection (score ≥ 0.40)
- ❄️ **Kylmä** (Cold) - Try something else (score < 0.40)

### Example Game

```
Category: Eläimet (Animals)
Target: ??? (it's "hirvi" = moose, but you don't know that!)

Your guesses:
→ auto (car)      0.37  ❄️ Kylmä         (unrelated!)
→ koira (dog)     0.56  🌡️ Lämmin        (another animal, getting warmer!)
→ eläin (animal)  0.79  🔥 KUUMA!        (the category itself - excellent!)
→ hirvi (moose)   1.00  🔥 KUUMA! 🎉     (perfect match!)
```

**🎓 You learned**: Category words like "eläin" (animal) score higher than specific instances. This teaches semantic hierarchies naturally!

## 🚀 Quick Start

```bash
# 1. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Build language indexes
python tools/build_index.py --lang fi --in data/fi/categories_fi.json --out artifacts/fi/
python tools/build_index.py --lang en --in data/en/categories_en.json --out artifacts/en/

# 3. Try the demo!
python examples/game_demo.py
```

## ✨ What Makes This Fun & Educational

### 1. **The Engine Knows Best Possible Guesses**

```python
Target: hirvi (moose)

Engine knowledge - Best possible guesses:
1. vuohi (goat)     → 0.79  🔥  (another hoofed animal!)
2. orava (squirrel) → 0.76  🔥  (forest animal)
3. kettu (fox)      → 0.71  🌡️  (wild animal)
...

Player guesses:
→ koira (dog)  0.56  🌡️  "Could be better! Gap: 0.23"
→ eläin        0.79  🔥  "Excellent! Very close to optimal!"
```

### 2. **Smart Hint System**

```python
# Hint Level 1: Category
"💡 Vihje: Sana kuuluu kategoriaan 'Eläimet'"

# Hint Level 2: Top 3 words
"💡 Kolme parasta arvausta:
   1. vuohi (0.79)
   2. orava (0.76)
   3. kettu (0.71)"

# Hint Level 3: Best word
"💡 Paras mahdollinen arvaus: 'vuohi' (0.79)"
```

### 3. **Natural Language Learning**

Players discover:
- **Semantic hierarchies**: eläin (animal) → hirvi (moose)
- **Word relationships**: Suomi ↔ Helsinki (country-capital)
- **Common pairings**: leipä ↔ voi (bread-butter)
- **Semantic fields**: Related concepts cluster together

## 📁 Project Structure

```
semanticseek3/
├── src/
│   ├── core/          # Game engine with hint system
│   │   ├── engine.py  # Game logic + compute_top_words(), get_hint()
│   │   ├── model.py   # Embedding model (paraphrase-multilingual)
│   │   ├── scoring.py # Similarity scoring & feedback
│   │   ├── index.py   # HNSW search index
│   │   └── data.py    # Category loader
│   ├── cli/           # Command-line interface
│   └── web/           # FastAPI REST API
├── data/
│   ├── fi/            # Finnish categories (180 words)
│   └── en/            # English categories (179 words)
├── artifacts/         # Generated embeddings & indexes
├── tools/             # Build & analysis scripts
├── examples/          # Game demos
├── docs/              # Documentation
└── tests/             # Unit tests
```

## 🔧 Development

### Build Indexes
```bash
# Finnish
python tools/build_index.py --lang fi \
  --in data/fi/categories_fi.json \
  --out artifacts/fi/

# English
python tools/build_index.py --lang en \
  --in data/en/categories_en.json \
  --out artifacts/en/
```

### Run Examples
```bash
# Interactive game demo with hints
python examples/game_demo.py

# Test word similarities
python tools/similarity.py similarity kissa koira
```

### API Server
```bash
uvicorn src.web.api:app --reload --port 8080
# Visit http://localhost:8080/docs
```

### Run Tests
```bash
pytest -q
```

### Analyze & Calibrate
```bash
# Compare models and tune thresholds
python tools/calibrate_thresholds.py --compare --sample-size 40
```

## 🧠 How It Works

### Model
**`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
- Optimized for word-level similarity (not sentence retrieval)
- 384-dimensional embeddings (fast & efficient)
- Multilingual (50+ languages including Finnish)
- L2-normalized → cosine similarity = dot product

### Why This Model?

Previous model (E5-base) gave unrealistic scores:
```
cat ↔ car:  0.86  🌡️ Warm   ❌ Misleading!
cat ↔ dog:  0.90  🔥 Hot    (barely different from "car"!)
```

New model (paraphrase-multilingual):
```
cat ↔ car:  0.35  ❄️ Cold   ✅ Correctly low!
cat ↔ dog:  0.30  ❄️ Cold   (different animals)
animal:      0.79  🔥 Hot!   (category thinking!)
```

**Result**: 10x better discrimination, more fun gameplay! See [docs/MODEL_ANALYSIS.md](docs/MODEL_ANALYSIS.md)

### Scoring System

Thresholds calibrated for educational gameplay:
```python
FEEDBACK_BINS = [0.75, 0.55, 0.40]

0.75+  → 🔥 Hot!    Very strong relationship
0.55+  → 🌡️ Warm   Clear connection, same category
0.40+  → 😊 Mild   Some relationship, interesting!
<0.40  → ❄️ Cold   Unrelated, try different direction
```

## 🎯 Game Features

### Hint System
```python
engine.get_hint(state, "category")    # Show category
engine.get_hint(state, "top_3")       # Show 3 best words
engine.get_hint(state, "best_word")   # Show optimal guess

engine.compute_top_words(state, k=10) # Get best possible guesses
engine.get_best_possible_score(state) # Maximum achievable score
```

### Game Modes (Current & Planned)

- ✅ **Classic**: Unlimited guesses, hints available
- 🔜 **Daily Challenge**: Same word for everyone, limited guesses
- 🔜 **Learning Mode**: Auto-hints after N wrong guesses
- 🔜 **Expert Mode**: No hints, exploration only

## 📊 Performance

| Metric | Value | Why It Matters |
|--------|-------|----------------|
| Score Range | 0.06 - 0.94 | Wide spread = interesting gameplay |
| Std Deviation | 0.193 | High discrimination (10x vs E5!) |
| Embedding Dim | 384 | 2x faster than E5's 768 |
| Languages | 50+ | True multilingual support |

## 🌍 Adding New Languages

1. Create `data/{lang}/categories_{lang}.json`:
```json
{
  "language": "sv",
  "categories": {
    "Djur": ["hund", "katt", "älg", ...],
    "Mat": ["bröd", "ost", "smör", ...]
  }
}
```

2. Build index:
```bash
python tools/build_index.py --lang sv \\
  --in data/sv/categories_sv.json \\
  --out artifacts/sv/
```

3. Add feedback text in `src/core/scoring.py`

## 📚 Documentation

- **[README_MODEL_UPGRADE.md](docs/README_MODEL_UPGRADE.md)** - Model upgrade summary
- **[MODEL_ANALYSIS.md](docs/MODEL_ANALYSIS.md)** - Technical analysis
- **[CHANGES.md](docs/CHANGES.md)** - Complete changelog
- **[BEFORE_AFTER_COMPARISON.md](docs/BEFORE_AFTER_COMPARISON.md)** - Side-by-side comparison
- **[semantic_seek_v_3.md](docs/semantic_seek_v_3.md)** - Original design

## 🛠️ Configuration

**configs/settings.yaml**:
```yaml
model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
batch_size: 64
hnsw:
  space: cosine
  M: 32
  ef_construction: 200
  ef_search: 256
```

**src/core/scoring.py**:
```python
FEEDBACK_BINS = [0.75, 0.55, 0.40]  # Tune for difficulty
```

## 🧪 Tools

- **build_index.py** - Generate embeddings & HNSW index
- **validate_data.py** - Check category data quality
- **calibrate_thresholds.py** - Analyze vocabulary, tune thresholds
- **similarity.py** - Test word similarities interactively

## 📝 Requirements

- Python 3.11+
- 2GB+ RAM
- GPU optional (faster embedding generation)

## 📄 License

MIT License - see LICENSE file

## 🎨 Credits

**Created by:** ChatGPT, Caneli, and H11rustan

**Technologies:**
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [HNSW](https://github.com/nmslib/hnswlib) - Fast approximate search
- [FastAPI](https://fastapi.tiangolo.com/) - REST API
- [Typer](https://typer.tiangolo.com/) - CLI framework

---

**🎮 Ready to play!** Run `python examples/game_demo.py` to see it in action!
