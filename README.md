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

A multilingual semantic word similarity game powered by transformer embeddings and approximate nearest neighbor search.

## Overview

**Semantic Seek** is a word-guessing game where players try to find a target word by making semantically similar guesses. Each guess is scored based on cosine similarity between word embeddings, providing feedback on how close the guess is to the target.

**Key Features:**
- Multilingual support (Finnish, English, and more)
- Powered by `intfloat/multilingual-e5-base` embeddings
- Efficient approximate nearest neighbor search using HNSW
- Both CLI and REST API interfaces
- Out-of-vocabulary word support (dynamic embedding)

## Quick Start

### 1. Setup Environment

```bash
make setup
```

This creates a virtual environment and installs all dependencies.

### 2. Build Index

Build the embeddings and search index for your desired language:

```bash
make build-fi   # Finnish
make build-en   # English
```

This will:
- Load category vocabulary
- Generate normalized embeddings using sentence-transformers
- Build an HNSW index for fast similarity search
- Save artifacts to `artifacts/{lang}/`

### 3. Play the Game

**CLI Mode:**
```bash
make play-fi    # Play in Finnish
make play-en    # Play in English
```

**API Mode:**
```bash
make api        # Start the API server on port 8080
```

Then visit http://localhost:8080/docs for the interactive API documentation.

## Usage

### CLI Commands

```bash
# Play with specific options
python -m src.cli.app play --lang fi --category Eläimet --seed 42

# View available categories
python -m src.cli.app info --lang fi
```

### API Endpoints

**Start a new game:**
```bash
curl -X POST http://localhost:8080/start \
  -H "Content-Type: application/json" \
  -d '{"language": "fi", "category": "Eläimet"}'
```

**Submit a guess:**
```bash
curl -X POST http://localhost:8080/guess \
  -H "Content-Type: application/json" \
  -d '{"session_id": "fi:Eläimet:0:0", "word": "koira"}'
```

**Get suggestions (reveals target):**
```bash
curl http://localhost:8080/suggest?session_id=fi:Eläimet:0:0&k=10
```

**List categories:**
```bash
curl http://localhost:8080/categories/fi
```

## Project Structure

```
semantic-seek/
├── src/
│   ├── core/           # Core game engine
│   │   ├── model.py    # Embedding model wrapper
│   │   ├── index.py    # HNSW index wrapper
│   │   ├── data.py     # Category data loader
│   │   ├── scoring.py  # Similarity scoring & feedback
│   │   └── engine.py   # Game logic
│   ├── cli/            # Command-line interface
│   │   └── app.py
│   └── web/            # REST API
│       └── api.py
├── tools/              # Build & validation scripts
│   ├── build_index.py
│   └── validate_data.py
├── data/               # Category definitions
│   ├── fi/
│   │   └── categories_fi.json
│   └── en/
│       └── categories_en.json
├── artifacts/          # Generated embeddings & indexes
├── configs/            # Configuration files
├── tests/              # Unit tests
└── Makefile           # Development commands
```

## Development

### Running Tests

```bash
make test
```

### Validating Category Data

```bash
python tools/validate_data.py data/fi/categories_fi.json
```

### Adding a New Language

1. Create `data/{lang}/categories_{lang}.json` with your vocabulary
2. Run `make build-{lang}` (you may need to add this to the Makefile)
3. Update the CLI/API to support the new language code

### Configuration

Edit `configs/settings.yaml` to customize:
- Embedding model
- Batch size
- HNSW parameters (M, ef_construction, ef_search)

## Technical Details

### Embeddings
- Model: `intfloat/multilingual-e5-base` (768 dimensions)
- All embeddings are L2-normalized
- Cosine similarity = dot product for normalized vectors

### Search Index
- Algorithm: HNSW (Hierarchical Navigable Small World)
- Distance metric: Cosine
- Similarity score: `1 - distance`

### Scoring Thresholds (Finnish)
- 0.90+: "MAHTAVA: lähes identtinen merkitys"
- 0.80-0.89: "Erittäin lähellä"
- 0.70-0.79: "Lähellä"
- <0.70: "Kaukana"

## Requirements

- Python 3.11+
- 4GB+ RAM (for loading models)
- GPU optional (but recommended for faster embedding)

## License

See project documentation for license details.

## Credits

**Created by:** ChatGPT, Caneli, and H11rustan

**Technologies:**
- [Sentence Transformers](https://www.sbert.net/)
- [HNSW](https://github.com/nmslib/hnswlib)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Typer](https://typer.tiangolo.com/)

---

**Enjoy playing Semantic Seek!**
