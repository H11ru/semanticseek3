# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Semantic Seek v3.0** is a multilingual semantic word similarity game using sentence transformers and approximate nearest neighbor search. Players try to guess a target word by making semantically similar guesses, receiving cosine similarity scores as feedback.

**Tech Stack**: Python 3.11+, sentence-transformers (multilingual-e5-base), hnswlib, FastAPI, Typer

## Development Commands

### Initial Setup
```bash
make setup          # Create venv and install dependencies
```

### Building Artifacts
```bash
make build-fi       # Build Finnish embeddings and HNSW index
make build-en       # Build English embeddings and HNSW index
```

### Running the Application
```bash
make play-fi        # Launch CLI game in Finnish (classic mode)
make api            # Start FastAPI server on port 8080
make test           # Run pytest tests
```

### Low-level Commands (when make isn't available)
```bash
# Build index for a language
python tools/build_index.py --lang fi --in data/fi/categories_fi.json --out artifacts/fi/

# Play CLI game
python -m src.cli.app play --lang fi --mode classic

# Start API server
uvicorn src.web.api:app --reload --port 8080

# Validate category data
python tools/validate_data.py data/fi/categories_fi.json
```

## Architecture

### Core Components (`src/core/`)

**model.py** - `EmbeddingModel` wrapper around sentence-transformers
- Uses `intfloat/multilingual-e5-base` by default
- Returns L2-normalized float32 embeddings
- Handles batch encoding with configurable batch size

**index.py** - `VectorIndex` wrapper around hnswlib
- Cosine similarity space (similarity = 1 - distance)
- Default params: M=32, ef_construction=200, ef_search=256
- Supports save/load for persistent indexes

**data.py** - `CategoryStore` for loading category JSON files
- Extracts vocabulary from categories with deduplication
- Maintains stable word order
- Tracks language metadata

**scoring.py** - Scoring logic and feedback bins
- `cosine_score()`: computes dot product (vectors are pre-normalized)
- `feedback()`: maps scores to Finnish feedback strings using thresholds [0.90, 0.80, 0.70]

**engine.py** - `GameEngine` and `GameState` core game logic
- `start()`: selects random target word from category
- `guess()`: scores guesses (handles OOV by embedding on-the-fly)
- `top_suggestions()`: retrieves nearest neighbors via index search
- GameState tracks: target word/vector, guesses, best score

### Interfaces

**src/cli/app.py** - Typer-based CLI
- Commands: `play` (with --lang, --mode, --category, --seed options)
- Interactive REPL loop for guessing
- Shows suggestions after game ends

**src/web/api.py** - FastAPI REST API
- POST `/start` - initialize game session
- POST `/guess` - submit guess and get score/feedback
- GET `/suggest` - get top-k suggestions
- GET `/healthz` - health check
- Sessions stored in-memory (SESSIONS dict)

### Tools (`tools/`)

**build_index.py** - Offline index builder
- Loads category JSON → extracts vocab → embeds → builds HNSW index
- Outputs: vocab.json, embeddings.npy, index_hnsw.bin
- Writes model provenance to models/model_id.txt

**validate_data.py** - Data quality checker
- Detects empty categories and duplicate words

## Data Flow

1. **Build Phase**: Category JSON → CategoryStore → vocab list → EmbeddingModel → embeddings.npy + HNSW index
2. **Game Start**: Load vocab + embeddings → select target word from category → create GameState
3. **Guess Loop**: User guess → embed (cached or OOV) → cosine_score → feedback → update best score
4. **Suggestions**: Query HNSW index with target vector → return top-k neighbors (excluding target)

## Key Design Decisions

- **Normalized Embeddings**: All vectors are L2-normalized, so cosine similarity equals dot product
- **Cosine Space in HNSW**: Index uses cosine distance; convert to similarity via `1 - distance`
- **OOV Handling**: CLI/API embed unknown words on-the-fly (not added to index)
- **Session Management**: API uses naive in-memory dict (no persistence yet)
- **Multilingual Support**: E5-base model handles multiple languages; data organized by language code

## Configuration

**configs/settings.yaml** - Model and index parameters
- model_name, batch_size, hnsw params (space, M, ef_construction, ef_search)

## Testing Strategy

Focus tests on:
- **Scoring logic**: cosine calculations, feedback bins, edge cases
- **OOV path**: verify on-the-fly embedding in engine.guess()
- **Engine state**: verify best_score/best_guess updates correctly
- **Index search**: validate k-NN returns expected neighbors

## Artifacts Structure

```
artifacts/
├── fi/
│   ├── vocab.json          # Ordered list of Finnish words
│   ├── embeddings.npy      # (N, 768) float32 array (E5-base dim)
│   └── index_hnsw.bin      # Serialized HNSW index
└── en/
    └── (same structure)
```

## Future Development Roadmap

Planned enhancements (see semantic_seek_v_3.md for details):
1. Daily challenge mode (date-seeded targets)
2. Hint system (H1-H3 hints in engine)
3. Web UI (React/Tailwind frontend)
4. Session persistence (SQLite)
5. Additional data tooling (merge categories, curator checks)
