# Quick Command Reference

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Build Indexes

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

## Run Game

```bash
# Interactive demo
python examples/game_demo.py

# CLI game (if working, has typer issue currently)
python -m src.cli.app play --lang fi --mode classic

# Start API server
uvicorn src.web.api:app --reload --port 8080
# Visit: http://localhost:8080/docs
```

## Development

```bash
# Run tests
pytest -q

# Validate category data
python tools/validate_data.py data/fi/categories_fi.json

# Test word similarity
python tools/similarity.py similarity kissa koira

# Calibrate thresholds
python tools/calibrate_thresholds.py --compare --sample-size 40
```

## Common Tasks

```bash
# Rebuild both language indexes
python tools/build_index.py --lang fi --in data/fi/categories_fi.json --out artifacts/fi/
python tools/build_index.py --lang en --in data/en/categories_en.json --out artifacts/en/

# Quick test new model
python tools/similarity.py similarity cat car  # Should be ~0.35 (cold)
python tools/similarity.py similarity cat dog  # Should be ~0.30 (cold)

# Run game demo to see hint system
python examples/game_demo.py
```

## Git Workflow

```bash
# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "Your message"

# Push to remote
git push
```

## Directory Structure

```
semanticseek3/
├── src/           # Source code
│   ├── core/      # Game engine
│   ├── cli/       # CLI interface
│   └── web/       # API server
├── tools/         # Build & utility scripts
├── examples/      # Demo scripts
├── docs/          # Documentation
├── data/          # Category data
│   ├── fi/        # Finnish
│   └── en/        # English
├── artifacts/     # Generated indexes
│   ├── fi/        # Finnish embeddings
│   └── en/        # English embeddings
└── tests/         # Unit tests
```

## Notes

- **No Makefile**: We use direct Python commands for transparency and cross-platform compatibility
- **Model**: Using `paraphrase-multilingual-MiniLM-L12-v2` for better word-level similarity
- **Thresholds**: [0.75, 0.55, 0.40] calibrated for educational gameplay
- **Hint system**: Engine knows best possible guesses and can provide hints
