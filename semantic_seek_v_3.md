# Semantic Seek v3.0 — Dev Starter Package

> **Purpose**: Give the team a ready-to-run skeleton with CLI + API, embeddings + ANN index, and clean boundaries. Start here, iterate fast.

---

## TL;DR (for the team)
- **Stack**: Python 3.11+, `sentence-transformers` (multilingual), `hnswlib`, `fastapi`, `typer`.
- **Commands**: `make setup` → `make build-fi` → `make play-fi` or `make api`.
- **Where to code**: `src/core/*` (engine), `src/cli/app.py` (CLI), `src/web/api.py` (HTTP), `tools/*` (offline ops).

---

## ASCII Banner (keep this!)
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

---

## Repository Layout
```
semantic-seek/
├─ pyproject.toml
├─ requirements.txt
├─ Makefile
├─ README.md
├─ .env.example
├─ configs/
│  └─ settings.yaml
├─ data/
│  ├─ fi/
│  │  ├─ categories_fi.json
│  │  └─ stopwords_fi.txt (optional)
│  └─ en/
│     └─ categories_en.json
├─ models/
│  └─ model_id.txt
├─ artifacts/                # built embeddings + indexes go here
│  ├─ fi/
│  │  ├─ vocab.json
│  │  ├─ embeddings.npy
│  │  └─ index_hnsw.bin
│  └─ en/
├─ src/
│  ├─ core/
│  │  ├─ model.py
│  │  ├─ index.py
│  │  ├─ data.py
│  │  ├─ scoring.py
│  │  └─ engine.py
│  ├─ cli/
│  │  └─ app.py
│  └─ web/
│     └─ api.py
├─ tools/
│  ├─ build_index.py
│  └─ validate_data.py
└─ tests/
   ├─ test_scoring.py
   └─ test_engine.py
```

---

## `requirements.txt`
```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
typer==0.12.5
pydantic==2.8.2
numpy==2.1.3
pandas==2.2.3
hnswlib==0.8.0
sentence-transformers==3.1.1
python-dotenv==1.0.1
pyyaml==6.0.2
```

## `pyproject.toml` (optional; if using pip-tools/poetry skip this)
```toml
[project]
name = "semantic-seek"
version = "3.0.0"
description = "Semantic Seek word similarity game"
requires-python = ">=3.11"
```

## `Makefile`
```makefile
PY=python3
PKG=semantic_seek

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

build-fi:
	. .venv/bin/activate && $(PY) tools/build_index.py --lang fi --in data/fi/categories_fi.json --out artifacts/fi/

build-en:
	. .venv/bin/activate && $(PY) tools/build_index.py --lang en --in data/en/categories_en.json --out artifacts/en/

play-fi:
	. .venv/bin/activate && $(PY) -m src.cli.app play --lang fi --mode classic

api:
	. .venv/bin/activate && uvicorn src.web.api:app --reload --port 8080

test:
	. .venv/bin/activate && pytest -q
```

## `configs/settings.yaml`
```yaml
model_name: intfloat/multilingual-e5-base
batch_size: 64
hnsw:
  space: cosine
  M: 32
  ef_construction: 200
  ef_search: 256
```

## Sample `data/fi/categories_fi.json`
```json
{
  "language": "fi",
  "version": "2025-11-07",
  "categories": {
    "Eläimet": ["koira", "kissa", "susi", "hirvi", "orava", "jänis"],
    "Ruoka": ["leipä", "juusto", "voi", "keitto", "omena", "marja"],
    "Luonto": ["metsä", "järvi", "joki", "vuori", "kallio", "niitty"]
  }
}
```

---

# Source Skeletons

### `src/core/model.py`
```python
from __future__ import annotations
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        # E5 expects instruction prefixes for best results; keep plain for v1
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)
```

### `src/core/index.py`
```python
from __future__ import annotations
import os
import numpy as np
import hnswlib
from typing import Tuple

class VectorIndex:
    def __init__(self, dim: int, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self._size = 0

    def build(self, vectors: np.ndarray, M: int = 32, ef_construction: int = 200, ef_search: int = 256):
        n = vectors.shape[0]
        self.index.init_index(max_elements=n, M=M, ef_construction=ef_construction)
        self.index.add_items(vectors)
        self.index.set_ef(ef_search)
        self._size = n

    def add(self, vectors: np.ndarray):
        # Optional append; ensure capacity or re-init outside if needed
        self.index.add_items(vectors)
        self._size += vectors.shape[0]

    def search(self, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        labels, dists = self.index.knn_query(qvec, k=k)
        sims = 1.0 - dists  # cosine space: similarity = 1 - distance
        return labels[0], sims[0]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.index.save_index(path)

    def load(self, path: str):
        self.index.load_index(path)
```

### `src/core/data.py`
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

class CategoryStore:
    def __init__(self, json_path: str):
        self.path = Path(json_path)
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.language: str = payload.get("language", "fi")
        self.categories: Dict[str, List[str]] = payload["categories"]

    @property
    def vocab(self) -> List[str]:
        words = []
        for lst in self.categories.values():
            words.extend(lst)
        # de-duplicate, stable
        seen = set()
        dedup = []
        for w in words:
            if w not in seen:
                seen.add(w)
                dedup.append(w)
        return dedup
```

### `src/core/scoring.py`
```python
from __future__ import annotations
import numpy as np

FEEDBACK_BINS = [0.90, 0.80, 0.70]
FEEDBACK_TEXT = [
    "MAHTAVA: lähes identtinen merkitys",
    "Erittäin lähellä",
    "Lähellä",
    "Kaukana"
]

def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    # a and b are expected L2-normalized. Dot equals cosine.
    return float(np.clip(np.dot(a, b), 0.0, 1.0))

def feedback(score: float) -> str:
    for i, thr in enumerate(FEEDBACK_BINS):
        if score >= thr:
            return FEEDBACK_TEXT[i]
    return FEEDBACK_TEXT[-1]
```

### `src/core/engine.py`
```python
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from .scoring import cosine_score, feedback

@dataclass
class GameState:
    language: str
    category: str
    target_word: str
    target_vec: np.ndarray
    guesses: List[str] = field(default_factory=list)
    best_score: float = 0.0
    best_guess: str = ""
    seed: int = 0

class GameEngine:
    def __init__(self, vocab: List[str], vectors: np.ndarray, word_to_ix: dict[str,int]):
        self.vocab = vocab
        self.vectors = vectors  # L2-normalized vectors
        self.word_to_ix = word_to_ix

    def start(self, language: str, category: str | None, words_in_category: List[str], seed: int | None = None) -> GameState:
        rng = random.Random(seed)
        target_word = rng.choice(words_in_category)
        target_vec = self.vectors[self.word_to_ix[target_word]]
        return GameState(language=language, category=category or "random", target_word=target_word, target_vec=target_vec, seed=seed or 0)

    def guess(self, state: GameState, word: str, embed_fn=None) -> Tuple[float, str]:
        state.guesses.append(word)
        if word in self.word_to_ix:
            vec = self.vectors[self.word_to_ix[word]]
        else:
            # OOV: embed on the fly
            assert embed_fn is not None, "embed_fn required for OOV words"
            vec = embed_fn([word])[0]
        s = cosine_score(state.target_vec, vec)
        if s > state.best_score:
            state.best_score = s
            state.best_guess = word
        return s, feedback(s)

    def top_suggestions(self, state: GameState, index_search_fn, k: int = 10) -> List[Tuple[str, float]]:
        idxs, sims = index_search_fn(state.target_vec.reshape(1, -1), k)
        pairs = []
        for i, sim in zip(idxs, sims):
            w = self.vocab[i]
            if w != state.target_word:
                pairs.append((w, float(sim)))
        return pairs
```

---

# Interfaces

### `src/cli/app.py`
```python
import typer
from typing import Optional
from pathlib import Path
import json
import numpy as np

from src.core.model import EmbeddingModel
from src.core.data import CategoryStore
from src.core.engine import GameEngine
from src.core.index import VectorIndex

app = typer.Typer(help="Semantic Seek CLI")

BANNER = r"""

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
"""

@app.command()
def play(lang: str = typer.Option("fi", help="Language code: fi|en"),
         mode: str = typer.Option("classic", help="Game mode"),
         category: Optional[str] = typer.Option(None, help="Category name"),
         seed: Optional[int] = typer.Option(None, help="Seed for reproducibility")):
    print(BANNER)

    data_path = Path(f"data/{lang}/categories_{lang}.json")
    store = CategoryStore(str(data_path))
    categories = store.categories

    if category is None:
        # random category
        category = list(categories.keys())[0]

    words = categories[category]

    # Load artifacts
    vocab = json.loads(Path(f"artifacts/{lang}/vocab.json").read_text(encoding="utf-8"))
    embeddings = np.load(f"artifacts/{lang}/embeddings.npy")

    # word_to_ix
    word_to_ix = {w: i for i, w in enumerate(vocab)}

    engine = GameEngine(vocab=vocab, vectors=embeddings, word_to_ix=word_to_ix)

    model = EmbeddingModel()  # for OOV
    state = engine.start(language=lang, category=category, words_in_category=words, seed=seed)

    print(f"\nKohdesana valittu kategoriasta '{category}'. Yritä arvata semanttisesti lähelle!\n")

    while True:
        guess = input("Arvaus (tai 'quit'): ").strip()
        if guess.lower() == "quit":
            break
        score, msg = engine.guess(state, guess, embed_fn=model.encode)
        print(f"Score: {score:.3f} — {msg}")

    # show suggestions
    from src.core.index import VectorIndex
    idx = VectorIndex(dim=embeddings.shape[1])
    idx.load(f"artifacts/{lang}/index_hnsw.bin")
    labels, sims = idx.search(state.target_vec.reshape(1, -1), k=10)

    print("\nParhaat vaihtoehdot olisivat olleet:")
    for l, s in zip(labels, sims):
        cand = vocab[int(l)]
        if cand == state.target_word:
            continue
        print(f" - {cand} ({s:.3f})")
    print(f"\nParas oma arvauksesi: '{state.best_guess}' ({state.best_score:.3f})")

if __name__ == "__main__":
    app()
```

### `src/web/api.py`
```python
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json
import numpy as np

from src.core.engine import GameEngine
from src.core.model import EmbeddingModel
from src.core.index import VectorIndex
from src.core.data import CategoryStore

app = FastAPI(title="Semantic Seek API", version="3.0.0")

# naive in-memory session store
SESSIONS = {}

class StartReq(BaseModel):
    language: str = "fi"
    category: str | None = None
    seed: int | None = None

class GuessReq(BaseModel):
    session_id: str
    word: str

@app.post("/start")
def start(req: StartReq):
    lang = req.language
    store = CategoryStore(f"data/{lang}/categories_{lang}.json")
    categories = store.categories
    category = req.category or list(categories.keys())[0]
    words = categories[category]

    vocab = json.loads(Path(f"artifacts/{lang}/vocab.json").read_text(encoding="utf-8"))
    embeddings = np.load(f"artifacts/{lang}/embeddings.npy")
    word_to_ix = {w: i for i, w in enumerate(vocab)}

    engine = GameEngine(vocab=vocab, vectors=embeddings, word_to_ix=word_to_ix)
    model = EmbeddingModel()

    state = engine.start(language=lang, category=category, words_in_category=words, seed=req.seed)

    # build search index
    idx = VectorIndex(dim=embeddings.shape[1])
    idx.load(f"artifacts/{lang}/index_hnsw.bin")

    session_id = f"{lang}:{category}:{req.seed or 0}"
    SESSIONS[session_id] = {
        "engine": engine,
        "state": state,
        "model": model,
        "index": idx,
        "vocab": vocab,
    }

    return {"session_id": session_id, "category": category}

@app.post("/guess")
def guess(req: GuessReq):
    s = SESSIONS[req.session_id]
    engine = s["engine"]
    state = s["state"]
    model = s["model"]
    score, msg = engine.guess(state, req.word, embed_fn=model.encode)
    return {"score": score, "feedback": msg, "best": {"word": state.best_guess, "score": state.best_score}}

@app.get("/suggest")
def suggest(session_id: str, k: int = 10):
    s = SESSIONS[session_id]
    idx = s["index"]
    state = s["state"]
    labels, sims = idx.search(state.target_vec.reshape(1, -1), k=k)
    vocab = s["vocab"]
    pairs = [
        {"word": vocab[int(l)], "similarity": float(sim)}
        for l, sim in zip(labels, sims)
        if vocab[int(l)] != state.target_word
    ]
    return {"target": state.target_word, "suggestions": pairs}

@app.get("/healthz")
def healthz():
    return {"ok": True}
```

---

# Tools

### `tools/build_index.py`
```python
import argparse
import json
from pathlib import Path
import numpy as np

from src.core.model import EmbeddingModel
from src.core.data import CategoryStore
from src.core.index import VectorIndex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True)
    p.add_argument("--in", dest="input_json", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    args = p.parse_args()

    store = CategoryStore(args.input_json)
    vocab = store.vocab

    print(f"Embedding {len(vocab)} words…")
    model = EmbeddingModel()
    vecs = model.encode(vocab)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out_dir / "embeddings.npy", vecs)

    # Build index
    idx = VectorIndex(dim=vecs.shape[1])
    idx.build(vecs)
    idx.save(str(out_dir / "index_hnsw.bin"))

    # Model id for provenance
    Path("models").mkdir(exist_ok=True)
    (Path("models") / "model_id.txt").write_text("intfloat/multilingual-e5-base\n", encoding="utf-8")

    print("Done.")

if __name__ == "__main__":
    main()
```

### `tools/validate_data.py`
```python
import json
import sys
from collections import Counter

path = sys.argv[1]
payload = json.loads(open(path, encoding="utf-8").read())
words = []
for k, lst in payload["categories"].items():
    if not lst:
        print(f"Empty category: {k}")
    words.extend(lst)

cnt = Counter(words)
dups = [w for w, c in cnt.items() if c > 1]
if dups:
    print("Duplicates:", dups[:20], ("…" if len(dups) > 20 else ""))
else:
    print("OK: no duplicates")
```

---

# README.md (snippet)
```md
# Semantic Seek v3.0

## Quickstart
```bash
make setup
make build-fi
make play-fi
# or
make api
```

## Notes
- Uses `intfloat/multilingual-e5-base` with normalized embeddings.
- ANN via `hnswlib` in cosine space. Similarity reported as `1 - distance` (== dot product for unit vectors).
- CLI allows OOV guesses (embedded on the fly), not added to the index.
```

---

# Next Tasks (Dev Board)
1. **Engine tests**: finalize unit tests for scoring + OOV path.
2. **Daily Challenge**: date-seeded target; add `cli app --mode daily`.
3. **Hint system**: implement H1–H3 in `engine.py` + expose in CLI/API.
4. **Web UI**: simple React/Tailwind client consuming `/start`, `/guess`, `/suggest`.
5. **Persistence**: optional SQLite for sessions, highscores.
6. **Data tooling**: add `tools/merge_categories.py` and curator checks (min category size, profanity filter if needed).

---

**Ship it.**

