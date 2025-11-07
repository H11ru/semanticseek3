from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Optional

from src.core.engine import GameEngine, GameState
from src.core.model import EmbeddingModel
from src.core.index import VectorIndex
from src.core.data import CategoryStore
from src.core.scoring import score_to_percentage

app = FastAPI(
    title="Semantic Seek API",
    version="3.0.0",
    description="REST API for the Semantic Seek word similarity game"
)

# In-memory session store (naive implementation)
# In production, use Redis or database with TTL
SESSIONS: Dict[str, dict] = {}


class StartRequest(BaseModel):
    """Request to start a new game."""
    language: str = "fi"
    category: Optional[str] = None
    seed: Optional[int] = None


class GuessRequest(BaseModel):
    """Request to submit a guess."""
    session_id: str
    word: str


class GameResponse(BaseModel):
    """Response after game action."""
    session_id: str
    category: str
    attempts: int
    best_score: float
    best_guess: Optional[str] = None


class GuessResponse(BaseModel):
    """Response after a guess."""
    word: str
    score: float
    percentage: int
    feedback: str
    best_score: float
    best_guess: str
    is_win: bool


class SuggestionItem(BaseModel):
    """A single suggestion."""
    word: str
    similarity: float


class SuggestionsResponse(BaseModel):
    """Response with suggestions."""
    target: str
    suggestions: List[SuggestionItem]


@app.post("/start", response_model=GameResponse)
def start_game(req: StartRequest):
    """
    Start a new game session.

    Returns a session ID that must be used for subsequent requests.
    """
    lang = req.language

    # Load category data
    data_path = Path(f"data/{lang}/categories_{lang}.json")
    if not data_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Language '{lang}' not supported or data not available"
        )

    store = CategoryStore(str(data_path))
    categories = store.categories

    # Select category
    if req.category is None:
        import random
        category = random.choice(list(categories.keys()))
    elif req.category not in categories:
        raise HTTPException(
            status_code=400,
            detail=f"Category '{req.category}' not found. Available: {list(categories.keys())}"
        )
    else:
        category = req.category

    words = categories[category]

    # Load artifacts
    vocab_path = Path(f"artifacts/{lang}/vocab.json")
    embeddings_path = Path(f"artifacts/{lang}/embeddings.npy")
    index_path = Path(f"artifacts/{lang}/index_hnsw.bin")

    if not all(p.exists() for p in [vocab_path, embeddings_path, index_path]):
        raise HTTPException(
            status_code=500,
            detail=f"Artifacts not built for language '{lang}'. Run build script first."
        )

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    word_to_ix = {w: i for i, w in enumerate(vocab)}

    # Initialize components
    engine = GameEngine(vocab=vocab, vectors=embeddings, word_to_ix=word_to_ix)
    model = EmbeddingModel()

    state = engine.start(language=lang, category=category, words_in_category=words, seed=req.seed)

    # Build search index
    idx = VectorIndex(dim=embeddings.shape[1])
    idx.load(str(index_path))

    # Create session
    session_id = f"{lang}:{category}:{req.seed or 0}:{len(SESSIONS)}"
    SESSIONS[session_id] = {
        "engine": engine,
        "state": state,
        "model": model,
        "index": idx,
        "vocab": vocab,
    }

    return GameResponse(
        session_id=session_id,
        category=category,
        attempts=0,
        best_score=0.0,
        best_guess=None
    )


@app.post("/guess", response_model=GuessResponse)
def submit_guess(req: GuessRequest):
    """
    Submit a guess for an ongoing game.

    Returns the score and feedback for the guess.
    """
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[req.session_id]
    engine: GameEngine = session["engine"]
    state: GameState = session["state"]
    model: EmbeddingModel = session["model"]

    try:
        score, feedback_text = engine.guess(state, req.word, embed_fn=model.encode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    is_win = engine.check_win(state)

    return GuessResponse(
        word=req.word,
        score=score,
        percentage=score_to_percentage(score),
        feedback=feedback_text,
        best_score=state.best_score,
        best_guess=state.best_guess,
        is_win=is_win
    )


@app.get("/suggest", response_model=SuggestionsResponse)
def get_suggestions(session_id: str, k: int = 10):
    """
    Get top-k suggestions for the target word.

    This reveals the target word, so typically used after game ends.
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[session_id]
    engine: GameEngine = session["engine"]
    state: GameState = session["state"]
    idx: VectorIndex = session["index"]
    vocab: List[str] = session["vocab"]

    suggestions = engine.top_suggestions(
        state,
        index_search_fn=lambda qvec, k: idx.search(qvec, k),
        k=k
    )

    return SuggestionsResponse(
        target=state.target_word,
        suggestions=[
            SuggestionItem(word=word, similarity=sim)
            for word, sim in suggestions
        ]
    )


@app.get("/session/{session_id}", response_model=GameResponse)
def get_session(session_id: str):
    """Get current state of a game session."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[session_id]
    state: GameState = session["state"]

    return GameResponse(
        session_id=session_id,
        category=state.category,
        attempts=len(state.guesses),
        best_score=state.best_score,
        best_guess=state.best_guess if state.best_guess else None
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a game session."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    del SESSIONS[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/categories/{language}")
def list_categories(language: str):
    """List available categories for a language."""
    data_path = Path(f"data/{language}/categories_{language}.json")
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Language '{language}' not found"
        )

    store = CategoryStore(str(data_path))
    return {
        "language": store.language,
        "version": store.version,
        "categories": [
            {"name": name, "word_count": len(words)}
            for name, words in store.categories.items()
        ]
    }


@app.get("/healthz")
def healthcheck():
    """Health check endpoint."""
    return {"status": "ok", "version": "3.0.0"}


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Semantic Seek API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/healthz"
    }
