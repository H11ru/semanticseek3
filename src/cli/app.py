import typer
from typing import Optional
from pathlib import Path
import json
import numpy as np

from src.core.model import EmbeddingModel
from src.core.data import CategoryStore
from src.core.engine import GameEngine
from src.core.index import VectorIndex

app = typer.Typer(help="Semantic Seek CLI - A semantic word similarity game")

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
def play(
    lang: str = typer.Option("fi", "--lang", "-l", help="Language code: fi|en"),
    mode: str = typer.Option("classic", "--mode", "-m", help="Game mode: classic|daily"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Category name (random if not specified)"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Seed for reproducibility"),
):
    """Play the Semantic Seek word guessing game."""
    print(BANNER)

    # Load category data
    data_path = Path(f"data/{lang}/categories_{lang}.json")
    if not data_path.exists():
        typer.echo(f"Error: Category data not found at {data_path}", err=True)
        typer.echo(f"Run 'make build-{lang}' first to build the index.", err=True)
        raise typer.Exit(1)

    store = CategoryStore(str(data_path))
    categories = store.categories

    # Select category
    if category is None:
        import random
        category = random.choice(list(categories.keys()))
    elif category not in categories:
        typer.echo(f"Error: Category '{category}' not found.", err=True)
        typer.echo(f"Available categories: {', '.join(categories.keys())}", err=True)
        raise typer.Exit(1)

    words = categories[category]

    # Load artifacts
    vocab_path = Path(f"artifacts/{lang}/vocab.json")
    embeddings_path = Path(f"artifacts/{lang}/embeddings.npy")
    index_path = Path(f"artifacts/{lang}/index_hnsw.bin")

    if not vocab_path.exists() or not embeddings_path.exists() or not index_path.exists():
        typer.echo(f"Error: Artifacts not found for language '{lang}'", err=True)
        typer.echo(f"Run 'make build-{lang}' first to build the index.", err=True)
        raise typer.Exit(1)

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)

    # Build word to index mapping
    word_to_ix = {w: i for i, w in enumerate(vocab)}

    # Initialize game engine
    engine = GameEngine(vocab=vocab, vectors=embeddings, word_to_ix=word_to_ix)

    # Initialize model for OOV words
    typer.echo("Loading embedding model...")
    model = EmbeddingModel()

    # Start game
    state = engine.start(language=lang, category=category, words_in_category=words, seed=seed)

    typer.echo(f"\nTarget word selected from category '{category}'.")
    typer.echo("Try to guess semantically similar words!\n")

    # Game loop
    attempt = 0
    while True:
        attempt += 1
        guess = typer.prompt(f"Guess #{attempt} (or 'quit' to give up)", type=str).strip()

        if guess.lower() in ["quit", "exit", "q"]:
            typer.echo("\nGiving up...")
            break

        if not guess:
            continue

        try:
            score, msg = engine.guess(state, guess, embed_fn=model.encode)
            typer.echo(f"  Score: {score:.3f} ({int(score*100)}%) — {msg}")

            if engine.check_win(state):
                typer.echo(f"\n🎉 Congratulations! You won in {attempt} attempts!")
                break

        except Exception as e:
            typer.echo(f"  Error: {e}", err=True)

    # Load index for suggestions
    idx = VectorIndex(dim=embeddings.shape[1])
    idx.load(str(index_path))

    # Show suggestions
    typer.echo(f"\n{'='*60}")
    typer.echo(f"TARGET WORD: {state.target_word}")
    typer.echo(f"{'='*60}")

    suggestions = engine.top_suggestions(
        state,
        index_search_fn=lambda qvec, k: idx.search(qvec, k),
        k=10
    )

    typer.echo("\nTop 10 most similar words:")
    for i, (word, sim) in enumerate(suggestions, 1):
        typer.echo(f"  {i:2d}. {word:20s} ({sim:.3f})")

    if state.best_guess:
        typer.echo(f"\nYour best guess: '{state.best_guess}' ({state.best_score:.3f})")
    typer.echo(f"Total attempts: {len(state.guesses)}")


@app.command()
def info(lang: str = typer.Option("fi", "--lang", "-l", help="Language code")):
    """Show information about available categories and vocabulary."""
    data_path = Path(f"data/{lang}/categories_{lang}.json")

    if not data_path.exists():
        typer.echo(f"Error: Category data not found at {data_path}", err=True)
        raise typer.Exit(1)

    store = CategoryStore(str(data_path))

    typer.echo(f"Language: {store.language}")
    typer.echo(f"Version: {store.version}")
    typer.echo(f"\nCategories ({len(store.categories)}):")

    for cat_name, words in store.categories.items():
        typer.echo(f"  • {cat_name}: {len(words)} words")

    typer.echo(f"\nTotal vocabulary: {len(store.vocab)} unique words")


if __name__ == "__main__":
    app()
