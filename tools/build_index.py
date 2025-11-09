#!/usr/bin/env python3
"""
Build embeddings and HNSW index for a given language.

Usage:
    python tools/build_index.py --lang fi --in data/fi/categories_fi.json --out artifacts/fi/
"""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml

from src.core.model import EmbeddingModel
from src.core.data import CategoryStore
from src.core.index import VectorIndex


def main():
    parser = argparse.ArgumentParser(description="Build embeddings and HNSW index")
    parser.add_argument("--lang", required=True, help="Language code (fi, en, etc.)")
    parser.add_argument("--in", dest="input_json", required=True, help="Input categories JSON file")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--config", default="configs/settings.yaml", help="Config file path")
    parser.add_argument("--device", default=None, help="Device (cpu, cuda, mps)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    model_name = config.get("model_name", "intfloat/multilingual-e5-base")
    batch_size = config.get("batch_size", 64)
    hnsw_config = config.get("hnsw", {})

    print(f"Building index for language: {args.lang}")
    print(f"Model: {model_name}")

    # Load category data
    print(f"Loading categories from {args.input_json}...")
    store = CategoryStore(args.input_json)
    vocab = store.vocab

    print(f"Found {len(vocab)} unique words across {len(store.categories)} categories")

    # Embed vocabulary
    print(f"Encoding {len(vocab)} words...")
    model = EmbeddingModel(model_name=model_name, device=args.device)
    embeddings = model.encode(vocab, batch_size=batch_size)

    print(f"Embeddings shape: {embeddings.shape}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save vocabulary
    vocab_path = out_dir / "vocab.json"
    print(f"Saving vocabulary to {vocab_path}...")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Save embeddings
    embeddings_path = out_dir / "embeddings.npy"
    print(f"Saving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)

    # Build HNSW index
    print("Building HNSW index...")
    index = VectorIndex(dim=embeddings.shape[1], space=hnsw_config.get("space", "cosine"))
    index.build(
        embeddings,
        M=hnsw_config.get("M", 32),
        ef_construction=hnsw_config.get("ef_construction", 200),
        ef_search=hnsw_config.get("ef_search", 256)
    )

    # Save index
    index_path = out_dir / "index_hnsw.bin"
    print(f"Saving index to {index_path}...")
    index.save(str(index_path))

    # Save model provenance
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_id_path = models_dir / "model_id.txt"
    print(f"Saving model provenance to {model_id_path}...")
    with open(model_id_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name}\n")

    print("\n✓ Build complete!")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Index: {index_path}")


if __name__ == "__main__":
    main()
