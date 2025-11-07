PY=python3
PKG=semantic_seek

.PHONY: setup build-fi build-en play-fi play-en api test clean

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

build-fi:
	. .venv/bin/activate && $(PY) tools/build_index.py --lang fi --in data/fi/categories_fi.json --out artifacts/fi/

build-en:
	. .venv/bin/activate && $(PY) tools/build_index.py --lang en --in data/en/categories_en.json --out artifacts/en/

play-fi:
	. .venv/bin/activate && $(PY) -m src.cli.app play --lang fi --mode classic

play-en:
	. .venv/bin/activate && $(PY) -m src.cli.app play --lang en --mode classic

api:
	. .venv/bin/activate && uvicorn src.web.api:app --reload --port 8080

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf .venv artifacts/*/embeddings.npy artifacts/*/index_hnsw.bin artifacts/*/vocab.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
