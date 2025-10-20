# Codebase Indexer

This small script creates a JSON index of the repository under `data/code_index.json`.

Usage

1. Ensure you have Python 3.8+ available.
2. From the repository root run:

    python scripts/index_codebase.py

What it does

- Walks the repository (skips common build and env folders).
- Records path, size, sha256, language, line count, first 20 lines, and simple symbol extraction for supported languages.

Next steps

- Add CLI flags (exclude/include globs, update existing index, incremental updates).
- Integrate with vectorstore/semantic search pipeline.
