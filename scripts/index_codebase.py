"""
Enhanced codebase indexer for the devsage project with:
- Incremental indexing (skips unchanged files)
- Extended language support with proper parsers
- Integration with vectorstore
- CLI flags for includes/excludes
"""
import os
import sys
import ast
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import argparse
import fnmatch
import logging

# Optional imports for enhanced parsing
try:
    import esprima  # for JS/TS
    HAS_ESPRIMA = True
except ImportError:
    HAS_ESPRIMA = False

try:
    from tree_sitter import Language, Parser  # for other languages
    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False

try:
    from sentence_transformers import SentenceTransformer
    from vectorstore.embeddings import get_embeddings_model
    from vectorstore.chunking import split_code_to_chunks
    HAS_VECTORSTORE = True
except ImportError:
    HAS_VECTORSTORE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "code_index.json"
DEFAULT_VECTOR_OUTPUT = REPO_ROOT / "data" / "vectorstores" / "codebase.faiss"

LANGUAGE_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.cpp': 'cpp',
    '.hpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.rb': 'ruby',
    '.php': 'php',
    '.cs': 'csharp',
    '.scala': 'scala',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.md': 'markdown',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
    '.sh': 'shell',
    '.ps1': 'powershell',
    '.psm1': 'powershell',
    '.html': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sql': 'sql',
}

DEFAULT_EXCLUDES = {
    '**/__pycache__/**',
    '**/.git/**',
    '**/node_modules/**',
    '**/venv/**',
    '**/.venv/**',
    '**/dist/**',
    '**/build/**',
    '**/.pytest_cache/**',
    '**/.coverage',
    '**/*.pyc',
}


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_python_symbols(content: str) -> List[dict]:
    """Extract Python symbols using ast."""
    try:
        tree = ast.parse(content)
        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno,
                })
            elif isinstance(node, ast.ClassDef):
                symbols.append({
                    'type': 'class',
                    'name': node.name,
                    'line': node.lineno,
                })
        return symbols
    except Exception as e:
        logger.warning(f"Failed to parse Python code: {e}")
        return []


def extract_js_symbols(content: str) -> List[dict]:
    """Extract JavaScript/TypeScript symbols using esprima."""
    if not HAS_ESPRIMA:
        return []
    
    try:
        tree = esprima.parseModule(content)
        symbols = []
        for node in tree.body:
            if node.type == 'FunctionDeclaration':
                symbols.append({
                    'type': 'function',
                    'name': node.id.name,
                    'line': node.location.start.line,
                })
            elif node.type == 'ClassDeclaration':
                symbols.append({
                    'type': 'class',
                    'name': node.id.name,
                    'line': node.location.start.line,
                })
        return symbols
    except Exception as e:
        logger.warning(f"Failed to parse JS/TS code: {e}")
        return []


def extract_symbols(content: str, language: str) -> List[dict]:
    """Extract symbols based on language."""
    if language == 'python':
        return extract_python_symbols(content)
    elif language in ('javascript', 'typescript'):
        return extract_js_symbols(content)
    return []


def should_exclude(path: Path, excludes: Set[str], includes: Set[str]) -> bool:
    """Check if path should be excluded based on glob patterns."""
    str_path = str(path).replace('\\', '/')
    
    # Check includes first - if we have includes and none match, exclude
    if includes and not any(fnmatch.fnmatch(str_path, pat) for pat in includes):
        return True
    
    # Then check excludes
    return any(fnmatch.fnmatch(str_path, pat) for pat in excludes)


def process_file(path: Path, prev_index: Optional[dict] = None) -> Optional[dict]:
    """Process a single file, returning None if unchanged."""
    try:
        # Get file stats
        stats = path.stat()
        curr_mtime = stats.st_mtime
        curr_size = stats.st_size
        
        # Check if file changed (using mtime first for speed)
        if prev_index is not None:
            prev_item = prev_index.get(str(path), {})
            if (prev_item.get('mtime') == curr_mtime and 
                prev_item.get('size') == curr_size):
                return None  # Unchanged
        
        # Compute full hash only if needed
        curr_hash = compute_file_hash(path)
        if prev_index is not None and prev_item.get('sha256') == curr_hash:
            return None  # Unchanged
        
        # Read and process file
        rel_path = path.relative_to(REPO_ROOT)
        ext = path.suffix.lower()
        language = LANGUAGE_MAP.get(ext, 'text')
        
        with path.open('r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        symbols = extract_symbols(content, language)
        first_lines = '\n'.join(content.splitlines()[:20])
        
        return {
            'path': str(rel_path).replace('\\', '/'),
            'size': curr_size,
            'mtime': curr_mtime,
            'sha256': curr_hash,
            'language': language,
            'line_count': content.count('\n') + 1,
            'first_lines': first_lines,
            'symbols': symbols,
            'indexed_at': datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.warning(f"Failed to process {path}: {e}")
        return None


def create_vectorstore(index: dict, output_path: Path):
    """Create vector embeddings for indexed files."""
    if not HAS_VECTORSTORE:
        logger.warning("Vectorstore integration not available")
        return
    
    try:
        model = get_embeddings_model()
        chunks = []
        metadatas = []
        
        for item in index['items']:
            try:
                with open(REPO_ROOT / item['path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                file_chunks = split_code_to_chunks(content)
                chunks.extend(file_chunks)
                metadatas.extend([{
                    'path': item['path'],
                    'language': item['language'],
                    'chunk': i,
                } for i in range(len(file_chunks))])
            except Exception as e:
                logger.warning(f"Failed to chunk {item['path']}: {e}")
        
        embeddings = model.encode(chunks)
        
        # Save to FAISS index
        import faiss
        import numpy as np
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))
        
        # Save metadata
        with (output_path.parent / 'metadata.json').open('w') as f:
            json.dump({
                'chunks': chunks,
                'metadata': metadatas,
            }, f)
        
        logger.info(f"Created vector index with {len(chunks)} chunks")
    
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")


def index_repo(
    root: Path,
    output: Path,
    excludes: Set[str],
    includes: Set[str],
    vectorize: bool = False,
    vector_output: Optional[Path] = None,
) -> dict:
    """Index the repository."""
    start_time = time.time()
    
    # Load previous index if it exists
    prev_index = None
    if output.exists():
        try:
            with output.open('r', encoding='utf-8') as f:
                prev_data = json.load(f)
                prev_index = {item['path']: item for item in prev_data.get('items', [])}
        except Exception as e:
            logger.warning(f"Failed to load previous index: {e}")
    
    # Process files
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirp = Path(dirpath)
        
        # Skip excluded directories early
        dirnames[:] = [d for d in dirnames 
                      if not should_exclude(dirp / d, excludes, includes)]
        
        for fname in filenames:
            path = dirp / fname
            if should_exclude(path, excludes, includes):
                continue
            
            item = process_file(path, prev_index)
            if item:
                items.append(item)
            elif prev_index and str(path) in prev_index:
                items.append(prev_index[str(path)])
    
    # Create output
    data = {
        'summary': {
            'repo': str(root),
            'file_count': len(items),
            'languages': {lang: sum(1 for i in items if i['language'] == lang)
                         for lang in {i['language'] for i in items}},
            'total_lines': sum(i['line_count'] for i in items),
            'indexed_at': datetime.now().isoformat(),
            'elapsed_seconds': round(time.time() - start_time, 2),
        },
        'items': items,
    }
    
    # Save index
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    # Create vector index if requested
    if vectorize and vector_output:
        create_vectorstore(data, vector_output)
    
    return data


def search_index(
    index: dict,
    query: str,
    field: str = 'path',
    language: Optional[str] = None,
) -> List[dict]:
    """Search the index."""
    q = query.lower()
    results = []
    
    for item in index.get('items', []):
        # Filter by language first if specified
        if language and item.get('language') != language:
            continue
        
        # Search in the specified field
        if field == 'path':
            val = item.get('path', '')
        elif field == 'language':
            val = item.get('language', '')
        elif field == 'symbol':
            val = ' '.join(s.get('name', '') for s in item.get('symbols', []))
        elif field == 'content':
            val = item.get('first_lines', '')
        else:
            val = ''
        
        if q in str(val).lower():
            results.append(item)
    
    return results


def cli():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Index and search repository code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate the index:
  python scripts/index_codebase.py --reindex
  
  # Search for files:
  python scripts/index_codebase.py --search api --field path
  
  # Search Python files only:
  python scripts/index_codebase.py --search class --field symbol --language python
  
  # Index with custom paths:
  python scripts/index_codebase.py --reindex --include "src/**" --exclude "**/*.test.*"
  
  # Create vector embeddings:
  python scripts/index_codebase.py --reindex --vectorize
""")
    
    parser.add_argument('--reindex', action='store_true',
                       help='Regenerate the code index')
    parser.add_argument('--search', type=str,
                       help='Search query')
    parser.add_argument('--field', type=str,
                       choices=['path', 'language', 'symbol', 'content'],
                       default='path',
                       help='Field to search in')
    parser.add_argument('--language', type=str,
                       help='Filter by programming language')
    parser.add_argument('--output', type=str,
                       default=str(DEFAULT_OUTPUT),
                       help='Output JSON file path')
    parser.add_argument('--exclude', type=str, action='append',
                       help='Glob patterns to exclude')
    parser.add_argument('--include', type=str, action='append',
                       help='Glob patterns to include (takes precedence)')
    parser.add_argument('--vectorize', action='store_true',
                       help='Create vector embeddings index')
    parser.add_argument('--vector-output', type=str,
                       default=str(DEFAULT_VECTOR_OUTPUT),
                       help='Vector index output path')
    
    args = parser.parse_args()
    
    # Set up exclude/include patterns
    excludes = set(DEFAULT_EXCLUDES)
    if args.exclude:
        excludes.update(args.exclude)
    
    includes = set()
    if args.include:
        includes.update(args.include)
    
    # Load or regenerate index
    output_path = Path(args.output)
    if args.reindex:
        logger.info(f"Indexing repository at {REPO_ROOT}")
        data = index_repo(
            REPO_ROOT,
            output_path,
            excludes,
            includes,
            args.vectorize,
            Path(args.vector_output) if args.vectorize else None,
        )
        logger.info(f"Wrote index for {len(data['items'])} files to {output_path}")
        if args.vectorize:
            logger.info(f"Wrote vector index to {args.vector_output}")
        return
    
    # Load existing index
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load index from {output_path}: {e}")
        logger.info("Run with --reindex to generate the index")
        return
    
    # Search or show summary
    if args.search:
        results = search_index(data, args.search, args.field, args.language)
        print(f"Found {len(results)} results for '{args.search}' "
              f"in field '{args.field}'"
              f"{f' (language: {args.language})' if args.language else ''}")
        
        for r in results[:50]:
            symbols = [f"{s['type']} {s['name']}" for s in r.get('symbols', [])]
            print(f"- {r['path']} ({r['language']}) "
                  f"lines={r['line_count']} "
                  f"symbols=[{', '.join(symbols)}]")
    else:
        print(json.dumps({'summary': data.get('summary')}, indent=2))


def pre_commit_hook():
    """Run as a pre-commit hook to update the index."""
    try:
        logger.setLevel(logging.WARNING)  # Reduce noise in pre-commit
        data = index_repo(
            REPO_ROOT,
            DEFAULT_OUTPUT,
            DEFAULT_EXCLUDES,
            set(),  # no includes
            False,  # no vectorization in pre-commit
            None
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to update index in pre-commit: {e}")
        return 1


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--pre-commit':
        sys.exit(pre_commit_hook())
    cli()
