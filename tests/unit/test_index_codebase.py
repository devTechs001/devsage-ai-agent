"""Tests for the code indexer."""
import json
import os
from pathlib import Path
import pytest

try:
    import esprima
    HAS_ESPRIMA = True
except ImportError:
    HAS_ESPRIMA = False
from scripts.index_codebase import (
    compute_file_hash,
    extract_python_symbols,
    extract_js_symbols,
    extract_symbols,
    should_exclude,
    process_file,
    index_repo,
    search_index,
)

FIXTURES = Path(__file__).parent / 'fixtures'
SAMPLE_FILES = FIXTURES / 'sample_files'


def setup_module():
    """Create sample files for testing."""
    FIXTURES.mkdir(exist_ok=True)
    SAMPLE_FILES.mkdir(exist_ok=True)
    
    # Python file with symbols
    with open(SAMPLE_FILES / 'test.py', 'w') as f:
        f.write('''
def hello():
    print("Hello")

class Person:
    def __init__(self, name):
        self.name = name
''')
    
    # JavaScript file with symbols
    with open(SAMPLE_FILES / 'test.js', 'w') as f:
        f.write('''
function greet(name) {
    console.log(`Hello ${name}`);
}

class User {
    constructor(name) {
        this.name = name;
    }
}
''')
    
    # Plain text file
    with open(SAMPLE_FILES / 'test.txt', 'w') as f:
        f.write('Hello, world!')


def test_compute_file_hash():
    path = SAMPLE_FILES / 'test.txt'
    hash1 = compute_file_hash(path)
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 is 64 hex chars
    
    # Test unchanged file has same hash
    hash2 = compute_file_hash(path)
    assert hash1 == hash2
    
    # Test changed file has different hash
    with open(path, 'a') as f:
        f.write('\nMore text')
    hash3 = compute_file_hash(path)
    assert hash1 != hash3


def test_extract_python_symbols():
    with open(SAMPLE_FILES / 'test.py') as f:
        content = f.read()
    
    symbols = extract_python_symbols(content)
    assert len(symbols) == 3  # hello, Person, and __init__
    assert symbols[0]['type'] == 'function'
    assert symbols[0]['name'] == 'hello'
    assert symbols[1]['type'] == 'class'
    assert symbols[1]['name'] == 'Person'
    assert symbols[2]['type'] == 'function'
    assert symbols[2]['name'] == '__init__'
    assert symbols[0]['type'] == 'function'
    assert symbols[0]['name'] == 'hello'
    assert symbols[1]['type'] == 'class'
    assert symbols[1]['name'] == 'Person'


@pytest.mark.skipif(not HAS_ESPRIMA, reason="esprima not installed")
def test_extract_js_symbols():
    with open(SAMPLE_FILES / 'test.js') as f:
        content = f.read()
    
    symbols = extract_js_symbols(content)
    assert len(symbols) == 2
    assert symbols[0]['type'] == 'function'
    assert symbols[0]['name'] == 'greet'
    assert symbols[1]['type'] == 'class'
    assert symbols[1]['name'] == 'User'


def test_should_exclude():
    excludes = {'**/__pycache__/**', '**/node_modules/**'}
    includes = {'src/**/*.py'}
    
    # Test excludes
    assert should_exclude(Path('foo/__pycache__/bar.py'), excludes, set())
    assert should_exclude(Path('node_modules/foo.js'), excludes, set())
    assert not should_exclude(Path('src/foo.py'), excludes, set())
    
    # Test includes
    assert not should_exclude(Path('src/foo.py'), excludes, includes)
    assert should_exclude(Path('lib/foo.py'), excludes, includes)


def test_process_file():
    path = SAMPLE_FILES / 'test.py'
    result = process_file(path)
    
    assert result is not None
    assert result['path'] == 'test.py'
    assert result['language'] == 'python'
    assert result['line_count'] > 0
    assert len(result['symbols']) == 3  # hello, Person, and __init__
    assert [s['name'] for s in result['symbols']] == ['hello', 'Person', '__init__']
    assert isinstance(result['mtime'], float)
    assert isinstance(result['indexed_at'], str)


def test_index_repo():
    result = index_repo(
        SAMPLE_FILES,
        FIXTURES / 'test_index.json',
        {'**/ignored/**'},
        set()
    )
    
    assert result['summary']['file_count'] == 3
    assert 'python' in result['summary']['languages']
    assert 'javascript' in result['summary']['languages']
    assert result['summary']['total_lines'] > 0
    assert isinstance(result['summary']['elapsed_seconds'], (int, float))


def test_search_index():
    index = {
        'items': [
            {'path': 'foo.py', 'language': 'python', 'symbols': [
                {'type': 'function', 'name': 'hello'}
            ]},
            {'path': 'bar.js', 'language': 'javascript', 'symbols': [
                {'type': 'class', 'name': 'User'}
            ]},
        ]
    }
    
    # Search by path
    results = search_index(index, 'foo', 'path')
    assert len(results) == 1
    assert results[0]['path'] == 'foo.py'
    
    # Search by language
    results = search_index(index, 'python', 'language')
    assert len(results) == 1
    assert results[0]['path'] == 'foo.py'
    
    # Search by symbol
    results = search_index(index, 'User', 'symbol')
    assert len(results) == 1
    assert results[0]['path'] == 'bar.js'
    
    # Search with language filter
    results = search_index(index, 'foo', 'path', language='python')
    assert len(results) == 1
    results = search_index(index, 'foo', 'path', language='javascript')
    assert len(results) == 0