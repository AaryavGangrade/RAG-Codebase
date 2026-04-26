import hashlib
import json

def generate_hash(content: str) -> str:
    """Generate a stable hash for a chunk of code."""
    return hashlib.md5(content.encode()).hexdigest()

def save_json(data: list, filepath: str):
    """Save data to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> list:
    """Load data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
