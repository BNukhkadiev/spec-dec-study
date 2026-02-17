"""Benchmark loading and management."""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterator

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


class Benchmark:
    """Load and manage benchmarks from JSONL files."""
    
    def __init__(self, file_path: str | Path):
        """
        Initialize benchmark from JSONL file.
        
        Args:
            file_path: Path to JSONL benchmark file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        
        self.entries: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self):
        """Load entries from JSONL file."""
        if HAS_ORJSON:
            with open(self.file_path, 'rb') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = orjson.loads(line)
                        # Ensure required fields
                        if 'prompt' not in entry:
                            raise ValueError(f"Entry {line_num} missing 'prompt' field")
                        if 'id' not in entry:
                            entry['id'] = f"entry_{line_num}"
                        self.entries.append(entry)
                    except orjson.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Ensure required fields
                        if 'prompt' not in entry:
                            raise ValueError(f"Entry {line_num} missing 'prompt' field")
                        if 'id' not in entry:
                            entry['id'] = f"entry_{line_num}"
                        self.entries.append(entry)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    
    def __len__(self) -> int:
        """Return number of benchmark entries."""
        return len(self.entries)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over benchmark entries."""
        return iter(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get benchmark entry by index."""
        return self.entries[idx]
    
    def get_prompts(self) -> List[str]:
        """Get all prompts as a list."""
        return [entry['prompt'] for entry in self.entries]
    
    def get_ids(self) -> List[str]:
        """Get all entry IDs as a list."""
        return [entry['id'] for entry in self.entries]
