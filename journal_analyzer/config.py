# File path: journal_analyzer/config.py
"""
Configuration management for the journal analyzer.
"""

from typing import Dict, Any
from pathlib import Path

class Config:
    """Manages configuration settings."""
    
    DEFAULT_CONFIG = {
        'embedding_model': 'text-embedding-3-small',
        'embedding_dimensions': 256,
        'max_tokens': 8192,
        'batch_size': 50
    }
    
    def __init__(self, custom_config: Dict[str, Any] = None):
        self.config = {**self.DEFAULT_CONFIG, **(custom_config or {})}
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
        
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values."""
        self.config.update(updates)