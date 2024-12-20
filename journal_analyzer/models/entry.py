"""
Core data models for journal entries and emotional patterns.

File path: journal_analyzer/models/entry.py
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class JournalEntry:
    """Represents a single journal entry with metadata and analysis."""
    
    date: datetime
    content: str
    day_of_week: str
    month: int
    year: int
    word_count: int
    embedding: Optional[List[float]] = None
    
    # Analysis results
    sentiment_score: Optional[float] = None
    emotional_metrics: Optional[Dict[str, float]] = None
    detected_patterns: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary format."""
        # TODO: Implement conversion logic
        pass