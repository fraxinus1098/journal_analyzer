"""
Main analysis pipeline for processing journal entries, generating embeddings,
and detecting emotional patterns.

File path: journal_analyzer/core/analyzer.py
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from openai import OpenAI
import numpy as np
import pandas as pd

from ..models.entry import JournalEntry, EmotionalPattern
from ..security.input_validator import InputValidator
from ..security.prompt_guard import PromptGuard
from ..visualization.emotional_timeline import EmotionalTimeline
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)

from typing import Dict, Any, List
from pathlib import Path
import asyncio
import logging
from datetime import datetime
import json

from .embeddings import EmbeddingGenerator
from .pattern_detector import PatternDetector
from ..security.input_validator import InputValidator
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)

class JournalAnalyzer:
    """Main class for analyzing journal entries."""
    
    def __init__(
        self,
        api_key: str,
        data_dir: str,
        embedding_dimension: int = 256
    ):
        """
        Initialize the journal analyzer.
        
        Args:
            api_key: OpenAI API key
            data_dir: Base directory for data storage
            embedding_dimension: Dimension for embeddings
        """
        self.data_dir = Path(data_dir)
        self.embedding_generator = EmbeddingGenerator(
            api_key=api_key,
            dimension=embedding_dimension
        )
        self.pattern_detector = PatternDetector()
        self.file_handler = FileHandler(str(self.data_dir))
        self.input_validator = InputValidator()
        
    async def process_year(self, year: int) -> None:
        """
        Process all journal entries for a given year.
        
        Args:
            year: Year to process
        """
        logger.info(f"Processing journal entries for year {year}")
        
        # Create directory structure
        self.file_handler.ensure_directory_structure()
        
        # Process each month
        for month in range(1, 13):
            await self.process_month(year, month)
            
    async def process_month(self, year: int, month: int) -> None:
        """
        Process journal entries for a specific month.
        
        Args:
            year: Year to process
            month: Month to process
        """
        # Load entries
        entries = self._load_entries(year, month)
        if not entries:
            logger.info(f"No entries found for {year}-{month}")
            return
            
        # Generate embeddings
        embeddings = await self.embedding_generator.generate_embeddings(entries)
        
        # Save embeddings
        self.embedding_generator.save_embeddings(
            embeddings,
            self.data_dir / "embeddings",
            year,
            month
        )
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(entries, embeddings)
        
        # Save patterns
        self.pattern_detector.save_patterns(
            patterns,
            self.data_dir / "patterns",
            year,
            month
        )
        
        logger.info(
            f"Processed {len(entries)} entries for {year}-{month}, "
            f"found {len(patterns)} patterns"
        )
    
    def _load_entries(self, year: int, month: int) -> List[Dict[str, Any]]:
        """Load journal entries from file system."""
        input_file = self.data_dir / "raw" / f"{year}_{month:02d}.json"
        
        if not input_file.exists():
            return []
            
        with open(input_file) as f:
            entries = json.load(f)
            
        # Validate entries
        validated_entries = [
            entry for entry in entries
            if self.input_validator.validate_entry(entry)
        ]
        
        return validated_entries

# Example usage
async def main():
    analyzer = JournalAnalyzer(
        api_key="your-api-key",
        data_dir="path/to/data"
    )
    
    # Process year 2020
    await analyzer.process_year(2020)

if __name__ == "__main__":
    asyncio.run(main())