"""
Training data generation for fine-tuning pipeline.

File path: journal_analyzer/core/training_data.py
"""

import logging
from typing import List, Dict, Any, Tuple, Iterator
from pathlib import Path
import json
import random
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..models.entry import JournalEntry
from ..models.patterns import EmotionalPattern
from ..models.training import TrainingExample, DatasetConfig, ValidationMetrics
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """Generates training data from journal entries and detected patterns."""
    
    def __init__(
        self,
        config: DatasetConfig,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize the training data generator.
        
        Args:
            config: Configuration for dataset generation
            embedding_generator: Instance of EmbeddingGenerator for computing embeddings
        """
        self.config = config
        self.embedding_generator = embedding_generator
        self.entries_by_date = {}  # Cache for quick date-based lookup
        
    async def prepare_training_data(
        self,
        entries: List[Dict[str, Any]],
        patterns: List[EmotionalPattern]
    ) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """
        Prepare training and validation datasets.
        
        Args:
            entries: List of journal entries
            patterns: List of detected emotional patterns
            
        Returns:
            Tuple of (training_examples, validation_examples)
        """
        logger.info("Preparing training data...")
        
        # Initialize entry cache
        self._build_entry_cache(entries)
        
        # Create training examples
        examples = []
        for entry_dict in tqdm(entries, desc="Creating training examples"):
            entry = JournalEntry(
                date=datetime.fromisoformat(entry_dict["date"]),
                content=entry_dict["content"],
                day_of_week=entry_dict["day_of_week"],
                word_count=entry_dict["word_count"]
            )
            
            # Find patterns that include this entry
            entry_patterns = self._find_patterns_for_entry(entry, patterns)
            
            # Get context entries
            context_entries = self._get_context_entries(entry)
            
            # Create example(s)
            for pattern in entry_patterns:
                if pattern.confidence_score >= self.config.min_pattern_confidence:
                    example = TrainingExample(
                        entry=entry,
                        context_entries=context_entries,
                        pattern=pattern,
                        metadata={
                            "date": entry.date.isoformat(),
                            "pattern_id": pattern.pattern_id
                        }
                    )
                    examples.append(example)
            
            # Also include some examples without patterns
            if not entry_patterns:
                example = TrainingExample(
                    entry=entry,
                    context_entries=context_entries,
                    pattern=None,
                    metadata={"date": entry.date.isoformat()}
                )
                examples.append(example)
        
        # Split into training and validation
        random.seed(self.config.seed)
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - self.config.validation_split))
        
        train_examples = examples[:split_idx]
        valid_examples = examples[split_idx:]
        
        logger.info(f"Created {len(train_examples)} training and {len(valid_examples)} validation examples")
        return train_examples, valid_examples
    
    def export_to_jsonl(
        self,
        examples: List[TrainingExample],
        output_path: Path
    ) -> None:
        """
        Export training examples to JSONL format for OpenAI fine-tuning.
        
        Args:
            examples: List of training examples
            output_path: Path to save the JSONL file
        """
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example.to_fine_tuning_format()) + '\n')
                
        logger.info(f"Exported {len(examples)} examples to {output_path}")
    
    def _build_entry_cache(self, entries: List[Dict[str, Any]]) -> None:
        """Build cache of entries by date for quick lookup."""
        self.entries_by_date = {
            entry["date"]: entry for entry in entries
        }
    
    def _find_patterns_for_entry(
        self,
        entry: JournalEntry,
        patterns: List[EmotionalPattern]
    ) -> List[EmotionalPattern]:
        """Find all patterns that include the given entry."""
        matching_patterns = []
        entry_date = entry.date
        
        for pattern in patterns:
            pattern_entries = [e.date for e in pattern.entries]
            if entry_date in pattern_entries:
                matching_patterns.append(pattern)
                
        return matching_patterns
    
    def _get_context_entries(self, entry: JournalEntry) -> List[JournalEntry]:
        """Get context entries before and after the given entry."""
        context_entries = []
        entry_date = entry.date
        
        # Get entries before
        for i in range(1, self.config.context_window_size + 1):
            prev_date = entry_date - timedelta(days=i)
            prev_entry = self.entries_by_date.get(prev_date.isoformat())
            if prev_entry:
                context_entries.append(JournalEntry(
                    date=prev_date,
                    content=prev_entry["content"],
                    day_of_week=prev_entry["day_of_week"],
                    word_count=prev_entry["word_count"]
                ))
        
        # Get entries after
        for i in range(1, self.config.context_window_size + 1):
            next_date = entry_date + timedelta(days=i)
            next_entry = self.entries_by_date.get(next_date.isoformat())
            if next_entry:
                context_entries.append(JournalEntry(
                    date=next_date,
                    content=next_entry["content"],
                    day_of_week=next_entry["day_of_week"],
                    word_count=next_entry["word_count"]
                ))
        
        return sorted(context_entries, key=lambda x: x.date)