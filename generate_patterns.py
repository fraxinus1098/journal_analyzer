#!/usr/bin/env python3
"""
Script to generate emotional pattern files from raw journal entries.
Usage: python generate_patterns.py --year 2020 [--month 1] [--data-dir ./data]

This script:
1. Loads raw journal entries
2. Generates embeddings
3. Detects emotional patterns
4. Saves patterns to YYYY_MM.patterns.json files
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

from openai import OpenAI
import numpy as np
from tqdm import tqdm

from journal_analyzer.core.pattern_detector import PatternDetector
from journal_analyzer.core.embeddings import EmbeddingGenerator
from journal_analyzer.config import Config
from journal_analyzer.models.entry import JournalEntry
from journal_analyzer.models.patterns import EmotionalPattern
from journal_analyzer.utils.file_handler import FileHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatternGenerator:
    """Generates pattern files from raw journal entries."""
    
    def __init__(
        self,
        api_key: str,
        data_dir: Path,
        config: Config
    ):
        self.data_dir = Path(data_dir)
        self.embedding_generator = EmbeddingGenerator(
            api_key=api_key,
            dimension=config.get("embedding_dimensions", 256)
        )
        self.pattern_detector = PatternDetector(
            min_cluster_size=config.get("min_cluster_size", 5),
            min_samples=config.get("min_samples", 3),
            temporal_weight=config.get("temporal_weight", 0.3)
        )
        self.file_handler = FileHandler(str(data_dir))
        
    async def generate_patterns(self, year: int, month: int) -> None:
        """Generate patterns for a specific month."""
        try:
            # Load raw entries
            entries = self._load_entries(year, month)
            logger.info(f"Loaded {len(entries)} entries for {year}-{month}")

            if not entries:
                logger.warning(f"No entries found for {year}-{month}")
                return
                
            # Generate embeddings
            embeddings = await self.embedding_generator.generate_embeddings(entries)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Save embeddings
            self.embedding_generator.save_embeddings(
                embeddings,
                self.data_dir / "embeddings",
                year,
                month
            )
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns(entries, embeddings)
            logger.info(f"Detected {len(patterns)} patterns with min_cluster_size={self.pattern_detector.min_cluster_size}") 
            
            # Save patterns
            await self._save_patterns(patterns, year, month)
            
            logger.info(f"Successfully generated patterns for {year}-{month}")
            
        except Exception as e:
            logger.error(f"Error generating patterns for {year}-{month}: {str(e)}")
            raise
    
    def _load_entries(self, year: int, month: int) -> List[Dict[str, Any]]:
        """Load raw journal entries."""
        input_file = self.data_dir / "raw" / f"{year}_{month:02d}.json"
        if not input_file.exists():
            return []
            
        try:
            with open(input_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading entries from {input_file}: {str(e)}")
            return []
    
    async def _save_patterns(
        self,
        patterns: List[EmotionalPattern],
        year: int,
        month: int
    ) -> None:
        """Save patterns to JSON file."""
        output_dir = self.data_dir / "patterns"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{year}_{month:02d}.patterns.json"
        
        try:
            # Convert patterns to serializable format
            pattern_data = [
                {
                    "pattern_id": p.pattern_id,
                    "description": p.description,
                    "entries": [
                        {
                            "date": e.date.isoformat(),
                            "content": e.content,
                            "day_of_week": e.day_of_week,
                            "word_count": e.word_count
                        }
                        for e in p.entries
                    ],
                    "timespan": {
                        "start_date": p.timespan.start_date.isoformat(),
                        "end_date": p.timespan.end_date.isoformat(),
                        "duration_days": p.timespan.duration_days,
                        "recurring": p.timespan.recurring
                    },
                    "confidence_score": p.confidence_score,
                    "emotion_type": p.emotion_type,
                    "intensity": {
                        "baseline": float(p.intensity.baseline),
                        "peak": float(p.intensity.peak),
                        "variance": float(p.intensity.variance),
                        "progression_rate": float(p.intensity.progression_rate)
                    }
                }
                for p in patterns
            ]
            
            with open(output_file, 'w') as f:
                json.dump(pattern_data, f, indent=2)
                
            logger.info(f"Saved {len(patterns)} patterns to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving patterns to {output_file}: {str(e)}")
            raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate emotional pattern files from raw journal entries"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process (e.g., 2020)"
    )
    
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        help="Specific month to process (1-12). If not provided, processes entire year"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for data storage"
    )
    
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Create configuration
    config = Config({
        "embedding_dimensions": 256,
        "min_cluster_size": 2,
        "min_samples": 2,
        "temporal_weight": 0.15
    })
    
    # Initialize generator
    generator = PatternGenerator(
        api_key=api_key,
        data_dir=Path(args.data_dir),
        config=config
    )
    
    try:
        if args.month:
            # Process specific month
            await generator.generate_patterns(args.year, args.month)
        else:
            # Process entire year
            for month in range(1, 13):
                await generator.generate_patterns(args.year, month)
                
    except Exception as e:
        logger.error(f"Error in pattern generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())