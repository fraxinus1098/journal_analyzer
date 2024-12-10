#!/usr/bin/env python3
"""
Script to generate emotional pattern files from raw journal entries using GPT-4o-mini.
Usage: python generate_patterns.py --year 2020 [--month 1] [--data-dir ./data]

This script:
1. Loads raw journal entries
2. Generates embeddings using text-embedding-3-small
3. Detects emotional patterns using HDBSCAN
4. Analyzes emotions using GPT-4o-mini
5. Saves patterns to YYYY_MM.patterns.json files
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

from openai import OpenAI
import numpy as np
from tqdm import tqdm

from journal_analyzer.core.pattern_detector import PatternDetector
from journal_analyzer.core.embeddings import EmbeddingGenerator
from journal_analyzer.core.emotion_analyzer import EmotionAnalyzer
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
        data_dir: Path
    ):
        """
        Initialize the pattern generator.
        
        Args:
            api_key: OpenAI API key
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.client = OpenAI(api_key=api_key)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            api_key=api_key,
            dimension=256,  # Using recommended dimension for efficiency
            batch_size=50
        )
        
        self.pattern_detector = PatternDetector(
            client=self.client,
            min_cluster_size=2,
            min_samples=2,
            temporal_weight=0.15
        )
        
        self.file_handler = FileHandler(str(data_dir))
        
    async def generate_patterns(self, year: int, month: int) -> None:
        """Generate patterns for a specific month."""
        try:
            start_time = datetime.now()
            logger.info(f"Starting pattern generation for {year}-{month}")
            
            # Load raw entries
            entries = self._load_entries(year, month)
            if not entries:
                logger.warning(f"No entries found for {year}-{month}")
                return
                
            logger.info(f"Processing {len(entries)} entries")
            
            # Create necessary directories
            self.data_dir.joinpath("embeddings").mkdir(parents=True, exist_ok=True)
            self.data_dir.joinpath("patterns").mkdir(parents=True, exist_ok=True)
            
            # Check if embeddings already exist
            embedding_file = self.data_dir / "embeddings" / f"{year}_{month:02d}.embeddings.json"
            if embedding_file.exists():
                logger.info("Loading existing embeddings")
                with open(embedding_file) as f:
                    embeddings = json.load(f)
            else:
                # Generate embeddings
                logger.info("Generating new embeddings")
                embeddings = await self.embedding_generator.generate_embeddings(entries)
                
                # Save embeddings
                self.embedding_generator.save_embeddings(
                    embeddings,
                    self.data_dir / "embeddings",
                    year,
                    month
                )
            
            # Detect patterns
            logger.info("Detecting and analyzing patterns")
            patterns = await self.pattern_detector.detect_patterns(entries, embeddings)
            
            # Save patterns
            await self._save_patterns(patterns, year, month)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(
                f"Successfully generated {len(patterns)} patterns for {year}-{month} "
                f"in {duration:.1f} seconds"
            )
            
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
            pattern_data = []
            for p in patterns:
                pattern_dict = {
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
                    "confidence_score": float(p.confidence_score),
                    "emotion_type": p.emotion_type,
                    "secondary_description": p.description,  # Added for emotional detail
                    "intensity": {
                        "baseline": float(p.intensity.baseline),
                        "peak": float(p.intensity.peak),
                        "variance": float(p.intensity.variance),
                        "progression_rate": float(p.intensity.progression_rate)
                    }
                }
                pattern_data.append(pattern_dict)
            
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
    
    try:
        # Initialize generator
        generator = PatternGenerator(
            api_key=api_key,
            data_dir=Path(args.data_dir)
        )
        
        if args.month:
            # Process specific month
            await generator.generate_patterns(args.year, args.month)
        else:
            # Process entire year
            for month in range(1, 13):
                logger.info(f"Processing month {month}")
                await generator.generate_patterns(args.year, month)
                await asyncio.sleep(1)  # Small delay between months
                
    except Exception as e:
        logger.error(f"Error in pattern generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())