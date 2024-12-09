#!/usr/bin/env python3
"""
CLI script for running the journal fine-tuning pipeline.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from journal_analyzer.core.fine_tuner import JournalFineTuner
from journal_analyzer.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the journal fine-tuning pipeline"
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
        "--suffix",
        type=str,
        help="Custom suffix for the fine-tuned model name"
    )
    
    parser.add_argument(
        "--context-size",
        type=int,
        default=3,
        help="Number of context entries before/after each entry"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
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
        "context_window_size": args.context_size,
        "n_epochs": args.epochs
    })
    
    # Initialize fine-tuner
    tuner = JournalFineTuner(
        config=config,
        api_key=api_key,
        data_dir=Path(args.data_dir)
    )
    
    try:
        # Run pipeline
        model_id = await tuner.run_fine_tuning_pipeline(
            year=args.year,
            month=args.month,
            model_suffix=args.suffix
        )
        
        if model_id:
            logger.info(f"Fine-tuning completed successfully. Model ID: {model_id}")
        else:
            logger.error("Fine-tuning failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running fine-tuning pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())