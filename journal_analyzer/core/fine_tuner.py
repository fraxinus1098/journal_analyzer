"""
High-level coordinator for the journal fine-tuning pipeline.

File path: journal_analyzer/core/fine_tuner.py
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import asyncio
from datetime import datetime

from openai import OpenAI
import numpy as np
from tqdm import tqdm

from .training_data import TrainingDataGenerator
from .fine_tuning import FineTuningManager
from .embeddings import EmbeddingGenerator
from ..models.training import TrainingExample, DatasetConfig, ValidationMetrics
from ..models.patterns import EmotionalPattern
from ..config import Config
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)

class JournalFineTuner:
    """Coordinates the end-to-end fine-tuning process for journal analysis."""
    
    def __init__(
        self,
        config: Config,
        api_key: str,
        data_dir: Path,
        base_model: str = "gpt-4o-mini-2024-07-18"
    ):
        """
        Initialize the journal fine-tuner.
        
        Args:
            config: Application configuration
            api_key: OpenAI API key
            data_dir: Base directory for data storage
            base_model: Base model to fine-tune from
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.client = OpenAI(api_key=api_key)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            api_key=api_key,
            dimension=config.get("embedding_dimensions", 256)
        )
        
        self.training_data_generator = TrainingDataGenerator(
            config=DatasetConfig(
                context_window_size=config.get("context_window_size", 3),
                min_pattern_confidence=config.get("min_pattern_confidence", 0.7),
                validation_split=config.get("validation_split", 0.2),
                max_tokens_per_example=config.get("max_tokens_per_example", 4096)
            ),
            embedding_generator=self.embedding_generator
        )
        
        self.fine_tuning_manager = FineTuningManager(
            client=self.client,
            config=config,
            base_model=base_model
        )
        
        self.file_handler = FileHandler(str(data_dir))
        
    async def run_fine_tuning_pipeline(
        self,
        year: int,
        month: Optional[int] = None,
        model_suffix: Optional[str] = None
    ) -> Optional[str]:
        """
        Run the complete fine-tuning pipeline for a specific time period.
        
        Args:
            year: Year to process
            month: Optional month to process (if None, processes entire year)
            model_suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            Fine-tuned model ID if successful, None otherwise
        """
        try:
            # Load data
            entries, patterns = await self._load_data(year, month)
            if not entries or not patterns:
                logger.error("No data available for fine-tuning")
                return None
            
            # Prepare training data
            train_examples, valid_examples = await self.training_data_generator.prepare_training_data(
                entries=entries,
                patterns=patterns
            )
            
            if len(train_examples) < 10:  # OpenAI minimum requirement
                logger.error("Insufficient training examples")
                return None
            
            # Export to JSONL
            training_file_path = self.data_dir / "training" / f"{year}_{month if month else 'full'}_train.jsonl"
            validation_file_path = self.data_dir / "training" / f"{year}_{month if month else 'full'}_valid.jsonl"
            
            self.training_data_generator.export_to_jsonl(train_examples, training_file_path)
            self.training_data_generator.export_to_jsonl(valid_examples, validation_file_path)
            
            # Upload files to OpenAI
            training_file = await self._upload_file(training_file_path)
            validation_file = await self._upload_file(validation_file_path)
            
            # Create and monitor fine-tuning job
            job_id = await self.fine_tuning_manager.create_fine_tuning_job(
                training_file_id=training_file.id,
                validation_file_id=validation_file.id,
                suffix=model_suffix or f"journal_{year}_{month if month else 'full'}",
                hyperparameters=self._get_hyperparameters()
            )
            
            model_id = await self.fine_tuning_manager.monitor_job(job_id)
            if not model_id:
                logger.error("Fine-tuning failed")
                return None
            
            # Evaluate model
            metrics = await self.fine_tuning_manager.evaluate_model(
                model_id=model_id,
                validation_examples=valid_examples
            )
            
            # Save evaluation results
            await self._save_evaluation_results(
                model_id=model_id,
                metrics=metrics,
                year=year,
                month=month
            )
            
            logger.info(f"Fine-tuning pipeline completed successfully. Model ID: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error in fine-tuning pipeline: {str(e)}")
            return None
    
    async def _load_data(
        self,
        year: int,
        month: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[EmotionalPattern]]:
        """Load journal entries and patterns."""
        entries = []
        patterns = []
        
        if month:
            # Load specific month
            entries = self._load_month_data(year, month)
            patterns = self._load_month_patterns(year, month)
        else:
            # Load entire year
            for m in range(1, 13):
                month_entries = self._load_month_data(year, m)
                month_patterns = self._load_month_patterns(year, m)
                entries.extend(month_entries)
                patterns.extend(month_patterns)
        
        return entries, patterns
    
    def _load_month_data(self, year: int, month: int) -> List[Dict[str, Any]]:
        """Load journal entries for a specific month."""
        try:
            input_file = self.data_dir / "raw" / f"{year}_{month:02d}.json"
            if not input_file.exists():
                return []
            
            with open(input_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data for {year}-{month}: {str(e)}")
            return []
    
    def _load_month_patterns(self, year: int, month: int) -> List[EmotionalPattern]:
        """Load patterns for a specific month."""
        try:
            pattern_file = self.data_dir / "patterns" / f"{year}_{month:02d}.patterns.json"
            if not pattern_file.exists():
                return []
            
            with open(pattern_file) as f:
                pattern_data = json.load(f)
                # Convert JSON to EmotionalPattern objects
                return [EmotionalPattern(**p) for p in pattern_data]
        except Exception as e:
            logger.error(f"Error loading patterns for {year}-{month}: {str(e)}")
            return []
    
    async def _upload_file(self, file_path: Path) -> Any:
        """Upload a file to OpenAI."""
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            logger.info(f"Uploaded file {file_path} with ID: {response.id}")
            return response
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {str(e)}")
            raise
    
    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters from config or use defaults."""
        return {
            "n_epochs": self.config.get("n_epochs", 3),
            "batch_size": self.config.get("batch_size", "auto"),
            "learning_rate_multiplier": self.config.get("learning_rate_multiplier", "auto")
        }
    
    async def _save_evaluation_results(
        self,
        model_id: str,
        metrics: ValidationMetrics,
        year: int,
        month: Optional[int] = None
    ) -> None:
        """Save evaluation results to file."""
        try:
            results = {
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data_period": {
                    "year": year,
                    "month": month
                },
                "metrics": metrics.to_dict()
            }
            
            output_path = self.data_dir / "evaluation" / f"{model_id}_eval.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved evaluation results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")