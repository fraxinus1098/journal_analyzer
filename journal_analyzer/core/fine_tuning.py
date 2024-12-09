"""
Fine-tuning job management and model evaluation.

File path: journal_analyzer/core/fine_tuning.py
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
import time
from datetime import datetime
import json

from openai import OpenAI
import numpy as np
from tqdm import tqdm

from ..models.training import TrainingExample, ValidationMetrics
from ..models.patterns import EmotionalPattern
from .training_data import TrainingDataGenerator
from ..config import Config

logger = logging.getLogger(__name__)

class FineTuningManager:
    """Manages fine-tuning jobs and model evaluation."""
    
    def __init__(
        self,
        client: OpenAI,
        config: Config,
        base_model: str = "gpt-4o-mini-2024-07-18"
    ):
        """
        Initialize the fine-tuning manager.
        
        Args:
            client: OpenAI client instance
            config: Application configuration
            base_model: Base model to fine-tune from
        """
        self.client = client
        self.config = config
        self.base_model = base_model
        
    async def create_fine_tuning_job(
        self,
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new fine-tuning job.
        
        Args:
            training_file_id: ID of uploaded training file
            validation_file_id: Optional ID of validation file
            suffix: Optional suffix for model name
            hyperparameters: Optional hyperparameters
            
        Returns:
            Fine-tuning job ID
        """
        try:
            # Set default hyperparameters if none provided
            if hyperparameters is None:
                hyperparameters = {
                    "n_epochs": 3,
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto"
                }
            
            # Create fine-tuning job
            job = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                validation_file=validation_file_id,
                model=self.base_model,
                hyperparameters=hyperparameters,
                suffix=suffix
            )
            
            logger.info(f"Created fine-tuning job: {job.id}")
            return job.id
            
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {str(e)}")
            raise
    
    async def monitor_job(
        self,
        job_id: str,
        poll_interval: int = 60
    ) -> Optional[str]:
        """
        Monitor a fine-tuning job until completion.
        
        Args:
            job_id: Fine-tuning job ID
            poll_interval: Seconds between status checks
            
        Returns:
            Fine-tuned model ID if successful, None if failed
        """
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                
                # Log any new events
                events = self.client.fine_tuning.jobs.list_events(
                    job_id,
                    limit=50
                )
                for event in events.data:
                    logger.info(f"Job {job_id} event: {event.message}")
                
                # Check job status
                if job.status == "succeeded":
                    logger.info(f"Fine-tuning job {job_id} completed successfully")
                    return job.fine_tuned_model
                    
                elif job.status == "failed":
                    logger.error(f"Fine-tuning job {job_id} failed")
                    if job.error:
                        logger.error(f"Error: {job.error}")
                    return None
                    
                elif job.status == "cancelled":
                    logger.warning(f"Fine-tuning job {job_id} was cancelled")
                    return None
                
                # Job still running
                logger.info(f"Job {job_id} status: {job.status}")
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring job {job_id}: {str(e)}")
                await asyncio.sleep(poll_interval)
    
    async def evaluate_model(
        self,
        model_id: str,
        validation_examples: List[TrainingExample]
    ) -> ValidationMetrics:
        """
        Evaluate a fine-tuned model's performance.
        
        Args:
            model_id: Fine-tuned model ID
            validation_examples: List of validation examples
            
        Returns:
            Validation metrics
        """
        predictions = []
        targets = []
        
        for example in tqdm(validation_examples, desc="Evaluating model"):
            try:
                # Get model prediction
                response = await self._get_model_prediction(model_id, example)
                predictions.append(response)
                
                # Get target values
                target = {
                    "pattern_detected": example.pattern is not None,
                    "emotion_type": example.pattern.emotion_type if example.pattern else None,
                    "intensity": example.pattern.intensity if example.pattern else None,
                    "confidence_score": example.pattern.confidence_score if example.pattern else 0.0
                }
                targets.append(target)
                
            except Exception as e:
                logger.error(f"Error evaluating example: {str(e)}")
                continue
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        logger.info(f"Validation metrics: {metrics.to_dict()}")
        
        return metrics
    
    async def _get_model_prediction(
        self,
        model_id: str,
        example: TrainingExample
    ) -> Dict[str, Any]:
        """Get prediction from fine-tuned model."""
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=example.to_fine_tuning_format()["messages"],
                temperature=0,  # Use deterministic outputs for evaluation
                max_tokens=500
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error getting model prediction: {str(e)}")
            raise
    
    def _calculate_metrics(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]]
    ) -> ValidationMetrics:
        """Calculate validation metrics."""
        # Pattern detection accuracy
        pattern_detection = [
            pred["pattern_detected"] == target["pattern_detected"]
            for pred, target in zip(predictions, targets)
            if "pattern_detected" in pred
        ]
        detection_accuracy = np.mean(pattern_detection)
        
        # Pattern classification accuracy (for detected patterns)
        pattern_classification = [
            pred.get("emotion_type") == target["emotion_type"]
            for pred, target in zip(predictions, targets)
            if pred.get("pattern_detected") and target["pattern_detected"]
        ]
        classification_accuracy = np.mean(pattern_classification) if pattern_classification else 0.0
        
        # Intensity MAE
        intensity_errors = []
        for pred, target in zip(predictions, targets):
            if pred.get("pattern_detected") and target["pattern_detected"]:
                pred_intensity = pred.get("intensity", {})
                target_intensity = target["intensity"]
                if pred_intensity and target_intensity:
                    mae = np.mean([
                        abs(pred_intensity["baseline"] - target_intensity.baseline),
                        abs(pred_intensity["peak"] - target_intensity.peak),
                        abs(pred_intensity["progression_rate"] - target_intensity.progression_rate)
                    ])
                    intensity_errors.append(mae)
        
        intensity_mae = np.mean(intensity_errors) if intensity_errors else 0.0
        
        # Confidence RMSE
        confidence_errors = [
            (pred.get("confidence_score", 0) - target["confidence_score"]) ** 2
            for pred, target in zip(predictions, targets)
            if pred.get("pattern_detected") and target["pattern_detected"]
        ]
        confidence_rmse = np.sqrt(np.mean(confidence_errors)) if confidence_errors else 0.0
        
        return ValidationMetrics(
            pattern_detection_accuracy=detection_accuracy,
            pattern_classification_accuracy=classification_accuracy,
            intensity_mae=intensity_mae,
            confidence_rmse=confidence_rmse
        )