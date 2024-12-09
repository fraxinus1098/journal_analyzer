"""
Training data models for fine-tuning pipeline.

File path: journal_analyzer/models/training.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .patterns import EmotionalPattern
from .entry import JournalEntry

@dataclass
class TrainingExample:
    """Represents a single training example for fine-tuning."""
    
    entry: JournalEntry
    context_entries: List[JournalEntry]  # Previous/next entries for context
    pattern: Optional[EmotionalPattern]  # Associated emotional pattern if any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_fine_tuning_format(self) -> Dict[str, Any]:
        """Convert to format required for OpenAI fine-tuning."""
        messages = []
        
        # System message with instruction
        messages.append({
            "role": "system",
            "content": "You are an emotional pattern analyzer for journal entries. "
                      "Detect emotional patterns, their intensity, and their development over time."
        })
        
        # Add context entries as user messages
        for context_entry in self.context_entries:
            messages.append({
                "role": "user",
                "content": f"Previous entry from {context_entry.date}: {context_entry.content}"
            })
        
        # Add main entry
        messages.append({
            "role": "user",
            "content": f"Analyze this entry from {self.entry.date}: {self.entry.content}"
        })
        
        # Add pattern detection as assistant message
        if self.pattern:
            response_content = {
                "pattern_detected": True,
                "pattern_description": self.pattern.description,
                "emotional_type": self.pattern.emotion_type,
                "intensity": {
                    "baseline": self.pattern.intensity.baseline,
                    "peak": self.pattern.intensity.peak,
                    "progression_rate": self.pattern.intensity.progression_rate
                },
                "confidence_score": self.pattern.confidence_score
            }
            messages.append({
                "role": "assistant",
                "content": json.dumps(response_content, indent=2)
            })
        else:
            messages.append({
                "role": "assistant",
                "content": json.dumps({"pattern_detected": False}, indent=2)
            })
            
        return {"messages": messages}

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    
    context_window_size: int = 3  # Number of entries before/after for context
    min_pattern_confidence: float = 0.7  # Minimum confidence score for patterns
    validation_split: float = 0.2  # Fraction of data to use for validation
    max_tokens_per_example: int = 4096  # Maximum tokens per example
    seed: int = 42  # Random seed for reproducibility

@dataclass
class ValidationMetrics:
    """Metrics for evaluating fine-tuning performance."""
    
    pattern_detection_accuracy: float
    pattern_classification_accuracy: float
    intensity_mae: float  # Mean absolute error for intensity prediction
    confidence_rmse: float  # Root mean squared error for confidence scores
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "pattern_detection_accuracy": self.pattern_detection_accuracy,
            "pattern_classification_accuracy": self.pattern_classification_accuracy,
            "intensity_mae": self.intensity_mae,
            "confidence_rmse": self.confidence_rmse
        }