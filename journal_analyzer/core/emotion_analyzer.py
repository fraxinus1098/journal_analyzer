"""
Module for analyzing emotions in journal entries using GPT-4o-mini.

File path: journal_analyzer/core/emotion_analyzer.py
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import asyncio
import logging
from datetime import datetime

from openai import OpenAI
import numpy as np

from ..models.patterns import EmotionalPattern, EmotionalIntensity

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """Analyzes emotions and topics in journal entries using GPT-4o-mini."""
    
    PRIMARY_EMOTIONS = [
        "joy", "sadness", "anger", "fear", 
        "surprise", "anticipation", "trust", "disgust"
    ]
    
    def __init__(self, client: OpenAI):
        """Initialize emotion analyzer."""
        self.client = client
        
    async def analyze_pattern(
        self,
        entries: List[Dict[str, Any]],
        pattern_id: str
    ) -> Tuple[str, str, float, str]:
        """
        Analyze emotions and topic in a group of related entries.
        
        Args:
            entries: List of journal entries in pattern
            pattern_id: Pattern identifier
            
        Returns:
            Tuple of (primary_emotion, topic_description, confidence, detailed_analysis)
        """
        # Construct analysis prompt
        prompt = self._construct_prompt(entries)
        
        try:
            # Get GPT-4o-mini analysis
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Parse response
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            # Validate primary emotion
            primary_emotion = analysis["primary_emotion"].lower()
            if primary_emotion not in self.PRIMARY_EMOTIONS:
                logger.warning(
                    f"Invalid primary emotion '{primary_emotion}' for pattern {pattern_id}. "
                    f"Defaulting to closest match."
                )
                primary_emotion = self._find_closest_emotion(primary_emotion)
            
            return (
                primary_emotion,
                analysis["topic"],
                float(analysis["confidence"]),
                analysis["detailed_analysis"]
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pattern {pattern_id}: {str(e)}")
            return ("mixed", "Unknown topic", 0.0, "Analysis failed")
    
    async def calculate_intensity(
        self,
        entries: List[Dict[str, Any]],
        embeddings: Dict[str, Any]
    ) -> EmotionalIntensity:
        """
        Calculate emotional intensity metrics for a pattern.
        
        Args:
            entries: List of journal entries
            embeddings: Dictionary of entry embeddings
            
        Returns:
            EmotionalIntensity object with normalized values
        """
        try:
            # Extract embeddings for entries
            pattern_embeddings = []
            for entry in entries:
                date = entry["date"]
                if isinstance(date, datetime):
                    date = date.strftime("%Y-%m-%d")
                embedding = embeddings[date]["embedding"]
                pattern_embeddings.append(embedding)
            
            pattern_embeddings = np.array(pattern_embeddings)
            
            # Calculate raw intensity metrics
            norms = np.linalg.norm(pattern_embeddings, axis=1)
            raw_baseline = float(np.mean(norms))
            raw_peak = float(np.max(norms))
            raw_variance = float(np.var(norms))
            
            # Calculate progression rate (normalized slope)
            x = np.arange(len(norms))
            slope = float(np.polyfit(x, norms, deg=1)[0])
            max_possible_slope = raw_peak / len(norms)
            progression_rate = slope / max_possible_slope if max_possible_slope != 0 else 0.0
            
            # Normalize values to 0-1 range
            norm_factor = max(raw_peak, 1e-6)  # Avoid division by zero
            baseline = raw_baseline / norm_factor
            peak = raw_peak / norm_factor
            variance = min(raw_variance / (norm_factor ** 2), 1.0)
            progression_rate = max(min(progression_rate, 1.0), -1.0)  # Clamp to [-1, 1]
            
            return EmotionalIntensity(
                baseline=baseline,
                peak=peak,
                variance=variance,
                progression_rate=progression_rate
            )
            
        except Exception as e:
            logger.error(f"Error calculating intensity metrics: {str(e)}")
            return EmotionalIntensity(
                baseline=0.0,
                peak=0.0,
                variance=0.0,
                progression_rate=0.0
            )
    
    def _construct_prompt(self, entries: List[Dict[str, Any]]) -> str:
        """Construct analysis prompt for entries."""
        entries_text = "\n\n".join([
            f"Entry {i+1} ({entry['date']}): {entry['content']}"
            for i, entry in enumerate(entries)
        ])
        
        return f"""Analyze these related journal entries and provide:
1. The primary emotion expressed (MUST be one of: {", ".join(self.PRIMARY_EMOTIONS)})
2. A clear, specific topic label (e.g., "Career Change Decision", "Family Vacation Planning")
3. A confidence score (0.0 to 1.0)
4. A detailed analysis of the emotional progression and key themes

Journal entries:
{entries_text}

Respond in this EXACT JSON format:
{{
    "primary_emotion": "<emotion>",
    "topic": "<specific topic label>",
    "confidence": <number>,
    "detailed_analysis": "<your analysis>"
}}"""

    def _get_system_prompt(self) -> str:
        """Get system prompt for emotion and topic analysis."""
        return """You are an expert in analyzing personal journals to identify emotional patterns and topics.
For each group of related entries, you will:
1. Identify the primary emotion
2. Determine a specific, meaningful topic label
3. Provide a confidence score
4. Generate a detailed analysis
Respond ONLY with the required JSON format, no other text."""

    def _find_closest_emotion(self, emotion: str) -> str:
        """Find closest matching primary emotion."""
        # Simple string matching - could be improved with embeddings
        emotion = emotion.lower()
        for primary in self.PRIMARY_EMOTIONS:
            if primary in emotion or emotion in primary:
                return primary
        return "mixed"