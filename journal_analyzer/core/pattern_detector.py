"""
Embedding generation and management using OpenAI's API and HBDSCAN clustering for pattern detection.

File path: journal_analyzer/core/pattern_detector.py
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import asdict
import logging
from pathlib import Path
import json

from ..models.patterns import Pattern, EmotionalPattern, PatternTimespan, EmotionalIntensity
from ..models.entry import JournalEntry

logger = logging.getLogger(__name__)

class PatternDetector:
    """Detects emotional patterns in journal entries using HDBSCAN clustering."""
    
    def __init__(
        self,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        temporal_weight: float = 0.1
    ):
        """
        Initialize pattern detector.
        
        Args:
            min_cluster_size: Minimum number of entries to form a pattern
            min_samples: HDBSCAN min_samples parameter
            temporal_weight: Weight given to temporal proximity (0-1)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.temporal_weight = temporal_weight
        self.scaler = StandardScaler()
        
    def detect_patterns(
        self,
        entries: List[Dict[str, Any]],
        embeddings: Dict[str, Any]
    ) -> List[EmotionalPattern]:
        """
        Detect emotional patterns in journal entries.
        
        Args:
            entries: List of journal entries
            embeddings: Dictionary of embeddings and metadata
            
        Returns:
            List of detected emotional patterns
        """
        # Convert string dates to datetime objects
        entries = self._ensure_datetime_dates(entries)
        
        # Prepare data for clustering
        dates, embedding_matrix, temporal_features = self._prepare_clustering_data(
            entries, embeddings
        )
        
        # Perform clustering
        clusters = self._perform_clustering(embedding_matrix, temporal_features)
        
        # Extract patterns from clusters
        patterns = self._extract_patterns(dates, entries, embeddings, clusters)
        
        return patterns
    
    def _ensure_datetime_dates(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure all dates are datetime objects."""
        for entry in entries:
            if isinstance(entry['date'], str):
                entry['date'] = datetime.fromisoformat(entry['date'].replace('T', ' '))
        return entries
    
    def _prepare_clustering_data(
        self,
        entries: List[Dict[str, Any]],
        embeddings: Dict[str, Any]
    ) -> Tuple[List[datetime], np.ndarray, np.ndarray]:
        """Prepare entry data for clustering."""
        # Sort entries by date
        sorted_entries = sorted(entries, key=lambda x: x["date"])
        dates = [entry["date"] for entry in sorted_entries]
        
        # Create embedding matrix
        embedding_matrix = np.array([
            embeddings[date.strftime('%Y-%m-%d')]["embedding"]
            for date in dates
                ])
        
        # Create temporal features
        temporal_features = self._create_temporal_features(dates)
        
        return dates, embedding_matrix, temporal_features
    
    def _create_temporal_features(self, dates: List[datetime]) -> np.ndarray:
        """Create temporal proximity features."""
        # Convert dates to timestamps
        timestamps = np.array([
            date.timestamp()
            for date in dates
        ])
        
        # Scale to 0-1 range
        min_time = timestamps.min()
        max_time = timestamps.max()
        time_range = max_time - min_time
        
        if time_range == 0:
            return np.zeros((len(dates), 1))
            
        scaled_time = (timestamps - min_time) / time_range
        return scaled_time.reshape(-1, 1)
    
    def _perform_clustering(
        self,
        embedding_matrix: np.ndarray,
        temporal_features: np.ndarray
    ) -> np.ndarray:
        """Perform HDBSCAN clustering on combined features."""
        # Normalize embedding matrix
        normalized_embeddings = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1)[:, np.newaxis]
        
        # Scale temporal features
        temporal_scaled = (temporal_features - temporal_features.min()) / (temporal_features.max() - temporal_features.min())
        
        # Combine features
        combined_features = np.hstack([
            normalized_embeddings,
            temporal_scaled * self.temporal_weight
        ])
        
        # Perform clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.6,
            alpha=0.3,
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(combined_features)
        
        # Log clustering statistics
        n_noise = sum(cluster_labels == -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"Clustering stats: {n_clusters} clusters found, {n_noise} noise points")
        
        return cluster_labels
    
    def _extract_patterns(
        self,
        dates: List[datetime],
        entries: List[Dict[str, Any]],
        embeddings: Dict[str, Any],
        clusters: np.ndarray
    ) -> List[EmotionalPattern]:
        """Extract emotional patterns from clustering results."""
        patterns = []
        
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get entries in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_dates = [dates[i] for i in cluster_indices]
            cluster_entries = [entries[i] for i in cluster_indices]
            
            # Calculate pattern timespan
            timespan = self._calculate_timespan(cluster_dates)
            
            # Calculate emotional intensity metrics
            intensity = self._calculate_intensity(cluster_dates, embeddings)
            
            # Create pattern object
            pattern = EmotionalPattern(
                pattern_id=f"pattern_{cluster_id}",
                description=self._generate_description(cluster_entries),
                entries=[JournalEntry(
                    date=entry["date"],
                    content=entry["content"],
                    day_of_week=entry["day_of_week"],
                    word_count=entry["word_count"],
                    month=entry["date"].month,
                    year=entry["date"].year
                ) for entry in cluster_entries],
                timespan=timespan,
                confidence_score=self._calculate_confidence(cluster_indices, clusters),
                emotion_type=self._detect_emotion_type(cluster_entries),
                intensity=intensity
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_timespan(self, dates: List[datetime]) -> PatternTimespan:
        """Calculate pattern timespan metrics."""
        sorted_dates = sorted(dates)
        start_date = sorted_dates[0]
        end_date = sorted_dates[-1]
        duration = (end_date - start_date).days
        
        return PatternTimespan(
            start_date=start_date,
            end_date=end_date,
            duration_days=duration,
            recurring=self._check_recurrence(dates)
        )
    
    def _calculate_intensity(
        self,
        dates: List[datetime],
        embeddings: Dict[str, Any]
    ) -> EmotionalIntensity:
        """Calculate emotional intensity metrics."""
        # Extract embeddings for cluster entries
        cluster_embeddings = np.array([
            embeddings[date.strftime('%Y-%m-%d')]["embedding"]
            for date in dates
        ])
        
        # Calculate intensity metrics
        baseline = float(np.mean(np.linalg.norm(cluster_embeddings, axis=1)))
        peak = float(np.max(np.linalg.norm(cluster_embeddings, axis=1)))
        variance = float(np.var(np.linalg.norm(cluster_embeddings, axis=1)))
        progression = float(np.polyfit(
            range(len(cluster_embeddings)),
            np.linalg.norm(cluster_embeddings, axis=1),
            deg=1
        )[0])
        
        return EmotionalIntensity(
            baseline=baseline,
            peak=peak,
            variance=variance,
            progression_rate=progression
        )
    
    def _check_recurrence(self, dates: List[datetime]) -> bool:
        """Check if pattern shows recurring behavior."""
        if len(dates) < 3:
            return False
            
        # Calculate intervals between consecutive dates
        intervals = []
        sorted_dates = sorted(dates)
        for i in range(1, len(sorted_dates)):
            interval = (sorted_dates[i] - sorted_dates[i-1]).days
            intervals.append(interval)
            
        # Check for regularity in intervals
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        return bool(std_interval < (mean_interval * 0.25))
    
    def _calculate_confidence(
        self,
        cluster_indices: np.ndarray,
        clusters: np.ndarray
    ) -> float:
        """Calculate confidence score for pattern."""
        cluster_size = len(cluster_indices)
        total_points = len(clusters)
        
        size_score = min(cluster_size / self.min_cluster_size, 1.0)
        density_score = cluster_size / total_points
        
        return float(size_score * 0.5 + density_score * 0.5)
    
    def _detect_emotion_type(self, entries: List[Dict[str, Any]]) -> str:
        """Detect primary emotion type for pattern."""
        return "mixed"
    
    def _generate_description(self, entries: List[Dict[str, Any]]) -> str:
        """Generate human-readable pattern description."""
        start_date = min(entries, key=lambda x: x["date"])["date"]
        end_date = max(entries, key=lambda x: x["date"])["date"]
        num_entries = len(entries)
        
        return (
            f"Pattern spanning {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, "
            f"comprising {num_entries} journal entries"
        )
    
    def save_patterns(
        self,
        patterns: List[EmotionalPattern],
        output_dir: Path,
        year: int,
        month: int
    ) -> None:
        """Save detected patterns to file system."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{year}_{month:02d}.patterns.json"
        
        try:
            # Convert patterns to serializable format
            serializable_patterns = []
            for pattern in patterns:
                pattern_dict = {
                    'pattern_id': pattern.pattern_id,
                    'description': pattern.description,
                    'entries': [{
                        'date': entry.date.strftime('%Y-%m-%d'),
                        'content': entry.content,
                        'day_of_week': entry.day_of_week,
                        'word_count': int(entry.word_count),
                        'month': int(entry.month),
                        'year': int(entry.year)
                    } for entry in pattern.entries],
                    'timespan': {
                        'start_date': pattern.timespan.start_date.strftime('%Y-%m-%d'),
                        'end_date': pattern.timespan.end_date.strftime('%Y-%m-%d'),
                        'duration_days': int(pattern.timespan.duration_days),
                        'recurring': bool(pattern.timespan.recurring),
                        'frequency': pattern.timespan.frequency
                    },
                    'confidence_score': float(pattern.confidence_score),
                    'emotion_type': str(pattern.emotion_type),
                    'intensity': {
                        'baseline': float(pattern.intensity.baseline),
                        'peak': float(pattern.intensity.peak),
                        'variance': float(pattern.intensity.variance),
                        'progression_rate': float(pattern.intensity.progression_rate)
                    }
                }
                serializable_patterns.append(pattern_dict)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_patterns, f, indent=2)
                
            logger.info(f"Saved {len(patterns)} patterns to {output_file}")
                
        except Exception as e:
            logger.error(f"Error saving patterns to {output_file}: {str(e)}")
            raise