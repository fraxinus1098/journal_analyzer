# File path: journal_analyzer/core/pattern_detector.py
"""
Embedding generation and management using OpenAI's API and HBDSCAN clustering for pattern detection.
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
        min_cluster_size: int = 5,
        min_samples: int = 3,
        temporal_weight: float = 0.3
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
        # Prepare data for clustering
        dates, embedding_matrix, temporal_features = self._prepare_clustering_data(
            entries, embeddings
        )
        
        # Perform clustering
        clusters = self._perform_clustering(embedding_matrix, temporal_features)
        
        # Extract patterns from clusters
        patterns = self._extract_patterns(dates, entries, embeddings, clusters)
        
        return patterns
    
    def _prepare_clustering_data(
        self,
        entries: List[Dict[str, Any]],
        embeddings: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Prepare entry data for clustering."""
        # Sort entries by date
        sorted_entries = sorted(entries, key=lambda x: x["date"])
        dates = [entry["date"] for entry in sorted_entries]
        
        # Create embedding matrix
        embedding_matrix = np.array([
            embeddings[date]["embedding"]
            for date in dates
        ])
        
        # Create temporal features
        temporal_features = self._create_temporal_features(dates)
        
        return dates, embedding_matrix, temporal_features
    
    def _create_temporal_features(self, dates: List[str]) -> np.ndarray:
        """Create temporal proximity features."""
        # Convert dates to timestamps
        timestamps = np.array([
            datetime.fromisoformat(date).timestamp()
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
        # Normalize embedding matrix (using L2 norm)
        normalized_embeddings = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1)[:, np.newaxis]
        
        # Scale temporal features to [0,1] range
        temporal_scaled = (temporal_features - temporal_features.min()) / (temporal_features.max() - temporal_features.min())
        
        # Combine features with weighted temporal component
        combined_features = np.hstack([
            normalized_embeddings,
            temporal_scaled * self.temporal_weight
        ])
        
        # Perform clustering with adjusted parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',  # Changed from 'cosine' to 'euclidean'
            cluster_selection_epsilon=0.3,
            alpha=1.0,
            cluster_selection_method='eom'
        )
        
        # Fit and predict clusters
        cluster_labels = clusterer.fit_predict(combined_features)
        
        # Log clustering statistics
        n_noise = sum(cluster_labels == -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"Clustering stats: {n_clusters} clusters found, {n_noise} noise points")
        
        return cluster_labels
    
    def _extract_patterns(
        self,
        dates: List[str],
        entries: List[Dict[str, Any]],
        embeddings: Dict[str, Any],
        clusters: np.ndarray
    ) -> List[EmotionalPattern]:
        """Extract emotional patterns from clustering results."""
        patterns = []
        
        # Process each cluster
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
                entries=[JournalEntry(**entry) for entry in cluster_entries],
                timespan=timespan,
                confidence_score=self._calculate_confidence(cluster_indices, clusters),
                emotion_type=self._detect_emotion_type(cluster_entries),
                intensity=intensity
            )
            
            patterns.append(pattern)
            
        return patterns
    
    def _calculate_timespan(self, dates: List[str]) -> PatternTimespan:
        """Calculate pattern timespan metrics."""
        sorted_dates = sorted(dates)
        start_date = datetime.fromisoformat(sorted_dates[0])
        end_date = datetime.fromisoformat(sorted_dates[-1])
        duration = (end_date - start_date).days
        
        return PatternTimespan(
            start_date=start_date,
            end_date=end_date,
            duration_days=duration,
            recurring=self._check_recurrence(dates)
        )
    
    def _calculate_intensity(
        self,
        dates: List[str],
        embeddings: Dict[str, Any]
    ) -> EmotionalIntensity:
        """Calculate emotional intensity metrics."""
        # Extract embeddings for cluster entries
        cluster_embeddings = np.array([
            embeddings[date]["embedding"] for date in dates
        ])
        
        # Calculate intensity metrics
        baseline = np.mean(np.linalg.norm(cluster_embeddings, axis=1))
        peak = np.max(np.linalg.norm(cluster_embeddings, axis=1))
        variance = np.var(np.linalg.norm(cluster_embeddings, axis=1))
        
        # Calculate progression rate (change in intensity over time)
        progression = np.polyfit(
            range(len(cluster_embeddings)),
            np.linalg.norm(cluster_embeddings, axis=1),
            deg=1
        )[0]
        
        return EmotionalIntensity(
            baseline=float(baseline),
            peak=float(peak),
            variance=float(variance),
            progression_rate=float(progression)
        )
    
    def _check_recurrence(self, dates: List[str]) -> bool:
        """Check if pattern shows recurring behavior."""
        if len(dates) < 3:
            return False
            
        # Convert to datetime objects
        dt_dates = [datetime.fromisoformat(date) for date in dates]
        
        # Calculate intervals between consecutive dates
        intervals = []
        for i in range(1, len(dt_dates)):
            interval = (dt_dates[i] - dt_dates[i-1]).days
            intervals.append(interval)
            
        # Check for regularity in intervals
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Consider pattern recurring if standard deviation is less than 25% of mean
        return std_interval < (mean_interval * 0.25)
    
    def _calculate_confidence(
        self,
        cluster_indices: np.ndarray,
        clusters: np.ndarray
    ) -> float:
        """Calculate confidence score for pattern."""
        # Calculate cluster density and size metrics
        cluster_size = len(cluster_indices)
        total_points = len(clusters)
        
        # Higher confidence for larger, denser clusters
        size_score = min(cluster_size / self.min_cluster_size, 1.0)
        density_score = cluster_size / total_points
        
        return (size_score * 0.7 + density_score * 0.3)
    
    def _detect_emotion_type(self, entries: List[Dict[str, Any]]) -> str:
        """Detect primary emotion type for pattern."""
        # TODO: Implement more sophisticated emotion detection
        # For now, using a simple placeholder
        return "mixed"
    
    def _generate_description(self, entries: List[Dict[str, Any]]) -> str:
        """Generate human-readable pattern description."""
        start_date = min(entries, key=lambda x: x["date"])["date"]
        end_date = max(entries, key=lambda x: x["date"])["date"]
        num_entries = len(entries)
        
        return (
            f"Pattern spanning {start_date} to {end_date}, "
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
        
        # Convert patterns to serializable format
        serializable_patterns = [
            {
                **asdict(pattern),
                'entries': [asdict(entry) for entry in pattern.entries],
                'timespan': asdict(pattern.timespan),
                'intensity': asdict(pattern.intensity)
            }
            for pattern in patterns
        ]
        
        with open(output_file, 'w') as f:
            json.dump(serializable_patterns, f, indent=2)