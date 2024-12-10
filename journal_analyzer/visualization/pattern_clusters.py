"""
Visualization of pattern clusters and relationships.

File path: journal_analyzer/visualization/pattern_clusters.py
"""

from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from ..models.patterns import EmotionalPattern, EmotionalIntensity

class PatternVisualizer:
    """Creates visualizations for emotional patterns and clusters."""
    
    def __init__(
        self,
        patterns: List[EmotionalPattern],
        color_scale: Optional[Dict[str, str]] = None
    ):
        """
        Initialize pattern visualizer.
        
        Args:
            patterns: List of emotional patterns to visualize
            color_scale: Optional custom color scale mapping
        """
        self.patterns = patterns
        self.fig = None
        
        # Default color scale
        self.color_scale = color_scale or {
            'positive': '#1f77b4',  # Blue
            'negative': '#ff7f0e',  # Orange
            'neutral': '#7f7f7f',   # Gray
            'mixed': '#2ca02c',     # Green
            'highlight': '#d62728'  # Red
        }
        
        # Create pattern DataFrame
        self.df = self._create_pattern_dataframe()
        
    def _create_pattern_dataframe(self) -> pd.DataFrame:
        """Convert patterns to DataFrame for visualization."""
        data = []
        for pattern in self.patterns:
            # Calculate pattern metrics
            duration = pattern.timespan.duration_days
            avg_intensity = (pattern.intensity.baseline + pattern.intensity.peak) / 2
            entry_count = len(pattern.entries)
            
            data.append({
                'pattern_id': pattern.pattern_id,
                'start_date': pattern.timespan.start_date,
                'end_date': pattern.timespan.end_date,
                'duration': duration,
                'confidence': pattern.confidence_score,
                'emotion_type': pattern.emotion_type,
                'intensity_baseline': pattern.intensity.baseline,
                'intensity_peak': pattern.intensity.peak,
                'intensity_variance': pattern.intensity.variance,
                'progression_rate': pattern.intensity.progression_rate,
                'entry_count': entry_count,
                'avg_intensity': avg_intensity,
                'recurring': pattern.timespan.recurring
            })
            
        return pd.DataFrame(data)
        
    def create_cluster_visualization(self) -> None:
        """Create interactive cluster visualization."""
        # Create figure with subplots
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pattern Intensity Distribution',
                'Pattern Duration vs Confidence',
                'Pattern Timeline',
                'Pattern Relationships'
            ),
            specs=[
                [{"type": "box"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Box plot of intensity distributions
        self.fig.add_trace(
            go.Box(
                x=self.df['emotion_type'],
                y=self.df['avg_intensity'],
                name='Intensity Distribution',
                marker_color=self.color_scale['positive'],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=1, col=1
        )
        
        # 2. Scatter plot of duration vs confidence
        self.fig.add_trace(
            go.Scatter(
                x=self.df['duration'],
                y=self.df['confidence'],
                mode='markers+text',
                text=self.df['pattern_id'],
                textposition="top center",
                name='Duration vs Confidence',
                marker=dict(
                    size=self.df['entry_count']*3,
                    color=self.df['avg_intensity'],
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title='Avg Intensity')
                ),
                hovertemplate=(
                    '<b>Pattern:</b> %{text}<br>' +
                    '<b>Duration:</b> %{x} days<br>' +
                    '<b>Confidence:</b> %{y:.2f}<br>' +
                    '<b>Entries:</b> %{marker.size/3}<br>' +
                    '<extra></extra>'
                )
            ),
            row=1, col=2
        )
        
        # 3. Pattern timeline
        for _, pattern in self.df.iterrows():
            self.fig.add_trace(
                go.Scatter(
                    x=[pattern['start_date'], pattern['end_date']],
                    y=[pattern['avg_intensity'], pattern['avg_intensity']],
                    mode='lines+markers',
                    name=pattern['pattern_id'],
                    line=dict(
                        width=4,
                        color=self.color_scale['mixed']
                    ),
                    hovertemplate=(
                        f'<b>{pattern["pattern_id"]}</b><br>' +
                        '<b>Period:</b> %{x}<br>' +
                        '<b>Intensity:</b> %{y:.2f}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=2, col=1
            )
            
        # 4. Pattern relationship heatmap
        relationship_matrix = self._calculate_pattern_relationships()
        self.fig.add_trace(
            go.Heatmap(
                z=relationship_matrix,
                x=self.df['pattern_id'],
                y=self.df['pattern_id'],
                colorscale='RdBu',
                name='Pattern Relationships',
                hoverongaps=False,
                hovertemplate=(
                    '<b>Patterns:</b> %{x} - %{y}<br>' +
                    '<b>Relationship:</b> %{z:.2f}<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        self.fig.update_layout(
            title='Pattern Analysis Dashboard',
            height=1000,
            width=1200,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axes labels
        self.fig.update_xaxes(title_text="Emotion Type", row=1, col=1)
        self.fig.update_yaxes(title_text="Average Intensity", row=1, col=1)
        self.fig.update_xaxes(title_text="Duration (days)", row=1, col=2)
        self.fig.update_yaxes(title_text="Confidence Score", row=1, col=2)
        self.fig.update_xaxes(title_text="Date", row=2, col=1)
        self.fig.update_yaxes(title_text="Average Intensity", row=2, col=1)
        
    def _calculate_pattern_relationships(self) -> np.ndarray:
        """Calculate relationship scores between patterns."""
        n_patterns = len(self.patterns)
        relationship_matrix = np.zeros((n_patterns, n_patterns))
        
        for i, p1 in enumerate(self.patterns):
            for j, p2 in enumerate(self.patterns):
                if i == j:
                    relationship_matrix[i,j] = 1.0
                else:
                    # Calculate temporal overlap
                    overlap = self._calculate_temporal_overlap(p1, p2)
                    
                    # Calculate intensity similarity
                    intensity_sim = self._calculate_intensity_similarity(p1, p2)
                    
                    # Combine scores
                    relationship_matrix[i,j] = (overlap + intensity_sim) / 2
                    
        return relationship_matrix
        
    def _calculate_temporal_overlap(
        self,
        p1: EmotionalPattern,
        p2: EmotionalPattern
    ) -> float:
        """Calculate temporal overlap between two patterns."""
        start1 = p1.timespan.start_date
        end1 = p1.timespan.end_date
        start2 = p2.timespan.start_date
        end2 = p2.timespan.end_date
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
            
        overlap_days = (overlap_end - overlap_start).days
        total_days = max((end1 - start1).days, (end2 - start2).days)
        
        return overlap_days / total_days
        
    def _calculate_intensity_similarity(
        self,
        p1: EmotionalPattern,
        p2: EmotionalPattern
    ) -> float:
        """Calculate similarity in emotional intensity patterns."""
        # Compare baseline and peak intensities
        baseline_diff = abs(p1.intensity.baseline - p2.intensity.baseline)
        peak_diff = abs(p1.intensity.peak - p2.intensity.peak)
        
        # Compare progression rates
        rate_diff = abs(p1.intensity.progression_rate - p2.intensity.progression_rate)
        
        # Normalize differences
        max_intensity = max(p1.intensity.peak, p2.intensity.peak)
        if max_intensity == 0:
            return 0.0
            
        baseline_sim = 1 - (baseline_diff / max_intensity)
        peak_sim = 1 - (peak_diff / max_intensity)
        rate_sim = 1 - min(rate_diff, 1.0)  # Cap rate difference at 1.0
        
        return (baseline_sim + peak_sim + rate_sim) / 3
        
    def add_temporal_links(self) -> None:
        """Add temporal relationship visualization."""
        if not self.fig:
            self.create_cluster_visualization()
            
        # Add temporal connections between overlapping patterns
        for i, p1 in enumerate(self.patterns):
            for j, p2 in enumerate(self.patterns[i+1:], i+1):
                overlap = self._calculate_temporal_overlap(p1, p2)
                if overlap > 0:
                    # Add connection line with opacity based on overlap
                    self.fig.add_trace(
                        go.Scatter(
                            x=[p1.timespan.start_date, p2.timespan.start_date],
                            y=[
                                (p1.intensity.baseline + p1.intensity.peak) / 2,
                                (p2.intensity.baseline + p2.intensity.peak) / 2
                            ],
                            mode='lines',
                            line=dict(
                                width=1,
                                dash='dot',
                                color='rgba(64, 124, 213, 0.5)'  # Custom blue with 50% transparency
                            ),
                            showlegend=False,
                            hovertemplate=(
                                f'<b>Connection</b><br>' +
                                f'Patterns: {p1.pattern_id} - {p2.pattern_id}<br>' +
                                f'Overlap: {overlap:.2f}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        row=2, col=1
                    )
                    
    def export_html(self, output_path: str) -> None:
        """Export visualization as interactive HTML."""
        if not self.fig:
            self.create_cluster_visualization()
            
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to HTML with configuration
        self.fig.write_html(
            output_path,
            include_plotlyjs=True,
            full_html=True,
            include_mathjax=False,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'eraseshape']
            }
        )