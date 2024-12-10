"""
Interactive visualization components for emotional timelines and pattern visualization.

File path: journal_analyzer/visualization/emotional_timeline.py
"""

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path
import json

from ..models.entry import JournalEntry
from ..models.patterns import EmotionalPattern, PatternTimespan

class EmotionalTimeline:
    """Creates interactive visualizations of emotional patterns over time."""
    
    def __init__(
        self,
        entries: List[JournalEntry],
        patterns: List[EmotionalPattern],
        color_scale: Optional[Dict[str, str]] = None
    ):
        """
        Initialize with analyzed entries and patterns.
        
        Args:
            entries: List of journal entries
            patterns: List of emotional patterns
            color_scale: Optional custom color scale mapping
        """
        self.entries = sorted(entries, key=lambda x: x.date)
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
        
        # Convert entries to DataFrame for easier manipulation
        self.df = self._create_dataframe()
        
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert entries to DataFrame with emotional metrics."""
        data = []
        for entry in self.entries:
            # Get patterns that include this entry
            entry_patterns = [
                p for p in self.patterns
                if any(e.date == entry.date for e in p.entries)
            ]
            
            # Calculate aggregate emotional intensity
            intensity = 0
            if entry_patterns:
                intensities = [p.intensity.peak for p in entry_patterns]
                intensity = max(intensities)
            
            data.append({
                'date': entry.date,
                'content': entry.content,
                'word_count': entry.word_count,
                'intensity': intensity,
                'pattern_ids': [p.pattern_id for p in entry_patterns],
                'pattern_count': len(entry_patterns)
            })
            
        return pd.DataFrame(data)
        
    def create_timeline(self) -> None:
        """Create main emotional journey timeline."""
        # Create figure with secondary y-axis
        self.fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Add emotional intensity line
        self.fig.add_trace(
            go.Scatter(
                x=self.df['date'],
                y=self.df['intensity'],
                mode='lines+markers',
                name='Emotional Intensity',
                line=dict(
                    color=self.color_scale['positive'],
                    width=2
                ),
                marker=dict(
                    size=8,
                    color=self.df['intensity'],
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(
                        title='Emotional<br>Intensity'
                    )
                ),
                hovertemplate=(
                    '<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                    '<b>Intensity:</b> %{y:.2f}<br>' +
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # Add word count bars
        self.fig.add_trace(
            go.Bar(
                x=self.df['date'],
                y=self.df['word_count'],
                name='Word Count',
                marker_color=self.color_scale['neutral'],
                opacity=0.5,
                hovertemplate=(
                    '<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                    '<b>Words:</b> %{y}<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        # Update layout
        self.fig.update_layout(
            title='Emotional Journey Timeline',
            height=800,
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(title='Emotional Intensity'),
            yaxis2=dict(title='Word Count'),
            template='plotly_white'
        )
        
    def add_pattern_overlays(self) -> None:
        """Add pattern highlight overlays to timeline."""
        if not self.fig:
            self.create_timeline()
            
        for pattern in self.patterns:
            # Create pattern span
            start = pattern.timespan.start_date
            end = pattern.timespan.end_date
            
            # Add pattern highlight
            self.fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=self.color_scale['highlight'],
                opacity=0.1,
                layer='below',
                line_width=0,
                annotation_text=f"Pattern {pattern.pattern_id}",
                annotation_position="top left",
                row=1, col=1
            )
            
            # Add pattern details to hover
            pattern_entries = self.df[
                (self.df['date'] >= start) &
                (self.df['date'] <= end)
            ]
            
            self.fig.add_trace(
                go.Scatter(
                    x=pattern_entries['date'],
                    y=pattern_entries['intensity'],
                    mode='markers',
                    marker=dict(size=1, opacity=0),
                    name=f'Pattern {pattern.pattern_id}',
                    hovertemplate=(
                        f'<b>Pattern {pattern.pattern_id}</b><br>' +
                        f'<b>Type:</b> {pattern.emotion_type}<br>' +
                        f'<b>Confidence:</b> {pattern.confidence_score:.2f}<br>' +
                        f'<b>Duration:</b> {pattern.timespan.duration_days} days<br>' +
                        '<extra></extra>'
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
            
    def add_interaction_elements(self) -> None:
        """Add interactive elements to visualization."""
        if not self.fig:
            self.create_timeline()
            
        # Add range selector
        self.fig.update_xaxes(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Add pattern selection buttons
        pattern_buttons = []
        for pattern in self.patterns:
            pattern_buttons.append(
                dict(
                    label=f"Pattern {pattern.pattern_id}",
                    method="update",
                    args=[
                        {"visible": [True] * len(self.fig.data)},
                        {
                            "xaxis.range": [
                                pattern.timespan.start_date,
                                pattern.timespan.end_date
                            ]
                        }
                    ]
                )
            )
            
        self.fig.update_layout(
            updatemenus=[
                dict(
                    buttons=pattern_buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.1,
                    name="Pattern Selection"
                )
            ]
        )
        
    def add_monthly_annotations(self) -> None:
        """Add monthly summary annotations."""
        if not self.fig:
            self.create_timeline()
            
        # Group by month
        monthly = self.df.set_index('date').resample('M').agg({
            'intensity': 'mean',
            'pattern_count': 'sum',
            'word_count': 'sum'
        })
        
        for date, row in monthly.iterrows():
            self.fig.add_annotation(
                x=date,
                y=row['intensity'],
                text=(
                    f"Monthly Summary<br>" +
                    f"Avg Intensity: {row['intensity']:.2f}<br>" +
                    f"Patterns: {int(row['pattern_count'])}<br>" +
                    f"Words: {int(row['word_count'])}"
                ),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.color_scale['neutral'],
                ax=0,
                ay=-40,
                row=1, col=1
            )
            
    def export_html(self, output_path: str) -> None:
        """Export visualization as interactive HTML."""
        if not self.fig:
            self.create_timeline()
            
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to HTML
        self.fig.write_html(
            output_path,
            include_plotlyjs=True,
            full_html=True,
            include_mathjax=False
        )
        
    def update_data(
        self,
        entries: Optional[List[JournalEntry]] = None,
        patterns: Optional[List[EmotionalPattern]] = None
    ) -> None:
        """Update visualization with new data."""
        if entries:
            self.entries = sorted(entries, key=lambda x: x.date)
        if patterns:
            self.patterns = patterns
            
        # Recreate DataFrame
        self.df = self._create_dataframe()
        
        # Recreate visualization
        self.create_timeline()
        self.add_pattern_overlays()
        self.add_interaction_elements()
        self.add_monthly_annotations()