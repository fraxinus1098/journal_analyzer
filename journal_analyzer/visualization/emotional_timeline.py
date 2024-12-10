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
        patterns: List[EmotionalPattern]
    ):
        """Initialize with analyzed entries and patterns."""
        self.entries = sorted(entries, key=lambda x: x.date)
        self.patterns = patterns
        self.fig = None
        
        # Enhanced color scale mapping for emotions
        self.color_scale = {
            'joy': '#2ecc71',         # Green
            'sadness': '#3498db',     # Blue
            'anger': '#e74c3c',       # Red
            'fear': '#9b59b6',        # Purple
            'surprise': '#f1c40f',    # Yellow
            'anticipation': '#e67e22', # Orange
            'trust': '#1abc9c',       # Turquoise
            'disgust': '#95a5a6',     # Gray
            'mixed': '#34495e',       # Dark Gray
            'neutral': '#bdc3c7'      # Light Gray
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
            
            # Get primary emotion and topic for entry
            primary_emotion = entry_patterns[0].emotion_type if entry_patterns else 'neutral'
            
            data.append({
                'date': entry.date,
                'content': entry.content,
                'word_count': entry.word_count,
                'intensity': intensity,
                'pattern_ids': [p.pattern_id for p in entry_patterns],
                'pattern_count': len(entry_patterns),
                'primary_emotion': primary_emotion,
                'topics': [p.pattern_id.split('_')[0] for p in entry_patterns]
            })
            
        return pd.DataFrame(data)
        
    def create_timeline(self) -> None:
        """Create main emotional journey timeline."""
        # Create figure with secondary y-axis
        self.fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                'Emotional Intensity Timeline',
                'Entry Length & Pattern Distribution'
            )
        )
        
        # Add emotional intensity line with color mapping
        for emotion in self.color_scale:
            emotion_data = self.df[self.df['primary_emotion'] == emotion]
            if len(emotion_data) > 0:
                self.fig.add_trace(
                    go.Scatter(
                        x=emotion_data['date'],
                        y=emotion_data['intensity'],
                        mode='lines+markers',
                        name=emotion.capitalize(),
                        line=dict(
                            color=self.color_scale[emotion],
                            width=2
                        ),
                        marker=dict(
                            size=8,
                            color=self.color_scale[emotion]
                        ),
                        hovertemplate=(
                            '<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                            '<b>Intensity:</b> %{y:.2f}<br>' +
                            '<b>Emotion:</b> ' + emotion.capitalize() + '<br>' +
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
                marker_color='rgba(100, 100, 100, 0.5)',
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
        """Add enhanced pattern highlight overlays to timeline."""
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
                fillcolor=self.color_scale[pattern.emotion_type],
                opacity=0.1,
                layer='below',
                line_width=0,
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
                    name=pattern.pattern_id,
                    hovertemplate=(
                        f'<b>Pattern:</b> {pattern.pattern_id}<br>' +
                        f'<b>Emotion:</b> {pattern.emotion_type}<br>' +
                        f'<b>Confidence:</b> {pattern.confidence_score:.2f}<br>' +
                        f'<b>Duration:</b> {pattern.timespan.duration_days} days<br>' +
                        f'<b>Description:</b> {pattern.description}<br>' +
                        '<extra></extra>'
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
            
    def add_interaction_elements(self) -> None:
        """Add enhanced interactive elements to visualization."""
        if not self.fig:
            self.create_timeline()
            
        # Add range selector
        self.fig.update_xaxes(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="Full Year")
                ])
            )
        )

         # Add emotion type filter
        emotion_buttons = [dict(
            args=[{"visible": [True] * len(self.fig.data)}],
            label="All Emotions",
            method="update"
        )]
        
        for emotion in self.color_scale:
            visible = [
                True if (emotion in trace.name.lower() or 'word count' in trace.name.lower())
                else False
                for trace in self.fig.data
            ]
            emotion_buttons.append(dict(
                args=[{"visible": visible}],
                label=emotion.capitalize(),
                method="update"
            ))
            
        # Add pattern filter dropdown
        pattern_buttons = []
        for pattern in self.patterns:
            pattern_buttons.append(dict(
                args=[{
                    "visible": [True] * len(self.fig.data),
                    "xaxis.range": [
                        pattern.timespan.start_date,
                        pattern.timespan.end_date
                    ]
                }],
                label=f"{pattern.pattern_id} ({pattern.emotion_type})",
                method="update"
            ))
        
        # Add buttons to layout
        self.fig.update_layout(
            updatemenus=[
                dict(
                    buttons=emotion_buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.15,
                    name="Emotion Filter"
                ),
                dict(
                    buttons=pattern_buttons,
                    direction="down",
                    showactive=True,
                    x=0.3,
                    y=1.15,
                    name="Pattern Focus"
                )
            ]
        )
        
    def add_monthly_annotations(self) -> None:
        """Add enhanced monthly summary annotations."""
        if not self.fig:
            self.create_timeline()
            
        # Group by month using 'ME' instead of 'M'
        monthly = self.df.set_index('date').resample('ME').agg({
            'intensity': 'mean',
            'pattern_count': 'sum',
            'word_count': 'sum',
            'primary_emotion': lambda x: x.mode().iloc[0] if not x.empty else 'neutral'
        })
        
        for date, row in monthly.iterrows():
            # Count patterns in this month
            month_patterns = [
                p for p in self.patterns
                if (p.timespan.start_date.month == date.month and
                    p.timespan.start_date.year == date.year)
            ]
            
            # Get unique topics
            topics = {p.pattern_id.split('_')[0] for p in month_patterns}
            
            self.fig.add_annotation(
                x=date,
                y=row['intensity'],
                text=(
                    f"<b>{date.strftime('%B %Y')}</b><br>" +
                    f"Avg Intensity: {row['intensity']:.2f}<br>" +
                    f"Dominant Emotion: {row['primary_emotion'].capitalize()}<br>" +
                    f"Active Patterns: {row['pattern_count']}<br>" +
                    f"Key Topics: {', '.join(topics)}<br>" +
                    f"Total Words: {int(row['word_count'])}"
                ),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.color_scale[row['primary_emotion']],
                ax=0,
                ay=-40,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=self.color_scale[row['primary_emotion']],
                borderwidth=2,
                borderpad=4,
                row=1, col=1
            )
            
    def export_html(self, output_path: str) -> None:
        """Export visualization as interactive HTML."""
        if not self.fig:
            self.create_timeline()
            
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to HTML with enhanced configuration
        self.fig.write_html(
            output_path,
            include_plotlyjs=True,
            full_html=True,
            include_mathjax=False,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': [
                    'drawline',
                    'drawopenpath',
                    'drawclosedpath',
                    'drawcircle',
                    'eraseshape'
                ],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'emotional_timeline',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
        )