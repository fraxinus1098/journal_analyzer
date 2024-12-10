"""
HTML report generation and export functionality.

File path: journal_analyzer/visualization/html_export.py
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import json
from datetime import datetime
import logging
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.utils
import pandas as pd
import numpy as np

from .emotional_timeline import EmotionalTimeline
from .pattern_clusters import PatternVisualizer
from ..models.entry import JournalEntry
from ..models.patterns import EmotionalPattern

logger = logging.getLogger(__name__)

class HTMLExporter:
    """Generates interactive HTML reports from analysis results."""
    
    YEAR_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            /* Base styles */
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
                background-color: #f5f5f5;
            }
            
            /* Enhanced container styling */
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            
            /* Year summary section */
            .year-summary {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            
            .summary-card {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            /* Emotion distribution chart */
            .emotion-chart {
                height: 400px;
                margin-bottom: 40px;
            }
            
            /* Pattern details section */
            .pattern-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            
            .pattern-card {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Topic analysis section */
            .topic-analysis {
                margin-bottom: 40px;
            }
            
            .topic-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }
            
            /* Monthly breakdown section */
            .monthly-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 30px;
                margin-bottom: 40px;
            }
            
            .month-card {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Navigation and filters */
            .nav-controls {
                position: sticky;
                top: 0;
                background-color: rgba(255, 255, 255, 0.95);
                padding: 10px 0;
                margin-bottom: 20px;
                z-index: 100;
                border-bottom: 1px solid #eee;
            }
            
            /* Enhanced typography */
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 40px;
            }
            
            h2 {
                color: #34495e;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            }
            
            h3 {
                color: #7f8c8d;
                margin-bottom: 15px;
            }
        </style>
    </head>
    <body>
        <!-- Header Section -->
        <div class="container">
            <h1>{{ title }}</h1>
            <div class="year-summary">
                {% for stat in summary_stats %}
                <div class="summary-card">
                    <h3>{{ stat.label }}</h3>
                    <div class="stat-value">{{ stat.value }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Timeline Visualization -->
        <div class="container">
            <h2>Emotional Journey Timeline</h2>
            <div class="viz-container">
                {{ timeline_plot }}
            </div>
        </div>
        
        <!-- Pattern Analysis -->
        <div class="container">
            <h2>Pattern Analysis</h2>
            <div class="viz-container">
                {{ pattern_plot }}
            </div>
            
            <div class="pattern-grid">
                {% for pattern in detailed_patterns %}
                <div class="pattern-card" style="border-left: 4px solid {{ pattern.color }}">
                    <h3>{{ pattern.topic }}</h3>
                    <p><strong>Emotion:</strong> {{ pattern.emotion_type }}</p>
                    <p><strong>Period:</strong> {{ pattern.period }}</p>
                    <p><strong>Description:</strong> {{ pattern.description }}</p>
                    <div class="pattern-metrics">
                        {% for metric in pattern.metrics %}
                        <div class="metric">{{ metric }}</div>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Topic Analysis -->
        <div class="container">
            <h2>Topic Analysis</h2>
            <div class="topic-analysis">
                {{ topic_distribution_plot }}
            </div>
            <div class="topic-grid">
                {% for topic in topic_analysis %}
                <div class="topic-card">
                    <h3>{{ topic.name }}</h3>
                    <p>{{ topic.description }}</p>
                    <ul>
                        {% for insight in topic.insights %}
                        <li>{{ insight }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Monthly Breakdown -->
        <div class="container">
            <h2>Monthly Breakdown</h2>
            <div class="monthly-grid">
                {% for month in monthly_breakdown %}
                <div class="month-card">
                    <h3>{{ month.name }}</h3>
                    <div class="month-stats">
                        <p><strong>Entries:</strong> {{ month.entry_count }}</p>
                        <p><strong>Patterns:</strong> {{ month.pattern_count }}</p>
                        <p><strong>Primary Emotions:</strong> {{ month.primary_emotions }}</p>
                        <p><strong>Key Topics:</strong> {{ month.key_topics }}</p>
                    </div>
                    {{ month.mini_plot }}
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Footer -->
        <div class="container">
            <p class="footer">Generated on {{ generation_date }}</p>
        </div>
    </body>
    </html>
    """
    
    def __init__(self):
        """Initialize the HTML exporter."""
        self.env = Environment(loader=FileSystemLoader("."))
        
    def __init__(self):
        """Initialize the HTML exporter."""
        self.env = Environment(loader=FileSystemLoader("."))
        
    def export(self, html_content: str, output_path: str) -> None:
        """
        Export HTML content to a file.
        
        Args:
            html_content: Generated HTML content
            output_path: Path to save the HTML file
        """
        try:
            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write HTML content to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Successfully exported HTML report to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting HTML report to {output_path}: {str(e)}")
            raise
    
    def _figure_to_html(self, fig: go.Figure) -> str:
        """
        Convert a Plotly figure to HTML string.
        
        Args:
            fig: Plotly figure object
            
        Returns:
            HTML string representation of the figure
        """
        if fig is None:
            return ""
            
        try:
            # Convert figure to HTML with specific configuration
            html = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                include_mathjax=False,
                config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'responsive': True
                }
            )
            return html
        except Exception as e:
            logger.error(f"Error converting figure to HTML: {str(e)}")
            return f"<div class='error'>Error generating visualization: {str(e)}</div>"
        
    def generate_year_report(
        self,
        entries: List[JournalEntry],
        patterns: List[EmotionalPattern],
        year: int,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Generate comprehensive year-level HTML report."""
        # Create main visualizations
        timeline = EmotionalTimeline(entries, patterns)
        timeline.create_timeline()
        timeline.add_pattern_overlays()
        timeline.add_interaction_elements()
        timeline.add_monthly_annotations()
        
        pattern_viz = PatternVisualizer(patterns)
        pattern_viz.create_cluster_visualization()
        pattern_viz.add_temporal_links()
        
        # Calculate summary statistics
        summary_stats = self._calculate_year_summary(entries, patterns)
        
        # Prepare pattern details
        detailed_patterns = self._prepare_pattern_details(patterns)
        
        # Prepare topic analysis
        topic_analysis = self._analyze_topics(patterns)
        
        # Prepare monthly breakdown
        monthly_breakdown = self._prepare_monthly_breakdown(entries, patterns)
        
        # Convert plots to HTML
        timeline_plot = self._figure_to_html(timeline.fig)
        pattern_plot = self._figure_to_html(pattern_viz.fig)
        topic_plot = self._create_topic_distribution_plot(patterns)
        
        # Render template
        template = Environment().from_string(self.YEAR_TEMPLATE)
        html = template.render(
            title=f"Journal Analysis Report - {year}",
            summary_stats=summary_stats,
            timeline_plot=timeline_plot,
            pattern_plot=pattern_plot,
            detailed_patterns=detailed_patterns,
            topic_analysis=topic_analysis,
            topic_distribution_plot=topic_plot,
            monthly_breakdown=monthly_breakdown,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return html
    def _calculate_year_summary(
        self,
        entries: List[JournalEntry],
        patterns: List[EmotionalPattern]
    ) -> List[Dict[str, str]]:
        """Calculate year-level summary statistics."""
        # Basic counts
        total_entries = len(entries)
        total_patterns = len(patterns)
        total_words = sum(entry.word_count for entry in entries)
        
        # Emotion distribution
        emotion_counts = {}
        for pattern in patterns:
            emotion_counts[pattern.emotion_type] = emotion_counts.get(pattern.emotion_type, 0) + 1
        primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Pattern statistics
        avg_pattern_duration = np.mean([p.timespan.duration_days for p in patterns])
        avg_pattern_confidence = np.mean([p.confidence_score for p in patterns])
        
        # Topic analysis
        topics = set(p.pattern_id.split('_')[0] for p in patterns)
        
        return [
            {"label": "Total Journal Entries", "value": f"{total_entries:,}"},
            {"label": "Total Words Written", "value": f"{total_words:,}"},
            {"label": "Average Words per Entry", "value": f"{total_words/total_entries:.0f}"},
            {"label": "Emotional Patterns Detected", "value": str(total_patterns)},
            {"label": "Primary Emotion", "value": primary_emotion.capitalize()},
            {"label": "Unique Topics", "value": str(len(topics))},
            {"label": "Average Pattern Duration", "value": f"{avg_pattern_duration:.1f} days"},
            {"label": "Pattern Detection Confidence", "value": f"{avg_pattern_confidence:.2%}"}
        ]
    
    def _prepare_pattern_details(
        self,
        patterns: List[EmotionalPattern]
    ) -> List[Dict[str, Any]]:
        """Prepare detailed pattern information with enhanced metrics."""
        detailed_patterns = []
        
        # Color scale for emotions
        color_scale = {
            'joy': '#2ecc71',
            'sadness': '#3498db',
            'anger': '#e74c3c',
            'fear': '#9b59b6',
            'surprise': '#f1c40f',
            'anticipation': '#e67e22',
            'trust': '#1abc9c',
            'disgust': '#95a5a6',
            'mixed': '#34495e'
        }
        
        for pattern in patterns:
            # Extract topic from pattern_id
            topic = ' '.join(pattern.pattern_id.split('_')[0].split('-')).title()
            
            metrics = [
                f"Confidence: {pattern.confidence_score:.2%}",
                f"Duration: {pattern.timespan.duration_days} days",
                f"Entry Count: {len(pattern.entries)}",
                f"Intensity Range: {pattern.intensity.baseline:.2f} - {pattern.intensity.peak:.2f}",
                f"Progression Rate: {pattern.intensity.progression_rate:+.2f}",
                f"Recurring: {'Yes' if pattern.timespan.recurring else 'No'}"
            ]
            
            detailed_patterns.append({
                "topic": topic,
                "emotion_type": pattern.emotion_type.capitalize(),
                "period": (
                    f"{pattern.timespan.start_date.strftime('%B %d')} - "
                    f"{pattern.timespan.end_date.strftime('%B %d, %Y')}"
                ),
                "description": pattern.description,
                "metrics": metrics,
                "color": color_scale[pattern.emotion_type]
            })
        
        return detailed_patterns
    
    def _analyze_topics(
        self,
        patterns: List[EmotionalPattern]
    ) -> List[Dict[str, Any]]:
        """Analyze topics and their emotional characteristics."""
        if not patterns:
            return []
        
        # Group patterns by topic
        topic_patterns = defaultdict(list)
        for pattern in patterns:
            if not pattern or not pattern.pattern_id:
                continue
            topic = ' '.join(pattern.pattern_id.split('_')[0].split('-')).title()
            topic_patterns[topic].append(pattern)
        
        topic_analysis = []
        for topic, topic_list in topic_patterns.items():
            # Calculate topic statistics
            emotion_counts = Counter(p.emotion_type for p in topic_list)
            primary_emotion = emotion_counts.most_common(1)[0][0]
            avg_confidence = np.mean([p.confidence_score for p in topic_list])
            avg_duration = np.mean([p.timespan.duration_days for p in topic_list])
            
            # Generate insights
            insights = [
                f"Most common emotion: {primary_emotion.capitalize()}",
                f"Average pattern duration: {avg_duration:.1f} days",
                f"Pattern detection confidence: {avg_confidence:.2%}",
                f"Number of related patterns: {len(topic_list)}"
            ]
            
            topic_analysis.append({
                "name": topic,
                "description": self._generate_topic_description(topic_list),
                "insights": insights
            })
        
        return topic_analysis
    
    def _prepare_monthly_breakdown(
        self,
        entries: List[JournalEntry],
        patterns: List[EmotionalPattern]
    ) -> List[Dict[str, Any]]:
        """Prepare monthly statistics and visualizations."""
        monthly_data = []
        
        # Group entries and patterns by month
        for month in range(1, 13):
            month_entries = [e for e in entries if e.date.month == month]
            if not month_entries:
                continue
                
            month_patterns = [
                p for p in patterns
                if any(e.date.month == month for e in p.entries)
            ]
            
            # Calculate monthly statistics
            emotion_counts = Counter(p.emotion_type for p in month_patterns)
            primary_emotions = [e for e, _ in emotion_counts.most_common(2)]
            
            # Get unique topics
            topics = set(p.pattern_id.split('_')[0] for p in month_patterns)
            
            # Create mini visualization
            mini_plot = self._create_monthly_mini_plot(month_entries, month_patterns)
            
            monthly_data.append({
                "name": month_entries[0].date.strftime("%B"),
                "entry_count": len(month_entries),
                "pattern_count": len(month_patterns),
                "primary_emotions": ", ".join(e.capitalize() for e in primary_emotions),
                "key_topics": ", ".join(t.title() for t in topics),
                "mini_plot": self._figure_to_html(mini_plot)
            })
        
        return monthly_data
    
    def _create_monthly_mini_plot(
        self,
        entries: List[JournalEntry],
        patterns: List[EmotionalPattern]
    ) -> go.Figure:
        """Create small monthly summary visualization."""
        fig = go.Figure()
        
        # Add entry intensity line
        dates = [e.date for e in entries]
        intensities = []
        for entry in entries:
            entry_patterns = [p for p in patterns if any(e.date == entry.date for e in p.entries)]
            intensity = max([p.intensity.peak for p in entry_patterns]) if entry_patterns else 0
            intensities.append(intensity)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=intensities,
                mode='lines',
                line=dict(width=2),
                showlegend=False
            )
        )
        
        # Update layout for small size
        fig.update_layout(
            height=150,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _create_topic_distribution_plot(
        self,
        patterns: List[EmotionalPattern]
    ) -> str:
        """Create topic distribution and relationship visualization."""
        # Create network graph of topic relationships
        topic_links = defaultdict(int)
        for p1 in patterns:
            for p2 in patterns:
                if p1 != p2:
                    topic1 = p1.pattern_id.split('_')[0]
                    topic2 = p2.pattern_id.split('_')[0]
                    if topic1 < topic2:  # Avoid duplicates
                        overlap = self._calculate_pattern_overlap(p1, p2)
                        if overlap > 0:
                            topic_links[(topic1, topic2)] = overlap
        
        # Create plot using Plotly
        fig = go.Figure()
        
        # Add nodes (topics)
        topics = set(p.pattern_id.split('_')[0] for p in patterns)
        positions = self._calculate_topic_positions(topics)
        
        for topic, pos in positions.items():
            topic_patterns = [p for p in patterns if p.pattern_id.split('_')[0] == topic]
            
            fig.add_trace(
                go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    name=topic.title(),
                    text=[topic.title()],
                    marker=dict(size=len(topic_patterns)*5),
                    textposition="middle center"
                )
            )
        
        # Add edges (relationships)
        for (topic1, topic2), weight in topic_links.items():
            pos1 = positions[topic1]
            pos2 = positions[topic2]
            
            fig.add_trace(
                go.Scatter(
                    x=[pos1[0], pos2[0]],
                    y=[pos1[1], pos2[1]],
                    mode='lines',
                    line=dict(
                        width=weight*5,
                        color='rgba(128, 128, 128, 0.5)'
                    ),
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title="Topic Relationships",
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        
        return self._figure_to_html(fig)
    
    def _calculate_pattern_overlap(
        self,
        p1: EmotionalPattern,
        p2: EmotionalPattern
    ) -> float:
        """Calculate temporal overlap between patterns."""
        start1 = p1.timespan.start_date
        end1 = p1.timespan.end_date
        start2 = p2.timespan.start_date
        end2 = p2.timespan.end_date
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
            
        overlap_days = (overlap_end - overlap_start).days
        total_days = max((end1 - start1).days, (end2 - start2).days)
        
        return overlap_days / total_days
    
    def _calculate_topic_positions(self, topics: Set[str]) -> Dict[str, Tuple[float, float]]:
        """Calculate positions for topic network visualization."""
        n = len(topics)
        positions = {}
        
        # Arrange topics in a circle
        for i, topic in enumerate(sorted(topics)):
            angle = 2 * np.pi * i / n
            x = np.cos(angle)
            y = np.sin(angle)
            positions[topic] = (x, y)
        
        return positions
    
    def _generate_topic_description(self, patterns: List[EmotionalPattern]) -> str:
        """Generate a summary description for a topic."""
        # Get most common emotions
        emotions = Counter(p.emotion_type for p in patterns)
        primary_emotions = [e for e, _ in emotions.most_common(2)]
        
        # Calculate averages
        avg_duration = np.mean([p.timespan.duration_days for p in patterns])
        avg_intensity = np.mean([p.intensity.peak for p in patterns])
        
        # Handle different numbers of emotions
        if not primary_emotions:
            return "No clear emotional patterns detected."
        elif len(primary_emotions) == 1:
            emotion_text = f"Primarily associated with {primary_emotions[0]}"
        else:
            emotion_text = f"Primarily associated with {primary_emotions[0]} and {primary_emotions[1]}"
        
        return (
            f"{emotion_text} emotions. "
            f"Patterns typically last {avg_duration:.1f} days with "
            f"moderate to {'high' if avg_intensity > 0.7 else 'medium'} intensity."
        )