"""
HTML report generation and export functionality.

File path: journal_analyzer/visualization/html_export.py
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
import logging
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.utils
import pandas as pd

from .emotional_timeline import EmotionalTimeline
from .pattern_clusters import PatternVisualizer
from ..models.entry import JournalEntry
from ..models.patterns import EmotionalPattern

logger = logging.getLogger(__name__)

class HTMLExporter:
    """Generates interactive HTML reports from analysis results."""
    
    TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .section {
                margin-bottom: 40px;
            }
            .viz-container {
                margin-bottom: 30px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
            .stat-label {
                color: #7f8c8d;
                font-size: 14px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f8f9fa;
            }
            .pattern-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
            }
            .footer {
                text-align: center;
                color: #7f8c8d;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Analysis Period: {{ period }}</p>
        </div>
        
        <div class="container">
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="stats-grid">
                    {% for stat in summary_stats %}
                    <div class="stat-box">
                        <div class="stat-value">{{ stat.value }}</div>
                        <div class="stat-label">{{ stat.label }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2>Emotional Timeline</h2>
                <div class="viz-container" id="timeline-viz">
                    {{ timeline_plot }}
                </div>
            </div>
            
            <div class="section">
                <h2>Pattern Analysis</h2>
                <div class="viz-container" id="pattern-viz">
                    {{ pattern_plot }}
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Pattern Analysis</h2>
                {% for pattern in detailed_patterns %}
                <div class="pattern-card">
                    <h3>Pattern {{ pattern.id }}</h3>
                    <p><strong>Period:</strong> {{ pattern.period }}</p>
                    <p><strong>Emotion Type:</strong> {{ pattern.emotion_type }}</p>
                    <p><strong>Confidence:</strong> {{ pattern.confidence }}</p>
                    <p><strong>Key Metrics:</strong></p>
                    <ul>
                        {% for metric in pattern.metrics %}
                        <li>{{ metric }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {{ generation_date }}</p>
        </div>
    </body>
    </html>
    """
    
    def __init__(self):
        """Initialize the HTML exporter."""
        self.env = Environment(loader=FileSystemLoader("."))
        
    def generate_report(
        self,
        entries: List[JournalEntry],
        patterns: List[EmotionalPattern],
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """
        Generate HTML report from analysis data.
        
        Args:
            entries: List of journal entries
            patterns: List of emotional patterns
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Generated HTML string
        """
        # Create visualizations
        timeline = EmotionalTimeline(entries, patterns)
        timeline.create_timeline()
        timeline.add_pattern_overlays()
        timeline.add_interaction_elements()
        timeline.add_monthly_annotations()
        
        pattern_viz = PatternVisualizer(patterns)
        pattern_viz.create_cluster_visualization()
        pattern_viz.add_temporal_links()
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(entries, patterns)
        
        # Prepare pattern details
        detailed_patterns = self._prepare_pattern_details(patterns)
        
        # Convert plots to HTML
        timeline_plot = self._figure_to_html(timeline.fig)
        pattern_plot = self._figure_to_html(pattern_viz.fig)
        
        # Render template
        template = Environment().from_string(self.TEMPLATE)
        html = template.render(
            title="Journal Analysis Report",
            period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            summary_stats=summary_stats,
            timeline_plot=timeline_plot,
            pattern_plot=pattern_plot,
            detailed_patterns=detailed_patterns,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return html
        
    def _calculate_summary_stats(
        self,
        entries: List[JournalEntry],
        patterns: List[EmotionalPattern]
    ) -> List[Dict[str, str]]:
        """Calculate summary statistics for the report."""
        total_entries = len(entries)
        total_patterns = len(patterns)
        avg_pattern_confidence = np.mean([p.confidence_score for p in patterns]) if patterns else 0
        
        # Calculate word statistics
        total_words = sum(entry.word_count for entry in entries)
        avg_words_per_entry = total_words / total_entries if total_entries > 0 else 0
        
        # Calculate pattern duration statistics
        pattern_durations = [p.timespan.duration_days for p in patterns]
        avg_pattern_duration = np.mean(pattern_durations) if pattern_durations else 0
        
        return [
            {"label": "Total Entries", "value": str(total_entries)},
            {"label": "Total Words", "value": f"{total_words:,}"},
            {"label": "Avg Words per Entry", "value": f"{avg_words_per_entry:.1f}"},
            {"label": "Patterns Detected", "value": str(total_patterns)},
            {"label": "Avg Pattern Duration", "value": f"{avg_pattern_duration:.1f} days"},
            {"label": "Avg Pattern Confidence", "value": f"{avg_pattern_confidence:.2f}"}
        ]
        
    def _prepare_pattern_details(
        self,
        patterns: List[EmotionalPattern]
    ) -> List[Dict[str, Any]]:
        """Prepare detailed pattern information for the report."""
        detailed_patterns = []
        
        for pattern in patterns:
            metrics = [
                f"Baseline Intensity: {pattern.intensity.baseline:.2f}",
                f"Peak Intensity: {pattern.intensity.peak:.2f}",
                f"Variance: {pattern.intensity.variance:.2f}",
                f"Progression Rate: {pattern.intensity.progression_rate:.2f}",
                f"Duration: {pattern.timespan.duration_days} days",
                f"Entry Count: {len(pattern.entries)}"
            ]
            
            if pattern.timespan.recurring:
                metrics.append("Type: Recurring Pattern")
            
            detailed_patterns.append({
                "id": pattern.pattern_id,
                "period": (
                    f"{pattern.timespan.start_date.strftime('%Y-%m-%d')} to "
                    f"{pattern.timespan.end_date.strftime('%Y-%m-%d')}"
                ),
                "emotion_type": pattern.emotion_type,
                "confidence": f"{pattern.confidence_score:.2f}",
                "metrics": metrics
            })
            
        return detailed_patterns
        
    def _figure_to_html(self, fig: go.Figure) -> str:
        """Convert Plotly figure to HTML string."""
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'eraseshape']
            }
        )
        
    def export(
        self,
        html: str,
        output_path: str
    ) -> None:
        """
        Export HTML report to file.
        
        Args:
            html: Generated HTML string
            output_path: Path to save the report
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
                
            logger.info(f"Successfully exported report to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting report to {output_path}: {str(e)}")
            raise