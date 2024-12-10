"""
Generate an HTML report for journal analysis.
Example usage: python generate_report.py --year 2020 --month 1 --output-dir reports
"""

import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json

from journal_analyzer.visualization.html_export import HTMLExporter
from journal_analyzer.models.entry import JournalEntry
from journal_analyzer.models.patterns import EmotionalPattern, PatternTimespan, EmotionalIntensity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_entries(data_dir: Path, year: int, month: int) -> list:
    """Load journal entries from JSON file."""
    file_path = data_dir / "raw" / f"{year}_{month:02d}.json"
    
    with open(file_path) as f:
        raw_entries = json.load(f)
    
    # Convert to JournalEntry objects
    entries = []
    for entry in raw_entries:
        entries.append(JournalEntry(
            date=datetime.fromisoformat(entry["date"]),
            content=entry["content"],
            day_of_week=entry["day_of_week"],
            word_count=entry["word_count"],
            month=entry["month"],
            year=entry["year"]
        ))
    
    return entries

def load_patterns(data_dir: Path, year: int, month: int) -> list:
    """Load patterns from JSON file."""
    file_path = data_dir / "patterns" / f"{year}_{month:02d}.patterns.json"
    
    with open(file_path) as f:
        raw_patterns = json.load(f)
    
    # Convert to EmotionalPattern objects
    patterns = []
    for p in raw_patterns:
        # Convert entries
        entries = [
            JournalEntry(
                date=datetime.fromisoformat(e["date"].replace('T', ' ')),
                content=e["content"],
                day_of_week=e["day_of_week"],
                word_count=e["word_count"],
                month=datetime.fromisoformat(e["date"].replace('T', ' ')).month,
                year=datetime.fromisoformat(e["date"].replace('T', ' ')).year
            ) for e in p["entries"]
        ]
        
        # Create timespan
        timespan = PatternTimespan(
            start_date=datetime.fromisoformat(p["timespan"]["start_date"].replace('T', ' ')),
            end_date=datetime.fromisoformat(p["timespan"]["end_date"].replace('T', ' ')),
            duration_days=p["timespan"]["duration_days"],
            recurring=p["timespan"]["recurring"]
        )
        
        # Create intensity
        intensity = EmotionalIntensity(
            baseline=float(p["intensity"]["baseline"]),
            peak=float(p["intensity"]["peak"]),
            variance=float(p["intensity"]["variance"]),
            progression_rate=float(p["intensity"]["progression_rate"])
        )
        
        # Create pattern
        pattern = EmotionalPattern(
            pattern_id=p["pattern_id"],
            description=p["description"],
            entries=entries,
            timespan=timespan,
            confidence_score=float(p["confidence_score"]),
            emotion_type=p["emotion_type"],
            intensity=intensity
        )
        
        patterns.append(pattern)
    
    return patterns

def generate_report(
    data_dir: Path,
    output_dir: Path,
    year: int,
    month: int
) -> None:
    """Generate HTML report for specified month."""
    try:
        # Load data
        entries = load_entries(data_dir, year, month)
        patterns = load_patterns(data_dir, year, month)
        
        logger.info(f"Loaded {len(entries)} entries and {len(patterns)} patterns")
        
        # Create report
        exporter = HTMLExporter()
        
        # Get date range
        start_date = min(entries, key=lambda x: x.date).date
        end_date = max(entries, key=lambda x: x.date).date
        
        # Generate HTML
        html = exporter.generate_report(
            entries=entries,
            patterns=patterns,
            start_date=start_date,
            end_date=end_date
        )
        
        # Export report
        output_path = output_dir / f"journal_analysis_{year}_{month:02d}.html"
        exporter.export(html, str(output_path))
        
        logger.info(f"Successfully generated report: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate HTML report for journal analysis"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to analyze"
    )
    
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month to analyze"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing journal data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save generated reports"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    generate_report(
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        year=args.year,
        month=args.month
    )

if __name__ == "__main__":
    main()