#!/usr/bin/env python3
"""
Generate a comprehensive HTML report for journal analysis across multiple months.
Example usage: python generate_report.py --year 2020 --output-dir reports
"""

import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any, Optional

from journal_analyzer.visualization.html_export import HTMLExporter
from journal_analyzer.models.entry import JournalEntry
from journal_analyzer.models.patterns import EmotionalPattern, PatternTimespan, EmotionalIntensity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YearReportGenerator:
    """Generates comprehensive year-level HTML reports from monthly data files."""
    
    def __init__(self, data_dir: Path, year: int):
        """Initialize the report generator."""
        self.data_dir = Path(data_dir)
        self.year = year
        self.entries: List[JournalEntry] = []
        self.patterns: List[EmotionalPattern] = []
        
    async def generate(self, output_dir: Path) -> None:
        """Generate comprehensive year report."""
        try:
            # Load all data for the year
            await self._load_full_year_data()
            
            if not self.entries or not self.patterns:
                logger.error(f"No data found for year {self.year}")
                return
                
            # Create report
            exporter = HTMLExporter()
            
            # Get date range
            start_date = min(self.entries, key=lambda x: x.date).date
            end_date = max(self.entries, key=lambda x: x.date).date
            
            # Generate HTML
            html = exporter.generate_year_report(
                entries=self.entries,
                patterns=self.patterns,
                year=self.year,
                start_date=start_date,
                end_date=end_date
            )
            
            # Export report
            output_path = output_dir / f"journal_analysis_{self.year}_full.html"
            exporter.export(html, str(output_path))
            
            logger.info(f"Successfully generated year report: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating year report: {str(e)}")
            raise
    
    async def _load_full_year_data(self) -> None:
        """Load and combine all monthly data for the year."""
        for month in range(1, 13):
            try:
                # Load entries
                entries = self._load_month_entries(month)
                if entries:
                    self.entries.extend(entries)
                    
                # Load patterns
                patterns = self._load_month_patterns(month)
                if patterns:
                    self.patterns.extend(patterns)
                    
            except Exception as e:
                logger.error(f"Error loading data for {self.year}-{month}: {str(e)}")
                continue
        
        # Sort entries by date
        self.entries.sort(key=lambda x: x.date)
        
        # Sort patterns by start date
        self.patterns.sort(key=lambda x: x.timespan.start_date)
        
        logger.info(
            f"Loaded {len(self.entries)} entries and {len(self.patterns)} patterns "
            f"for year {self.year}"
        )
    
    def _load_month_entries(self, month: int) -> List[JournalEntry]:
        """Load journal entries for a specific month."""
        file_path = self.data_dir / "raw" / f"{self.year}_{month:02d}.json"
        if not file_path.exists():
            return []
            
        with open(file_path) as f:
            raw_entries = json.load(f)
            
        return [
            JournalEntry(
                date=datetime.fromisoformat(e["date"]),
                content=e["content"],
                day_of_week=e["day_of_week"],
                word_count=e["word_count"],
                month=datetime.fromisoformat(e["date"]).month,
                year=datetime.fromisoformat(e["date"]).year
            ) for e in raw_entries
        ]
    
    def _load_month_patterns(self, month: int) -> List[EmotionalPattern]:
        """Load patterns for a specific month."""
        file_path = self.data_dir / "patterns" / f"{self.year}_{month:02d}.patterns.json"
        if not file_path.exists():
            return []
            
        with open(file_path) as f:
            raw_patterns = json.load(f)
            
        patterns = []
        for p in raw_patterns:
            # Convert entries
            entries = [
                JournalEntry(
                    date=datetime.fromisoformat(e["date"]),
                    content=e["content"],
                    day_of_week=e["day_of_week"],
                    word_count=e["word_count"],
                    month=datetime.fromisoformat(e["date"]).month,
                    year=datetime.fromisoformat(e["date"]).year
                ) for e in p["entries"]
            ]
            
            # Create timespan
            timespan = PatternTimespan(
                start_date=datetime.fromisoformat(p["timespan"]["start_date"]),
                end_date=datetime.fromisoformat(p["timespan"]["end_date"]),
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
                description=p.get("description", ""),  # Handle legacy files
                entries=entries,
                timespan=timespan,
                confidence_score=float(p["confidence_score"]),
                emotion_type=p["emotion_type"],
                intensity=intensity
            )
            
            patterns.append(pattern)
            
        return patterns

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive year-level HTML report for journal analysis"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to analyze"
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

async def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    generator = YearReportGenerator(
        data_dir=Path(args.data_dir),
        year=args.year
    )
    
    await generator.generate(output_dir)

if __name__ == "__main__":
    asyncio.run(main())