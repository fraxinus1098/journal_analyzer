# File path: journal_analyzer/core/embeddings.py
"""
Embedding generation and management using OpenAI's API.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates and manages embeddings for journal entries using OpenAI's API."""
    
    def __init__(
        self,
        api_key: str,
        dimension: int = 256,
        batch_size: int = 50,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key
            dimension: Desired embedding dimension (default 256 for efficiency)
            batch_size: Number of texts to process in each batch
            model: OpenAI embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.dimension = dimension
        self.batch_size = batch_size
        self.model = model
        
    async def generate_embeddings(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for a list of journal entries.
        
        Args:
            entries: List of journal entry dictionaries
            
        Returns:
            Dictionary mapping entry dates to their embeddings and metadata
        """
        results = {}
        
        # Process entries in batches
        for i in tqdm(range(0, len(entries), self.batch_size), desc="Generating embeddings"):
            batch = entries[i:i + self.batch_size]
            try:
                # Extract text content from batch
                texts = [entry["content"] for entry in batch]
                
                # Generate embeddings for batch
                response = await self._generate_batch_embeddings(texts)
                
                # Process and store results
                for entry, embedding_data in zip(batch, response.data):
                    results[entry["date"]] = {
                        "embedding": embedding_data.embedding,
                        "metadata": {
                            "date": entry["date"],
                            "day_of_week": entry["day_of_week"],
                            "word_count": entry["word_count"],
                            "month": entry["month"],
                            "year": entry["year"]
                        }
                    }
                
                # Add small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                continue
                
        return results
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> Any:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            OpenAI API response containing embeddings
        """
        try:
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                input=texts,
                model=self.model,
                dimensions=self.dimension
            )
            return response
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings: Dict[str, Any], output_dir: Path, year: int, month: int) -> None:
        """
        Save embeddings to file system.
        
        Args:
            embeddings: Dictionary of embeddings and metadata
            output_dir: Directory to save embeddings
            year: Year of entries
            month: Month of entries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        output_file = output_dir / f"{year}_{month:02d}.embeddings.json"
        with open(output_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_embeddings = {
                date: {
                    "embedding": np.array(data["embedding"]).tolist(),
                    "metadata": data["metadata"]
                }
                for date, data in embeddings.items()
            }
            json.dump(serializable_embeddings, f, indent=2)
    
    def load_embeddings(self, input_dir: Path, year: int, month: int) -> Dict[str, Any]:
        """
        Load embeddings from file system.
        
        Args:
            input_dir: Directory containing embedding files
            year: Year to load
            month: Month to load
            
        Returns:
            Dictionary of embeddings and metadata
        """
        input_file = input_dir / f"{year}_{month:02d}.embeddings.json"
        
        if not input_file.exists():
            logger.warning(f"No embeddings file found for {year}-{month}")
            return {}
            
        with open(input_file) as f:
            embeddings = json.load(f)
            
        # Convert lists back to numpy arrays
        return {
            date: {
                "embedding": np.array(data["embedding"]),
                "metadata": data["metadata"]
            }
            for date, data in embeddings.items()
        }