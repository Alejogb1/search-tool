import pandas as pd
import json
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
import logging
import os 
import sys
import time
from integrations.llm_client import EnhancedGeminiKeywordGenerator


class ChunkProcessor:
    def __init__(self, output_dir: str = "processed_chunks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('chunk_processing.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    async def process_csv_to_chunks(self, csv_path: str, chunk_size: int = 500) -> List[Dict]:
        """Process CSV file into chunks and save as JSON files"""
        try:
            df = pd.read_csv(csv_path)
            chunks = []
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size].to_dict('records')
                chunk_file = self.output_dir / f"chunk_{i//chunk_size}.json"
                
                with open(chunk_file, 'w') as f:
                    json.dump(chunk, f)
                
                self.logger.info(f"Saved chunk {i//chunk_size} to {chunk_file}")
                chunks.append(chunk)
                
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
            raise

    async def analyze_chunk(self, chunk: List[Dict], previous_clusters: Optional[List] = None) -> Dict:
        """Analyze a single chunk of data with context from previous clusters using LLM"""
        analysis = {
            "chunk_id": hash(str(chunk)),
            "total_terms": len(chunk),
            "clusters": [],
            "metrics": {
                "avg_search_volume": 0,
                "avg_cpc": 0,
                "commercial_intent_ratio": 0
            },
            "llm_analysis": {}
        }

        # Skip empty chunks
        if not chunk:
            return analysis

        # Initialize LLM client
        api_keys = os.getenv("GOOGLE_API_KEYS", "").split(",")
        if not api_keys:
            self.logger.warning("No API keys found, falling back to basic analysis")
            return await self._basic_analysis(chunk, previous_clusters)

        llm_client = EnhancedGeminiKeywordGenerator(api_keys=api_keys)

        try:
            # Get keywords from chunk
            keywords = [term['keyword'] for term in chunk if 'keyword' in term]
            
            # Generate LLM analysis
            llm_analysis = await self._get_llm_analysis(llm_client, keywords, previous_clusters)
            analysis['llm_analysis'] = llm_analysis

            # Enhanced clustering with LLM
            analysis['clusters'] = await self._cluster_with_llm(llm_client, keywords, previous_clusters)

            # Calculate metrics
            analysis.update(await self._calculate_metrics(chunk))

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return await self._basic_analysis(chunk, previous_clusters)

        return analysis

    async def _get_llm_analysis(self, llm_client, keywords: List[str], previous_clusters: Optional[List]) -> Dict:
        """Get semantic analysis from LLM"""
        prompt = f"""
        Analyze these search terms and provide semantic clustering and intent classification:
        
        Terms: {keywords[:100]}  # Limit to first 100 terms
        
        Analysis Requirements:
        1. Group by semantic similarity
        2. Identify primary intent for each group (commercial, informational, navigational)
        3. Note any emerging themes
        4. Highlight potential gaps in coverage
        
        Previous clusters context: {previous_clusters[:5] if previous_clusters else 'None'}
        
        Return JSON format with:
        - clusters: array of grouped terms with themes
        - intent_distribution: counts by intent type
        - insights: key observations
        """
        
        response = await llm_client.generate_content(prompt)
        return json.loads(response.text) if response else {}

    async def _cluster_with_llm(self, llm_client, keywords: List[str], previous_clusters: Optional[List]) -> List[Dict]:
        """Get enhanced clustering from LLM"""
        prompt = f"""
        Cluster these search terms into meaningful groups based on:
        - Semantic similarity
        - User intent
        - Commercial potential
        
        Terms: {keywords[:100]}
        Previous clusters: {previous_clusters[:5] if previous_clusters else 'None'}
        
        Return JSON array where each cluster has:
        - theme: main topic
        - terms: array of terms
        - intent: primary intent
        - commercial_score: 1-10
        """
        
        response = await llm_client.generate_content(prompt)
        return json.loads(response.text) if response else []

    async def _calculate_metrics(self, chunk: List[Dict]) -> Dict:
        """Calculate standard metrics"""
        metrics = {
            "avg_search_volume": 0,
            "avg_cpc": 0,
            "commercial_intent_ratio": 0
        }

        search_volumes = [term.get('search_volume', 0) for term in chunk]
        cpcs = [term.get('cpc', 0) for term in chunk]
        
        metrics['avg_search_volume'] = sum(search_volumes) / len(search_volumes)
        metrics['avg_cpc'] = sum(cpcs) / len(cpcs)

        commercial_terms = [t for t in chunk if t.get('commercial_intent', False)]
        if commercial_terms:
            metrics['commercial_intent_ratio'] = len(commercial_terms) / len(chunk)

        return metrics

    async def _basic_analysis(self, chunk: List[Dict], previous_clusters: Optional[List]) -> Dict:
        """Fallback basic analysis without LLM"""
        analysis = {
            "chunk_id": hash(str(chunk)),
            "total_terms": len(chunk),
            "clusters": [],
            "metrics": await self._calculate_metrics(chunk),
            "llm_analysis": {"error": "LLM analysis not available"}
        }

        # Simple keyword-based clustering
        clusters = {}
        for term in chunk:
            keyword = term.get('keyword', '').lower()
            cluster_key = keyword[:3]  
            if cluster_key not in clusters:
                clusters[cluster_key] = {'terms': []}
            clusters[cluster_key]['terms'].append(term)

        for cluster_key, cluster_data in clusters.items():
            analysis['clusters'].append({
                'key': cluster_key,
                'sample_terms': [t['keyword'] for t in cluster_data['terms'][:3]],
                'size': len(cluster_data['terms'])
            })

        return analysis

    async def process_with_context(self, csv_path: str):
        """Process CSV with context passing between chunks"""
        chunks = await self.process_csv_to_chunks(csv_path)
        cluster_history = []
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            analysis = await self.analyze_chunk(chunk, cluster_history)
            cluster_history.extend(analysis.get('clusters', []))
            
            # Save analysis results
            result_file = self.output_dir / f"analysis_{i}.json"
            with open(result_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
        return cluster_history

def validate_input_file(file_path: str) -> bool:
    """Validate the input CSV file has required columns"""
    try:
        df = pd.read_csv(file_path, nrows=1)
        required_cols = {'keyword', 'search_volume', 'cpc'}
        return required_cols.issubset(set(df.columns))
    except Exception:
        return False

async def main():
    if len(sys.argv) < 2:
        print("Usage: python chunk_processor.py <input_csv> [output_dir] [chunk_size]")
        print("Required CSV columns: keyword, search_volume, cpc")
        return 1

    input_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_chunks"
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found")
        return 1

    if not validate_input_file(input_csv):
        print("Error: Input CSV missing required columns (keyword, search_volume, cpc)")
        return 1

    processor = ChunkProcessor(output_dir)
    try:
        print(f"Processing {input_csv} in chunks of {chunk_size}...")
        start_time = time.time()
        
        cluster_history = await processor.process_with_context(input_csv)
        
        elapsed = time.time() - start_time
        print(f"\nProcessing complete in {elapsed:.2f} seconds")
        print(f"Generated {len(cluster_history)} total clusters")
        print(f"Results saved to: {output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
