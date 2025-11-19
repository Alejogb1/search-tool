import google.generativeai as genai
from openai import OpenAI
import asyncio
import time
import logging
import os
import csv
import json
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# OpenRouter models for fallback
OPENROUTER_MODELS = [
    {"name": "deepseek/deepseek-r1-0528:free", "rpm": 30},
    {"name": "google/gemini-2.5-pro-exp-03-25", "rpm": 15},
    {"name": "qwen/qwen3-235b-a22b:free", "rpm": 30},
]

class AnalysisType(Enum):
    PAIN_POINT_EXTRACTION = "pain_point_extraction"
    SOLUTION_IDENTIFICATION = "solution_identification"
    GAP_ANALYSIS = "gap_analysis"
    TECHNICAL_DEPTH = "technical_depth"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUERY_EXPANSION = "query_expansion"

@dataclass
class SearchResult:
    query: str
    url: str
    title: str
    text: str

@dataclass
class AnalysisResult:
    query: str
    url: str
    pain_points: List[str]
    solutions_mentioned: List[str]
    gaps_identified: List[str]
    technical_terms: List[str]
    sentiment: str
    confidence_level: str
    new_query_suggestions: List[str]
    key_insights: List[str]

class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.key_usage_count = {}
        self.key_last_used = {}
        self.key_rate_limited_until = {}
        self.rate_limit_delay = 60
        self.min_interval = 1
        self.logger = logging.getLogger(__name__)

        for key in api_keys:
            self.key_usage_count[key] = 0
            self.key_last_used[key] = 0
            self.key_rate_limited_until[key] = 0

    def get_next_key(self) -> str:
        current_time = time.time()
        best_key = None
        min_usage = float('inf')
        
        for i in range(len(self.api_keys)):
            key_index = (self.current_key_index + i) % len(self.api_keys)
            key = self.api_keys[key_index]
            
            if current_time < self.key_rate_limited_until[key]:
                continue
                
            if current_time - self.key_last_used[key] < self.min_interval:
                continue
                
            if self.key_usage_count[key] < min_usage:
                best_key = key
                min_usage = self.key_usage_count[key]
        
        if best_key:
            self.current_key_index = (self.api_keys.index(best_key) + 1) % len(self.api_keys)
            self.key_last_used[best_key] = current_time
            self.key_usage_count[best_key] += 1
            return best_key
        
        self.logger.warning("All API keys in use or rate limited - waiting before retry")
        time.sleep(2)
        return self.get_next_key()
    
    def mark_rate_limited(self, api_key: str):
        current_time = time.time()
        self.key_rate_limited_until[api_key] = current_time + self.rate_limit_delay
        self.key_last_used[api_key] = current_time

class SearchResultAnalyzer:
    def __init__(self, api_keys: List[str], model_configs: List[Dict] = None):
        self.api_key_manager = APIKeyManager(api_keys)
        self.model_configs = model_configs if model_configs else [{'name': 'gemini-2.0-flash-exp', 'rpm': 200}]
        self.current_model_config = self.model_configs[0]
        self.logger = logging.getLogger(__name__)
        self.clients = {}
        self.model_request_counts = {}
        self.analyzed_urls = set()
        
        for cfg in self.model_configs:
            self.model_request_counts[cfg['name']] = []

        # Initialize OpenRouter clients
        self.openrouter_keys = os.getenv("OPENROUTER_API_KEYS", "").split(",") if os.getenv("OPENROUTER_API_KEYS") else []
        self.openrouter_clients = {}
        self.openrouter_model_index = 0
        self.openrouter_request_counts = {}
        
        if self.openrouter_keys and any(key.strip() for key in self.openrouter_keys):
            self.openrouter_keys = [key.strip() for key in self.openrouter_keys if key.strip()]
            self.logger.info(f"Initialized {len(self.openrouter_keys)} OpenRouter API keys")

        for key in self.openrouter_keys:
            self.openrouter_clients[key] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
            )

        for model_config in OPENROUTER_MODELS:
            self.openrouter_request_counts[model_config['name']] = []
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('search_analysis.log'),
                logging.StreamHandler()
            ]
        )
        
    def _get_client(self, api_key: str):
        model_name = self.current_model_config['name']
        if api_key not in self.clients:
            self.clients[api_key] = {}
        
        if model_name not in self.clients[api_key]:
            genai.configure(api_key=api_key)
            self.clients[api_key][model_name] = genai.GenerativeModel(model_name)
        return self.clients[api_key][model_name]

    def _can_make_request(self) -> bool:
        model_name = self.current_model_config['name']
        rpm_limit = self.current_model_config['rpm']
        current_time = time.time()

        self.model_request_counts[model_name] = [
            t for t in self.model_request_counts.get(model_name, []) if current_time - t < 60
        ]

        if len(self.model_request_counts[model_name]) < rpm_limit:
            return True
        else:
            self.logger.warning(f"Model {model_name} RPM limit reached ({rpm_limit} RPM).")
            self._switch_model()
            return False

    def _record_request(self):
        model_name = self.current_model_config['name']
        self.model_request_counts[model_name].append(time.time())

    def _switch_model(self):
        current_model_index = self.model_configs.index(self.current_model_config)
        next_model_index = (current_model_index + 1) % len(self.model_configs)
        self.current_model_config = self.model_configs[next_model_index]
        
        self.model_request_counts[self.current_model_config['name']] = []
        
        cooldown = 5 if len(self.model_request_counts) > 1 else 0
        time.sleep(cooldown)
        
        self.logger.info(f"Switched to model: {self.current_model_config['name']} (RPM: {self.current_model_config['rpm']})")

    def _can_make_openrouter_request(self, model_name: str) -> bool:
        current_time = time.time()
        self.openrouter_request_counts[model_name] = [
            t for t in self.openrouter_request_counts.get(model_name, []) if current_time - t < 60
        ]

        rpm_limit = next((model['rpm'] for model in OPENROUTER_MODELS if model['name'] == model_name), 30)
        if len(self.openrouter_request_counts[model_name]) < rpm_limit:
            return True
        else:
            self.logger.warning(f"OpenRouter model {model_name} RPM limit reached ({rpm_limit} RPM).")
            return False

    def _record_openrouter_request(self, model_name: str):
        self.openrouter_request_counts[model_name].append(time.time())

    def _get_analysis_prompt(self, result: SearchResult) -> str:
        """
        Multi-faceted analysis prompt for extracting deep insights
        """
        return f"""
You are an expert Research Analyst and Pain Point Discovery Specialist. Analyze this search result deeply to extract actionable insights.

ORIGINAL QUERY: "{result.query}"
URL: {result.url}
TITLE: {result.title}
CONTENT:
{result.text[:8000]}

YOUR TASK: Perform a comprehensive analysis and return ONLY a JSON object with the following structure:

{{
    "pain_points": [
        // List 3-8 specific pain points, problems, or frustrations mentioned
        // Examples: "CUDA out of memory errors during training", "Model convergence takes too long"
        // Focus on ACTIONABLE problems that researchers/practitioners face
    ],
    "solutions_mentioned": [
        // List 2-6 solutions, workarounds, or fixes discussed
        // Examples: "Reduce batch size to 8", "Use gradient checkpointing"
        // Include both temporary fixes and proper solutions
    ],
    "gaps_identified": [
        // List 2-5 GAPS - problems mentioned WITHOUT clear solutions
        // These are gold mines for discovering unmet needs
        // Examples: "No clear way to estimate optimal batch size", "Lack of tools to predict memory usage"
    ],
    "technical_terms": [
        // Extract 5-10 technical terms, frameworks, or specific technologies mentioned
        // Examples: "PyTorch", "A6000 GPU", "gradient checkpointing", "mixed precision training"
        // Include versions if mentioned
    ],
    "sentiment": "frustrated" | "neutral" | "satisfied" | "confused",
    // Overall emotional tone of the content
    
    "confidence_level": "high" | "medium" | "low",
    // How confident is the author about the information shared?
    
    "new_query_suggestions": [
        // Generate 4-8 NEW search queries that would logically follow from this content
        // Think: "What would someone search NEXT after reading this?"
        // Examples based on gaps identified, related problems, deeper technical questions
        // These should be DIFFERENT from the original query but related
    ],
    "key_insights": [
        // List 2-4 high-level insights or patterns
        // Examples: "Memory issues are more common with transformer models > 1B parameters"
        // "Community relies heavily on trial-and-error for batch size tuning"
    ]
}}

CRITICAL RULES:
1. Be SPECIFIC - avoid generic statements
2. Extract ACTUAL information from the text, don't invent
3. For gaps_identified: only list problems where solutions are NOT clearly provided
4. For new_query_suggestions: generate queries that go DEEPER or SIDEWAYS from current topic
5. Focus on ACTIONABLE insights that could guide further research
6. Return ONLY valid JSON, no explanations outside the JSON

STRATEGIC FOCUS:
- Prioritize identifying UNMET NEEDS and KNOWLEDGE GAPS
- Look for phrases like "I don't know", "unclear", "no documentation", "trial and error"
- Identify patterns in what people are struggling with
- Suggest queries that could uncover systematic bottlenecks
"""

    async def _analyze_single_result_async(self, result: SearchResult) -> Optional[AnalysisResult]:
        """
        Analyze a single search result using LLM
        """
        
        prompt = self._get_analysis_prompt(result)
        
        max_api_retries = len(self.api_key_manager.api_keys) * len(self.model_configs) * 2

        # Try Gemini first
        for api_attempt in range(max_api_retries):
            try:
                if not self._can_make_request():
                    self._switch_model()
                    await asyncio.sleep(5)
                    continue

                api_key = self.api_key_manager.get_next_key()
                client = self._get_client(api_key)

                response = await asyncio.to_thread(
                    client.generate_content,
                    contents=prompt
                )

                self._record_request()

                if response and response.text:
                    analysis_data = self._parse_analysis_response(response.text)
                    if analysis_data:
                        return AnalysisResult(
                            query=result.query,
                            url=result.url,
                            pain_points=analysis_data.get("pain_points", []),
                            solutions_mentioned=analysis_data.get("solutions_mentioned", []),
                            gaps_identified=analysis_data.get("gaps_identified", []),
                            technical_terms=analysis_data.get("technical_terms", []),
                            sentiment=analysis_data.get("sentiment", "neutral"),
                            confidence_level=analysis_data.get("confidence_level", "medium"),
                            new_query_suggestions=analysis_data.get("new_query_suggestions", []),
                            key_insights=analysis_data.get("key_insights", [])
                        )

                self.logger.warning(f"Empty/invalid response from API")
                self._switch_model()
                await asyncio.sleep(5)
                continue

            except Exception as e:
                error_msg = str(e).lower()

                if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                    self.api_key_manager.mark_rate_limited(api_key)
                    await asyncio.sleep(5)
                    continue

                elif '504' in error_msg or 'deadline' in error_msg:
                    self._switch_model()
                    await asyncio.sleep(5)
                    continue

                elif 'api' in error_msg or 'request' in error_msg:
                    self._switch_model()
                    await asyncio.sleep(3)
                    continue

                else:
                    self.logger.error(f"Unexpected error: {e}")
                    raise e

        # Fallback to OpenRouter
        if self.openrouter_keys:
            print(f"    🔄 Trying OpenRouter for: {result.query}")
            
            max_total_attempts = 6
            attempts = 0

            for model_config in OPENROUTER_MODELS:
                if attempts >= max_total_attempts:
                    break

                model_name = model_config['name']

                if not self._can_make_openrouter_request(model_name):
                    continue

                keys_tried = 0
                for openrouter_key in self.openrouter_keys:
                    if keys_tried >= 2 or attempts >= max_total_attempts:
                        break

                    keys_tried += 1
                    attempts += 1

                    try:
                        client = self.openrouter_clients[openrouter_key]

                        response = await asyncio.wait_for(
                            asyncio.to_thread(
                                client.chat.completions.create,
                                model=model_name,
                                messages=[
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=4000,
                                temperature=0.3
                            ),
                            timeout=30.0
                        )

                        self._record_openrouter_request(model_name)

                        if response and response.choices and len(response.choices) > 0:
                            response_text = response.choices[0].message.content
                            analysis_data = self._parse_analysis_response(response_text)
                            if analysis_data:
                                return AnalysisResult(
                                    query=result.query,
                                    url=result.url,
                                    pain_points=analysis_data.get("pain_points", []),
                                    solutions_mentioned=analysis_data.get("solutions_mentioned", []),
                                    gaps_identified=analysis_data.get("gaps_identified", []),
                                    technical_terms=analysis_data.get("technical_terms", []),
                                    sentiment=analysis_data.get("sentiment", "neutral"),
                                    confidence_level=analysis_data.get("confidence_level", "medium"),
                                    new_query_suggestions=analysis_data.get("new_query_suggestions", []),
                                    key_insights=analysis_data.get("key_insights", [])
                                )

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'rate limit' in error_msg or '429' in error_msg:
                            break
                        else:
                            continue

        return None

    def _parse_analysis_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response into structured data"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                self.logger.warning("No JSON found in response")
                return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return None

    def _load_search_results_from_csv(self, csv_file: str) -> List[SearchResult]:
        """Load search results from CSV file (one result per query)"""
        results = []
        seen_queries = set()
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query = row.get('query', '').strip()
                    
                    # Only take first result per query
                    if query and query not in seen_queries:
                        results.append(SearchResult(
                            query=query,
                            url=row.get('url', '').strip(),
                            title=row.get('title', '').strip(),
                            text=row.get('text', '').strip()
                        ))
                        seen_queries.add(query)
                        
        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {csv_file}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return []
        
        return results

    def _save_analysis_to_json(self, analyses: List[AnalysisResult], output_file: str):
        """Save analysis results to JSON file"""
        try:
            data = []
            for analysis in analyses:
                data.append({
                    "query": analysis.query,
                    "url": analysis.url,
                    "pain_points": analysis.pain_points,
                    "solutions_mentioned": analysis.solutions_mentioned,
                    "gaps_identified": analysis.gaps_identified,
                    "technical_terms": analysis.technical_terms,
                    "sentiment": analysis.sentiment,
                    "confidence_level": analysis.confidence_level,
                    "new_query_suggestions": analysis.new_query_suggestions,
                    "key_insights": analysis.key_insights
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved analysis to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")

    def _extract_new_queries_from_analyses(self, analyses: List[AnalysisResult]) -> Set[str]:
        """Extract all new query suggestions from analyses"""
        new_queries = set()
        
        for analysis in analyses:
            for query in analysis.new_query_suggestions:
                # Clean and normalize
                query = query.strip().lower()
                if query and len(query.split()) <= 10:
                    new_queries.add(query)
        
        return new_queries

    def _save_new_queries_to_file(self, queries: Set[str], output_file: str):
        """Save new query suggestions to text file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for query in sorted(queries):
                    f.write(query + '\n')
            
            print(f"✅ Saved {len(queries)} new queries to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving queries: {e}")

    def _generate_summary_report(self, analyses: List[AnalysisResult]) -> str:
        """Generate a summary report of findings"""
        
        total_results = len(analyses)
        total_pain_points = sum(len(a.pain_points) for a in analyses)
        total_gaps = sum(len(a.gaps_identified) for a in analyses)
        total_solutions = sum(len(a.solutions_mentioned) for a in analyses)
        total_new_queries = sum(len(a.new_query_suggestions) for a in analyses)
        
        # Sentiment distribution
        sentiments = {}
        for analysis in analyses:
            sentiments[analysis.sentiment] = sentiments.get(analysis.sentiment, 0) + 1
        
        # Most common technical terms
        all_terms = {}
        for analysis in analyses:
            for term in analysis.technical_terms:
                all_terms[term] = all_terms.get(term, 0) + 1
        
        top_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Most common pain points
        all_pain_points = {}
        for analysis in analyses:
            for pain in analysis.pain_points:
                # Normalize to first 50 chars for grouping
                pain_key = pain[:50].lower()
                all_pain_points[pain_key] = all_pain_points.get(pain_key, 0) + 1
        
        top_pains = sorted(all_pain_points.items(), key=lambda x: x[1], reverse=True)[:5]
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║           SEARCH RESULT ANALYSIS SUMMARY                   ║
╠════════════════════════════════════════════════════════════╣
║ Total Results Analyzed:        {str(total_results).ljust(28)} ║
║ Total Pain Points Identified:  {str(total_pain_points).ljust(28)} ║
║ Total Knowledge Gaps Found:    {str(total_gaps).ljust(28)} ║
║ Total Solutions Mentioned:     {str(total_solutions).ljust(28)} ║
║ Total New Queries Generated:   {str(total_new_queries).ljust(28)} ║
╠════════════════════════════════════════════════════════════╣
║ SENTIMENT DISTRIBUTION:                                    ║
"""
        for sentiment, count in sorted(sentiments.items()):
            report += f"║   {sentiment.capitalize().ljust(20)}: {str(count).ljust(26)} ║\n"
        
        report += f"""╠════════════════════════════════════════════════════════════╣
║ TOP TECHNICAL TERMS:                                       ║
"""
        for i, (term, count) in enumerate(top_terms[:5], 1):
            term_display = term[:35]
            report += f"║   {str(i)}.  {term_display.ljust(35)} ({count}){''.ljust(10)} ║\n"
        
        report += f"""╠════════════════════════════════════════════════════════════╣
║ MOST COMMON PAIN POINTS:                                   ║
"""
        for i, (pain, count) in enumerate(top_pains, 1):
            pain_display = pain[:35]
            report += f"║   {str(i)}.  {pain_display.ljust(35)} ({count}){''.ljust(10)} ║\n"
        
        report += "╚════════════════════════════════════════════════════════════╝\n"
        
        return report

    async def analyze_search_results(self, 
                                     input_csv: str, 
                                     output_json: str = 'analysis_results.json',
                                     output_queries: str = 'new_queries.txt',
                                     max_results: int = None) -> str:
        """
        Main analysis function
        
        Args:
            input_csv: CSV file with columns: query, url, title, text
            output_json: Output file for detailed analysis
            output_queries: Output file for new query suggestions
            max_results: Maximum number of results to process (None = all)
        """
        
        start_time = time.time()
        
        print(f"\n=== SEARCH RESULT ANALYSIS STARTED ===")
        print(f"Input CSV: {input_csv}")
        print(f"Output JSON: {output_json}")
        print(f"Output Queries: {output_queries}")
        
        # Load search results
        search_results = self._load_search_results_from_csv(input_csv)
        
        if not search_results:
            return "Error: No search results found in CSV"
        
        if max_results:
            search_results = search_results[:max_results]
        
        print(f"Loaded {len(search_results)} unique queries to analyze")
        
        # Analyze each result
        analyses = []
        
        for idx, result in enumerate(search_results, 1):
            print(f"\n--- Analyzing {idx}/{len(search_results)}: '{result.query}' ---")
            
            try:
                analysis = await self._analyze_single_result_async(result)
                
                if analysis:
                    analyses.append(analysis)
                    print(f"  ✅ Pain points: {len(analysis.pain_points)}")
                    print(f"  ✅ Gaps identified: {len(analysis.gaps_identified)}")
                    print(f"  ✅ New queries: {len(analysis.new_query_suggestions)}")
                else:
                    print(f"  ❌ Analysis failed")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing '{result.query}': {e}")
                print(f"  ❌ Error: {str(e)}")
                continue
            
            # Progress update
            progress = idx / len(search_results) * 100
            print(f"\nOverall progress: {idx}/{len(search_results)} ({progress:.1f}%)")
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        # Save results
        if analyses:
            self._save_analysis_to_json(analyses, output_json)
            
            # Extract and save new queries
            new_queries = self._extract_new_queries_from_analyses(analyses)
            self._save_new_queries_to_file(new_queries, output_queries)
            
            # Generate summary report
            summary = self._generate_summary_report(analyses)
            print(summary)
            
            # Save summary to file
            summary_file = output_json.replace('.json', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"✅ Saved summary to {summary_file}")
        
        # Final timing
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        return (
            f"\n=== ANALYSIS COMPLETE ===\n"
            f"Analyzed: {len(analyses)} results\n"
            f"New queries generated: {len(new_queries) if analyses else 0}\n"
            f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"Outputs:\n"
            f"  - Detailed analysis: {output_json}\n"
            f"  - New queries: {output_queries}\n"
            f"  - Summary: {summary_file if analyses else 'N/A'}"
        )


# Main execution function
async def analyze_search_results(input_csv: str, 
                                 output_json: str = 'analysis_results.json',
                                 output_queries: str = 'new_queries.txt',
                                 api_keys: List[str] = None,
                                 model_configs: List[Dict] = None,
                                 max_results: int = None):
    """
    Analyze search results from CSV file
    """
    if not api_keys:
        raise ValueError("At least one API key is required")
    
    analyzer = SearchResultAnalyzer(
        api_keys=api_keys,
        model_configs=model_configs
    )
    
    return await analyzer.analyze_search_results(
        input_csv=input_csv,
        output_json=output_json,
        output_queries=output_queries,
        max_results=max_results
    )


# Example usage
if __name__ == "__main__":
    # Load API keys from .env file
    api_keys_str = os.getenv("GOOGLE_API_KEYS")
    if not api_keys_str:
        print("ERROR: GOOGLE_API_KEYS not found in .env file.")
        exit(1)

    API_KEYS = [key.strip() for key in api_keys_str.split(',')]

    MODEL_CONFIGS = [
        {"name": "gemini-2.0-flash-exp", "rpm": 200},
        {"name": "gemini-2.0-flash-lite", "rpm": 30},
    ]

    async def run_analysis():
        # Analyze search results from CSV
        result = await analyze_search_results(
            input_csv='search_results.csv',  # Your CSV file
            output_json='deep_analysis.json',
            output_queries='discovered_queries.txt',
            api_keys=API_KEYS,
            model_configs=MODEL_CONFIGS,
            max_results=100  # Process first 100 unique queries
        )
        print(result)

    asyncio.run(run_analysis())