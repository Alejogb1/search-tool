from google import genai
from google.genai import types
from web_scraper import Webscraper
import asyncio
import json
import logging
import time
import re
import random
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib

class KeywordCategory(Enum):
    CORE_SERVICES = "core_services"
    PROBLEM_STATEMENTS = "problem_statements"
    TECHNOLOGY_TERMS = "technology_terms"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    FEATURE_SPECIFIC = "feature_specific"
    PROCESS_RELATED = "process_related"
    INDUSTRY_TERMINOLOGY = "industry_terminology"
    ALTERNATIVE_PHRASINGS = "alternative_phrasings"
    EDUCATIONAL_TERMS = "educational_terms"
    LONG_TAIL_VARIATIONS = "long_tail_variations"

@dataclass
class KeywordBatch:
    category: KeywordCategory
    seed_terms: List[str]
    target_count: int
    context: str
    variation_strategy: str

class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.key_usage_count = {}
        self.key_last_used = {}
        self.rate_limit_delay = 60  # seconds to wait after rate limit
        
        # Initialize tracking for each key
        for key in api_keys:
            self.key_usage_count[key] = 0
            self.key_last_used[key] = 0
    
    def get_next_key(self) -> str:
        """Get the next available API key with rotation"""
        current_time = time.time()
        
        # Try to find a key that hasn't been used recently
        for i in range(len(self.api_keys)):
            key_index = (self.current_key_index + i) % len(self.api_keys)
            key = self.api_keys[key_index]
            
            # Check if enough time has passed since last use
            if current_time - self.key_last_used[key] >= 1:  # 1 second minimum between uses
                self.current_key_index = (key_index + 1) % len(self.api_keys)
                self.key_last_used[key] = current_time
                return key
        
        # If all keys used recently, wait and return the next one
        time.sleep(2)
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.key_last_used[key] = current_time
        return key
    
    def mark_rate_limited(self, api_key: str):
        """Mark a key as rate limited"""
        self.key_last_used[api_key] = time.time() + self.rate_limit_delay
        print(f"API key ending in ...{api_key[-8:]} is rate limited. Switching to next key.")

class EnhancedGeminiKeywordGenerator:
    def __init__(self, api_keys: List[str], model_name: str = 'gemini-2.0-flash-exp'):
        self.api_key_manager = APIKeyManager(api_keys)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.generated_keywords = set()
        self.clients = {}  # Cache clients for each API key
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('keyword_generation.log'),
                logging.StreamHandler()
            ]
        )
        
    def _get_client(self, api_key: str):
        """Get or create a client for the given API key"""
        if api_key not in self.clients:
            self.clients[api_key] = genai.Client(api_key=api_key)
        return self.clients[api_key]
    
    def _append_keywords_to_file(self, keywords: Set[str], filename: str):
        """Append new keywords to file immediately"""
        if not keywords:
            return
            
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                for keyword in keywords:
                    f.write(keyword + '\n')
            print(f"Appended {len(keywords)} keywords to {filename}")
        except Exception as e:
            self.logger.error(f"Error appending keywords to file: {e}")
    
    def _load_existing_keywords(self, filename: str) -> Set[str]:
        """Load existing keywords from file to avoid duplicates"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            return set()
        except Exception as e:
            self.logger.error(f"Error loading existing keywords: {e}")
            return set()
    
    def generate_30k_keywords(self, domain_url: str, output_file: str = 'input-keywords.txt') -> str:
        """
        Generate 30,000 keywords using the enhanced multi-stage approach with progress tracking
        """
        try:
            # Initialize progress tracking
            start_time = time.time()
            total_generated = 0
            
            # Load existing keywords to avoid duplicates
            existing_keywords = self._load_existing_keywords(output_file)
            self.generated_keywords.update(existing_keywords)
            
            print(f"Starting keyword generation for {domain_url}")
            print(f"Found {len(existing_keywords)} existing keywords")
            
            # Stage 1: Scrape domain content
            print("Stage 1: Scraping domain content...")
            scraper_response = Webscraper(domain_url)
            domain_content = scraper_response.text
            print(f"Scraped {len(domain_content)} characters of content")
            
            # Stage 2: Analyze domain content
            print("Stage 2: Analyzing domain content...")
            domain_analysis = self._analyze_domain_content(domain_content)
            print("Domain analysis completed")
            
            # Stage 3: Create keyword generation batches
            print("Stage 3: Creating keyword batches...")
            batches = self._create_keyword_batches(domain_analysis, domain_url)
            print(f"Created {len(batches)} keyword batches")
            
            # Stage 4: Generate keywords for each batch with progress tracking
            print("Stage 4: Generating keywords by batch...")
            for batch_idx, batch in enumerate(batches, 1):
                batch_start_time = time.time()
                print(f"\n--- Batch {batch_idx}/{len(batches)}: {batch.category.value} ---")
                print(f"Target: {batch.target_count} keywords")
                
                try:
                    batch_keywords = self._generate_keywords_for_batch_with_progress(
                        batch, domain_content, output_file
                    )
                    
                    # Update totals
                    new_keywords = batch_keywords - self.generated_keywords
                    self.generated_keywords.update(new_keywords)
                    total_generated += len(new_keywords)
                    
                    batch_time = time.time() - batch_start_time
                    print(f"Batch {batch_idx} completed: {len(new_keywords)} new keywords in {batch_time:.1f}s")
                    print(f"Total progress: {total_generated}/{30000} keywords ({total_generated/30000*100:.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx} ({batch.category.value}): {e}")
                    print(f"Batch {batch_idx} failed, continuing with next batch...")
                    continue
            
            # Final summary
            total_time = time.time() - start_time
            print(f"\n=== GENERATION COMPLETE ===")
            print(f"Total keywords generated: {total_generated}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average rate: {total_generated/total_time:.1f} keywords/second")
            print(f"Keywords saved to: {output_file}")
            
            return f"Generated {total_generated} keywords and saved to {output_file}"
            
        except Exception as e:
            self.logger.error(f"Error in main generation process: {e}")
            return f"Error: {e}"
    
    def _generate_keywords_for_batch_with_progress(self, batch: KeywordBatch, domain_content: str, output_file: str) -> Set[str]:
        """
        Generate keywords for a specific batch with progress tracking and immediate file writing
        """
        all_keywords = set()
        
        # Calculate smaller sub-batches for free API limits (250 keywords per call)
        sub_batch_size = 250
        num_sub_batches = (batch.target_count + sub_batch_size - 1) // sub_batch_size
        
        print(f"Splitting into {num_sub_batches} sub-batches of {sub_batch_size} keywords each")
        
        for i in range(num_sub_batches):
            current_target = min(sub_batch_size, batch.target_count - len(all_keywords))
            if current_target <= 0:
                break
                
            print(f"  Sub-batch {i+1}/{num_sub_batches}: Generating {current_target} keywords...")
            
            # Retry logic for each sub-batch
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    sub_keywords = self._generate_sub_batch_with_retry(
                        batch, current_target, i, domain_content
                    )
                    
                    # Remove duplicates and filter new keywords
                    new_keywords = sub_keywords - self.generated_keywords - all_keywords
                    
                    if new_keywords:
                        all_keywords.update(new_keywords)
                        # Immediately append to file
                        self._append_keywords_to_file(new_keywords, output_file)
                        print(f"    Generated {len(new_keywords)} new keywords")
                    else:
                        print(f"    No new keywords generated (all duplicates)")
                    
                    break  # Success, break retry loop
                    
                except Exception as e:
                    self.logger.error(f"Attempt {attempt+1} failed for sub-batch {i+1}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"    Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"    Sub-batch {i+1} failed after {max_retries} attempts")
            
            # Progress update
            progress = len(all_keywords) / batch.target_count * 100
            print(f"  Batch progress: {len(all_keywords)}/{batch.target_count} ({progress:.1f}%)")
            
            # Delay between sub-batches to respect rate limits
            time.sleep(2)
        
        return all_keywords
    
    def _generate_sub_batch_with_retry(self, batch: KeywordBatch, count: int, iteration: int, domain_content: str) -> Set[str]:
        """
        Generate a sub-batch of keywords with API key rotation and retry logic
        """
        variation_strategies = [
            "Focus on 1-2 word short-tail keywords",
            "Focus on 3-4 word mid-tail keywords", 
            "Focus on 5+ word long-tail keywords",
            "Focus on question-based and conversational keywords"
        ]
        
        current_strategy = variation_strategies[iteration % len(variation_strategies)]
        
        prompt = f"""
        You are a Keyword Research Specialist. Generate exactly {count} unique keywords for keyword research.
        
        CATEGORY: {batch.category.value}
        SEED TERMS: {', '.join(batch.seed_terms[:8])}
        CONTEXT: {batch.context}
        VARIATION STRATEGY: {current_strategy}
        
        DOMAIN CONTENT SAMPLE:
        {domain_content[:2000]}
        
        KEYWORD GENERATION RULES:
        - Generate exactly {count} unique keywords
        - Use actual terminology from the domain, not assumed phrases
        - Include variations of real terms found on the site
        - Add logical extensions of actual content themes
        - Cannot use the domain's own brand name
        - All keywords must relate to actual domain content
        - No artificially constructed phrases
        - No repetitive keyword stuffing patterns
        - Each keyword serves distinct search intent
        - Mix of 1-5 word phrases
        - Focus on generic/solution terms, some competitor brand names allowed
        - Prioritize problem-solution keywords based on actual site content
        
        OUTPUT FORMAT:
        Return only the keywords, one per line, no explanations, no headers, no numbering.
        
        Generate keywords based on what the domain actually offers and discusses, not theoretical user search patterns.
        """
        
        max_api_retries = len(self.api_key_manager.api_keys) * 2
        
        for api_attempt in range(max_api_retries):
            try:
                # Get next API key
                api_key = self.api_key_manager.get_next_key()
                client = self._get_client(api_key)
                
                # Make API call
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                # Parse response
                keywords = self._parse_keyword_response(response.text)
                
                if len(keywords) > 0:
                    return keywords
                else:
                    print(f"    Empty response from API, retrying...")
                    continue
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit errors
                if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                    self.api_key_manager.mark_rate_limited(api_key)
                    print(f"    Rate limited, switching API key...")
                    time.sleep(5)
                    continue
                    
                # Check for other API errors
                elif 'api' in error_msg or 'request' in error_msg:
                    print(f"    API error: {e}")
                    time.sleep(3)
                    continue
                    
                else:
                    # Non-API error, re-raise
                    raise e
        
        # If we get here, all API attempts failed
        self.logger.error(f"All API attempts failed for sub-batch")
        return set()
    
    def _analyze_domain_content(self, content: str) -> Dict:
        """
        Analyze domain content to extract structured data using Gemini with retry logic
        """
        analysis_prompt = f"""
        Analyze this domain content and extract structured data for keyword generation.
        
        CONTENT:
        {content[:6000]}  # Truncate for token limits
        
        Extract and return ONLY a JSON object with these exact keys:
        {{
            "core_services": ["service1", "service2"],
            "problems_mentioned": ["problem1", "problem2"],
            "technologies": ["tech1", "tech2"],
            "competitors_referenced": ["comp1", "comp2"],
            "features_described": ["feature1", "feature2"],
            "processes_explained": ["process1", "process2"],
            "industry_terms": ["term1", "term2"],
            "target_audience": ["audience1", "audience2"],
            "value_propositions": ["value1", "value2"],
            "use_cases": ["case1", "case2"]
        }}
        
        Only include terms actually found or strongly implied in the content.
        Return ONLY the JSON, no explanations.
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                api_key = self.api_key_manager.get_next_key()
                client = self._get_client(api_key)
                
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=analysis_prompt
                )
                
                # Clean the response text
                response_text = response.text.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                else:
                    if attempt < max_retries - 1:
                        print(f"JSON parsing failed, retrying... (attempt {attempt + 1})")
                        time.sleep(2)
                        continue
                    else:
                        return self._create_fallback_analysis(content)
                        
            except Exception as e:
                self.logger.error(f"Error analyzing domain content (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    return self._create_fallback_analysis(content)
    
    def _create_keyword_batches(self, analysis: Dict, domain_url: str) -> List[KeywordBatch]:
        """
        Create strategic batches for keyword generation
        """
        batches = []
        
        # Core Services Batch (High Priority)
        batches.append(KeywordBatch(
            category=KeywordCategory.CORE_SERVICES,
            seed_terms=analysis.get("core_services", [])[:10],
            target_count=4000,
            context=f"Generate keywords for core services offered by {domain_url}. Focus on direct service terms, variations, and related searches.",
            variation_strategy="Focus on 1-3 word service-related keywords"
        ))
        
        # Problem-Solution Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.PROBLEM_STATEMENTS,
            seed_terms=analysis.get("problems_mentioned", [])[:8],
            target_count=3500,
            context="Generate keywords around problems users are trying to solve. Include pain points, challenges, and solution-seeking terms.",
            variation_strategy="Focus on problem-solution and pain point keywords"
        ))
        
        # Technology & Features Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.TECHNOLOGY_TERMS,
            seed_terms=analysis.get("technologies", [])[:8] + analysis.get("features_described", [])[:8],
            target_count=3000,
            context="Generate technical keywords, feature-specific terms, and technology-related searches.",
            variation_strategy="Focus on technical and feature-specific terminology"
        ))
        
        # Long-tail Variations Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.LONG_TAIL_VARIATIONS,
            seed_terms=analysis.get("use_cases", [])[:10],
            target_count=5000,
            context="Generate long-tail keyword variations, specific use cases, and niche applications.",
            variation_strategy="Focus on 4-5 word long-tail variations"
        ))
        
        # Competitor Analysis Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.COMPETITOR_ANALYSIS,
            seed_terms=analysis.get("competitors_referenced", [])[:5],
            target_count=2500,
            context="Generate competitor-related keywords, comparison terms, and alternative solution searches.",
            variation_strategy="Focus on competitor names and comparison terms"
        ))
        
        # Industry Terms Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.INDUSTRY_TERMINOLOGY,
            seed_terms=analysis.get("industry_terms", [])[:10],
            target_count=3000,
            context="Generate industry-specific terminology, jargon, and professional language variations.",
            variation_strategy="Focus on industry jargon and professional terms"
        ))
        
        # Process & How-to Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.PROCESS_RELATED,
            seed_terms=analysis.get("processes_explained", [])[:8],
            target_count=2500,
            context="Generate process-related keywords, how-to searches, and procedural terms.",
            variation_strategy="Focus on process and how-to keywords"
        ))
        
        # Alternative Phrasings Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.ALTERNATIVE_PHRASINGS,
            seed_terms=analysis.get("value_propositions", [])[:8],
            target_count=2500,
            context="Generate alternative phrasings, synonyms, and different ways to express the same concepts.",
            variation_strategy="Focus on synonyms and alternative phrasings"
        ))
        
        # Educational Terms Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.EDUCATIONAL_TERMS,
            seed_terms=analysis.get("industry_terms", [])[:5] + analysis.get("technologies", [])[:5],
            target_count=2000,
            context="Generate educational keywords, learning-focused terms, and informational searches.",
            variation_strategy="Focus on educational and informational terms"
        ))
        
        # Audience-specific Batch
        batches.append(KeywordBatch(
            category=KeywordCategory.LONG_TAIL_VARIATIONS,
            seed_terms=analysis.get("target_audience", [])[:8],
            target_count=2000,
            context="Generate audience-specific keywords and remaining variations to reach target.",
            variation_strategy="Focus on audience-specific and niche terms"
        ))
        
        return batches
    
    def _parse_keyword_response(self, response: str) -> Set[str]:
        """
        Parse Gemini response into clean keyword set
        """
        lines = response.strip().split('\n')
        keywords = set()
        
        for line in lines:
            # Clean the line
            keyword = line.strip().lower()
            
            # Remove numbering, bullets, quotes
            keyword = self._clean_keyword(keyword)
            
            # Validate keyword
            if self._is_valid_keyword(keyword):
                keywords.add(keyword)
        
        return keywords
    
    def _clean_keyword(self, keyword: str) -> str:
        """
        Clean and normalize keyword
        """
        # Remove common prefixes and suffixes
        prefixes = ['- ', '• ', '* ', '"', "'", '`']
        for i in range(1, 21):  # Remove numbered lists 1-20
            prefixes.append(f'{i}. ')
            prefixes.append(f'{i}) ')
        
        for prefix in prefixes:
            if keyword.startswith(prefix):
                keyword = keyword[len(prefix):]
        
        # Remove trailing punctuation
        keyword = keyword.rstrip('.,;:"\'`')
        
        return keyword.strip()
    
    def _is_valid_keyword(self, keyword: str) -> bool:
        """
        Validate if keyword meets quality criteria
        """
        if not keyword or len(keyword) < 3:
            return False
        
        # Check word count (1-5 words)
        word_count = len(keyword.split())
        if word_count < 1 or word_count > 5:
            return False
        
        # Avoid obvious spam patterns
        if keyword.count(' ') > 4:  # Too many words
            return False
        
        # Check for repetitive patterns
        words = keyword.split()
        if len(words) > 1 and len(words) != len(set(words)):  # Duplicate words
            return False
        
        # Avoid keywords that are too short or too long
        if len(keyword) < 3 or len(keyword) > 80:
            return False
        
        return True
    
    def _create_fallback_analysis(self, content: str) -> Dict:
        """
        Create fallback analysis if JSON parsing fails
        """
        # Simple keyword extraction from content
        words = content.lower().split()
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'a', 'an', 'this', 'that', 'these', 'those'}
        
        # Extract potential terms
        potential_terms = []
        for word in words:
            if len(word) > 3 and word not in common_words and word.isalpha():
                potential_terms.append(word)
        
        # Remove duplicates and take unique terms
        unique_terms = list(set(potential_terms))
        
        # Create basic analysis
        return {
            "core_services": unique_terms[:10],
            "problems_mentioned": unique_terms[10:18],
            "technologies": unique_terms[18:26],
            "competitors_referenced": unique_terms[26:31],
            "features_described": unique_terms[31:39],
            "processes_explained": unique_terms[39:47],
            "industry_terms": unique_terms[47:55],
            "target_audience": unique_terms[55:63],
            "value_propositions": unique_terms[63:71],
            "use_cases": unique_terms[71:81]
        }

# Enhanced generation function with multiple API keys
def generate_30k_keywords(domain_url: str, api_keys: List[str], output_file: str = 'input-keywords.txt'):
    """
    Generate 30,000 keywords using multiple API keys with rotation
    """
    generator = EnhancedGeminiKeywordGenerator(api_keys=api_keys)
    result = generator.generate_30k_keywords(domain_url, output_file)
    return result

# Backward compatibility - original function signature
def generate(domain_url: str, api_key: str = "", output_file: str = 'input-keywords.txt'):
    """
    Original function signature for backward compatibility
    Enhanced to generate 30k keywords instead of 200
    """
    if not api_key:
        raise ValueError("API key is required")
    
    # Convert single API key to list for compatibility
    api_keys = [api_key]
    return generate_30k_keywords(domain_url, api_keys, output_file)

# Multi-API key function
def generate_with_multiple_keys(domain_url: str, api_keys: List[str], output_file: str = 'input-keywords.txt'):
    """
    Generate 30k keywords using multiple API keys
    """
    if not api_keys:
        raise ValueError("At least one API key is required")
    
    return generate_30k_keywords(domain_url, api_keys, output_file)

# Example usage
if __name__ == "__main__":
    # Replace with your actual API keys
    API_KEYS = [
        "", # magnicetio
    ]
    
    # Generate 30k keywords with multiple API keys
    result = generate_with_multiple_keys("https://sakana.ai", API_KEYS)
    print(result)