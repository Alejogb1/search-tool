import google.generativeai as genai
from google.generativeai import types
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
import os
from dotenv import load_dotenv

load_dotenv()

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
        self.key_rate_limited_until = {}
        self.rate_limit_delay = 60  # seconds to wait after rate limit
        self.min_interval = 1  # minimum seconds between uses of same key
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking for each key
        for key in api_keys:
            self.key_usage_count[key] = 0
            self.key_last_used[key] = 0
            self.key_rate_limited_until[key] = 0
    
    def get_next_key(self) -> str:
        """Get the next available API key with intelligent rotation"""
        current_time = time.time()
        best_key = None
        min_usage = float('inf')
        
        # Find the best available key based on usage and rate limits
        for i in range(len(self.api_keys)):
            key_index = (self.current_key_index + i) % len(self.api_keys)
            key = self.api_keys[key_index]
            
            # Skip if key is currently rate limited
            if current_time < self.key_rate_limited_until[key]:
                continue
                
            # Check minimum interval between uses
            if current_time - self.key_last_used[key] < self.min_interval:
                continue
                
            # Prefer key with lowest usage count
            if self.key_usage_count[key] < min_usage:
                best_key = key
                min_usage = self.key_usage_count[key]
        
        if best_key:
            self.current_key_index = (self.api_keys.index(best_key) + 1) % len(self.api_keys)
            self.key_last_used[best_key] = current_time
            self.key_usage_count[best_key] += 1
            self.logger.debug(f"Selected API key ending in ...{best_key[-8:]} (usage: {self.key_usage_count[best_key]})")
            return best_key
        
        # If no key available, wait and try again
        self.logger.warning("All API keys in use or rate limited - waiting before retry")
        time.sleep(2)
        return self.get_next_key()
    
    def mark_rate_limited(self, api_key: str):
        """Mark a key as rate limited and update tracking"""
        current_time = time.time()
        self.key_rate_limited_until[api_key] = current_time + self.rate_limit_delay
        self.key_last_used[api_key] = current_time  # Reset last used time
        self.logger.warning(f"API key ending in ...{api_key[-8:]} rate limited until {self.key_rate_limited_until[api_key]}")

class EnhancedGeminiKeywordGenerator:
    def __init__(self, api_keys: List[str], model_configs: List[Dict] = None, parallel: bool = True, mode: str = "commercial", context_source: str = "domain", context_file: str = None):
        self.api_key_manager = APIKeyManager(api_keys)
        self.model_configs = model_configs if model_configs else [{'name': 'gemini-2.0-flash-exp', 'rpm': 200}]
        self.current_model_config = self.model_configs[0]
        self.logger = logging.getLogger(__name__)
        self.generated_keywords = set()
        self.clients = {}  # Cache clients for each API key
        self.file_lock = asyncio.Lock()  # For thread-safe file operations
        self.keyword_lock = asyncio.Lock()  # For thread-safe keyword tracking
        self.model_request_counts = {} # Track requests per minute for each model
        self.parallel = parallel
        self.mode = mode
        self.context_source = context_source
        self.context_file = context_file
        for cfg in self.model_configs:
            self.model_request_counts[cfg['name']] = []
        
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
        """Get or create a client for the given API key using the current model configuration"""
        model_name = self.current_model_config['name']
        if api_key not in self.clients:
            self.clients[api_key] = {}
        
        if model_name not in self.clients[api_key]:
            genai.configure(api_key=api_key)
            self.clients[api_key][model_name] = genai.GenerativeModel(model_name)
        return self.clients[api_key][model_name]

    def _can_make_request(self) -> bool:
        """Check if a request can be made with the current model without exceeding RPM."""
        model_name = self.current_model_config['name']
        rpm_limit = self.current_model_config['rpm']
        current_time = time.time()

        # Clean up old requests (older than 60 seconds)
        self.model_request_counts[model_name] = [
            t for t in self.model_request_counts.get(model_name, []) if current_time - t < 60
        ]

        if len(self.model_request_counts[model_name]) < rpm_limit:
            return True
        else:
            self.logger.warning(f"Model {model_name} RPM limit reached ({rpm_limit} RPM).")
            # Automatically switch to next model when limit reached
            self._switch_model()
            return False

    def _record_request(self):
        """Record a successful request for the current model."""
        model_name = self.current_model_config['name']
        self.model_request_counts[model_name].append(time.time())

    def _switch_model(self):
        """Switch to the next available model with proper cooldown."""
        current_model_index = self.model_configs.index(self.current_model_config)
        next_model_index = (current_model_index + 1) % len(self.model_configs)
        self.current_model_config = self.model_configs[next_model_index]
        
        # Reset request count for new model
        self.model_request_counts[self.current_model_config['name']] = []
        
        # Add cooldown period if previous model was rate limited
        cooldown = 5 if len(self.model_request_counts) > 1 else 0
        time.sleep(cooldown)
        
        self.logger.info(f"Switched to model: {self.current_model_config['name']} (RPM: {self.current_model_config['rpm']})")
    
    async def _append_keywords_to_file_async(self, keywords: Set[str], filename: str):
        """Thread-safe version of append_keywords_to_file"""
        if not keywords:
            return
            
        try:
            async with self.file_lock:
                with open(filename, 'a', encoding='utf-8') as f:
                    for keyword in keywords:
                        f.write(keyword + '\n')
                print(f"Appended {len(keywords)} keywords to {filename}")
        except Exception as e:
            self.logger.error(f"Error appending keywords to file: {e}")
            
    def _append_keywords_to_file(self, keywords: Set[str], filename: str):
        """Sync version for backward compatibility"""
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
    
    async def generate_30k_keywords_async(self, domain_url: str = None, output_file: str = 'input-keywords.txt') -> str:
        """
        Async version of generate_30k_keywords with parallel processing
        """
        # Ensure the output file exists
        with open(output_file, 'a') as f:
            pass
            
        try:
            # Initialize progress tracking
            start_time = time.time()
            total_generated = 0
            
            # Load existing keywords to avoid duplicates
            existing_keywords = self._load_existing_keywords(output_file)
            self.generated_keywords.update(existing_keywords)
            
            print(f"\n=== KEYWORD GENERATION STARTED ===")
            print(f"Target: 30,000 keywords")
            print(f"Existing keywords: {len(existing_keywords)}")
            print(f"API keys available: {len(self.api_key_manager.api_keys)}")
            print(f"Models available: {len(self.model_configs)}")
            print(f"Mode: {self.mode}, Context source: {self.context_source}")
            
            # Load content based on context source
            if self.context_source == "file" and self.context_file:
                print(f"Loading context from file: {self.context_file}")
                with open(self.context_file, 'r') as f:
                    domain_content = f.read()
                print(f"Loaded {len(domain_content)} characters from file")
                context_name = self.context_file
            else:
                if not domain_url:
                    raise ValueError("Domain URL is required when using domain context source")
                print(f"Stage 1: Scraping domain content from {domain_url}...")
                scraper_response = Webscraper(domain_url)
                scraper_response2 = Webscraper(domain_url)
                domain_content = scraper_response.text
                print(f"Scraped {len(domain_content)} characters of content")
                context_name = domain_url

            if self.mode == "scientific":
                print("Scientific mode: Using content without analysis")
                # Create a single batch with the entire content
                if self.context_source == "file":
                    print("Context source is file")
                    context_desc = f"Generate frontier-level research terms from the provided file: {self.context_file}"
                else:
                    print("Context source is domain")
                    context_desc = f"Generate frontier-level research terms from domain: {domain_url}"
                
                scientific_batch = KeywordBatch(
                    category=KeywordCategory.CORE_SERVICES,
                    seed_terms=[],
                    target_count=30000,
                    context=context_desc,
                    variation_strategy="Focus on long-tail, exploratory research phrases."
                )
                batches = [scientific_batch]
            else:
                # Commercial mode
                if self.context_source == "domain":
                    print("Stage 2: Analyzing domain content...")
                    domain_analysis = await self._analyze_domain_content(domain_content)
                    print("Domain analysis completed")
                    
                    print("Stage 3: Creating keyword batches...")
                    batches = self._create_keyword_batches(domain_analysis, context_name)
                else:
                    # For file-based context in commercial mode, use simplified batch
                    print("Creating keyword batches from file content...")
                    commercial_batch = KeywordBatch(
                        category=KeywordCategory.CORE_SERVICES,
                        seed_terms=[],
                        target_count=30000,
                        context="Generate commercial keywords from the provided text.",
                        variation_strategy="Focus on product and service keywords"
                    )
                    batches = [commercial_batch]
                
                print(f"Created {len(batches)} keyword batches")

            # Generate keywords for each batch
            print("Stage 4: Generating keywords by batch...")
            for batch_idx, batch in enumerate(batches, 1):
                batch_start_time = time.time()
                print(f"\n--- Batch {batch_idx}/{len(batches)}: {batch.category.value} ---")
                print(f"Target: {batch.target_count} keywords")
                
                try:
                    batch_keywords = await self._generate_keywords_for_batch_with_progress(
                        batch, domain_content, output_file, domain_url
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
            minutes, seconds = divmod(total_time, 60)
            hours, minutes = divmod(minutes, 60)
            
            self.logger.info(f"Main generation process finished. Total time: {total_time:.1f}s")
            print(f"\n=== GENERATION COMPLETE ===")
            print(f"╔════════════════════════════════════╗")
            print(f"║          KEYWORD GENERATION        ║")
            print(f"╠════════════════════════════════════╣")
            print(f"║ Total keywords generated: {str(total_generated).ljust(8)} ║")
            print(f"║ Existing keywords:       {str(len(existing_keywords)).ljust(8)} ║")
            print(f"║ New unique keywords:     {str(total_generated - len(existing_keywords)).ljust(8)} ║")
            print(f"║ Time elapsed:            {f'{int(hours)}h {int(minutes)}m {int(seconds)}s'.ljust(8)} ║")
            print(f"║ Generation rate:         {f'{total_generated/total_time:.1f} keywords/s'.ljust(8)} ║")
            print(f"║ Output file:             {output_file.ljust(8)} ║")
            print(f"╚════════════════════════════════════╝")
            
            return (
                f"Generated {total_generated} keywords ({total_generated - len(existing_keywords)} new unique)\n"
                f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
                f"Rate: {total_generated/total_time:.1f} keywords/s\n"
                f"Saved to: {output_file}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in main generation process: {e}")
            return f"Error: {e}"
        finally:
            self.logger.info("Script finished.")

    def generate_30k_keywords(self, domain_url: str, output_file: str = 'input-keywords.txt') -> str:
        """
        Generate 30,000 keywords using the enhanced multi-stage approach with progress tracking
        """
        return asyncio.run(self.generate_30k_keywords_async(domain_url, output_file))
    
    async def _generate_keywords_for_batch_with_progress(self, batch: KeywordBatch, domain_content: str, output_file: str, domain_url: str) -> Set[str]:
        """
        Generate keywords for a specific batch with progress tracking and immediate file writing (parallel version)
        """
        all_keywords = set()
        failed_sub_batches = 0
        
        # Calculate smaller sub-batches for free API limits (8000 tokens per call)
        sub_batch_size = 3000
        num_sub_batches = (batch.target_count + sub_batch_size - 1) // sub_batch_size
        
        print(f"\nProcessing batch: {batch.category.value}")
        print(f"Target keywords: {batch.target_count}")
        print(f"Splitting into {num_sub_batches} sub-batches of {sub_batch_size} keywords each")
        
        # Create a semaphore to limit concurrent requests based on available API keys
        semaphore = asyncio.Semaphore(len(self.api_key_manager.api_keys))
        
        async def process_sub_batch(i: int):
            nonlocal all_keywords, failed_sub_batches
            current_target = min(sub_batch_size, batch.target_count - len(all_keywords))
            if current_target <= 0:
                return
                
            print(f"\n  Sub-batch {i+1}/{num_sub_batches}: Generating {current_target} keywords...")
            
            # Retry logic for each sub-batch
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        sub_keywords = await self._generate_sub_batch_with_retry_async(
                            batch, current_target, i, domain_content, domain_url
                        )
                    
                    # Thread-safe keyword tracking
                    async with self.keyword_lock:
                        new_keywords = sub_keywords - self.generated_keywords - all_keywords
                        if new_keywords:
                            all_keywords.update(new_keywords)
                            # Thread-safe file writing
                            await self._append_keywords_to_file_async(new_keywords, output_file)
                            print(f"    ✅ Generated {len(new_keywords)} new keywords")
                        else:
                            print(f"    ⚠️ No new keywords generated (all duplicates)")
                    
                    return  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    self.logger.error(f"Attempt {attempt+1} failed for sub-batch {i+1}: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"    ⏳ Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
            
            # If we get here, all attempts failed
            failed_sub_batches += 1
            print(f"    ❌ Sub-batch {i+1} failed after {max_retries} attempts")
            if last_error:
                print(f"    Last error: {str(last_error)}")
            
            # Progress update
            progress = len(all_keywords) / batch.target_count * 100
            print(f"\n  Batch progress: {len(all_keywords)}/{batch.target_count} keywords ({progress:.1f}%)")
            print(f"  Failed sub-batches: {failed_sub_batches}/{num_sub_batches}")

        # Run sub-batches concurrently or sequentially based on the `parallel` flag
        if self.parallel:
            await asyncio.gather(*[process_sub_batch(i) for i in range(num_sub_batches)])
        else:
            for i in range(num_sub_batches):
                await process_sub_batch(i)
        
        self.logger.info(f"Finished processing batch {batch.category.value}. Total keywords: {len(all_keywords)}")
        return all_keywords
    
    def _get_commercial_prompt(self, batch: KeywordBatch, count: int, iteration: int, domain_content: str) -> str:
        variation_strategies = [
            "Focus on 1-2 word short-tail keywords",
            "Focus on 3-4 word mid-tail keywords",
            "Focus on 5+ word long-tail keywords",
            "Focus on question-based and conversational keywords"
        ]
        current_strategy = variation_strategies[iteration % len(variation_strategies)]
        return f"""
        You are a Keyword Research Specialist. Generate exactly {count} unique keywords for keyword research.
        
        CATEGORY: {batch.category.value}
        SEED TERMS: {', '.join(batch.seed_terms[:8])}
        CONTEXT: {batch.context}
        VARIATION STRATEGY: {current_strategy}
        
        DOMAIN CONTENT SAMPLE:
        {domain_content[:500000]}
        
        STRICT SEARCH TERM REQUIREMENTS:
        1. Must be actual search query patterns (not statements or commentary)
        2. No special characters except hyphens and apostrophes 
        3. No colons or semantic separators
        4. Must resemble real Google search queries
        5. Reject any terms similar to these problematic patterns:
           - "freelancers: ¿cómo recibir pagos?"
           - "saldo sin fronteras: seguridad"  
           - "tipo de cambio: análisis"
           - "¿cómo usar usd digitales?"
        
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
        - Mix of 1-5 word phrases (no more than 5 words)
        - Focus on generic/solution terms, some competitor brand names allowed
        - Prioritize problem-solution keywords based on actual site content
        - Questions are allowed but without explicit question marks
        - Must be directly searchable in Google Ads
        
        OUTPUT FORMAT:
        Return only the keywords, one per line, no explanations, no headers, no numbering.
        
        VALIDATION CRITERIA (ask yourself):
        - Would someone actually search this on Google?
        - Does this clearly relate to a product/service?
        - Is this specific enough to trigger relevant ads?
        
        Generate keywords based on what the domain actually offers and discusses, not theoretical user search patterns.
        """

    def _get_scientific_prompt(self, batch: KeywordBatch, count: int, iteration: int, domain_content: str) -> str:
        return f"""
    You are a Senior Scientific Research Specialist and Frontier Research Analyst.
    Your task is to generate exactly {count} **unique, long-tail, frontier-level research terms or phrases** that are semantically rich, academically rigorous, and suitable for advanced discovery across any scientific domain.

    CATEGORY: {batch.category.value}
    CONTEXT: {batch.context} + {domain_content[:500000]}

    INSTRUCTIONS AND REQUIREMENTS:
    1. Focus on generating terms that reflect **novel, open-ended, and exploratory research directions**, probing areas not fully defined or experimentally investigated.  
    2. Include **technical terminology, research methodologies, experimental design, mechanistic modeling, theoretical frameworks, empirical evaluation, and validation techniques**.  
    3. Integrate **semantic qualifiers that imply authority and rigor**, such as peer-reviewed, validated, reproducible, high-impact, recognized laboratory, canonical study, or foundational research.  
    4. Encourage terms that **span interdisciplinary and hybrid approaches**, capturing emergent phenomena, adaptive behaviors, co-evolution, multi-component interactions, and complex system dynamics.  
    5. Include **meta-level or conceptual terms** (e.g., self-organizing structures, recursive evaluation frameworks, hierarchical emergent behaviors, latent representation dynamics).  
    6. Terms may imply **experimental, computational, observational, or procedural implementations**, without being restricted to any specific domain.  
    7. Prioritize **accuracy, academic relevance, and novelty** over popularity or common usage.  
    8. Use **long-tail phrases**, typically 3–4 words, but you can also use shorter terms 1-3 words (half long and half short distribution) that are semantically precise, descriptive, and rich in context.  
    9. Avoid **commercial, marketing, tutorial-style, or superficial terms**; focus solely on scientific, technical, or experimental language.  
    10. Encourage **open-ended inquiry and discovery**, allowing terms to suggest uncharted areas, emerging mechanisms, or unexplored interactions.  
    11. Integrate the **combined layers of exploration, academic rigor, and authority** so that each term communicates both novelty and reliability.  
    12. Terms should be suitable for use in **scholarly databases, peer-reviewed publications, technical reports, or frontier experimental studies**.  
    13. Emphasize **mechanistic, empirical, or formal analysis perspectives**, while allowing conceptual and theoretical framing.  
    14. Where appropriate, include **phrases suggesting adaptive systems, multi-agent interactions, emergent behaviors, or discovery-oriented procedural frameworks**.  

    OUTPUT FORMAT:
    - Return only the terms or phrases.
    - One term per line.
    - No numbering, bullet points, or explanations.
    """

    async def _generate_sub_batch_with_retry_async(self, batch: KeywordBatch, count: int, iteration: int, domain_content: str, domain_url: str) -> Set[str]:
        """
        Generate a sub-batch of keywords with API key rotation and retry logic (async version)
        """
        start_time = time.time()
        
        if self.mode == "scientific":
            prompt = self._get_scientific_prompt(batch, count, iteration, domain_content)
        else:
            prompt = self._get_commercial_prompt(batch, count, iteration, domain_content)

        max_api_retries = len(self.api_key_manager.api_keys) * len(self.model_configs) * 2 # Consider retries for each model
        
        for api_attempt in range(max_api_retries):
            try:
                # Check and switch model if RPM limit is reached
                if not self._can_make_request():
                    self._switch_model()
                    self.logger.info(f"Switched to model: {self.current_model_config['name']}")
                    await asyncio.sleep(5) # Wait a bit before trying again with the new model
                    continue

                # Get next API key
                api_key = self.api_key_manager.get_next_key()
                client = self._get_client(api_key)
                
                # Make async API call
                response = await asyncio.to_thread(
                    client.generate_content,
                    contents=prompt
                )
                
                # Record the request regardless of the response
                self._record_request()

                # Parse response
                if response and response.text:
                    keywords = self._parse_keyword_response(response.text)
                    if len(keywords) > 0:
                        return keywords
                
                # Handle empty or invalid response
                self.logger.warning(f"Empty/invalid response from API for model {self.current_model_config['name']}")
                self._switch_model()
                self.logger.info(f"Switched to model: {self.current_model_config['name']} after empty response")
                await asyncio.sleep(5)
                continue
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit errors
                if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                    self.api_key_manager.mark_rate_limited(api_key)
                    print(f"    Rate limited, switching API key...")
                    await asyncio.sleep(5)
                    continue
                    
                # Check for timeout errors
                elif '504' in error_msg or 'deadline' in error_msg:
                    self._switch_model()
                    self.logger.warning(f"Timeout error, switched to model: {self.current_model_config['name']}")
                    await asyncio.sleep(5)
                    continue
                    
                # Check for other API errors
                elif 'api' in error_msg or 'request' in error_msg:
                    self._switch_model()
                    self.logger.warning(f"API error, switched to model: {self.current_model_config['name']}")
                    await asyncio.sleep(3)
                    continue
                    
                else:
                    # Non-API error, re-raise
                    raise e
        
        # If we get here, all API attempts failed
        elapsed = time.time() - start_time
        self.logger.error(f"All API attempts failed for sub-batch after {elapsed:.1f}s")
        self.logger.error(f"Failed batch details: Category={batch.category.value}, Target={count}, Strategy={current_strategy}")
        return set()
    
    async def _analyze_domain_content(self, content: str) -> Dict:
        """
        Analyze domain content to extract structured data using Gemini with retry logic
        """
        analysis_prompt = f"""
        Analyze this domain content and extract structured data for keyword generation.
        
        CONTENT:
        {content:500000}  # Truncate for token limits
        
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
                
                response = await asyncio.to_thread(
                    client.generate_content,
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
                        await asyncio.sleep(2)
                        continue
                    else:
                        return self._create_fallback_analysis(content)
                        
            except Exception as e:
                self.logger.error(f"Error analyzing domain content (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
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
            variation_strategy="Focus on 1-8 word service-related keywords"
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
            variation_strategy="Focus on 4-8 word long-tail variations"
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
        Parse Gemini response into clean keyword set with commercial optimization
        """
        lines = response.strip().split('\n')
        keywords = set()
        
        for line in lines:
            # Clean the line
            keyword = line.strip().lower()
            keyword = self._clean_keyword(keyword)
            
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
        Validate if keyword meets quality criteria based on the current mode.
        """
        word_count = len(keyword.split())
        if word_count < 1:
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
def generate_30k_keywords(domain_url: str = None, api_keys: List[str] = None, model_configs: List[Dict] = None, output_file: str = 'input-keywords.txt', parallel: bool = True, mode: str = "commercial", context_source: str = "domain", context_file: str = None):
    """
    Generate 30,000 keywords using multiple API keys with rotation
    """
    generator = EnhancedGeminiKeywordGenerator(
        api_keys=api_keys, 
        model_configs=model_configs, 
        parallel=parallel, 
        mode=mode,
        context_source=context_source,
        context_file=context_file
    )
    if parallel:
        result = asyncio.run(generator.generate_30k_keywords_async(domain_url, output_file))
    else:
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
    return generate_30k_keywords(domain_url=domain_url, api_keys=api_keys, output_file=output_file)

# Multi-API key function
def generate_with_multiple_keys(domain_url: str = None, api_keys: List[str] = None, model_configs: List[Dict] = None, output_file: str = 'input-keywords.txt', parallel: bool = True, mode: str = "commercial", context_source: str = "domain", context_file: str = None):
    """
    Generate 30k keywords using multiple API keys
    """
    if not api_keys:
        raise ValueError("At least one API key is required")
    
    return generate_30k_keywords(
        domain_url=domain_url,
        api_keys=api_keys,
        model_configs=model_configs,
        output_file=output_file,
        parallel=parallel,
        mode=mode,
        context_source=context_source,
        context_file=context_file
    )

# Example usage
if __name__ == "__main__":
    # Load API keys from .env file
    api_keys_str = os.getenv("GOOGLE_API_KEYS")
    if not api_keys_str:
        print("ERROR: GOOGLE_API_KEYS not found in .env file.")
        print("Please add your keys as a comma-separated list.")
        exit(1)
    
    API_KEYS = [key.strip() for key in api_keys_str.split(',')]
    
    MODEL_CONFIGS = [
        {"name": "gemini-2.5-pro", "rpm": 5},
        {"name": "gemini-2.5-flash", "rpm": 10},
        {"name": "gemini-2.5-flash-lite-preview-06-17", "rpm": 15},
        {"name": "gemini-2.0-flash", "rpm": 15},
        {"name": "gemini-2.0-flash-lite", "rpm": 30},
    ]
    
    # Generate 30k keywords with multiple API keys in parallel
    # Commercial mode example
    # result = generate_with_multiple_keys("https://synagroweb.com", API_KEYS, MODEL_CONFIGS, parallel=False, mode="commercial")
    # print(result)

    # Scientific mode examples
    # Domain-based    
    # File-based
    file_result = generate_with_multiple_keys(
        context_source="file",
        context_file="context.txt",
        api_keys=API_KEYS, 
        model_configs=MODEL_CONFIGS, 
        output_file='input-keywords-P-.txt', 
        domain_url=None,
        parallel=False, 
        mode="commercial"
    )
    print(file_result)
    
    # To run sequentially, set parallel=False
    # result_seq = generate_with_multiple_keys("https://agentazlon.com", API_KEYS, MODEL_CONFIGS, parallel=False)
    # print(result_seq)
