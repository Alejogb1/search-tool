import google.generativeai as genai
from openai import OpenAI
import asyncio
import time
import logging
import os
from typing import List, Set, Dict
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# OpenRouter models for fallback - free models in order of preference
OPENROUTER_MODELS = [
    {"name": "deepseek/deepseek-r1-0528:free", "rpm": 30},
    {"name": "google/gemini-2.5-pro-exp-03-25", "rpm": 15},
    {"name": "qwen/qwen3-235b-a22b:free", "rpm": 30},
]

class ExpansionStrategy(Enum):
    NATURAL_QUESTIONS = "natural_questions"  # What/how/why people actually search
    COMPARISONS = "comparisons"  # vs, versus, difference between
    IMPLEMENTATION_GUIDES = "implementation_guides"  # tutorials, guides, how-to
    ESSENTIAL_TROUBLESHOOTING = "essential_troubleshooting"  # critical errors only

@dataclass
class ExpansionBatch:
    seed_keyword: str
    strategy: ExpansionStrategy
    target_count: int

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
            self.logger.debug(f"Selected API key ending in ...{best_key[-8:]} (usage: {self.key_usage_count[best_key]})")
            return best_key
        
        self.logger.warning("All API keys in use or rate limited - waiting before retry")
        time.sleep(2)
        return self.get_next_key()
    
    def mark_rate_limited(self, api_key: str):
        current_time = time.time()
        self.key_rate_limited_until[api_key] = current_time + self.rate_limit_delay
        self.key_last_used[api_key] = current_time
        self.logger.warning(f"API key ending in ...{api_key[-8:]} rate limited until {self.key_rate_limited_until[api_key]}")

class KeywordExpander:
    def __init__(self, api_keys: List[str], model_configs: List[Dict] = None, parallel: bool = True):
        self.api_key_manager = APIKeyManager(api_keys)
        self.model_configs = model_configs if model_configs else [{'name': 'gemini-2.0-flash-exp', 'rpm': 200}]
        self.current_model_config = self.model_configs[0]
        self.logger = logging.getLogger(__name__)
        self.expanded_keywords = set()
        self.clients = {}
        self.file_lock = asyncio.Lock()
        self.keyword_lock = asyncio.Lock()
        self.model_request_counts = {}
        self.parallel = parallel
        
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
                logging.FileHandler('keyword_expansion.log'),
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

    def _get_expansion_prompt(self, keyword: str, strategy: ExpansionStrategy, count: int) -> str:
        """Generate expansion prompt based on strategy - focused on natural user search patterns"""

        strategy_prompts = {
            ExpansionStrategy.NATURAL_QUESTIONS: f"""
Based on "{keyword}", generate exactly {count} natural questions that real people actually search for.

Focus on genuine curiosity and learning:
- "what is..." - fundamental understanding
- "how does... work" - mechanism explanation
- "why does..." - cause and effect
- "when should I..." - practical application
- "where can I..." - resource location

AVOID artificial academic questions. Generate queries that sound like real Google searches from confused but motivated learners.

Example for "transformer attention":
what is attention in transformers
how does attention work in neural networks
why do transformers use attention
when to use multi head attention
where to learn about transformer attention

Output only the queries, one per line.
""",

            ExpansionStrategy.COMPARISONS: f"""
Based on "{keyword}", generate exactly {count} comparison queries that people actually search when choosing between options.

Focus on real decision-making scenarios:
- "... vs ..." - direct comparisons
- "difference between... and..." - understanding distinctions
- "which is better..." - choosing winners
- "... or ..." - alternative selection
- "pros and cons of..." - balanced evaluation

Generate queries that sound like someone researching before making a technical decision.

Example for "adam optimizer":
adam vs sgd which is better
difference between adam and adamw
adam optimizer pros and cons
adam or rmsprop for deep learning
which optimizer is faster adam or momentum

Output only the queries, one per line.
""",

            ExpansionStrategy.IMPLEMENTATION_GUIDES: f"""
Based on "{keyword}", generate exactly {count} practical implementation and tutorial queries that developers actually search.

Focus on getting things done:
- "how to implement..." - step-by-step guides
- "... tutorial" - learning resources
- "... example code" - practical examples
- "... best practices" - optimization tips
- "... in pytorch/tensorflow" - framework-specific

Generate queries that sound like someone trying to solve a real coding problem.

Example for "gradient checkpointing":
how to implement gradient checkpointing
gradient checkpointing tutorial pytorch
gradient checkpointing example code
best practices for gradient checkpointing
gradient checkpointing memory optimization guide

Output only the queries, one per line.
""",

            ExpansionStrategy.ESSENTIAL_TROUBLESHOOTING: f"""
Based on "{keyword}", generate exactly {count} critical troubleshooting queries for common failure scenarios.

Focus ONLY on frequent, high-impact problems (not edge cases):
- Memory errors and OOM issues
- Training instability and divergence
- Performance bottlenecks
- Common configuration mistakes
- Framework-specific gotchas

Generate queries that sound like someone debugging a real issue that's blocking their work.

Example for "transformer training":
transformer training out of memory
transformer model not converging
transformer training slow
fix transformer gradient explosion
transformer training nan loss

Output only the queries, one per line.
"""
        }

        return strategy_prompts.get(strategy, strategy_prompts[ExpansionStrategy.NATURAL_QUESTIONS])

    async def _expand_single_keyword_async(self, keyword: str, strategy: ExpansionStrategy, 
                                          count: int, output_file: str) -> Set[str]:
        """Expand a single keyword using the specified strategy"""
        
        prompt = self._get_expansion_prompt(keyword, strategy, count)
        
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
                    keywords = self._parse_keyword_response(response.text)
                    if len(keywords) > 0:
                        # Save immediately
                        async with self.keyword_lock:
                            new_keywords = keywords - self.expanded_keywords
                            if new_keywords:
                                self.expanded_keywords.update(new_keywords)
                                await self._append_keywords_to_file_async(new_keywords, output_file)
                        return keywords

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
                    raise e

        # Fallback to OpenRouter
        if self.openrouter_keys:
            print(f"    🔄 Trying OpenRouter for: {keyword}")
            
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
                                temperature=0.7
                            ),
                            timeout=30.0
                        )

                        self._record_openrouter_request(model_name)

                        if response and response.choices and len(response.choices) > 0:
                            response_text = response.choices[0].message.content
                            keywords = self._parse_keyword_response(response_text)
                            if len(keywords) > 0:
                                # Save immediately
                                async with self.keyword_lock:
                                    new_keywords = keywords - self.expanded_keywords
                                    if new_keywords:
                                        self.expanded_keywords.update(new_keywords)
                                        await self._append_keywords_to_file_async(new_keywords, output_file)
                                return keywords

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'rate limit' in error_msg or '429' in error_msg:
                            break
                        else:
                            continue

        return set()

    async def _append_keywords_to_file_async(self, keywords: Set[str], filename: str):
        """Thread-safe file writing"""
        if not keywords:
            return
            
        try:
            async with self.file_lock:
                with open(filename, 'a', encoding='utf-8') as f:
                    for keyword in keywords:
                        f.write(keyword + '\n')
        except Exception as e:
            self.logger.error(f"Error appending keywords to file: {e}")

    def _parse_keyword_response(self, response: str) -> Set[str]:
        """Parse LLM response into clean keyword set"""
        lines = response.strip().split('\n')
        keywords = set()
        
        for line in lines:
            keyword = line.strip().lower()
            keyword = self._clean_keyword(keyword)
            
            if self._is_valid_keyword(keyword):
                keywords.add(keyword)
        
        return keywords
    
    def _clean_keyword(self, keyword: str) -> str:
        """Clean and normalize keyword"""
        prefixes = ['- ', '• ', '* ', '"', "'", '`']
        for i in range(1, 51):
            prefixes.append(f'{i}. ')
            prefixes.append(f'{i}) ')
        
        for prefix in prefixes:
            if keyword.startswith(prefix):
                keyword = keyword[len(prefix):]
        
        keyword = keyword.rstrip('.,;:"\'`')
        
        return keyword.strip()
    
    def _is_valid_keyword(self, keyword: str) -> bool:
        """Validate keyword quality"""
        word_count = len(keyword.split())
        if word_count < 1 or word_count > 10:
            return False
        if ':' in keyword or ';' in keyword:
            return False
        return True

    def _load_existing_keywords(self, filename: str) -> Set[str]:
        """Load existing keywords to avoid duplicates"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            return set()
        except Exception as e:
            self.logger.error(f"Error loading existing keywords: {e}")
            return set()

    async def expand_keywords_from_file(self, input_file: str, output_file: str,
                                       strategies: List[ExpansionStrategy] = None,
                                       expansions_per_keyword: int = 3,
                                       max_keywords_to_process: int = None) -> str:
        """
        Main expansion function - reads keywords from file and expands them
        
        Args:
            input_file: File containing seed keywords (one per line)
            output_file: File to save expanded keywords
            strategies: List of expansion strategies to use
            expansions_per_keyword: How many expanded keywords to generate per seed
            max_keywords_to_process: Limit number of seed keywords to process (None = all)
        """
        
        start_time = time.time()
        
        # Default strategies if none provided - balanced mix for quality over quantity
        if strategies is None:
            strategies = [
                ExpansionStrategy.NATURAL_QUESTIONS,      # 40% - conceptual learning
                ExpansionStrategy.IMPLEMENTATION_GUIDES,  # 30% - practical how-to
                ExpansionStrategy.COMPARISONS,            # 20% - decision making
                ExpansionStrategy.ESSENTIAL_TROUBLESHOOTING  # 10% - critical errors only
            ]
        
        print(f"\n=== KEYWORD EXPANSION STARTED ===")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Strategies: {[s.value for s in strategies]}")
        print(f"Expansions per keyword: {expansions_per_keyword}")
        
        # Load seed keywords
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                seed_keywords = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return f"Error: Input file {input_file} not found"
        
        if max_keywords_to_process:
            seed_keywords = seed_keywords[:max_keywords_to_process]
        
        print(f"Loaded {len(seed_keywords)} seed keywords")
        
        # Load existing expanded keywords
        existing_keywords = self._load_existing_keywords(output_file)
        self.expanded_keywords.update(existing_keywords)
        print(f"Existing expanded keywords: {len(existing_keywords)}")
        
        # Create expansion tasks
        total_generated = 0
        
        for keyword_idx, seed_keyword in enumerate(seed_keywords, 1):
            print(f"\n--- Processing seed keyword {keyword_idx}/{len(seed_keywords)}: '{seed_keyword}' ---")
            
            for strategy in strategies:
                print(f"  Strategy: {strategy.value}")
                
                try:
                    expanded = await self._expand_single_keyword_async(
                        seed_keyword, 
                        strategy, 
                        expansions_per_keyword,
                        output_file
                    )
                    
                    new_count = len(expanded - existing_keywords)
                    total_generated += new_count
                    
                    print(f"    ✅ Generated {new_count} new keywords")
                    
                except Exception as e:
                    self.logger.error(f"Error expanding '{seed_keyword}' with {strategy.value}: {e}")
                    print(f"    ❌ Failed: {str(e)}")
                    continue
                
                # Small delay between strategies
                await asyncio.sleep(1)
            
            # Progress update
            progress = keyword_idx / len(seed_keywords) * 100
            print(f"\nOverall progress: {keyword_idx}/{len(seed_keywords)} seeds ({progress:.1f}%)")
            print(f"Total new keywords generated: {total_generated}")
        
        # Final summary
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        print(f"\n=== EXPANSION COMPLETE ===")
        print(f"╔════════════════════════════════════╗")
        print(f"║       KEYWORD EXPANSION RESULTS    ║")
        print(f"╠════════════════════════════════════╣")
        print(f"║ Seed keywords processed:  {str(len(seed_keywords)).ljust(8)} ║")
        print(f"║ Total keywords generated: {str(total_generated).ljust(8)} ║")
        print(f"║ Strategies used:         {str(len(strategies)).ljust(8)} ║")
        print(f"║ Time elapsed:            {f'{int(hours)}h {int(minutes)}m {int(seconds)}s'.ljust(8)} ║")
        print(f"║ Output file:             {output_file.ljust(8)} ║")
        print(f"╚════════════════════════════════════╝")
        
        return (
            f"Processed {len(seed_keywords)} seed keywords\n"
            f"Generated {total_generated} new expanded keywords\n"
            f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"Saved to: {output_file}"
        )


# Main execution functions
async def expand_keywords(input_file: str, output_file: str, api_keys: List[str],
                         model_configs: List[Dict] = None,
                         strategies: List[ExpansionStrategy] = None,
                         expansions_per_keyword: int = 3,
                         max_keywords: int = None,
                         parallel: bool = True):
    """
    Expand keywords from input file using multiple strategies
    """
    if not api_keys:
        raise ValueError("At least one API key is required")
    
    expander = KeywordExpander(
        api_keys=api_keys,
        model_configs=model_configs,
        parallel=parallel
    )
    
    return await expander.expand_keywords_from_file(
        input_file=input_file,
        output_file=output_file,
        strategies=strategies,
        expansions_per_keyword=expansions_per_keyword,
        max_keywords_to_process=max_keywords
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

    async def run_expansion():
        # Example 1: Basic expansion with default strategies (quality-focused)
        result = await expand_keywords(
            input_file='seed-keywords.txt',
            output_file='expanded-keywords-quality.txt',
            api_keys=API_KEYS,
            model_configs=MODEL_CONFIGS,
            expansions_per_keyword=3,  # Quality over quantity
            max_keywords=600  # Process first 10 keywords only for testing
        )
        print(result)

        # Example 2: Custom expansion with specific strategies
        # result = await expand_keywords(
        #     input_file='seed-keywords.txt',
        #     output_file='expanded-keywords-custom.txt',
        #     api_keys=API_KEYS,
        #     model_configs=MODEL_CONFIGS,
        #     strategies=[
        #         ExpansionStrategy.NATURAL_QUESTIONS,
        #         ExpansionStrategy.IMPLEMENTATION_GUIDES
        #     ],
        #     expansions_per_keyword=4,
        #     max_keywords=20
        # )
        # print(result)

    asyncio.run(run_expansion())
