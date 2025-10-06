import os
import hashlib
from typing import Optional
from integrations.llm_client import EnhancedGeminiKeywordGenerator, Webscraper, APIKeyManager

class SingleURLKeywordGenerator:
    def __init__(self, api_keys: list, output_dir: str = "output-keywords"):
        self.api_keys = api_keys
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.generator = EnhancedGeminiKeywordGenerator(api_keys)
        self.scraper = Webscraper()
        
    def generate_keywords(self, url: str, mode: str = "commercial") -> str:
        """Generate keywords for a single URL with automatically generated unique filename"""
        try:
            # Create filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            output_file = os.path.join(self.output_dir, f"keywords-{url_hash}.csv")
            
            # Scrape and process single URL
            domain_content = self.scraper(url).text
            result = self.generator.generate_30k_keywords(
                domain_url=url,
                output_file=output_file,
                mode=mode,
                context_source="domain"
            )
            
            return f"Generated keywords for {url}\nOutput: {output_file}\n{result}"
            
        except Exception as e:
            return f"Error processing {url}: {str(e)}"

def generate_from_urls(urls: list[str], api_keys: list[str], mode: str = "commercial") -> list[str]:
    """Batch process multiple URLs sequentially"""
    generator = SingleURLKeywordGenerator(api_keys)
    return [generator.generate_kws(url, mode) for url in urls]
