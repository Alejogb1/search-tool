import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import random
from typing import List

# Load environment variables from .env file
load_dotenv()

class APIKeyManager:
    """
    Manages a pool of API keys to rotate them and handle rate limits.
    """
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("API key list cannot be empty.")
        self.api_keys = api_keys
        self.current_key_index = 0

    def get_next_key(self):
        """Rotates to the next key in the list."""
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"  - Switching to API key ending in ...{key[-4:]}")
        return key

def generate_synonyms_for_batch(model, prompt_template, keyword_batch):
    """Generates synonym variations for a batch of keywords using the LLM."""
    keywords_str = "\n".join(keyword_batch)
    prompt = prompt_template.format(SEED_KEYWORDS=keywords_str)
    response = model.generate_content(prompt)
    if response.text:
        return {line.strip() for line in response.text.strip().split('\n') if line.strip()}
    return set()

def process_synonyms_in_batches(input_file, output_file, prompt_file, batch_size=20, max_retries=5):
    """
    Reads keywords, processes them in batches with API key rotation and retries,
    and writes the unique set of all keywords to an output file.
    """
    # 1. Configure API Key Manager
    api_keys_str = os.getenv("GOOGLE_API_KEYS")
    if not api_keys_str:
        print("ERROR: GOOGLE_API_KEYS not found in .env file.")
        return
    
    api_keys = [key.strip() for key in api_keys_str.split(',')]
    key_manager = APIKeyManager(api_keys)
    
    # 2. Read the prompt template
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found at '{prompt_file}'")
        return

    # 3. Read seed keywords
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            seed_keywords = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    all_keywords = set(seed_keywords)
    seed_list = list(seed_keywords)
    total_seeds = len(seed_list)
    print(f"Read {total_seeds} unique seed keywords from '{input_file}'.")
    print(f"Processing in batches of {batch_size} with {len(api_keys)} API keys...")

    # 4. Process keywords in batches
    for i in range(0, total_seeds, batch_size):
        batch = seed_list[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_seeds + batch_size - 1)//batch_size}...")
        
        for attempt in range(max_retries):
            try:
                # Get the next API key and configure the client
                current_key = key_manager.get_next_key()
                genai.configure(api_key=current_key)
                model = genai.GenerativeModel('gemini-1.5-pro-latest')

                synonyms = generate_synonyms_for_batch(model, prompt_template, batch)
                
                if synonyms:
                    new_synonyms = synonyms - all_keywords
                    if new_synonyms:
                        print(f"  + Generated {len(new_synonyms)} new unique synonyms.")
                        all_keywords.update(new_synonyms)
                
                # Success, break the retry loop
                time.sleep(1) # Small delay even on success
                break
            
            except Exception as e:
                error_message = str(e)
                if "429" in error_message or "quota" in error_message:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  - Rate limit hit on attempt {attempt + 1}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  - An unexpected error occurred: {e}")
                    # For non-rate-limit errors, maybe wait a bit and retry once
                    time.sleep(5)
            
            if attempt == max_retries - 1:
                print(f"  - Batch failed after {max_retries} attempts. Skipping this batch.")

    print(f"\nGenerated a total of {len(all_keywords)} unique keywords.")

    # 5. Write all keywords to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for keyword in sorted(list(all_keywords)):
                f.write(keyword + '\n')
        print(f"Successfully wrote all keywords to '{output_file}'.")
    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")

if __name__ == "__main__":
    input_filename = "permutated-keywords.txt"
    output_filename = "expanded-keywords.txt"
    prompt_filename = "synonym_prompt_batch.txt"
    process_synonyms_in_batches(input_filename, output_filename, prompt_filename)
