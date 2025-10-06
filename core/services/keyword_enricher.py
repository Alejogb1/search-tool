import csv
import logging
import asyncio
import time
from integrations.google_ads_client import GoogleAdsClient

logger = logging.getLogger(__name__)

class KeywordEnricher:
    def __init__(self, customer_id, output_file='output-keywords.csv'):
        self.google_ads_client = GoogleAdsClient()
        self.customer_id = customer_id
        self.output_file = output_file
        self._initialize_csv()

    def _initialize_csv(self):
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'avg_monthly_searches', 'competition'])

    def expand_keywords_incremental(self, seed_keywords, db, domain_id, timeout_seconds=300, progress_callback=None):
        """
        Expand keywords with incremental database saving and progress tracking.
        Saves each batch immediately after processing for fault tolerance.

        Args:
            seed_keywords: List of seed keywords to expand
            db: Database session for immediate saving
            domain_id: Domain ID for database operations
            timeout_seconds: Maximum time to wait for all batches (default 5 minutes)
            progress_callback: Optional callback function for progress updates
        """
        logger.info(f"Starting incremental keyword expansion for {len(seed_keywords)} seed keywords with {timeout_seconds}s timeout")
        if not seed_keywords:
            logger.warning("No seed keywords provided for expansion")
            return

        from data_access import repository

        batch_size = 20
        total_batches = (len(seed_keywords) + batch_size - 1) // batch_size
        logger.info(f"Processing in {total_batches} batches of size {batch_size}")

        start_time = time.time()
        total_keywords_saved = 0

        for batch_num in range(total_batches):
            # Check timeout before processing each batch
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                logger.warning(f"Timeout reached after {elapsed_time:.1f}s. Processed {batch_num}/{total_batches} batches.")
                break

            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(seed_keywords))
            batch = seed_keywords[start_idx:end_idx]

            logger.info(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch)} keywords (elapsed: {elapsed_time:.1f}s)")

            batch_start_time = time.time()
            try:
                logger.debug(f"Calling Google Ads API for batch {batch_num + 1}")

                # Add per-batch timeout (30 seconds per batch)
                keyword_ideas = self._call_google_ads_with_timeout(
                    customer_id=self.customer_id,
                    keyword_texts=batch,
                    timeout_seconds=30
                )

                batch_elapsed = time.time() - batch_start_time
                logger.debug(f"Batch {batch_num + 1} API call took {batch_elapsed:.1f}s")

                if keyword_ideas:
                    logger.info(f"Batch {batch_num + 1} returned {len(keyword_ideas)} keyword ideas")

                    # Create batch hash for tracking
                    batch_hash = self._create_batch_hash(keyword_ideas)

                    # Save this batch immediately to database with tracking
                    saved_count = repository.update_keywords_with_expanded_data(db, domain_id, batch_hash, keyword_ideas)
                    total_keywords_saved += saved_count

                    # Update CSV file
                    self._append_to_csv(keyword_ideas)

                    logger.info(f"✅ Batch {batch_num + 1} saved: {saved_count} new keywords (Total: {total_keywords_saved})")

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback({
                            'batch': batch_num + 1,
                            'total_batches': total_batches,
                            'keywords_in_batch': len(keyword_ideas),
                            'new_keywords_saved': saved_count,
                            'total_keywords': total_keywords_saved,
                            'elapsed_time': elapsed_time
                        })
                else:
                    logger.warning(f"Batch {batch_num + 1} returned no keyword ideas")

            except Exception as e:
                batch_elapsed = time.time() - batch_start_time
                logger.error(f"Error expanding keywords for batch {batch_num + 1} (took {batch_elapsed:.1f}s): {e}", exc_info=True)
                # Continue with next batch instead of failing completely
                logger.info(f"Continuing with next batch after error in batch {batch_num + 1}")

        total_elapsed = time.time() - start_time
        logger.info(f"Keyword expansion completed in {total_elapsed:.1f}s. Total keywords saved: {total_keywords_saved}")

    def expand_keywords(self, seed_keywords, timeout_seconds=300):
        """
        Legacy method for backward compatibility - collects all results in memory.
        """
        logger.warning("Using legacy expand_keywords method. Consider using expand_keywords_incremental for better performance.")
        if not seed_keywords:
            logger.warning("No seed keywords provided for expansion")
            return []

        all_keyword_ideas = []
        batch_size = 20
        total_batches = (len(seed_keywords) + batch_size - 1) // batch_size
        logger.info(f"Processing in {total_batches} batches of size {batch_size}")

        start_time = time.time()

        for batch_num in range(total_batches):
            # Check timeout before processing each batch
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                logger.warning(f"Timeout reached after {elapsed_time:.1f}s. Processed {batch_num}/{total_batches} batches.")
                break

            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(seed_keywords))
            batch = seed_keywords[start_idx:end_idx]

            logger.info(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch)} keywords (elapsed: {elapsed_time:.1f}s)")

            batch_start_time = time.time()
            try:
                logger.debug(f"Calling Google Ads API for batch {batch_num + 1}")

                # Add per-batch timeout (30 seconds per batch)
                keyword_ideas = self._call_google_ads_with_timeout(
                    customer_id=self.customer_id,
                    keyword_texts=batch,
                    timeout_seconds=30
                )

                batch_elapsed = time.time() - batch_start_time
                logger.debug(f"Batch {batch_num + 1} API call took {batch_elapsed:.1f}s")

                if keyword_ideas:
                    logger.info(f"Batch {batch_num + 1} returned {len(keyword_ideas)} keyword ideas")
                    logger.debug(f"Sample keyword ideas: {[kw.get('text', '') for kw in keyword_ideas[:3]]}")

                    self._append_to_csv(keyword_ideas)
                    all_keyword_ideas.extend(keyword_ideas)
                    logger.debug(f"Total keywords collected so far: {len(all_keyword_ideas)}")
                else:
                    logger.warning(f"Batch {batch_num + 1} returned no keyword ideas")

            except Exception as e:
                batch_elapsed = time.time() - batch_start_time
                logger.error(f"Error expanding keywords for batch {batch_num + 1} (took {batch_elapsed:.1f}s): {e}", exc_info=True)
                # Continue with next batch instead of failing completely
                logger.info(f"Continuing with next batch after error in batch {batch_num + 1}")

        total_elapsed = time.time() - start_time
        logger.info(f"Keyword expansion completed in {total_elapsed:.1f}s. Total keywords collected: {len(all_keyword_ideas)}")
        return all_keyword_ideas

    def _call_google_ads_with_timeout(self, customer_id, keyword_texts, timeout_seconds=30):
        """
        Call Google Ads API with timeout to prevent hanging.
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Google Ads API call timed out after {timeout_seconds} seconds")

        # Set up the timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            # Make the API call
            result = self.google_ads_client.generate_keyword_ideas(
                customer_id=customer_id,
                keyword_texts=keyword_texts
            )
            return result
        finally:
            # Clean up the timeout
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _create_batch_hash(self, keyword_ideas):
        """
        Create a unique hash for a batch of keywords for tracking purposes.
        """
        import hashlib

        # Create a string from sorted keyword texts for consistent hashing
        keyword_texts = sorted([kw.get('text', '') for kw in keyword_ideas])
        batch_string = '|'.join(keyword_texts)

        # Create SHA-256 hash
        return hashlib.sha256(batch_string.encode()).hexdigest()[:16]  # Use first 16 chars for shorter hash

    def _append_to_csv(self, keyword_ideas):
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'avg_monthly_searches', 'competition'])
            for idea in keyword_ideas:
                writer.writerow(idea)

if __name__ == '__main__':
    # This is an example of how to use the KeywordEnricher
    # You should replace 'your-customer-id' with your actual Google Ads customer ID.
    # You can also load it from a config file or environment variable.
    customer_id = "YOUR CUSTOMER ID"  # IMPORTANT: Replace with a valid customer ID
    
    # Read seed keywords from input-keywords.txt
    try:
        with open('input-keywords.txt', 'r') as f:
            initial_keywords = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: input-keywords.txt not found. Please create it with a list of seed keywords.")
        initial_keywords = []

    # Initialize the enricher
    enricher = KeywordEnricher(customer_id, output_file='output-keywords.csv')

    # Expand the initial keywords
    expanded_keywords = enricher.expand_keywords(initial_keywords)

    # Print the results
    if expanded_keywords:
        print("Expanded Keywords:")
        for keyword in expanded_keywords:
            print(
                f"- Text: {keyword['text']}, "
                f"Avg. Monthly Searches: {keyword['avg_monthly_searches']}, "
                f"Competition: {keyword['competition']}"
            )
    else:
        print("No keywords were expanded. Check your customer ID and API credentials.")
