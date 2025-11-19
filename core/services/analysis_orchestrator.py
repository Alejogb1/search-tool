import sys
import os
import csv
import uuid
import logging
import time
from contextlib import contextmanager
from core.services.keyword_enricher import KeywordEnricher
from data_access.database import SessionLocal
from data_access import repository, models

# Configure comprehensive logging to stdout for Render compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

@contextmanager
def get_database_session_with_timeout(timeout_seconds=300):
    """Context manager for database sessions with timeout handling"""
    db = SessionLocal()
    try:
        # Set a timeout for long-running operations
        start_time = time.time()
        yield db
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def execute_with_retry(operation, max_retries=3, delay=1):
    """Execute a database operation with retry logic"""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Operation failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff

def generate_csv_from_database(db, domain_id: int, domain_name: str) -> str:
    """Generate CSV directly from existing database EXPANDED keywords only"""
    logger.info(f"Generating CSV from database for domain {domain_id}")

    # Get ONLY expanded keywords for this domain (exclude seed-only keywords)
    expanded_keywords = db.query(models.Keyword).filter(
        models.Keyword.domain_id == domain_id,
        models.Keyword.keyword_type.in_([models.KeywordType.EXPANDED_ONLY, models.KeywordType.SEED_EXPANDED])
    ).all()

    if not expanded_keywords:
        raise ValueError(f"No expanded keywords found in database for domain {domain_id}")

    logger.info(f"Found {len(expanded_keywords)} expanded keywords in database")

    # Generate CSV filename
    consolidated_csv_filename = f"output-keywords-{domain_name}.csv"
    consolidated_csv_path = os.path.abspath(consolidated_csv_filename)

    # Create consolidated CSV output with ONLY expanded keywords
    logger.info("Creating consolidated CSV from expanded keywords only...")
    with open(consolidated_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['keyword', 'avg_monthly_searches', 'competition'])
        for kw in expanded_keywords:
            writer.writerow([
                kw.text,
                kw.avg_monthly_searches or '',
                kw.competition_level or ''
            ])

    logger.info(f"Successfully created CSV from database with {len(expanded_keywords)} expanded keywords")
    return consolidated_csv_path

def run_keyword_expansion_and_generate_csv(db, domain_id: int, seed_keywords: list, domain_name: str) -> str:
    """Run Google Ads API expansion and generate CSV"""
    logger.info("Running Google Ads API expansion...")

    # Generate consolidated CSV filename
    consolidated_csv_filename = f"output-keywords-{domain_name}.csv"
    consolidated_csv_path = os.path.abspath(consolidated_csv_filename)
    logger.info(f"Consolidated CSV path: {consolidated_csv_path}")

    # Expand keywords
    # Get customer ID from environment variable
    customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID", "dummy_id")
    logger.info(f"Using Google Ads customer ID: {customer_id}")

    logger.info("Initializing KeywordEnricher...")
    enricher = KeywordEnricher(customer_id=customer_id, output_file=consolidated_csv_path)

    logger.info("Starting keyword expansion via Google Ads API...")
    try:
        expanded_keywords = enricher.expand_keywords(seed_keywords)
        logger.info(f"Keyword expansion completed. Got {len(expanded_keywords)} expanded keywords")
    except Exception as expand_error:
        logger.error(f"Keyword expansion failed: {expand_error}", exc_info=True)
        from data_access import repository
        repository.update_domain_status_and_count(db, domain_id, models.JobStatus.FAILED)
        raise

    # Save expanded keywords to database (updates existing seed keywords with data)
    logger.info("Saving expanded keywords to database...")
    from data_access import repository
    repository.update_keywords_with_expanded_data(db, domain_id, expanded_keywords)
    logger.info("Expanded keywords saved to database")

    # Create consolidated CSV output
    logger.info("Creating consolidated CSV output...")
    try:
        with open(consolidated_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['keyword', 'avg_monthly_searches', 'competition'])
            for kw in expanded_keywords:
                writer.writerow([
                    kw.get('text', ''),
                    kw.get('avg_monthly_searches', ''),
                    kw.get('competition', '')
                ])
        logger.info(f"Successfully created consolidated CSV with {len(expanded_keywords)} keywords")
    except Exception as csv_error:
        logger.error(f"Consolidated CSV creation failed: {csv_error}", exc_info=True)
        raise

    return consolidated_csv_path

async def expand_input_keywords(input_file: str = 'input-keywords.txt') -> list:
    """Directly expand keywords from input file via Google Ads API"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input keywords file '{input_file}' not found")

    # Read input keywords
    with open(input_file, 'r') as f:
        initial_keywords = [line.strip() for line in f if line.strip()]

    if not initial_keywords:
        raise ValueError(f"No keywords found in '{input_file}'")

    # Expand keywords directly using KeywordEnricher
    # Get customer ID from environment variable
    customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID", "dummy_id")
    enricher = KeywordEnricher(customer_id=customer_id)
    expanded_keywords = enricher.expand_keywords(initial_keywords)

    return expanded_keywords

async def run_keyword_workflow(domain: str) -> str:
    """Optimized workflow returning raw CSV of expanded keywords with database integration"""
    logger.info(f"Starting keyword workflow for domain: {domain}")

    # Initialize database session
    db = SessionLocal()
    db_domain = None

    try:
        # Create or get domain record
        logger.info("Creating/finding domain record in database...")
        db_domain = repository.create_domain_if_not_exists(db, domain)
        logger.info(f"Domain record ready: ID {db_domain.id}, status {db_domain.status.value}")

        # Extract domain name for file lookup (remove protocol and www)
        domain_name = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('.')[0].upper()
        logger.info(f"Extracted domain name: {domain_name}")

        # Check for existing domain-specific keyword file
        domain_keyword_file = f"input-keywords-{domain_name}-.txt"
        logger.info(f"Checking for domain keyword file: {domain_keyword_file}")

        if os.path.exists(domain_keyword_file):
            # Use existing keywords for this domain
            logger.info(f"Found existing keywords for {domain_name}, using them directly...")
            with open(domain_keyword_file, 'r') as f:
                seed_keywords = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(seed_keywords)} seed keywords from existing file")
            logger.debug(f"Seed keywords: {seed_keywords[:5]}...")  # Log first 5 keywords
        else:
            # Generate new seed keywords from domain
            logger.info(f"No existing keywords found for {domain_name}, generating from domain...")
            from integrations.llm_client import generate_with_multiple_keys
            api_keys_str = os.getenv("GOOGLE_API_KEYS")
            if not api_keys_str:
                raise ValueError("GOOGLE_API_KEYS environment variable not set")
            API_KEYS = [key.strip() for key in api_keys_str.split(',')]
            logger.info(f"Using {len(API_KEYS)} API keys for LLM generation")

            MODEL_CONFIGS = [
                {"name": "gemini-2.5-pro", "rpm": 5},
                {"name": "gemini-2.5-flash", "rpm": 10},
                {"name": "gemini-2.5-flash-lite-preview-06-17", "rpm": 15},
                {"name": "gemini-2.0-flash", "rpm": 15},
                {"name": "gemini-2.0-flash-lite", "rpm": 30},
            ]

            logger.info("Starting LLM keyword generation...")
            seed_keywords = await generate_with_multiple_keys(
                context_source="domain",
                domain_url=domain,
                api_keys=API_KEYS,
                model_configs=MODEL_CONFIGS,
                output_file=domain_keyword_file,
                parallel=False,
                mode="commercial"
            )
            logger.info(f"Generated {len(seed_keywords)} seed keywords from LLM")

        if not seed_keywords:
            logger.error("No keywords available for expansion")
            repository.update_domain_status_and_count(db, db_domain.id, models.JobStatus.FAILED)
            raise ValueError("No keywords available for expansion")

        # Check existing seed keywords in database (threshold: 10,000)
        existing_seed_keywords = db.query(models.Keyword).filter(
            models.Keyword.domain_id == db_domain.id,
            models.Keyword.keyword_type == models.KeywordType.SEED_ONLY
        ).all()

        seed_count = len(existing_seed_keywords)
        logger.info(f"Domain has {seed_count} existing seed keywords in database")

        if seed_count > 10000:  # If already has 10k+ seed keywords, use them
            logger.info(f"Domain has sufficient seed keywords ({seed_count} > 10,000). Using existing seeds.")
            seed_keywords = [kw.text for kw in existing_seed_keywords]
        else:
            # Need more seed keywords, save current batch to database
            if seed_count > 0:
                logger.info(f"Domain has {seed_count} seed keywords, but needs more. Saving additional seeds.")
            else:
                logger.info("Domain has no seed keywords. Saving initial seed keywords to database.")

            # Save seed keywords to database
            repository.create_seed_keywords_for_domain_conditional(db, db_domain.id, seed_keywords)
            logger.info(f"Saved {len(seed_keywords)} seed keywords to database")

        logger.info(f"Proceeding with {len(seed_keywords)} seed keywords for expansion")

        # Check if domain already has substantial expanded keywords (threshold: 100,000)
        expanded_keywords_count = db.query(models.Keyword).filter(
            models.Keyword.domain_id == db_domain.id,
            models.Keyword.keyword_type.in_([models.KeywordType.EXPANDED_ONLY, models.KeywordType.SEED_EXPANDED])
        ).count()

        logger.info(f"Domain has {expanded_keywords_count} expanded keywords in database")

        if expanded_keywords_count > 100000:  # If already has 100K+ expanded keywords
            logger.info(f"Domain already has sufficient expanded keywords ({expanded_keywords_count} > 100,000). Generating CSV directly from database.")
            consolidated_csv_path = generate_csv_from_database(db, db_domain.id, domain_name)
        else:
            logger.info(f"Domain has {expanded_keywords_count} expanded keywords, needs more. Running Google Ads API expansion.")
            consolidated_csv_path = run_keyword_expansion_and_generate_csv(db, db_domain.id, seed_keywords, domain_name)

        # Verify file was created and log its details
        if os.path.exists(consolidated_csv_path):
            file_size = os.path.getsize(consolidated_csv_path)
            logger.info(f"CSV file verified - Path: {consolidated_csv_path}, Size: {file_size} bytes")
        else:
            logger.error(f"CSV file was not created at: {consolidated_csv_path}")
            raise FileNotFoundError(f"Failed to create CSV file at {consolidated_csv_path}")

        # Update domain status and count
        repository.update_domain_status_and_count(db, db_domain.id, models.JobStatus.COMPLETED)
        logger.info(f"Domain status updated to COMPLETED")

        logger.info(f"Workflow completed successfully. Returning consolidated CSV path: {consolidated_csv_path}")
        return consolidated_csv_path

    except Exception as e:
        logger.error(f"Workflow failed with error: {e}", exc_info=True)
        if db_domain:
            repository.update_domain_status_and_count(db, db_domain.id, models.JobStatus.FAILED)
        # Note: CSV file is preserved on error for debugging purposes
        raise
    finally:
        db.close()

# The legacy run_full_analysis method is kept for backward compatibility,
# but it's not directly used by the new API endpoints.
# It also contains FastAPI imports which are not relevant here.
# For clarity, it's commented out. If needed, it should be refactored
# to remove FastAPI dependencies or moved to an API-specific module.

# async def run_full_analysis(self, domain_url: str, customer_id: str):
#     """
#     Orchestrates the full analysis pipeline (Legacy):
#     1. Generate seed keywords.
#     2. Enrich keywords with metrics.
#     3. Cluster keywords.
#     4. Build knowledge graph (future step).
#     """
#     print(f"Starting full analysis for domain: {domain_url}")
#     db_domain = None
#     
#     try:
#         # Generate seed keywords using LLM
#         print("Step 1: Generating seed keywords with LLM...")
#         from integrations.llm_client import generate_seed_keywords
#         seed_keywords = generate_seed_keywords(domain_url)
#         
#         if not seed_keywords:
#             print("No seed keywords found in input-keywords.txt. Aborting analysis.")
#             return
#
#         # Step 2: Enrich Keywords (using the refactored service)
#         print("Step 2: Enriching keywords with search metrics...")
#         enricher = KeywordEnricher(customer_id=customer_id, output_file='output-keywords.csv')
#         enriched_keywords = enricher.expand_keywords(seed_keywords)
#         
#         if not enriched_keywords:
#             print("No keywords were enriched. Check API credentials and customer ID.")
#             # Decide if to proceed with only seed keywords
#         
#         # Store the enriched keywords in the database
#         # This logic will replace the old repository call
#         print("Storing keywords and metrics in the database...")
#         db_domain = repository.create_domain_if_not_exists(self.db, domain_url)
#         repository.create_keywords_for_domain(self.db, db_domain.id, enriched_keywords)
#         print(f"{len(enriched_keywords)} enriched keywords stored for domain {db_domain.id}.")
#
#         # Step 3: Cluster Keywords (Future Implementation)
#         print("Step 3: Clustering keywords (not yet implemented)...")
#         # embeddings = self.keyword_clusterer.generate_embeddings(enriched_keywords)
#         # clusters = self.keyword_clusterer.perform_clustering(embeddings)
#         # repository.store_clusters(self.db, clusters)
#
#         # Step 4: Build Knowledge Graph (Future Implementation)
#         print("Step 4: Building knowledge graph (not yet implemented)...")
#
#         print("Analysis pipeline completed successfully.")
#
#     except Exception as e:
#         print(f"An error occurred during the analysis pipeline: {e}")
#         # Optionally, add more robust error handling and rollback logic
#     finally:
#         print("Closing database session.")
#         self.db.close()
#
# async def main():
#     # This is an example of how to run the orchestrator
#     # In a real application, this would be triggered by an API call.
#     db_session = SessionLocal()
#     customer_id = "ID-123131"  # This should be managed securely (e.g., from config)
#     orchestrator = AnalysisOrchestrator(db_session)
#     await orchestrator.run_full_analysis(domain_url="https://www.getgalaxy.io", customer_id=customer_id)
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
