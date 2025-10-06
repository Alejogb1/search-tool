#!/usr/bin/env python3
"""
Keyword Workflow Manager

A robust, production-ready script for managing keyword analysis workflows.
Handles seed keyword generation, expansion via Google Ads API, and database operations
with proper error handling, timeouts, and duplicate prevention.

Features:
- Smart seed keyword management (prevents duplicates)
- Timeout protection for API calls
- Comprehensive error handling and logging
- Database integration with proper transaction management
- Progress tracking and status reporting

Usage:
    python keyword_workflow_manager.py --domain "example.com" --customer-id "1234567890"
"""

import sys
import os
import csv
import time
import signal
import logging
import argparse
from typing import List, Dict, Optional
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_access.database import SessionLocal
from data_access import repository, models
from core.services.keyword_enricher import KeywordEnricher
from integrations.llm_client import generate_with_multiple_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('keyword_workflow.log')
    ]
)
logger = logging.getLogger(__name__)


class KeywordWorkflowManager:
    """
    Manages the complete keyword analysis workflow with robust error handling.
    """

    def __init__(self, customer_id: str, timeout_seconds: int = 600):
        """
        Initialize the workflow manager.

        Args:
            customer_id: Google Ads customer ID
            timeout_seconds: Overall timeout for the workflow (default 10 minutes)
        """
        self.customer_id = customer_id
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.db = None

    def run_workflow(self, domain: str) -> Dict[str, any]:
        """
        Execute the complete keyword analysis workflow.

        Args:
            domain: Target domain for analysis

        Returns:
            Dict containing workflow results and status
        """
        self.start_time = time.time()
        result = {
            'success': False,
            'domain': domain,
            'csv_path': None,
            'keywords_processed': 0,
            'duration_seconds': 0,
            'error': None
        }

        try:
            # Set up timeout for entire workflow
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout_seconds)

            logger.info(f"🚀 Starting keyword workflow for domain: {domain}")

            # Initialize database session
            self.db = SessionLocal()

            # Step 1: Create/find domain record
            logger.info("📝 Step 1: Creating/finding domain record")
            db_domain = repository.create_domain_if_not_exists(self.db, domain)
            logger.info(f"✅ Domain ready: ID {db_domain.id}, status {db_domain.status.value}")

            # Step 2: Get or generate seed keywords
            logger.info("🔍 Step 2: Obtaining seed keywords")
            seed_keywords = self._get_seed_keywords(domain, db_domain.id)

            if not seed_keywords:
                raise ValueError("No seed keywords available for expansion")

            # Step 3: Determine expansion strategy based on domain status
            domain_obj = self.db.query(models.Domain).filter(models.Domain.id == db_domain.id).first()

            if domain_obj and domain_obj.status == models.JobStatus.COMPLETED:
                # Completed domain: expand existing keywords from database
                logger.info("🔄 Domain previously completed - using existing keywords for expansion")
                expansion_keywords = self._get_existing_keywords_for_expansion(db_domain.id)
                logger.info(f"� Found {len(expansion_keywords)} existing keywords to expand")
            else:
                # New/failed domain: save seeds and use them for expansion
                logger.info("💾 New domain - saving seed keywords and using for expansion")
                self._handle_seed_keywords(db_domain.id, seed_keywords)
                expansion_keywords = seed_keywords

            # Step 4: Expand keywords using the appropriate set
            logger.info(f"📈 Step 4: Expanding {len(expansion_keywords)} keywords via Google Ads API")
            csv_path = self._expand_keywords(domain, db_domain.id, expansion_keywords)

            # Step 5: Update domain status
            logger.info("✅ Step 5: Finalizing workflow")
            repository.update_domain_status_and_count(db_domain.id, models.JobStatus.COMPLETED)

            # Calculate duration
            duration = time.time() - self.start_time
            result.update({
                'success': True,
                'csv_path': csv_path,
                'keywords_processed': len(seed_keywords),
                'duration_seconds': round(duration, 2)
            })

            logger.info(f"⏱️  Workflow completed in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - self.start_time
            logger.error(f"❌ Workflow failed after {duration:.1f}s: {e}", exc_info=True)

            # Update domain status to failed if we have a domain
            if self.db and 'db_domain' in locals():
                try:
                    repository.update_domain_status_and_count(db_domain.id, models.JobStatus.FAILED)
                except Exception as db_error:
                    logger.error(f"Failed to update domain status: {db_error}")

            result.update({
                'error': str(e),
                'duration_seconds': round(duration, 2)
            })
            return result

        finally:
            # Clean up
            signal.alarm(0)  # Cancel timeout
            if self.db:
                self.db.close()

    def _get_seed_keywords(self, domain: str, domain_id: int) -> List[str]:
        """
        Get seed keywords either from existing files or generate new ones.
        """
        # Extract domain name for file lookup
        domain_name = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('.')[0].upper()

        # Check for existing domain-specific keyword file
        domain_keyword_file = f"input-keywords-{domain_name}-.txt"

        if os.path.exists(domain_keyword_file):
            logger.info(f"📁 Found existing keyword file: {domain_keyword_file}")
            with open(domain_keyword_file, 'r') as f:
                keywords = [line.strip() for line in f if line.strip()]
            logger.info(f"✅ Loaded {len(keywords)} keywords from file")
            return keywords

        # Generate new keywords if file doesn't exist
        logger.info(f"🤖 Generating new seed keywords for {domain_name}")
        return self._generate_seed_keywords(domain, domain_keyword_file)

    def _generate_seed_keywords(self, domain: str, output_file: str) -> List[str]:
        """
        Generate seed keywords using LLM.
        """
        api_keys_str = os.getenv("GOOGLE_API_KEYS")
        if not api_keys_str:
            raise ValueError("GOOGLE_API_KEYS environment variable not set")

        API_KEYS = [key.strip() for key in api_keys_str.split(',')]
        MODEL_CONFIGS = [
            {"name": "gemini-2.5-pro", "rpm": 5},
            {"name": "gemini-2.5-flash", "rpm": 10},
            {"name": "gemini-2.5-flash-lite-preview-06-17", "rpm": 15},
            {"name": "gemini-2.0-flash", "rpm": 15},
            {"name": "gemini-2.0-flash-lite", "rpm": 30},
        ]

        try:
            keywords = generate_with_multiple_keys(
                context_source="domain",
                domain_url=domain,
                api_keys=API_KEYS,
                model_configs=MODEL_CONFIGS,
                output_file=output_file,
                parallel=False,
                mode="commercial"
            )
            logger.info(f"✅ Generated {len(keywords)} seed keywords")
            return keywords
        except Exception as e:
            logger.error(f"❌ Failed to generate seed keywords: {e}")
            raise

    def _get_existing_keywords_for_expansion(self, domain_id: int) -> List[str]:
        """
        Get existing keywords from database for expansion (for completed domains).
        Returns list of keyword texts.
        """
        keywords = self.db.query(models.Keyword).filter(models.Keyword.domain_id == domain_id).all()
        return [kw.text for kw in keywords]

    def _handle_seed_keywords(self, domain_id: int, seed_keywords: List[str]) -> None:
        """
        Save seed keywords only for domains that haven't been successfully processed before.
        If domain already has keywords from previous successful runs, skip seed saving.
        """
        # Check if domain has been successfully processed before
        domain = self.db.query(models.Domain).filter(models.Domain.id == domain_id).first()

        if domain and domain.status == models.JobStatus.COMPLETED:
            logger.info(f"ℹ️  Domain was previously completed. Skipping seed keyword save.")
            return

        # For new domains or failed domains: save seed keywords
        repository.create_seed_keywords_for_domain_conditional(self.db, domain_id, seed_keywords)
        logger.info(f"✅ Saved seed keywords to database")

    def _expand_keywords(self, domain: str, domain_id: int, expansion_keywords: List[str]) -> str:
        """
        Expand keywords using Google Ads API with incremental saving and progress tracking.
        """
        domain_name = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('.')[0].upper()
        csv_filename = f"output-keywords-{domain_name}.csv"
        csv_path = os.path.abspath(csv_filename)

        logger.info(f"🔄 Starting incremental keyword expansion with real-time saving")

        # Initialize enricher
        enricher = KeywordEnricher(self.customer_id, output_file=csv_path)

        try:
            # Progress tracking callback
            def progress_callback(progress_data):
                logger.info(f"📈 Progress: Batch {progress_data['batch']}/{progress_data['total_batches']} - "
                          f"{progress_data['keywords_in_batch']} keywords saved "
                          f"(Total: {progress_data['total_keywords']})")

            # Use incremental expansion with immediate saving
            enricher.expand_keywords_incremental(
                seed_keywords=expansion_keywords,
                db=self.db,
                domain_id=domain_id,
                timeout_seconds=600,  # 10 minutes for incremental processing
                progress_callback=progress_callback
            )

            # Verify results
            final_count = self.db.query(models.Keyword).filter(models.Keyword.domain_id == domain_id).count()
            logger.info(f"✅ Expansion complete. Domain now has {final_count} total keywords")

            return csv_path

        except Exception as e:
            logger.error(f"❌ Keyword expansion failed: {e}")
            raise

    def _timeout_handler(self, signum, frame):
        """
        Handle workflow timeout.
        """
        duration = time.time() - self.start_time
        logger.error(f"⏰ Workflow timed out after {duration:.1f}s")
        raise TimeoutError(f"Workflow timed out after {self.timeout_seconds} seconds")


def main():
    """
    Main entry point for the keyword workflow manager.
    """
    parser = argparse.ArgumentParser(description="Keyword Workflow Manager")
    parser.add_argument("--domain", required=True, help="Target domain for analysis")
    parser.add_argument("--customer-id", required=True, help="Google Ads customer ID")
    parser.add_argument("--timeout", type=int, default=600, help="Workflow timeout in seconds (default: 600)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate environment
    if not os.getenv("DATABASE_URL"):
        logger.error("❌ DATABASE_URL environment variable not set")
        sys.exit(1)

    if not os.getenv("GOOGLE_API_KEYS"):
        logger.error("❌ GOOGLE_API_KEYS environment variable not set")
        sys.exit(1)

    # Run workflow
    manager = KeywordWorkflowManager(args.customer_id, args.timeout)
    result = manager.run_workflow(args.domain)

    # Print results
    if result['success']:
        print("\n🎉 Workflow completed successfully!")
        print(f"📊 Domain: {result['domain']}")
        print(f"📄 CSV Path: {result['csv_path']}")
        print(f"🔢 Keywords Processed: {result['keywords_processed']}")
        print(f"⏱️  Duration: {result['duration_seconds']:.2f}s")
        sys.exit(0)
    else:
        print(f"\n❌ Workflow failed: {result['error']}")
        print(f"⏱️  Duration: {result['duration_seconds']:.2f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
